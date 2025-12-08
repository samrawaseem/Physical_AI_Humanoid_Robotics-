---
title: Visual SLAM Navigation
sidebar_position: 4
---

# Module 3: The AI-Robot Brain (NVIDIA Isaac™)

## Visual SLAM for Autonomous Navigation

Visual Simultaneous Localization and Mapping (VSLAM) enables robots to build maps of their environment while simultaneously localizing themselves within those maps using visual sensors. This technology is crucial for autonomous navigation in unknown environments.

### Understanding Visual SLAM

Visual SLAM combines computer vision and robotics to solve two fundamental problems:
- **Mapping**: Creating a representation of the environment
- **Localization**: Determining the robot's position within that environment

The key components of VSLAM include:
- **Feature Detection and Matching**: Identifying and tracking visual features
- **Pose Estimation**: Calculating camera/robot pose from feature correspondences
- **Map Building**: Constructing a consistent map from multiple observations
- **Loop Closure**: Recognizing previously visited locations to correct drift

### VSLAM Approaches

#### Feature-Based VSLAM

Feature-based methods extract and track distinctive visual features:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

class FeatureBasedVSLAM(Node):
    def __init__(self):
        super().__init__('feature_based_vslam')

        # Create subscribers
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        # Publishers
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            '/vslam_pose',
            10
        )

        self.map_publisher = self.create_publisher(
            PointStamped,
            '/vslam_map_point',
            10
        )

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        # VSLAM state
        self.keyframes = []
        self.map_points = []
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.previous_image = None
        self.previous_keypoints = None
        self.orb = cv2.ORB_create(nfeatures=1000)

        # Feature matching
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def camera_info_callback(self, msg):
        """Store camera calibration parameters"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Process incoming images for VSLAM"""
        if self.camera_matrix is None:
            return

        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Detect keypoints and descriptors
        keypoints = self.orb.detect(cv_image, None)
        keypoints, descriptors = self.orb.compute(cv_image, keypoints)

        if self.previous_keypoints is not None and descriptors is not None:
            # Match features between current and previous frames
            matches = self.bf_matcher.knnMatch(
                self.previous_descriptors, descriptors, k=2
            )

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) >= 10:  # Minimum matches required
                # Extract matched points
                prev_pts = np.float32([
                    self.previous_keypoints[m.queryIdx].pt for m in good_matches
                ]).reshape(-1, 1, 2)
                curr_pts = np.float32([
                    keypoints[m.trainIdx].pt for m in good_matches
                ]).reshape(-1, 1, 2)

                # Estimate essential matrix
                E, mask = cv2.findEssentialMat(
                    curr_pts, prev_pts,
                    self.camera_matrix,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0
                )

                if E is not None:
                    # Recover pose
                    _, R, t, _ = cv2.recoverPose(
                        E, curr_pts, prev_pts, self.camera_matrix
                    )

                    # Create transformation matrix
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = t.flatten()

                    # Update current pose
                    self.current_pose = self.current_pose @ T

                    # Check if this frame should be a keyframe
                    if self.is_keyframe(prev_pts, curr_pts, R, t):
                        self.add_keyframe(cv_image, keypoints, descriptors)

                    # Publish current pose
                    self.publish_pose()

        # Update previous frame data
        self.previous_image = cv_image
        self.previous_keypoints = keypoints
        self.previous_descriptors = descriptors

    def is_keyframe(self, prev_pts, curr_pts, R, t):
        """Determine if current frame should be a keyframe"""
        # Criteria for keyframe selection:
        # 1. Sufficient translation/rotation
        translation_norm = np.linalg.norm(t)
        rotation_angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

        # 2. Sufficient number of tracked features
        num_tracked = len(prev_pts)

        return (translation_norm > 0.1 or rotation_angle > 0.1) and num_tracked > 50

    def add_keyframe(self, image, keypoints, descriptors):
        """Add current frame as a keyframe to the map"""
        keyframe = {
            'image': image,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': self.current_pose.copy(),
            'timestamp': self.get_clock().now().nanoseconds
        }
        self.keyframes.append(keyframe)

        # This is where you would add new map points
        # For simplicity, we'll just log the keyframe
        self.get_logger().info(f'Added keyframe {len(self.keyframes)}')

    def publish_pose(self):
        """Publish current estimated pose"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        # Extract position and orientation from transformation matrix
        position = self.current_pose[:3, 3]
        rotation_matrix = self.current_pose[:3, :3]

        # Convert rotation matrix to quaternion
        r = R.from_matrix(rotation_matrix)
        quaternion = r.as_quat()  # [x, y, z, w]

        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]

        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]

        self.pose_publisher.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    vslam_node = FeatureBasedVSLAM()

    try:
        rclpy.spin(vslam_node)
    except KeyboardInterrupt:
        pass
    finally:
        vslam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Direct VSLAM

Direct methods work directly with pixel intensities rather than features:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import cv2

class DirectVSLAM(Node):
    def __init__(self):
        super().__init__('direct_vslam')

        # Create subscriber
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        # Publisher
        self.pose_publisher = self.create_publisher(
            PoseStamped,
            '/direct_vslam_pose',
            10
        )

        self.bridge = CvBridge()
        self.reference_frame = None
        self.current_pose = np.eye(4)
        self.frame_count = 0

        # Photometric consistency parameters
        self.patch_size = 15
        self.max_iterations = 10
        self.convergence_threshold = 1e-4

    def image_callback(self, msg):
        """Process image using direct VSLAM"""
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

        # Convert to float for processing
        gray = cv_image.astype(np.float32)

        if self.reference_frame is None:
            # Initialize reference frame
            self.reference_frame = gray
            self.reference_points = self.select_tracking_points(gray)
            return

        # Track points using direct alignment
        pose_update = self.direct_alignment(
            self.reference_frame, gray, self.reference_points
        )

        if pose_update is not None:
            # Update current pose
            self.current_pose = self.current_pose @ pose_update

            # Publish pose
            self.publish_pose()

            # Update reference frame periodically
            self.frame_count += 1
            if self.frame_count % 10 == 0:  # Update every 10 frames
                self.reference_frame = gray
                self.reference_points = self.select_tracking_points(gray)

    def select_tracking_points(self, image):
        """Select points for tracking using Shi-Tomasi corner detection"""
        corners = cv2.goodFeaturesToTrack(
            image,
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=3
        )

        if corners is not None:
            points = corners.reshape(-1, 2)
            # Filter points to ensure they have valid neighborhoods
            valid_points = []
            for pt in points:
                x, y = int(pt[0]), int(pt[1])
                if (self.patch_size//2 <= x < image.shape[1] - self.patch_size//2 and
                    self.patch_size//2 <= y < image.shape[0] - self.patch_size//2):
                    valid_points.append([x, y])
            return np.array(valid_points, dtype=np.float32)
        return np.array([])

    def direct_alignment(self, ref_img, curr_img, points):
        """Perform direct image alignment to estimate pose"""
        if len(points) == 0:
            return None

        # Initialize pose update (6 DOF: 3 translation, 3 rotation)
        pose_update = np.zeros(6)

        for iteration in range(self.max_iterations):
            # Compute Jacobian and error
            J, error = self.compute_jacobian_and_error(
                ref_img, curr_img, points, pose_update
            )

            if len(error) == 0:
                break

            # Solve for pose update using Gauss-Newton
            try:
                H = J.T @ J
                b = J.T @ error
                delta_pose = np.linalg.solve(H, b)

                # Update pose
                pose_update += delta_pose

                # Check for convergence
                if np.linalg.norm(delta_pose) < self.convergence_threshold:
                    break
            except np.linalg.LinAlgError:
                # Matrix is singular, skip this iteration
                break

        # Convert pose update to transformation matrix
        T = self.pose_update_to_transform(pose_update)
        return T

    def compute_jacobian_and_error(self, ref_img, curr_img, points, pose_update):
        """Compute Jacobian matrix and error vector"""
        J = []
        error = []

        for pt in points:
            x, y = int(pt[0]), int(pt[1])

            # Get patch from reference image
            ref_patch = ref_img[
                y-self.patch_size//2:y+self.patch_size//2+1,
                x-self.patch_size//2:x+self.patch_size//2+1
            ]

            if ref_patch.shape[0] != self.patch_size or ref_patch.shape[1] != self.patch_size:
                continue

            # Compute image gradients
            grad_x = cv2.Sobel(ref_patch, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(ref_patch, cv2.CV_32F, 0, 1, ksize=3)

            # For each pixel in the patch, compute Jacobian
            for dy in range(-self.patch_size//2, self.patch_size//2+1):
                for dx in range(-self.patch_size//2, self.patch_size//2+1):
                    px, py = x + dx, y + dy

                    # Get gradient at this point
                    if 0 <= px < curr_img.shape[1] and 0 <= py < curr_img.shape[0]:
                        g_x = grad_x[dy + self.patch_size//2, dx + self.patch_size//2]
                        g_y = grad_y[dy + self.patch_size//2, dx + self.patch_size//2]

                        # Compute Jacobian of projection w.r.t. pose
                        # This is a simplified version - full implementation would be more complex
                        J_row = np.array([
                            g_x, g_y, 0,  # translation derivatives
                            0, 0, 0       # rotation derivatives (simplified)
                        ])

                        # Compute photometric error
                        ref_val = ref_patch[dy + self.patch_size//2, dx + self.patch_size//2]
                        curr_val = curr_img[py, px]
                        err = ref_val - curr_val

                        J.append(J_row)
                        error.append(err)

        return np.array(J), np.array(error)

    def pose_update_to_transform(self, pose_update):
        """Convert 6DOF pose update to 4x4 transformation matrix"""
        dx, dy, dz, rx, ry, rz = pose_update

        # Create skew-symmetric matrix for rotation
        rot_skew = np.array([
            [0, -rz, ry],
            [rz, 0, -rx],
            [-ry, rx, 0]
        ])

        # Rotation matrix using first order approximation
        R = np.eye(3) + rot_skew
        # Orthonormalize rotation matrix
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt

        # Translation vector
        t = np.array([dx, dy, dz])

        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def publish_pose(self):
        """Publish current estimated pose"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        # Extract position and orientation from transformation matrix
        position = self.current_pose[:3, 3]
        rotation_matrix = self.current_pose[:3, :3]

        # Convert rotation matrix to quaternion
        from scipy.spatial.transform import Rotation as R
        r = R.from_matrix(rotation_matrix)
        quaternion = r.as_quat()  # [x, y, z, w]

        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]

        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]

        self.pose_publisher.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    direct_vslam_node = DirectVSLAM()

    try:
        rclpy.spin(direct_vslam_node)
    except KeyboardInterrupt:
        pass
    finally:
        direct_vslam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Loop Closure Detection

Loop closure is essential for correcting drift in VSLAM systems:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

class LoopClosureDetector(Node):
    def __init__(self):
        super().__init__('loop_closure_detector')

        # Subscribe to poses from VSLAM
        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            '/vslam_pose',
            self.pose_callback,
            10
        )

        # Publisher for loop closure events
        self.loop_closure_publisher = self.create_publisher(
            Bool,
            '/loop_closure_detected',
            10
        )

        # Store poses and descriptors for loop closure detection
        self.poses = []
        self.descriptors = []
        self.positions = []
        self.timestamps = []

        # Loop closure parameters
        self.min_loop_candidates = 10
        self.position_threshold = 2.0  # meters
        self.time_threshold = 10.0e9  # nanoseconds (10 seconds)
        self.descriptor_threshold = 0.5  # similarity threshold

    def pose_callback(self, msg):
        """Process incoming poses for loop closure detection"""
        # Store current pose
        position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        self.positions.append(position)
        self.poses.append(msg)
        self.timestamps.append(msg.header.stamp.nanosec)

        # Generate simple descriptor (in practice, this would be more sophisticated)
        descriptor = self.generate_descriptor(position)
        self.descriptors.append(descriptor)

        # Check for potential loop closures
        if len(self.positions) >= self.min_loop_candidates:
            self.check_for_loop_closure()

    def generate_descriptor(self, position):
        """Generate a simple descriptor for the current location"""
        # In practice, this would use visual features, but for simplicity:
        # Create a descriptor based on position and local geometric context
        descriptor = np.zeros(64)  # 64-dimensional descriptor

        # Encode position information
        descriptor[0:3] = position / 10.0  # Normalize position

        # Add some randomness to make descriptors different
        descriptor[3:64] = np.random.rand(61)

        return descriptor

    def check_for_loop_closure(self):
        """Check if current location matches a previous location"""
        if len(self.positions) < self.min_loop_candidates:
            return

        current_pos = self.positions[-1]
        current_time = self.timestamps[-1]
        current_desc = self.descriptors[-1]

        # Find potential matches based on position proximity
        position_distances = cdist([current_pos], self.positions[:-10])[0]  # Skip recent poses
        potential_matches = np.where(position_distances < self.position_threshold)[0]

        # Check for loop closure among potential matches
        for idx in potential_matches:
            # Check time separation (avoid immediate revisits)
            time_diff = abs(current_time - self.timestamps[idx])
            if time_diff < self.time_threshold:
                continue

            # Check descriptor similarity
            desc_similarity = self.descriptor_similarity(
                current_desc, self.descriptors[idx]
            )

            if desc_similarity > self.descriptor_threshold:
                # Loop closure detected!
                self.get_logger().info(f'Loop closure detected! Match with pose {idx}')

                # Publish loop closure event
                loop_msg = Bool()
                loop_msg.data = True
                self.loop_closure_publisher.publish(loop_msg)

                # This is where you would trigger optimization
                self.optimize_map(idx)
                return

    def descriptor_similarity(self, desc1, desc2):
        """Calculate similarity between two descriptors"""
        # Use normalized cross-correlation
        norm_desc1 = desc1 / np.linalg.norm(desc1)
        norm_desc2 = desc2 / np.linalg.norm(desc2)
        similarity = np.dot(norm_desc1, norm_desc2)
        return similarity

    def optimize_map(self, loop_idx):
        """Optimize map based on loop closure"""
        # In a real system, this would use graph optimization (e.g., g2o, Ceres)
        # to correct drift by minimizing the error between the current pose
        # and the loop-closed pose
        self.get_logger().info(f'Optimizing map using loop closure with pose {loop_idx}')

def main(args=None):
    rclpy.init(args=args)
    loop_closure_node = LoopClosureDetector()

    try:
        rclpy.spin(loop_closure_node)
    except KeyboardInterrupt:
        pass
    finally:
        loop_closure_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Navigation with VSLAM Maps

#### Path Planning Using VSLAM Maps

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, OccupancyGrid
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import heapq
from scipy.spatial.distance import euclidean

class VSLAMNavigation(Node):
    def __init__(self):
        super().__init__('vslam_navigation')

        # Subscribers
        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            '/vslam_pose',
            self.pose_callback,
            10
        )

        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.path_publisher = self.create_publisher(
            Path,
            '/vslam_path',
            10
        )

        self.goal_publisher = self.create_publisher(
            PoseStamped,
            '/move_base_simple/goal',
            10
        )

        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/vslam_markers',
            10
        )

        # Navigation state
        self.current_pose = None
        self.map_points = []  # VSLAM map points
        self.path = []
        self.current_goal = None
        self.path_index = 0

        # Navigation parameters
        self.linear_speed = 0.3
        self.angular_speed = 0.5
        self.arrival_threshold = 0.3
        self.avoidance_distance = 0.5

    def pose_callback(self, msg):
        """Update current pose from VSLAM"""
        self.current_pose = msg

        # If we have a goal, navigate towards it
        if self.current_goal is not None:
            self.navigate_to_goal()

    def scan_callback(self, msg):
        """Process laser scan for obstacle avoidance"""
        # Check for obstacles in the robot's path
        front_scan = msg.ranges[len(msg.ranges)//2 - 15 : len(msg.ranges)//2 + 15]
        front_distances = [r for r in front_scan if not np.isnan(r) and r > 0]

        if front_distances:
            min_distance = min(front_distances)

            if min_distance < self.avoidance_distance:
                # Emergency stop or obstacle avoidance
                self.emergency_stop()
                self.get_logger().warn(f'Obstacle at {min_distance:.2f}m, executing avoidance')

    def set_goal(self, x, y, z=0.0):
        """Set navigation goal"""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = z
        goal_msg.pose.orientation.w = 1.0

        self.current_goal = goal_msg
        self.get_logger().info(f'Set goal to ({x:.2f}, {y:.2f})')

        # Plan path to goal using VSLAM map
        self.plan_path()

    def plan_path(self):
        """Plan path using VSLAM map points"""
        if self.current_pose is None or self.current_goal is None:
            return

        # In a real system, you would use the VSLAM map to plan a path
        # For simplicity, we'll create a straight line path
        start_pos = np.array([
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y
        ])

        goal_pos = np.array([
            self.current_goal.pose.position.x,
            self.current_goal.pose.position.y
        ])

        # Simple straight-line path (in practice, use A* or other path planning)
        num_waypoints = 20
        path_points = []

        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            point = start_pos + t * (goal_pos - start_pos)

            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0

            path_points.append(pose)

        self.path = path_points
        self.path_index = 0

        # Publish path for visualization
        self.publish_path()

    def navigate_to_goal(self):
        """Execute navigation to goal"""
        if not self.path or self.path_index >= len(self.path):
            return

        # Get current target on path
        target = self.path[self.path_index]
        current_pos = np.array([
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y
        ])
        target_pos = np.array([
            target.pose.position.x,
            target.pose.position.y
        ])

        # Calculate distance to target
        distance = euclidean(current_pos, target_pos)

        if distance < self.arrival_threshold:
            # Reached current waypoint, move to next
            self.path_index += 1
            if self.path_index >= len(self.path):
                # Reached goal
                self.get_logger().info('Reached goal!')
                self.stop_robot()
                return

        # Calculate control commands
        cmd_vel = Twist()

        # Direction to target
        direction = target_pos - current_pos
        distance_to_target = np.linalg.norm(direction)

        if distance_to_target > 0:
            direction = direction / distance_to_target

            # Calculate angle to target
            current_yaw = self.get_yaw_from_pose(self.current_pose.pose)
            target_yaw = np.arctan2(direction[1], direction[0])

            # Calculate angle difference
            angle_diff = target_yaw - current_yaw
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            # Set velocities
            cmd_vel.linear.x = min(self.linear_speed, distance_to_target * 0.5)
            cmd_vel.angular.z = angle_diff * self.angular_speed

        # Publish command
        self.cmd_vel_publisher.publish(cmd_vel)

    def get_yaw_from_pose(self, pose):
        """Extract yaw angle from pose quaternion"""
        quat = pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def publish_path(self):
        """Publish path for visualization"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for pose_stamped in self.path:
            path_msg.poses.append(pose_stamped)

        self.path_publisher.publish(path_msg)

    def emergency_stop(self):
        """Stop robot immediately"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_vel)

    def stop_robot(self):
        """Stop robot movement"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    nav_node = VSLAMNavigation()

    # Example: Set a goal after a delay
    def set_example_goal():
        nav_node.set_goal(2.0, 2.0)

    # Set goal after 2 seconds
    timer = nav_node.create_timer(2.0, set_example_goal)

    try:
        rclpy.spin(nav_node)
    except KeyboardInterrupt:
        pass
    finally:
        nav_node.stop_robot()
        nav_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Performance Considerations

#### Real-time VSLAM Optimization

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import numpy as np
import cv2
from threading import Thread, Lock
from queue import Queue
import time

class OptimizedVSLAM(Node):
    def __init__(self):
        super().__init__('optimized_vslam')

        # Create subscriber
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        # Performance monitoring
        self.fps_publisher = self.create_publisher(
            Float32,
            '/vslam_fps',
            10
        )

        self.bridge = CvBridge()

        # Threading for performance
        self.image_queue = Queue(maxsize=2)  # Only keep latest 2 images
        self.processing_thread = Thread(target=self.process_images, daemon=True)
        self.processing_lock = Lock()
        self.processing_thread.start()

        # Performance monitoring
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0

    def image_callback(self, msg):
        """Receive image and add to processing queue"""
        try:
            # Add to queue (oldest image is dropped if queue is full)
            if not self.image_queue.full():
                self.image_queue.put(msg)
            else:
                # Drop the oldest image and add new one
                try:
                    self.image_queue.get_nowait()
                    self.image_queue.put(msg)
                except:
                    pass  # Queue might be empty in the meantime
        except Exception as e:
            self.get_logger().error(f'Error adding image to queue: {e}')

    def process_images(self):
        """Process images in separate thread"""
        orb = cv2.ORB_create(nfeatures=500)  # Reduced features for speed
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        prev_kp_desc = None

        while rclpy.ok():
            try:
                # Get image from queue
                msg = self.image_queue.get(timeout=0.1)

                # Convert to OpenCV
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                # Process image (simplified VSLAM step)
                start_process_time = time.time()

                # Detect and compute features
                kp = orb.detect(cv_image, None)
                kp, desc = orb.compute(cv_image, kp)

                if prev_kp_desc is not None and desc is not None:
                    prev_kp, prev_desc = prev_kp_desc
                    matches = bf_matcher.knnMatch(prev_desc, desc, k=2)

                    # Process matches (simplified)
                    good_matches = []
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.75 * n.distance:
                                good_matches.append(m)

                    # Update pose estimation (simplified)
                    if len(good_matches) >= 10:
                        # In a real system, this would update the pose
                        pass

                # Store current keypoints and descriptors
                prev_kp_desc = (kp, desc)

                # Update performance metrics
                process_time = time.time() - start_process_time
                self.frame_count += 1

                if self.frame_count % 10 == 0:
                    current_time = time.time()
                    self.fps = 10.0 / (current_time - self.start_time)
                    self.start_time = current_time

                    # Publish FPS
                    fps_msg = Float32()
                    fps_msg.data = self.fps
                    self.fps_publisher.publish(fps_msg)

                    self.get_logger().info(f'VSLAM FPS: {self.fps:.2f}, Process time: {process_time:.3f}s')

            except Exception as e:
                # Queue is empty, continue
                continue

def main(args=None):
    rclpy.init(args=args)
    vslam_node = OptimizedVSLAM()

    try:
        rclpy.spin(vslam_node)
    except KeyboardInterrupt:
        pass
    finally:
        vslam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for VSLAM Implementation

1. **Feature Selection**: Use robust features that are invariant to lighting and viewpoint changes
2. **Real-time Performance**: Optimize algorithms for real-time execution
3. **Drift Correction**: Implement loop closure and global optimization
4. **Sensor Fusion**: Combine visual information with inertial and other sensors
5. **Map Management**: Efficiently store and update map points
6. **Failure Recovery**: Handle tracking failures gracefully
7. **Calibration**: Ensure proper camera calibration for accurate measurements

Visual SLAM is a powerful technology for enabling autonomous navigation in unknown environments, providing both localization and mapping capabilities essential for robotic applications.