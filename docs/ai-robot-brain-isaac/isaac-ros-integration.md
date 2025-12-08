---
title: Isaac ROS Integration
sidebar_position: 3
---

# Module 3: The AI-Robot Brain (NVIDIA Isaac™)

## Isaac ROS Integration for Perception and Navigation

Isaac ROS provides optimized ROS 2 packages for perception, navigation, and manipulation tasks. These packages leverage NVIDIA's hardware acceleration to provide high-performance robotic applications.

### Isaac ROS Architecture

Isaac ROS consists of several key components:

- **Hardware Accelerated Perception Nodes**: GPU-accelerated computer vision and sensor processing
- **Visual-Inertial Odometry (VIO)**: Real-time pose estimation using visual and inertial data
- **3D Perception and Reconstruction**: Point cloud processing and scene understanding
- **Navigation and Manipulation**: AI-driven navigation and manipulation capabilities
- **ROS 2 Bridge**: Seamless integration with standard ROS 2 ecosystem

### Installing Isaac ROS

#### System Requirements
- **GPU**: NVIDIA GPU with CUDA compute capability 6.0+ (RTX series recommended)
- **CUDA**: 11.8 or later
- **OS**: Ubuntu 22.04 LTS
- **ROS 2**: Humble Hawksbill

#### Installation Process

```bash
# Add NVIDIA package repository
wget https://repo.download.nvidia.com/nvidia-ml.gpg.key
sudo gpg --dearmor -o /usr/share/keyrings/nvidia-ml.gpg.key
echo "deb [signed-by=/usr/share/keyrings/nvidia-ml.gpg.key] https://repo.download.nvidia.com/$(lsb_release -cs)/all /" | sudo tee /etc/apt/sources.list.d/nvidia-ml.list

# Update package list
sudo apt update

# Install Isaac ROS packages
sudo apt install ros-humble-isaac-ros-gem ros-humble-isaac-ros-visual-odometry
sudo apt install ros-humble-isaac-ros-pointcloud-utils ros-humble-isaac-ros-cortex
sudo apt install ros-humble-isaac-ros-realsense ros-humble-isaac-ros-message-bridge
```

### Isaac ROS Perception Pipeline

#### RGB-D Processing

Isaac ROS provides accelerated RGB-D processing capabilities:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import numpy as np
import cv2

class IsaacRGBDProcessor(Node):
    def __init__(self):
        super().__init__('isaac_rgbd_processor')

        # Create subscribers for RGB and depth images
        self.rgb_subscriber = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.rgb_callback,
            10
        )

        self.depth_subscriber = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        # Create publisher for processed data
        self.point_publisher = self.create_publisher(
            PointStamped,
            '/detected_object_point',
            10
        )

        self.bridge = CvBridge()
        self.camera_intrinsics = None

    def camera_info_callback(self, msg):
        """Store camera intrinsics for 3D point calculation"""
        self.camera_intrinsics = {
            'fx': msg.k[0],  # Focal length x
            'fy': msg.k[4],  # Focal length y
            'cx': msg.k[2],  # Principal point x
            'cy': msg.k[5]   # Principal point y
        }

    def rgb_callback(self, msg):
        """Process RGB image for object detection"""
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Example: Detect red objects using color thresholding
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define range for red color
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:  # Minimum area threshold
                # Get the center of the contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # Publish the 2D point (will be converted to 3D with depth)
                    point_2d = (cX, cY)
                    self.process_3d_point(point_2d)

    def depth_callback(self, msg):
        """Process depth image for 3D information"""
        # This would typically be called in coordination with RGB callback
        # to get both color and depth information
        pass

    def process_3d_point(self, pixel_coords):
        """Convert 2D pixel coordinates to 3D world coordinates using depth"""
        if self.camera_intrinsics is None:
            return

        # This is a simplified example - in practice, you'd get the depth value
        # at the pixel coordinates from the depth image
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']

        # Example: assume a depth value of 1.0 meter for demonstration
        depth_value = 1.0  # This should come from the depth image
        x = (pixel_coords[0] - cx) * depth_value / fx
        y = (pixel_coords[1] - cy) * depth_value / fy
        z = depth_value

        # Create and publish 3D point
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = 'camera_rgb_optical_frame'
        point_msg.point.x = x
        point_msg.point.y = y
        point_msg.point.z = z

        self.point_publisher.publish(point_msg)
        self.get_logger().info(f'Published 3D point: ({x:.2f}, {y:.2f}, {z:.2f})')

def main(args=None):
    rclpy.init(args=args)
    processor = IsaacRGBDProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Point Cloud Processing

Isaac ROS provides efficient point cloud processing capabilities:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import numpy as np

class IsaacPointCloudProcessor(Node):
    def __init__(self):
        super().__init__('isaac_pointcloud_processor')

        # Subscribe to point cloud
        self.pc_subscriber = self.create_subscription(
            PointCloud2,
            '/camera/depth/points',
            self.pointcloud_callback,
            10
        )

        # Publisher for processed point cloud
        self.filtered_publisher = self.create_publisher(
            PointCloud2,
            '/filtered_points',
            10
        )

    def pointcloud_callback(self, msg):
        """Process incoming point cloud data"""
        # Convert PointCloud2 to list of points
        points = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))

        # Filter points based on distance
        filtered_points = []
        for point in points:
            x, y, z = point
            distance = np.sqrt(x**2 + y**2 + z**2)

            # Only keep points within 0.5 to 3.0 meters
            if 0.5 <= distance <= 3.0:
                filtered_points.append([x, y, z])

        # Create new PointCloud2 message with filtered points
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = msg.header.frame_id

        # Convert back to PointCloud2
        filtered_pc = point_cloud2.create_cloud_xyz32(header, filtered_points)

        # Publish filtered point cloud
        self.filtered_publisher.publish(filtered_pc)

def main(args=None):
    rclpy.init(args=args)
    processor = IsaacPointCloudProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Visual-Inertial Odometry (VIO)

Isaac ROS provides hardware-accelerated VIO for accurate pose estimation:

#### Creating a VIO Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
import numpy as np

class IsaacVIONode(Node):
    def __init__(self):
        super().__init__('isaac_vio_node')

        # Create subscribers for camera and IMU
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Publisher for odometry
        self.odom_publisher = self.create_publisher(
            Odometry,
            '/visual_odom',
            10
        )

        self.bridge = CvBridge()
        self.latest_imu = None
        self.frame_count = 0

        # Initialize pose
        self.position = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # w, x, y, z quaternion

    def image_callback(self, msg):
        """Process image for visual odometry"""
        # Convert ROS image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Extract features using OpenCV (simplified example)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Use ORB feature detector (Isaac ROS would use more advanced methods)
        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        # In a real VIO system, you would track features between frames
        # and integrate with IMU data for pose estimation
        if self.latest_imu is not None:
            self.update_pose_with_imu(self.latest_imu)

        self.frame_count += 1

        # Publish current pose estimate
        self.publish_odometry()

    def imu_callback(self, msg):
        """Process IMU data for inertial measurements"""
        self.latest_imu = msg

    def update_pose_with_imu(self, imu_msg):
        """Update pose using IMU data"""
        # Extract angular velocity and linear acceleration
        angular_vel = np.array([
            imu_msg.angular_velocity.x,
            imu_msg.angular_velocity.y,
            imu_msg.angular_velocity.z
        ])

        linear_acc = np.array([
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z
        ])

        # Integrate angular velocity to get orientation change
        dt = 1.0 / 30.0  # Assuming 30 Hz camera rate

        # Simple integration (in practice, more sophisticated methods are used)
        delta_angle = angular_vel * dt

        # Update orientation quaternion
        dq = self.angle_axis_to_quaternion(delta_angle)
        self.orientation = self.quaternion_multiply(self.orientation, dq)

        # Normalize quaternion
        norm = np.linalg.norm(self.orientation)
        if norm > 0:
            self.orientation /= norm

    def angle_axis_to_quaternion(self, angle_axis):
        """Convert angle-axis representation to quaternion"""
        angle = np.linalg.norm(angle_axis)
        if angle == 0:
            return np.array([1.0, 0.0, 0.0, 0.0])

        axis = angle_axis / angle
        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)

        return np.array([
            np.cos(half_angle),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half
        ])

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([w, x, y, z])

    def publish_odometry(self):
        """Publish current odometry estimate"""
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Set position
        odom_msg.pose.pose.position.x = self.position[0]
        odom_msg.pose.pose.position.y = self.position[1]
        odom_msg.pose.pose.position.z = self.position[2]

        # Set orientation
        odom_msg.pose.pose.orientation.w = self.orientation[0]
        odom_msg.pose.pose.orientation.x = self.orientation[1]
        odom_msg.pose.pose.orientation.y = self.orientation[2]
        odom_msg.pose.pose.orientation.z = self.orientation[3]

        # Publish odometry
        self.odom_publisher.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    vio_node = IsaacVIONode()

    try:
        rclpy.spin(vio_node)
    except KeyboardInterrupt:
        pass
    finally:
        vio_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac ROS Navigation

#### Setting up Navigation Stack

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, OccupancyGrid
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
import numpy as np

class IsaacNavigationNode(Node):
    def __init__(self):
        super().__init__('isaac_navigation_node')

        # Create subscribers
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # Publisher for velocity commands
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Publisher for navigation goals
        self.goal_publisher = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.current_goal = None
        self.path = None
        self.safe_distance = 0.5  # meters

    def scan_callback(self, msg):
        """Process laser scan for obstacle detection"""
        # Check for obstacles in front of robot
        front_scan = msg.ranges[len(msg.ranges)//2 - 30 : len(msg.ranges)//2 + 30]
        front_distances = [r for r in front_scan if not np.isnan(r) and r > 0]

        if front_distances:
            min_distance = min(front_distances)

            if min_distance < self.safe_distance:
                # Emergency stop
                self.stop_robot()
                self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m, stopping robot')
            else:
                # Continue with navigation
                self.execute_navigation()

    def map_callback(self, msg):
        """Process occupancy grid map"""
        # Store map for path planning
        self.map_info = msg.info
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))

    def execute_navigation(self):
        """Execute navigation logic"""
        if self.current_goal is None:
            return

        # Simple proportional controller for demonstration
        cmd_vel = Twist()

        # Get robot's current pose relative to map
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )

            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y

            # Calculate direction to goal
            goal_x = self.current_goal.pose.position.x
            goal_y = self.current_goal.pose.position.y

            dx = goal_x - robot_x
            dy = goal_y - robot_y
            distance = np.sqrt(dx**2 + dy**2)

            if distance > 0.1:  # If not close to goal
                # Set linear velocity proportional to distance
                cmd_vel.linear.x = min(0.3, distance * 0.5)

                # Set angular velocity to face goal
                current_yaw = self.get_yaw_from_quaternion(
                    transform.transform.rotation
                )
                goal_yaw = np.arctan2(dy, dx)
                angle_diff = goal_yaw - current_yaw

                # Normalize angle difference
                while angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                while angle_diff < -np.pi:
                    angle_diff += 2 * np.pi

                cmd_vel.angular.z = angle_diff * 1.0  # Proportional controller

        except Exception as e:
            self.get_logger().error(f'Transform lookup failed: {e}')

        # Publish velocity command
        self.cmd_vel_publisher.publish(cmd_vel)

    def get_yaw_from_quaternion(self, quat):
        """Extract yaw angle from quaternion"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def stop_robot(self):
        """Stop the robot"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_vel)

    def set_goal(self, x, y, z=0.0):
        """Set navigation goal"""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = z
        goal_msg.pose.orientation.w = 1.0  # No rotation

        self.current_goal = goal_msg
        self.goal_publisher.publish(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    nav_node = IsaacNavigationNode()

    # Set an example goal
    nav_node.set_goal(2.0, 2.0)

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

### Isaac ROS Manipulation

#### Manipulation with Isaac ROS

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
import numpy as np

class IsaacManipulationNode(Node):
    def __init__(self):
        super().__init__('isaac_manipulation_node')

        # Publisher for joint trajectory commands
        self.joint_trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Subscriber for joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Publisher for gripper control
        self.gripper_publisher = self.create_publisher(
            String,
            '/gripper_command',
            10
        )

        self.current_joint_positions = {}
        self.joint_names = []  # Will be populated from joint states

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        self.joint_names = msg.name
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]

    def move_to_joint_positions(self, joint_positions, duration=5.0):
        """Move manipulator to specified joint positions"""
        if not self.joint_names:
            self.get_logger().warn('No joint names available, cannot send trajectory')
            return

        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()

        # Set positions
        positions = []
        for joint_name in self.joint_names:
            if joint_name in joint_positions:
                positions.append(joint_positions[joint_name])
            else:
                # Keep current position if not specified
                positions.append(self.current_joint_positions.get(joint_name, 0.0))

        point.positions = positions

        # Set velocities to zero (optional)
        point.velocities = [0.0] * len(positions)

        # Set time from start
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)

        trajectory_msg.points.append(point)

        # Publish trajectory
        self.joint_trajectory_publisher.publish(trajectory_msg)

    def move_to_cartesian_pose(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        """Move end effector to Cartesian pose (simplified)"""
        # This would typically involve inverse kinematics
        # For demonstration, we'll just log the desired pose
        self.get_logger().info(f'Moving to Cartesian pose: ({x:.2f}, {y:.2f}, {z:.2f})')

        # In a real implementation, you would:
        # 1. Calculate inverse kinematics to get joint angles
        # 2. Call move_to_joint_positions with the calculated angles
        # 3. Handle any errors or constraints

    def open_gripper(self):
        """Open the gripper"""
        cmd_msg = String()
        cmd_msg.data = 'open'
        self.gripper_publisher.publish(cmd_msg)

    def close_gripper(self):
        """Close the gripper"""
        cmd_msg = String()
        cmd_msg.data = 'close'
        self.gripper_publisher.publish(cmd_msg)

    def pick_object(self, x, y, z):
        """Execute pick sequence"""
        self.get_logger().info(f'Picking object at ({x}, {y}, {z})')

        # Move to approach position (above object)
        approach_z = z + 0.1
        self.move_to_cartesian_pose(x, y, approach_z)

        # Move down to object
        self.move_to_cartesian_pose(x, y, z)

        # Close gripper
        self.close_gripper()

        # Lift object
        self.move_to_cartesian_pose(x, y, approach_z)

    def place_object(self, x, y, z):
        """Execute place sequence"""
        self.get_logger().info(f'Placing object at ({x}, {y}, {z})')

        # Move to approach position (above place location)
        approach_z = z + 0.1
        self.move_to_cartesian_pose(x, y, approach_z)

        # Move down to place location
        self.move_to_cartesian_pose(x, y, z)

        # Open gripper
        self.open_gripper()

        # Lift gripper
        self.move_to_cartesian_pose(x, y, approach_z)

def main(args=None):
    rclpy.init(args=args)
    manipulation_node = IsaacManipulationNode()

    # Example: Pick and place operation
    # manipulation_node.pick_object(0.5, 0.0, 0.1)
    # manipulation_node.place_object(0.0, 0.5, 0.1)

    try:
        rclpy.spin(manipulation_node)
    except KeyboardInterrupt:
        pass
    finally:
        manipulation_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac ROS Launch Files

#### Complete Isaac ROS System Launch

Create `isaac_ros_examples/launch/complete_isaac_ros_system.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    # Isaac ROS Perception Nodes
    visual_odometry_node = Node(
        package='isaac_ros_visual_odometry',
        executable='dvo_node',
        name='dvo_node',
        parameters=[{
            'use_sim_time': use_sim_time,
            'enable_rectification': True,
            'max_disparity': 64.0,
            'dense_stereo_algorithm': 'BLOCK_MATCHING',
            'output_qos': 'SENSOR_DATA'
        }],
        remappings=[
            ('left/image_rect', '/camera/rgb/image_raw'),
            ('right/image_rect', '/camera/depth/image_raw'),
            ('left/camera_info', '/camera/rgb/camera_info'),
            ('right/camera_info', '/camera/depth/camera_info'),
            ('visual_odometry', '/visual_odometry')
        ]
    )

    pointcloud_node = Node(
        package='isaac_ros_pointcloud_utils',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        parameters=[{
            'use_sim_time': use_sim_time,
            'target_frame': 'base_link',
            'transform_tolerance': 0.01,
            'min_height': 0.0,
            'max_height': 1.0,
            'angle_min': -1.57,
            'angle_max': 1.57,
            'angle_increment': 0.0087,
            'scan_time': 0.1,
            'range_min': 0.1,
            'range_max': 10.0
        }],
        remappings=[
            ('cloud_in', '/camera/depth/points'),
            ('scan', '/scan')
        ]
    )

    # Navigation nodes
    nav2_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )

    # TF publishers
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'urdf',
                'my_robot.urdf.xacro'
            ])
        }]
    )

    # Create launch description
    ld = LaunchDescription()

    # Add nodes
    ld.add_action(robot_state_publisher)
    ld.add_action(visual_odometry_node)
    ld.add_action(pointcloud_node)
    ld.add_action(nav2_bringup_launch)

    return ld
```

### Performance Optimization with Isaac ROS

#### GPU Acceleration Configuration

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
import torch
import torchvision.transforms as transforms

class IsaacROSGPUAcceleratedNode(Node):
    def __init__(self):
        super().__init__('isaac_ros_gpu_node')

        # Check for GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device = torch.device('cuda')
            self.get_logger().info('GPU acceleration enabled')
        else:
            self.device = torch.device('cpu')
            self.get_logger().info('Using CPU (GPU not available)')

        # Initialize neural network model
        self.initialize_model()

        # Create subscriber for camera images
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.gpu_image_callback,
            10
        )

        # Publisher for inference results
        self.result_publisher = self.create_publisher(
            Int32,
            '/inference_result',
            10
        )

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def initialize_model(self):
        """Initialize deep learning model for GPU acceleration"""
        try:
            # Example: Load a pre-trained model
            self.model = torch.hub.load('pytorch/vision:v0.10.0',
                                      'resnet18',
                                      pretrained=True)
            self.model.eval()

            # Move model to GPU if available
            self.model = self.model.to(self.device)
            self.get_logger().info('Model loaded and moved to device')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize model: {e}')
            self.model = None

    def gpu_image_callback(self, msg):
        """Process image using GPU-accelerated inference"""
        if self.model is None:
            return

        try:
            # Convert ROS image to tensor
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            tensor_image = self.transform(cv_image).unsqueeze(0)

            # Move to device
            tensor_image = tensor_image.to(self.device)

            # Run inference
            with torch.no_grad():
                output = self.model(tensor_image)
                predicted_class = torch.argmax(output, dim=1).item()

            # Publish result
            result_msg = Int32()
            result_msg.data = predicted_class
            self.result_publisher.publish(result_msg)

            self.get_logger().info(f'Inference result: {predicted_class}')

        except Exception as e:
            self.get_logger().error(f'GPU inference failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    gpu_node = IsaacROSGPUAcceleratedNode()

    try:
        rclpy.spin(gpu_node)
    except KeyboardInterrupt:
        pass
    finally:
        gpu_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for Isaac ROS Integration

1. **Hardware Acceleration**: Always leverage GPU acceleration when available
2. **Data Pipeline Optimization**: Minimize data copying between CPU and GPU
3. **Real-time Performance**: Ensure nodes meet timing requirements for real-time operation
4. **Resource Management**: Monitor GPU memory and compute usage
5. **Error Handling**: Implement robust error handling for hardware failures
6. **Calibration**: Properly calibrate sensors for accurate perception

Isaac ROS integration provides powerful capabilities for AI-driven robotics, combining NVIDIA's hardware acceleration with ROS 2's flexibility for building sophisticated robotic applications.