---
title: Hands-on Exercises
sidebar_position: 6
---

# Module 3: The AI-Robot Brain (NVIDIA Isaac™)

## Hands-on Exercises: Isaac Sim and ROS Integration

This section provides practical exercises to reinforce your understanding of NVIDIA Isaac Sim and Isaac ROS integration for advanced perception and AI-driven control.

### Exercise 1: Setting Up Isaac Sim Environment

**Objective**: Install and configure Isaac Sim with basic scene setup.

#### Step 1: Verify Isaac Sim Installation

First, verify that Isaac Sim is properly installed:

```bash
# Check Isaac Sim version
isaac-sim --version

# Verify GPU compatibility
nvidia-smi
```

#### Step 2: Create a Basic Scene Script

Create `isaac_exercises/basic_scene.py`:

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.robots import Robot
import carb

def setup_basic_scene():
    """Set up a basic Isaac Sim scene with ground plane and objects"""

    # Create world instance
    world = World(stage_units_in_meters=1.0)

    # Add default ground plane
    world.scene.add_default_ground_plane()

    # Add lighting
    create_prim(
        prim_path="/World/Light",
        prim_type="DistantLight",
        position=(0, 0, 10),
        attributes={"color": (0.8, 0.8, 0.8), "intensity": 3000}
    )

    # Add a simple cube
    create_prim(
        prim_path="/World/Cube",
        prim_type="Cube",
        position=(1.0, 0.0, 0.5),
        attributes={"size": 0.2},
        color=(0.8, 0.2, 0.2, 1.0)
    )

    # Add a sphere
    create_prim(
        prim_path="/World/Sphere",
        prim_type="Sphere",
        position=(-1.0, 0.5, 0.5),
        attributes={"radius": 0.15},
        color=(0.2, 0.8, 0.2, 1.0)
    )

    # Add a cylinder
    create_prim(
        prim_path="/World/Cylinder",
        prim_type="Cylinder",
        position=(0.0, -1.0, 0.3),
        attributes={"radius": 0.1, "height": 0.6},
        color=(0.2, 0.2, 0.8, 1.0)
    )

    print("Basic scene created successfully!")
    return world

def main():
    """Main function to run the basic scene setup"""
    try:
        world = setup_basic_scene()

        # Reset the world to initialize physics
        world.reset()

        print("Scene is ready. You can now interact with Isaac Sim.")
        print("Press Ctrl+C to exit.")

        # Run simulation for a few steps to see the scene
        for i in range(100):
            world.step(render=True)
            if i % 20 == 0:
                print(f"Simulation step {i} completed")

        world.clear()

    except Exception as e:
        carb.log_error(f"Error setting up scene: {e}")

if __name__ == "__main__":
    main()
```

#### Step 3: Run the Basic Scene

```bash
# Navigate to Isaac Sim directory and run the script
cd ~/isaac-sim
python ../isaac_exercises/basic_scene.py
```

### Exercise 2: Loading and Controlling a Robot in Isaac Sim

**Objective**: Load a robot model and implement basic control.

#### Step 1: Create Robot Control Script

Create `isaac_exercises/robot_control.py`:

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import ArticulationView
import numpy as np
import carb

class RobotController:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.assets_root_path = get_assets_root_path()
        self.robot = None
        self.setup_scene()

    def setup_scene(self):
        """Set up the scene with ground plane and robot"""
        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Add lighting
        from omni.isaac.core.utils.prims import create_prim
        create_prim(
            prim_path="/World/Light",
            prim_type="DistantLight",
            position=(0, 0, 10),
            attributes={"color": (0.8, 0.8, 0.8), "intensity": 3000}
        )

        # Add a simple robot (using a basic differential drive robot as example)
        if self.assets_root_path:
            # Try to load a simple wheeled robot
            robot_path = self.assets_root_path + "/Isaac/Robots/Turtlebot/turtlebot3_differential.usd"
            add_reference_to_stage(robot_path, "/World/Robot")

            # Add robot to world
            self.robot = self.world.scene.add(
                Robot(
                    prim_path="/World/Robot",
                    name="turtlebot_robot",
                    position=[0, 0, 0.1]
                )
            )
        else:
            # Fallback: create a simple robot representation
            create_prim(
                prim_path="/World/Robot",
                prim_type="Cylinder",
                position=(0, 0, 0.2),
                attributes={"radius": 0.15, "height": 0.4},
                color=(0.5, 0.5, 0.5, 1.0)
            )
            print("Using fallback simple robot due to missing assets")

    def move_robot(self, linear_velocity=0.1, angular_velocity=0.0):
        """Move the robot with specified velocities"""
        if self.robot and hasattr(self.robot, 'get_articulation_controller'):
            # For a real robot with joints
            try:
                # Get current joint positions
                joint_positions = self.robot.get_joint_positions()

                # Apply control (simplified example)
                # In a real implementation, you would use the articulation controller
                print(f"Moving robot with linear: {linear_velocity}, angular: {angular_velocity}")
            except Exception as e:
                print(f"Could not control robot: {e}")
        else:
            # For simple representation, just move the prim
            current_pos = self.world.scene.get_object("Robot").get_world_pos()
            new_x = current_pos[0] + linear_velocity * 0.01
            new_y = current_pos[1] + angular_velocity * 0.005  # Simplified turning
            # Update position (this is simplified)
            print(f"Simulated movement to: ({new_x:.3f}, {new_y:.3f})")

    def run_simulation(self, steps=500):
        """Run the simulation with robot control"""
        self.world.reset()

        for i in range(steps):
            # Apply some control pattern
            if i < 100:
                self.move_robot(linear_velocity=0.2, angular_velocity=0.0)  # Move forward
            elif i < 200:
                self.move_robot(linear_velocity=0.0, angular_velocity=0.5)  # Turn right
            elif i < 300:
                self.move_robot(linear_velocity=0.2, angular_velocity=0.0)  # Move forward
            elif i < 400:
                self.move_robot(linear_velocity=0.0, angular_velocity=-0.5)  # Turn left
            else:
                self.move_robot(linear_velocity=0.0, angular_velocity=0.0)  # Stop

            self.world.step(render=True)

            if i % 50 == 0:
                print(f"Simulation step {i} completed")

    def cleanup(self):
        """Clean up the simulation"""
        self.world.clear()

def main():
    """Main function to run robot control exercise"""
    controller = RobotController()

    try:
        print("Starting robot control simulation...")
        controller.run_simulation()
        print("Simulation completed!")
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    except Exception as e:
        carb.log_error(f"Error in robot control: {e}")
    finally:
        controller.cleanup()

if __name__ == "__main__":
    main()
```

#### Step 2: Run the Robot Control Exercise

```bash
# Run the robot control script
python ../isaac_exercises/robot_control.py
```

### Exercise 3: Isaac ROS Integration - Perception Pipeline

**Objective**: Create a ROS 2 node that interfaces with Isaac Sim for perception.

#### Step 1: Create Isaac ROS Perception Package

```bash
cd ~/physical_ai_ws/src
ros2 pkg create --build-type ament_python isaac_ros_exercises --dependencies rclpy sensor_msgs cv_bridge geometry_msgs
```

#### Step 2: Create Perception Node

Create `isaac_ros_exercises/isaac_ros_exercises/perception_node.py`:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import cv2

class IsaacPerceptionNode(Node):
    def __init__(self):
        super().__init__('isaac_perception_node')

        # Create subscribers for Isaac Sim camera data
        self.rgb_subscriber = self.create_subscription(
            Image,
            '/isaac_sim_camera_rgb',
            self.rgb_callback,
            10
        )

        self.depth_subscriber = self.create_subscription(
            Image,
            '/isaac_sim_camera_depth',
            self.depth_callback,
            10
        )

        self.pointcloud_subscriber = self.create_subscription(
            PointCloud2,
            '/isaac_sim_pointcloud',
            self.pointcloud_callback,
            10
        )

        # Create publishers for processed data
        self.object_detection_publisher = self.create_publisher(
            PoseStamped,
            '/detected_object_pose',
            10
        )

        self.processed_image_publisher = self.create_publisher(
            Image,
            '/processed_camera_image',
            10
        )

        self.bridge = CvBridge()
        self.camera_intrinsics = None

        self.get_logger().info('Isaac Perception Node initialized')

    def rgb_callback(self, msg):
        """Process RGB camera data from Isaac Sim"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Perform object detection (simple color-based detection for demo)
            processed_image = self.detect_objects(cv_image)

            # Publish processed image
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding='bgr8')
            processed_msg.header = msg.header
            self.processed_image_publisher.publish(processed_msg)

            # If objects detected, publish their poses
            if hasattr(self, 'detected_objects'):
                for obj_pose in self.detected_objects:
                    pose_msg = PoseStamped()
                    pose_msg.header = msg.header
                    pose_msg.pose = obj_pose
                    self.object_detection_publisher.publish(pose_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {e}')

    def depth_callback(self, msg):
        """Process depth camera data from Isaac Sim"""
        try:
            # Convert depth image to OpenCV
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

            # Process depth data (e.g., for obstacle detection)
            min_depth = np.min(depth_image[np.isfinite(depth_image)])
            max_depth = np.max(depth_image[np.isfinite(depth_image)])

            self.get_logger().info(f'Depth range: {min_depth:.2f} - {max_depth:.2f}m')

            # Example: detect obstacles within 1 meter
            obstacle_mask = (depth_image < 1.0) & (depth_image > 0.1)
            obstacle_count = np.sum(obstacle_mask)

            if obstacle_count > 100:  # Threshold for obstacle detection
                self.get_logger().warn(f'Obstacle detected! {obstacle_count} points within 1m')

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def pointcloud_callback(self, msg):
        """Process point cloud data from Isaac Sim"""
        try:
            # In a real implementation, you would use point_cloud2 functions
            # For this exercise, we'll just log the message info
            self.get_logger().info(f'Received point cloud with {msg.height * msg.width} points')

        except Exception as e:
            self.get_logger().error(f'Error processing point cloud: {e}')

    def detect_objects(self, image):
        """Detect objects in the image using color thresholding"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for red color (for demo purposes)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process contours to find objects
        processed_image = image.copy()
        self.detected_objects = []

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum area threshold
                # Draw contour
                cv2.drawContours(processed_image, [contour], -1, (0, 255, 0), 2)

                # Calculate center of object
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # Create a simple pose (in a real system, you'd use depth data for 3D position)
                    from geometry_msgs.msg import Pose
                    obj_pose = Pose()
                    obj_pose.position.x = cX / image.shape[1]  # Normalize to 0-1
                    obj_pose.position.y = cY / image.shape[0]  # Normalize to 0-1
                    obj_pose.position.z = 0.5  # Assumed depth
                    obj_pose.orientation.w = 1.0

                    self.detected_objects.append(obj_pose)

        return processed_image

def main(args=None):
    rclpy.init(args=args)
    perception_node = IsaacPerceptionNode()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        perception_node.get_logger().info('Perception node stopped by user')
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Update setup.py

Update `isaac_ros_exercises/setup.py`:

```python
from setuptools import find_packages, setup

package_name = 'isaac_ros_exercises'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Isaac ROS Exercises Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = isaac_ros_exercises.perception_node:main',
        ],
    },
)
```

### Exercise 4: Isaac ROS Navigation with VSLAM

**Objective**: Implement a navigation system using Isaac ROS visual odometry.

#### Step 1: Create Navigation Node

Create `isaac_ros_exercises/isaac_ros_exercises/navigation_node.py`:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from sensor_msgs.msg import LaserScan, Image, Imu
from nav_msgs.msg import Odometry, Path
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose
import tf2_geometry_msgs
import numpy as np
from scipy.spatial.distance import euclidean
import heapq

class IsaacNavigationNode(Node):
    def __init__(self):
        super().__init__('isaac_navigation_node')

        # Create subscribers
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/visual_odom',
            self.odom_callback,
            10
        )

        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Publisher for velocity commands
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Publisher for navigation path
        self.path_publisher = self.create_publisher(
            Path,
            '/navigation_path',
            10
        )

        # Publisher for goals
        self.goal_publisher = self.create_publisher(
            PoseStamped,
            '/move_base_simple/goal',
            10
        )

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Navigation state
        self.current_pose = None
        self.current_goal = None
        self.path = []
        self.path_index = 0
        self.latest_imu = None

        # Navigation parameters
        self.linear_speed = 0.3
        self.angular_speed = 0.5
        self.arrival_threshold = 0.3
        self.rotation_threshold = 0.1
        self.safe_distance = 0.5

        # Obstacle avoidance
        self.obstacle_detected = False

        self.get_logger().info('Isaac Navigation Node initialized')

    def odom_callback(self, msg):
        """Process odometry data from Isaac Sim"""
        self.current_pose = msg.pose.pose

        if self.current_goal is not None:
            self.execute_navigation()

    def scan_callback(self, msg):
        """Process laser scan for obstacle detection"""
        # Check for obstacles in front of robot (simplified)
        front_scan = msg.ranges[len(msg.ranges)//2 - 30 : len(msg.ranges)//2 + 30]
        front_distances = [r for r in front_scan if not np.isnan(r) and r > 0]

        if front_distances:
            min_distance = min(front_distances)
            self.obstacle_detected = min_distance < self.safe_distance

            if self.obstacle_detected:
                self.get_logger().warn(f'Obstacle detected at {min_distance:.2f}m')

    def imu_callback(self, msg):
        """Process IMU data"""
        self.latest_imu = msg

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

        # Plan path to goal (simplified - in real implementation use A*)
        self.plan_path_to_goal(x, y)

    def plan_path_to_goal(self, goal_x, goal_y):
        """Plan a simple path to goal (straight line)"""
        if self.current_pose is None:
            return

        start_x = self.current_pose.position.x
        start_y = self.current_pose.position.y

        # Create simple path (in real implementation, use proper path planning)
        num_waypoints = 20
        self.path = []

        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            x = start_x + t * (goal_x - start_x)
            y = start_y + t * (goal_y - start_y)

            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.header.frame_id = 'map'
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0

            self.path.append(pose_stamped)

        self.path_index = 0

        # Publish path
        self.publish_path()

    def publish_path(self):
        """Publish the planned path"""
        if not self.path:
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        path_msg.poses = self.path

        self.path_publisher.publish(path_msg)

    def execute_navigation(self):
        """Execute navigation to current goal"""
        if self.current_goal is None or self.current_pose is None:
            return

        # Check if we've reached the goal
        current_pos = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y
        ])
        goal_pos = np.array([
            self.current_goal.pose.position.x,
            self.current_goal.pose.position.y
        ])

        distance_to_goal = euclidean(current_pos, goal_pos)

        if distance_to_goal < self.arrival_threshold:
            self.get_logger().info('Reached goal!')
            self.stop_robot()
            self.current_goal = None
            return

        # Simple proportional controller
        cmd_vel = Twist()

        if not self.obstacle_detected:
            # Calculate direction to goal
            direction = goal_pos - current_pos
            distance = np.linalg.norm(direction)

            if distance > 0:
                direction = direction / distance

                # Calculate desired orientation
                desired_yaw = np.arctan2(direction[1], direction[0])

                # Get current orientation
                current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)

                # Calculate angle difference
                angle_diff = desired_yaw - current_yaw
                while angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                while angle_diff < -np.pi:
                    angle_diff += 2 * np.pi

                # Set velocities
                cmd_vel.linear.x = min(self.linear_speed, distance * 0.5)
                cmd_vel.angular.z = np.clip(angle_diff * self.angular_speed, -1.0, 1.0)
        else:
            # Obstacle avoidance behavior
            cmd_vel.linear.x = 0.0  # Stop
            cmd_vel.angular.z = 0.5  # Turn right slowly

        # Publish command
        self.cmd_vel_publisher.publish(cmd_vel)

    def get_yaw_from_quaternion(self, quat):
        """Extract yaw from quaternion"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def stop_robot(self):
        """Stop robot movement"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    nav_node = IsaacNavigationNode()

    # Set an example goal after 2 seconds
    def set_example_goal():
        nav_node.set_goal(2.0, 2.0)

    timer = nav_node.create_timer(2.0, set_example_goal)

    try:
        rclpy.spin(nav_node)
    except KeyboardInterrupt:
        nav_node.get_logger().info('Navigation node stopped by user')
    finally:
        nav_node.stop_robot()
        nav_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Exercise 5: Isaac ROS Manipulation

**Objective**: Implement manipulation capabilities using Isaac ROS.

#### Step 1: Create Manipulation Node

Create `isaac_ros_exercises/isaac_ros_exercises/manipulation_node.py`:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import String, Bool
from tf2_ros import TransformListener, Buffer
import numpy as np
import time

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

        # Publisher for manipulation status
        self.status_publisher = self.create_publisher(
            Bool,
            '/manipulation_status',
            10
        )

        # TF listener for end-effector pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Manipulation state
        self.current_joint_positions = {}
        self.joint_names = []
        self.end_effector_frame = 'end_effector'
        self.base_frame = 'base_link'

        # Predefined joint configurations
        self.home_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Example for 6-DOF arm
        self.pre_grasp_position = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.grasp_position = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]

        self.get_logger().info('Isaac Manipulation Node initialized')

    def joint_state_callback(self, msg):
        """Update current joint positions"""
        self.joint_names = msg.name
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.current_joint_positions[name] = msg.position[i]

    def move_to_joint_positions(self, joint_positions, duration=3.0):
        """Move manipulator to specified joint positions"""
        if not self.joint_names:
            self.get_logger().warn('No joint names available, cannot send trajectory')
            return False

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
        point.velocities = [0.0] * len(positions)
        point.accelerations = [0.0] * len(positions)

        # Set time from start
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration - int(duration)) * 1e9)

        trajectory_msg.points.append(point)

        # Publish trajectory
        self.joint_trajectory_publisher.publish(trajectory_msg)

        # Wait for movement to complete (simplified)
        time.sleep(duration)

        return True

    def open_gripper(self):
        """Open the gripper"""
        cmd_msg = String()
        cmd_msg.data = 'open'
        self.gripper_publisher.publish(cmd_msg)
        self.get_logger().info('Gripper opened')

    def close_gripper(self):
        """Close the gripper"""
        cmd_msg = String()
        cmd_msg.data = 'close'
        self.gripper_publisher.publish(cmd_msg)
        self.get_logger().info('Gripper closed')

    def move_to_cartesian_pose(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        """Move end effector to Cartesian pose (simplified)"""
        self.get_logger().info(f'Moving to Cartesian pose: ({x:.2f}, {y:.2f}, {z:.2f})')

        # In a real implementation, you would:
        # 1. Calculate inverse kinematics to get joint angles
        # 2. Call move_to_joint_positions with the calculated angles
        # For this exercise, we'll use predefined positions

        # This is a simplified approach - in real implementation, use IK solver
        if self.joint_names:
            # Example: move to pre-grasp position
            self.move_to_joint_positions(self.pre_grasp_position, duration=2.0)
            time.sleep(2.0)
            return True
        else:
            self.get_logger().warn('No joint names available')
            return False

    def pick_object(self, x, y, z):
        """Execute pick sequence"""
        self.get_logger().info(f'Picking object at ({x}, {y}, {z})')

        # Move to approach position (above object)
        approach_z = z + 0.1
        success = self.move_to_cartesian_pose(x, y, approach_z)
        if not success:
            return False

        # Move down to object
        self.move_to_cartesian_pose(x, y, z)

        # Close gripper
        self.close_gripper()
        time.sleep(1.0)

        # Lift object
        self.move_to_cartesian_pose(x, y, approach_z)

        # Publish success status
        status_msg = Bool()
        status_msg.data = True
        self.status_publisher.publish(status_msg)

        return True

    def place_object(self, x, y, z):
        """Execute place sequence"""
        self.get_logger().info(f'Placing object at ({x}, {y}, {z})')

        # Move to approach position (above place location)
        approach_z = z + 0.1
        success = self.move_to_cartesian_pose(x, y, approach_z)
        if not success:
            return False

        # Move down to place location
        self.move_to_cartesian_pose(x, y, z)

        # Open gripper
        self.open_gripper()
        time.sleep(1.0)

        # Lift gripper
        self.move_to_cartesian_pose(x, y, approach_z)

        # Return to home
        self.move_to_joint_positions(self.home_position)

        # Publish success status
        status_msg = Bool()
        status_msg.data = True
        self.status_publisher.publish(status_msg)

        return True

    def execute_demo_sequence(self):
        """Execute a demonstration manipulation sequence"""
        self.get_logger().info('Starting manipulation demo sequence...')

        # Open gripper
        self.open_gripper()
        time.sleep(1.0)

        # Move to home position
        self.move_to_joint_positions(self.home_position)
        time.sleep(2.0)

        # Pick an object
        if self.pick_object(0.5, 0.0, 0.1):
            self.get_logger().info('Successfully picked object')

            # Place object at new location
            if self.place_object(0.0, 0.5, 0.1):
                self.get_logger().info('Successfully placed object')
            else:
                self.get_logger().error('Failed to place object')
        else:
            self.get_logger().error('Failed to pick object')

        # Final return to home
        self.move_to_joint_positions(self.home_position)
        self.get_logger().info('Demo sequence completed')

def main(args=None):
    rclpy.init(args=args)
    manipulation_node = IsaacManipulationNode()

    # Execute demo sequence after 3 seconds
    def run_demo():
        manipulation_node.execute_demo_sequence()

    timer = manipulation_node.create_timer(3.0, run_demo)

    try:
        rclpy.spin(manipulation_node)
    except KeyboardInterrupt:
        manipulation_node.get_logger().info('Manipulation node stopped by user')
    finally:
        manipulation_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Exercise 6: Complete Isaac ROS System Integration

**Objective**: Create a launch file that integrates all Isaac ROS components.

Create `isaac_ros_exercises/launch/complete_isaac_system.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Perception node
    perception_node = Node(
        package='isaac_ros_exercises',
        executable='perception_node',
        name='isaac_perception_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Navigation node
    navigation_node = Node(
        package='isaac_ros_exercises',
        executable='navigation_node',
        name='isaac_navigation_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Manipulation node
    manipulation_node = Node(
        package='isaac_ros_exercises',
        executable='manipulation_node',
        name='isaac_manipulation_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Robot state publisher (if needed)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen'
    )

    # Create launch description
    ld = LaunchDescription()

    # Add all nodes
    ld.add_action(robot_state_publisher)
    ld.add_action(perception_node)
    ld.add_action(navigation_node)
    ld.add_action(manipulation_node)

    return ld
```

### Exercise 7: Testing and Validation

**Objective**: Create a validation script to test the complete system.

Create `isaac_ros_exercises/test_system.py`:

```python
#!/usr/bin/env python3
"""
System validation script for Isaac ROS exercises
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
import time

class SystemValidator(Node):
    def __init__(self):
        super().__init__('system_validator')

        # Subscribers to validate system components
        self.status_subscribers = {}
        self.system_status = {
            'perception': False,
            'navigation': False,
            'manipulation': False
        }

        # Publishers for testing
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Test subscribers
        self.test_subscribers = {
            'image': self.create_subscription(Image, '/isaac_sim_camera_rgb',
                                            lambda msg: self.update_status('perception', True), 10),
            'scan': self.create_subscription(LaserScan, '/scan',
                                           lambda msg: self.update_status('navigation', True), 10),
            'status': self.create_subscription(Bool, '/manipulation_status',
                                             lambda msg: self.update_status('manipulation', msg.data), 10)
        }

        # Timer for validation
        self.timer = self.create_timer(1.0, self.check_system_status)
        self.start_time = self.get_clock().now().nanoseconds / 1e9

        self.get_logger().info('System validator started')

    def update_status(self, component, status):
        """Update status of a system component"""
        self.system_status[component] = status
        self.get_logger().info(f'{component} status: {status}')

    def check_system_status(self):
        """Check overall system status"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        elapsed = current_time - self.start_time

        # Check if all components are active
        all_active = all(self.system_status.values())

        if all_active:
            self.get_logger().info('✓ All system components are active!')
            # Send a simple movement command to test integration
            cmd = Twist()
            cmd.linear.x = 0.1
            cmd.angular.z = 0.0
            self.cmd_vel_publisher.publish(cmd)
        else:
            active_components = [k for k, v in self.system_status.items() if v]
            inactive_components = [k for k, v in self.system_status.items() if not v]
            self.get_logger().info(f'Active: {active_components}, Inactive: {inactive_components}')

        if elapsed > 30:  # Stop after 30 seconds
            self.print_final_report()
            self.destroy_node()

    def print_final_report(self):
        """Print final validation report"""
        self.get_logger().info('=== System Validation Report ===')
        for component, status in self.system_status.items():
            status_str = '✓ PASS' if status else '✗ FAIL'
            self.get_logger().info(f'{component}: {status_str}')
        self.get_logger().info('===============================')

def main(args=None):
    rclpy.init(args=args)
    validator = SystemValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        validator.print_final_report()
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Exercise Completion Checklist

Complete the following to master Isaac Sim and ROS integration:

- [ ] Set up Isaac Sim environment with basic scene
- [ ] Load and control a robot in Isaac Sim
- [ ] Create Isaac ROS perception pipeline
- [ ] Implement navigation system with VSLAM
- [ ] Create manipulation system with Isaac ROS
- [ ] Integrate all components in a launch file
- [ ] Test and validate the complete system
- [ ] Understand sim-to-real transfer concepts

These exercises provide hands-on experience with NVIDIA Isaac Sim and Isaac ROS, preparing you for advanced AI-driven robotic applications.