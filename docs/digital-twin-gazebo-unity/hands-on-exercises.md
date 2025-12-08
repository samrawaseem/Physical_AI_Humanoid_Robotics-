---
title: Hands-on Exercises
sidebar_position: 6
---

# Module 2: The Digital Twin (Gazebo & Unity)

## Hands-on Exercises: Digital Twin Simulation

This section provides practical exercises to reinforce your understanding of digital twin concepts using Gazebo and Unity. Complete these exercises to gain hands-on experience with simulation environments, sensor integration, and visualization.

### Exercise 1: Creating a Basic Robot Model in Gazebo

**Objective**: Create a simple robot model and spawn it in Gazebo.

#### Step 1: Create a Robot Description Package

```bash
# Create robot description package
cd ~/physical_ai_ws/src
ros2 pkg create --build-type ament_cmake my_robot_description --dependencies urdf xacro

# Create directory structure
mkdir -p my_robot_description/urdf
mkdir -p my_robot_description/meshes
mkdir -p my_robot_description/config
```

#### Step 2: Create the Robot URDF

Create `my_robot_description/urdf/my_robot.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="my_robot">

  <!-- Materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>
  <material name="grey">
    <color rgba="0.5 0.5 0.5 1.0"/>
  </material>
  <material name="orange">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>
  <material name="brown">
    <color rgba="0.870588235294 0.811764705882 0.764705882353 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Wheel Links -->
  <joint name="base_to_wheel_fl" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_fl"/>
    <origin xyz="0.15 0.15 0.0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="wheel_fl">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="base_to_wheel_fr" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_fr"/>
    <origin xyz="0.15 -0.15 0.0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="wheel_fr">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="base_to_wheel_bl" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_bl"/>
    <origin xyz="-0.15 0.15 0.0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="wheel_bl">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="base_to_wheel_br" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_br"/>
    <origin xyz="-0.15 -0.15 0.0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="wheel_br">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Camera -->
  <joint name="base_to_camera" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.2 0.0 0.1" rpy="0 0 0"/>
  </joint>

  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Gazebo plugins -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="wheel_fl">
    <material>Gazebo/Black</material>
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
  </gazebo>

  <gazebo reference="wheel_fr">
    <material>Gazebo/Black</material>
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
  </gazebo>

  <gazebo reference="wheel_bl">
    <material>Gazebo/Black</material>
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
  </gazebo>

  <gazebo reference="wheel_br">
    <material>Gazebo/Black</material>
    <mu1>0.8</mu1>
    <mu2>0.8</mu2>
  </gazebo>

  <!-- Camera sensor -->
  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <update_rate>30</update_rate>
      <camera name="head_camera">
        <horizontal_fov>1.089</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <ros>
          <namespace>/my_robot</namespace>
          <remapping>~/image_raw:=/camera/image_raw</remapping>
          <remapping>~/camera_info:=/camera/camera_info</remapping>
        </ros>
        <camera_name>camera</camera_name>
        <frame_name>camera_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

#### Step 3: Create a Gazebo World

Create `my_robot_description/worlds/simple_world.world`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_world">
    <!-- Include a ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include a sun light -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Simple obstacles -->
    <model name="box1">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.2 0.2 1</ambient>
            <diffuse>0.8 0.2 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.166667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.166667</iyy>
            <iyz>0</iyz>
            <izz>0.166667</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <model name="box2">
      <pose>-2 1 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 0.5 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 0.5 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.2 0.8 0.2 1</ambient>
            <diffuse>0.2 0.8 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.041667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.041667</iyy>
            <iyz>0</iyz>
            <izz>0.041667</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

#### Step 4: Create Launch File

Create `my_robot_description/launch/robot_spawn.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    world = LaunchConfiguration('world')
    robot_name = LaunchConfiguration('robot_name')

    # Declare launch arguments
    declare_world_cmd = DeclareLaunchArgument(
        'world',
        default_value='simple_world.world',
        description='Choose one of the world files from `/my_robot_description/worlds`'
    )

    declare_robot_name_cmd = DeclareLaunchArgument(
        'robot_name',
        default_value='my_robot',
        description='Name of the robot'
    )

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'urdf',
                'my_robot.urdf.xacro'
            ])
        }]
    )

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'worlds',
                world
            ])
        }.items()
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', robot_name,
            '-x', '0', '-y', '0', '-z', '0.5'
        ],
        output='screen'
    )

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_world_cmd)
    ld.add_action(declare_robot_name_cmd)

    # Add nodes
    ld.add_action(robot_state_publisher)
    ld.add_action(gazebo)
    ld.add_action(spawn_entity)

    return ld
```

#### Step 5: Build and Test

```bash
# Build the package
cd ~/physical_ai_ws
colcon build --packages-select my_robot_description
source install/setup.bash

# Launch the simulation
ros2 launch my_robot_description robot_spawn.launch.py
```

### Exercise 2: Adding Differential Drive Controller

**Objective**: Add a differential drive controller to enable robot movement.

#### Step 1: Install Controller Packages

```bash
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-ros2-control ros-humble-ros2-controllers
```

#### Step 2: Update URDF with Transmission

Add to your URDF file (`my_robot_description/urdf/my_robot.urdf.xacro`):

```xml
<!-- Add after the robot declaration -->
<xacro:macro name="wheel_transmission" params="name">
  <transmission name="${name}_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="${name}">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    </joint>
    <actuator name="${name}_motor">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>
</xacro:macro>

<!-- Add transmissions for wheels -->
<xacro:wheel_transmission name="base_to_wheel_fl"/>
<xacro:wheel_transmission name="base_to_wheel_fr"/>
<xacro:wheel_transmission name="base_to_wheel_bl"/>
<xacro:wheel_transmission name="base_to_wheel_br"/>

<!-- Add Gazebo ROS 2 Control plugin -->
<gazebo>
  <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
    <parameters>$(find my_robot_description)/config/my_robot_controllers.yaml</parameters>
  </plugin>
</gazebo>
```

#### Step 3: Create Controller Configuration

Create `my_robot_description/config/my_robot_controllers.yaml`:

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    my_robot_velocity_controller:
      type: diff_drive_controller/DiffDriveController

joint_state_broadcaster:
  ros__parameters:
    interface_names:
      - base_to_wheel_fl
      - base_to_wheel_fr
      - base_to_wheel_bl
      - base_to_wheel_br

my_robot_velocity_controller:
  ros__parameters:
    left_wheel_names: ["base_to_wheel_fl", "base_to_wheel_bl"]
    right_wheel_names: ["base_to_wheel_fr", "base_to_wheel_br"]

    wheel_separation: 0.3  # Distance between left and right wheels
    wheel_radius: 0.1      # Wheel radius

    # Topic names
    cmd_vel_topic: "/cmd_vel"
    odom_topic: "/odom"
    publish_rate: 50.0

    # Frame names
    base_frame_id: "base_link"
    odom_frame_id: "odom"

    # Velocity and acceleration limits
    linear.x.has_velocity_limits: true
    linear.x.max_velocity: 1.0
    linear.x.has_acceleration_limits: true
    linear.x.max_acceleration: 2.0

    angular.z.has_velocity_limits: true
    angular.z.max_velocity: 1.5
    angular.z.has_acceleration_limits: true
    angular.z.max_acceleration: 3.0
```

#### Step 4: Create Control Launch File

Create `my_robot_description/launch/robot_control.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    world = LaunchConfiguration('world')
    robot_name = LaunchConfiguration('robot_name')

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'urdf',
                'my_robot.urdf.xacro'
            ])
        }]
    )

    # Gazebo launch
    gazebo = Node(
        package='gazebo_ros',
        executable='gzserver',
        arguments=[
            PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'worlds',
                'simple_world.world'
            ])
        ],
        output='screen'
    )

    gazebo_client = Node(
        package='gazebo_ros',
        executable='gzclient',
        output='screen'
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', robot_name,
            '-x', '0', '-y', '0', '-z', '0.5'
        ],
        output='screen'
    )

    # Load and activate controllers after spawn
    load_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )

    load_diff_drive_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['my_robot_velocity_controller'],
        output='screen'
    )

    # Create launch description
    ld = LaunchDescription()

    # Add nodes
    ld.add_action(robot_state_publisher)
    ld.add_action(gazebo)
    ld.add_action(gazebo_client)
    ld.add_action(spawn_entity)

    # Load controllers after robot is spawned
    ld.add_action(RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_entity,
            on_exit=[load_joint_state_broadcaster],
        )
    ))

    ld.add_action(RegisterEventHandler(
        OnProcessExit(
            target_action=load_joint_state_broadcaster,
            on_exit=[load_diff_drive_controller],
        )
    ))

    return ld
```

### Exercise 3: Sensor Integration and Data Processing

**Objective**: Integrate sensors and create a node to process sensor data.

#### Step 1: Create Sensor Processing Package

```bash
cd ~/physical_ai_ws/src
ros2 pkg create --build-type ament_python sensor_processing --dependencies rclpy sensor_msgs geometry_msgs cv_bridge
```

#### Step 2: Create Sensor Processing Node

Create `sensor_processing/sensor_processing/sensor_processor.py`:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class SensorProcessor(Node):
    def __init__(self):
        super().__init__('sensor_processor')

        # Create subscribers
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/my_robot/scan',
            self.lidar_callback,
            10
        )

        self.camera_subscription = self.create_subscription(
            Image,
            '/my_robot/camera/image_raw',
            self.camera_callback,
            10
        )

        # Create publisher for robot commands
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Create publisher for processed image
        self.processed_image_publisher = self.create_publisher(
            Image,
            '/processed_image',
            10
        )

        self.bridge = CvBridge()
        self.safe_distance = 1.0  # meters

    def lidar_callback(self, msg):
        # Process LiDAR data to detect obstacles
        ranges = np.array(msg.ranges)

        # Filter out invalid ranges
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)

            # Simple obstacle avoidance
            cmd_vel = Twist()

            if min_distance < self.safe_distance:
                # Stop and turn
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.5  # Turn right
            else:
                # Move forward
                cmd_vel.linear.x = 0.3
                cmd_vel.angular.z = 0.0

            self.cmd_vel_publisher.publish(cmd_vel)
            self.get_logger().info(f'Min distance: {min_distance:.2f}m, Command: v={cmd_vel.linear.x}, w={cmd_vel.angular.z}')

    def camera_callback(self, msg):
        # Process camera image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Simple color detection (red objects)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define range for red color
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on image
        result_image = cv_image.copy()
        cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

        # Convert back to ROS image and publish
        processed_msg = self.bridge.cv2_to_imgmsg(result_image, encoding='bgr8')
        processed_msg.header = msg.header
        self.processed_image_publisher.publish(processed_msg)

def main(args=None):
    rclpy.init(args=args)
    sensor_processor = SensorProcessor()

    try:
        rclpy.spin(sensor_processor)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Update setup.py

Update `sensor_processing/setup.py`:

```python
from setuptools import find_packages, setup

package_name = 'sensor_processing'

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
    description='Sensor processing for robot simulation',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_processor = sensor_processing.sensor_processor:main',
        ],
    },
)
```

#### Step 4: Test Sensor Integration

```bash
# Build the package
cd ~/physical_ai_ws
colcon build --packages-select sensor_processing
source install/setup.bash

# Launch robot with control
ros2 launch my_robot_description robot_control.launch.py

# In another terminal, run the sensor processor
ros2 run sensor_processing sensor_processor
```

### Exercise 4: Visualization with RViz

**Objective**: Create an RViz configuration to visualize robot state and sensor data.

Create `my_robot_description/config/robot_visualization.rviz`:

```yaml
Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /RobotModel1
        - /LaserScan1
        - /Image1
        - /TF1
      Splitter Ratio: 0.5
    Tree Height: 617
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Alpha: 1
      Class: rviz_default_plugins/RobotModel
      Collision Enabled: false
      Description File: ""
      Description Source: Topic
      Description Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /robot_description
      Enabled: true
      Links:
        All Links Enabled: true
        Expand Joint Details: false
        Expand Link Details: false
        Expand Tree: false
        Link Tree Style: Links in Alphabetic Order
        base_link:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        camera_link:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        wheel_bl:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        wheel_br:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        wheel_fl:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        wheel_fr:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
      Name: RobotModel
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
    - Alpha: 1
      Autocompute Intensity Bounds: true
      Autocompute Value Bounds:
        Max Value: 10
        Min Value: -10
        Value: true
      Axis: Z
      Channel Name: intensity
      Class: rviz_default_plugins/LaserScan
      Color: 255; 255; 255
      Color Transformer: Intensity
      Decay Time: 0
      Enabled: true
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Max Intensity: 0
      Min Color: 0; 0; 0
      Min Intensity: 0
      Name: LaserScan
      Position Transformer: XYZ
      Queue Size: 10
      Selectable: true
      Size (Pixels): 3
      Size (m): 0.009999999776482582
      Style: Points
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /my_robot/scan
      Use Fixed Frame: true
      Use rainbow: true
      Value: true
    - Class: rviz_default_plugins/Image
      Enabled: true
      Max Value: 1
      Min Value: 0
      Name: Image
      Overlay Alpha: 0.5
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /my_robot/camera/image_raw
      Value: true
    - Class: rviz_default_plugins/TF
      Enabled: true
      Frame Timeout: 15
      Frames:
        All Enabled: true
        base_link:
          Value: true
        camera_link:
          Value: true
        odom:
          Value: true
        wheel_bl:
          Value: true
        wheel_br:
          Value: true
        wheel_fl:
          Value: true
        wheel_fr:
          Value: true
      Marker Scale: 1
      Name: TF
      Show Arrows: true
      Show Axes: true
      Show Names: false
      Tree:
        odom:
          base_link:
            camera_link:
              {}
            wheel_bl:
              {}
            wheel_br:
              {}
            wheel_fl:
              {}
            wheel_fr:
              {}
      Update Interval: 0
      Value: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: odom
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 8.190780639648438
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0.6601600646972656
        Y: 0.14592909812927246
        Z: 0.6918399333953857
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 0.6003982424736023
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz)
      Yaw: 5.6435770988464355
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 846
  Width: 1200
```

### Exercise 5: Unity Integration (Optional)

**Objective**: Set up basic Unity integration with ROS 2.

#### Step 1: Install Unity and ROS-TCP-Connector

1. Install Unity Hub and Unity 2022.3 LTS
2. Create a new 3D project
3. Import ROS-TCP-Connector package from Unity Asset Store or GitHub

#### Step 2: Create Unity Robot Controller

Create a C# script in Unity (Assets/Scripts/RobotController.cs):

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class RobotController : MonoBehaviour
{
    ROSConnection ros;
    public float linearVelocity = 0f;
    public float angularVelocity = 0f;
    public float wheelRadius = 0.1f;
    public float wheelSeparation = 0.3f;

    void Start()
    {
        ros = ROSConnection.instance;
        ros.Subscribe<TwistMsg>("/cmd_vel", CmdVelCallback);
    }

    void CmdVelCallback(TwistMsg cmd)
    {
        linearVelocity = (float)cmd.linear.x;
        angularVelocity = (float)cmd.angular.z;

        // Calculate individual wheel velocities for differential drive
        float leftWheelVel = (linearVelocity - angularVelocity * wheelSeparation / 2) / wheelRadius;
        float rightWheelVel = (linearVelocity + angularVelocity * wheelSeparation / 2) / wheelRadius;

        // Apply rotation to wheels (simplified)
        Transform[] wheels = GetComponentsInChildren<Transform>();
        foreach (Transform wheel in wheels)
        {
            if (wheel.name.Contains("wheel"))
            {
                if (wheel.name.Contains("fl") || wheel.name.Contains("bl"))
                {
                    wheel.Rotate(Vector3.forward, leftWheelVel * Time.deltaTime * Mathf.Rad2Deg);
                }
                else
                {
                    wheel.Rotate(Vector3.forward, rightWheelVel * Time.deltaTime * Mathf.Rad2Deg);
                }
            }
        }

        // Move the robot body
        transform.Translate(Vector3.forward * linearVelocity * Time.deltaTime);
        transform.Rotate(Vector3.up, angularVelocity * Time.deltaTime * Mathf.Rad2Deg);
    }

    void Update()
    {
        // Publish robot position back to ROS
        var poseMsg = new PoseMsg();
        poseMsg.position = new Vector3Msg(
            transform.position.x,
            transform.position.y,
            transform.position.z
        );
        poseMsg.orientation = new QuaternionMsg(
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w
        );

        ros.Send("unity_robot_pose", poseMsg);
    }
}
```

### Exercise Completion Checklist

Complete the following to master digital twin concepts:

- [ ] Created a robot model with URDF and spawned it in Gazebo
- [ ] Added differential drive controller for robot movement
- [ ] Integrated sensors (LiDAR and camera) and processed their data
- [ ] Created an RViz configuration to visualize robot state
- [ ] Implemented sensor-based obstacle avoidance
- [ ] Tested the complete simulation system
- [ ] (Optional) Set up basic Unity integration

These exercises provide hands-on experience with digital twin simulation, preparing you for more advanced robotics applications in the following modules.