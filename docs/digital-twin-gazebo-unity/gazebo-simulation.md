---
title: Gazebo Simulation
sidebar_position: 2
---

# Module 2: The Digital Twin (Gazebo & Unity)

## Gazebo Simulation Environment

Gazebo is a physics-based simulation environment that provides realistic sensor simulation and dynamics. It's widely used in robotics research and development for testing algorithms before deployment on physical robots.

### Gazebo Architecture

Gazebo consists of several key components:

- **Physics Engine**: Provides realistic simulation of rigid body dynamics
- **Sensor Simulation**: Simulates various sensors (cameras, LiDAR, IMU, etc.)
- **Rendering Engine**: Provides visual rendering for simulation
- **Plugin System**: Allows custom functionality to be added
- **Communication Interface**: Provides interfaces to ROS/ROS 2

### Installing and Setting Up Gazebo

Gazebo Garden is the recommended version for ROS 2 Humble:

```bash
# Install Gazebo Garden
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-gazebo-plugins
sudo apt install ros-humble-gazebo-dev

# Verify installation
gz sim --version
```

### Basic Gazebo Commands

```bash
# Launch Gazebo with an empty world
gz sim -r

# Launch with a specific world file
gz sim -r -v 4 empty.sdf

# Launch with verbose output
gz sim -v 4 -r empty.sdf
```

### Creating World Files

World files define the environment in which robots operate. Here's a basic world file example:

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

    <!-- Define a simple box obstacle -->
    <model name="box">
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

    <!-- Define a simple table -->
    <model name="table">
      <pose>0 2 0.5 0 0 0</pose>
      <link name="table_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.5 1.0 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.5 1.0 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>1.833333</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>2.833333</iyy>
            <iyz>0</iyz>
            <izz>3.666667</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

### Integrating Robots with Gazebo

To use your URDF robot in Gazebo, you need to convert it to SDF format or use the gazebo_ros package plugins:

```xml
<!-- Add to your URDF to make it Gazebo-compatible -->
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="my_robot">
  <!-- Your existing URDF content -->

  <!-- Gazebo-specific plugins -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>$(find my_robot_description)/config/my_robot_controllers.yaml</parameters>
    </plugin>
  </gazebo>

  <!-- For each link, you can specify Gazebo-specific properties -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
  </gazebo>
</robot>
```

### Gazebo Plugins for ROS 2

Gazebo supports various plugins that enable ROS 2 integration:

#### Joint State Publisher
```xml
<gazebo>
  <plugin filename="libgazebo_ros_joint_state_publisher.so" name="joint_state_publisher">
    <ros>
      <namespace>/my_robot</namespace>
      <remapping>~/out:=/joint_states</remapping>
    </ros>
    <update_rate>30</update_rate>
    <joint_name>joint1</joint_name>
    <joint_name>joint2</joint_name>
  </plugin>
</gazebo>
```

#### Joint Position Controller
```xml
<gazebo>
  <plugin filename="libgazebo_ros_joint_position.so" name="joint_position_controller">
    <ros>
      <namespace>/my_robot</namespace>
      <remapping>~/commands:=/position_commands</remapping>
    </ros>
    <joint_name>my_joint</joint_name>
    <update_rate>100</update_rate>
    <command_topic>position_commands</command_topic>
    <state_topic>position_state</state_topic>
  </plugin>
</gazebo>
```

### Launching Gazebo with ROS 2

Create a launch file to start Gazebo with your robot:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    world = LaunchConfiguration('world')
    model = LaunchConfiguration('model')

    # Declare launch arguments
    declare_world_cmd = DeclareLaunchArgument(
        'world',
        default_value='empty.sdf',
        description='Choose one of the world files from `/gazebo_ros/worlds`'
    )

    declare_model_cmd = DeclareLaunchArgument(
        'model',
        default_value='my_robot',
        description='Choose the robot model to spawn'
    )

    # Start Gazebo server
    start_gazebo_server_cmd = Node(
        package='gazebo_ros',
        executable='gzserver',
        arguments=[
            '-v', '4',
            PathJoinSubstitution([
                FindPackageShare('my_robot_gazebo'),
                'worlds',
                world
            ])
        ],
        output='screen'
    )

    # Start Gazebo client
    start_gazebo_client_cmd = Node(
        package='gazebo_ros',
        executable='gzclient',
        output='screen'
    )

    # Spawn robot in Gazebo
    spawn_entity_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', model,
            '-topic', 'robot_description',
            '-x', '0', '-y', '0', '-z', '0.5'
        ],
        output='screen'
    )

    # Robot State Publisher
    robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'urdf',
                'my_robot.urdf'
            ])
        }]
    )

    # Create launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_world_cmd)
    ld.add_action(declare_model_cmd)

    # Add nodes
    ld.add_action(start_gazebo_server_cmd)
    ld.add_action(start_gazebo_client_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_entity_cmd)

    return ld
```

### Controlling Robots in Gazebo

To control your robot in Gazebo, you typically use ROS 2 controllers:

```yaml
# config/my_robot_controllers.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    velocity_controller:
      type: velocity_controllers/JointGroupVelocityController

    position_controller:
      type: position_controllers/JointGroupPositionController

velocity_controller:
  ros__parameters:
    joints:
      - joint1
      - joint2
      - joint3

position_controller:
  ros__parameters:
    joints:
      - joint1
      - joint2
      - joint3
```

### Sensor Simulation in Gazebo

Gazebo can simulate various sensors. Here's an example of a camera sensor:

```xml
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="head">
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
      <frame_name>camera_optical_frame</frame_name>
      <hack_baseline>0.07</hack_baseline>
    </plugin>
  </sensor>
</gazebo>
```

### Common Gazebo Issues and Solutions

#### Physics Issues
- **Robot falls through the ground**: Check collision geometries and ensure proper inertial properties
- **Jittery movement**: Adjust physics parameters in the world file (step size, solver parameters)
- **Robot tips over easily**: Verify mass and inertia properties are realistic

#### Performance Issues
- **Slow simulation**: Reduce update rates, simplify collision meshes, or use fewer sensors
- **High CPU usage**: Optimize rendering settings or reduce complexity of visual models

#### ROS 2 Integration Issues
- **Robot doesn't appear**: Verify URDF is correct and robot_description parameter is set
- **Controllers not working**: Check controller configuration and ensure proper plugin setup

### Best Practices

1. **Start Simple**: Begin with basic shapes before adding complex models
2. **Validate URDF**: Always check your URDF before importing to Gazebo
3. **Use Proper Inertial Properties**: Realistic mass and inertia improve simulation stability
4. **Optimize for Performance**: Use simplified collision meshes when possible
5. **Test Incrementally**: Add components one at a time to identify issues quickly

Gazebo provides a powerful simulation environment for testing robotic systems before deploying to real hardware. The next section will cover physics engine basics and how they affect robot simulation.