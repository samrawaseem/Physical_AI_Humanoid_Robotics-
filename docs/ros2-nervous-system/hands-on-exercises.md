---
title: Hands-on Exercises
sidebar_position: 6
---

# Module 1: The Robotic Nervous System (ROS 2)

## Hands-on Exercises: ROS 2 Fundamentals

This section provides practical exercises to reinforce your understanding of ROS 2 concepts. Complete these exercises to gain hands-on experience with nodes, topics, services, and URDF.

### Exercise 1: Create a Simple Publisher-Subscriber System

**Objective**: Create a publisher that sends messages and a subscriber that receives them.

#### Step 1: Create a ROS 2 Package
```bash
# Create a new package
ros2 pkg create --build-type ament_python ros2_exercises
cd ros2_exercises
```

#### Step 2: Create the Publisher Node
Create `ros2_exercises/simple_publisher.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SimplePublisher(Node):
    def __init__(self):
        super().__init__('simple_publisher')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    simple_publisher = SimplePublisher()

    try:
        rclpy.spin(simple_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        simple_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Create the Subscriber Node
Create `ros2_exercises/simple_subscriber.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SimpleSubscriber(Node):
    def __init__(self):
        super().__init__('simple_subscriber')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    simple_subscriber = SimpleSubscriber()

    try:
        rclpy.spin(simple_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        simple_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 4: Update setup.py
Add the executables to your `setup.py`:

```python
entry_points={
    'console_scripts': [
        'simple_publisher = ros2_exercises.simple_publisher:main',
        'simple_subscriber = ros2_exercises.simple_subscriber:main',
    ],
},
```

#### Step 5: Build and Test
```bash
# Build the package
cd ~/physical_ai_ws
colcon build --packages-select ros2_exercises

# Source the workspace
source install/setup.bash

# Terminal 1: Run the publisher
ros2 run ros2_exercises simple_publisher

# Terminal 2: Run the subscriber
ros2 run ros2_exercises simple_subscriber
```

### Exercise 2: Create a Service Client-Server System

**Objective**: Implement a service that performs calculations and a client that requests them.

#### Step 1: Create the Service Server
Create `ros2_exercises/calculator_service.py`:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class CalculatorService(Node):
    def __init__(self):
        super().__init__('calculator_service')
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback
        )
        self.get_logger().info('Calculator service is ready')

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Adding {request.a} + {request.b} = {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    calculator_service = CalculatorService()

    try:
        rclpy.spin(calculator_service)
    except KeyboardInterrupt:
        pass
    finally:
        calculator_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 2: Create the Service Client
Create `ros2_exercises/calculator_client.py`:

```python
import sys
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class CalculatorClient(Node):
    def __init__(self):
        super().__init__('calculator_client')
        self.client = self.create_client(AddTwoInts, 'add_two_ints')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.request = AddTwoInts.Request()

    def send_request(self, a, b):
        self.request.a = a
        self.request.b = b
        self.future = self.client.call_async(self.request)
        return self.future

def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) != 3:
        print('Usage: ros2 run ros2_exercises calculator_client <int1> <int2>')
        return

    calculator_client = CalculatorClient()
    a = int(sys.argv[1])
    b = int(sys.argv[2])

    future = calculator_client.send_request(a, b)

    try:
        rclpy.spin_until_future_complete(calculator_client, future)
        response = future.result()
        calculator_client.get_logger().info(
            f'Result of {a} + {b} = {response.sum}'
        )
    except Exception as e:
        calculator_client.get_logger().error(f'Service call failed: {e}')
    finally:
        calculator_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Step 3: Update setup.py
Add the new executables:

```python
entry_points={
    'console_scripts': [
        'simple_publisher = ros2_exercises.simple_publisher:main',
        'simple_subscriber = ros2_exercises.simple_subscriber:main',
        'calculator_service = ros2_exercises.calculator_service:main',
        'calculator_client = ros2_exercises.calculator_client:main',
    ],
},
```

#### Step 4: Test the Service
```bash
# Terminal 1: Run the service
ros2 run ros2_exercises calculator_service

# Terminal 2: Call the service
ros2 run ros2_exercises calculator_client 10 20
```

### Exercise 3: Create a URDF Robot Model

**Objective**: Create a simple robot model and visualize it.

#### Step 1: Create URDF Files
Create a directory structure:
```bash
mkdir -p ros2_exercises/urdf
```

Create `ros2_exercises/urdf/simple_robot.urdf`:

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>

  <!-- Base link -->
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
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Wheel links -->
  <joint name="base_to_wheel_front_left" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_front_left"/>
    <origin xyz="0.2 0.15 0.0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="wheel_front_left">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="red"/>
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

  <joint name="base_to_wheel_front_right" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_front_right"/>
    <origin xyz="0.2 -0.15 0.0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="wheel_front_right">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="red"/>
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

  <joint name="base_to_wheel_back_left" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_back_left"/>
    <origin xyz="-0.2 0.15 0.0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="wheel_back_left">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="red"/>
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

  <joint name="base_to_wheel_back_right" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_back_right"/>
    <origin xyz="-0.2 -0.15 0.0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="wheel_back_right">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="red"/>
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
</robot>
```

#### Step 2: Validate the URDF
```bash
# Check URDF syntax
check_urdf ros2_exercises/urdf/simple_robot.urdf

# Generate graph of URDF structure
urdf_to_graphiz ros2_exercises/urdf/simple_robot.urdf
```

### Exercise 4: Robot State Publisher

Create a launch file to visualize the robot:

Create `ros2_exercises/launch/robot_visualization.launch.py`:

```python
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Get URDF file path
    robot_description_path = PathJoinSubstitution([
        FindPackageShare('ros2_exercises'),
        'urdf',
        'simple_robot.urdf'
    ])

    return LaunchDescription([
        # Robot State Publisher node
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': PathJoinSubstitution([
                    FindPackageShare('ros2_exercises'),
                    'urdf',
                    'simple_robot.urdf'
                ])
            }]
        ),

        # Joint State Publisher (GUI for moving joints)
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui'
        ),

        # RViz2 for visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', PathJoinSubstitution([
                FindPackageShare('ros2_exercises'),
                'config',
                'robot_visualization.rviz'
            ])]
        )
    ])
```

Create the RViz configuration file `ros2_exercises/config/robot_visualization.rviz`:

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
        wheel_back_left:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        wheel_back_right:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        wheel_front_left:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        wheel_front_right:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
      Name: RobotModel
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: base_link
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
      Distance: 1.758040428161621
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 0.5203983187675476
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz)
      Yaw: 5.843581676483154
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 846
  Width: 1200
```

### Exercise 5: Build and Run the Complete System

1. Update `setup.py` to include the launch and config directories:

```python
import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'ros2_exercises'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
        # Include URDF files
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='ROS 2 Exercises Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_publisher = ros2_exercises.simple_publisher:main',
            'simple_subscriber = ros2_exercises.simple_subscriber:main',
            'calculator_service = ros2_exercises.calculator_service:main',
            'calculator_client = ros2_exercises.calculator_client:main',
        ],
    },
)
```

2. Build and run the visualization:
```bash
# Build the package
cd ~/physical_ai_ws
colcon build --packages-select ros2_exercises
source install/setup.bash

# Run the visualization
ros2 launch ros2_exercises robot_visualization.launch.py
```

### Exercise 6: Parameter Server Integration

Create a node that uses parameters:

Create `ros2_exercises/parameter_node.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('publish_frequency', 1.0)
        self.declare_parameter('message_prefix', 'Hello from')

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.publish_frequency = self.get_parameter('publish_frequency').value
        self.message_prefix = self.get_parameter('message_prefix').value

        # Create publisher
        self.publisher = self.create_publisher(String, 'parameter_chatter', 10)

        # Create timer with parameter-based frequency
        self.timer = self.create_timer(
            1.0 / self.publish_frequency,
            self.timer_callback
        )

        self.get_logger().info(
            f'Parameter node initialized with: '
            f'robot_name={self.robot_name}, '
            f'frequency={self.publish_frequency}Hz, '
            f'prefix="{self.message_prefix}"'
        )

        self.counter = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'{self.message_prefix} {self.robot_name}: {self.counter}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Update `setup.py` to include this new executable:

```python
entry_points={
    'console_scripts': [
        'simple_publisher = ros2_exercises.simple_publisher:main',
        'simple_subscriber = ros2_exercises.simple_subscriber:main',
        'calculator_service = ros2_exercises.calculator_service:main',
        'calculator_client = ros2_exercises.calculator_client:main',
        'parameter_node = ros2_exercises.parameter_node:main',
    ],
},
```

Test the parameter node:
```bash
# Build the package
cd ~/physical_ai_ws
colcon build --packages-select ros2_exercises
source install/setup.bash

# Run with default parameters
ros2 run ros2_exercises parameter_node

# Run with custom parameters
ros2 run ros2_exercises parameter_node --ros-args -p robot_name:=test_robot -p publish_frequency:=2.0
```

### Exercise Completion Checklist

Complete the following to master ROS 2 fundamentals:

- [ ] Created and ran a publisher-subscriber system
- [ ] Implemented and tested a service-client system
- [ ] Created a URDF robot model and validated it
- [ ] Visualized the robot in RViz
- [ ] Created and used parameters in a ROS 2 node
- [ ] Ran the complete system with launch files
- [ ] Used ROS 2 command-line tools to inspect topics and services

These exercises provide hands-on experience with the core concepts of ROS 2, preparing you for more advanced topics in the following modules.