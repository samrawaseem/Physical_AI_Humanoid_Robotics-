---
title: rclpy Python Nodes
sidebar_position: 4
---

# Module 1: The Robotic Nervous System (ROS 2)

## Programming ROS 2 Nodes with rclpy

The `rclpy` library provides Python bindings for ROS 2, allowing you to create nodes, publishers, subscribers, services, and actions using Python. This section covers the fundamentals of creating and managing ROS 2 nodes in Python.

### Setting Up Your Python Environment

Before creating ROS 2 nodes, ensure your Python environment is properly configured:

```bash
# Activate your Python virtual environment
source ~/physical_ai_env/bin/activate

# Verify rclpy is available
python3 -c "import rclpy; print('rclpy version:', rclpy.get_rclpy_type())"
```

### Basic Node Structure

Every ROS 2 Python node follows a standard structure:

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('node_name')
        # Initialize node components here
        self.get_logger().info('Node initialized')

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()

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

### Creating Publishers

Publishers send messages to topics:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
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
    talker = Talker()

    try:
        rclpy.spin(talker)
    except KeyboardInterrupt:
        pass
    finally:
        talker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating Subscribers

Subscribers receive messages from topics:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):
    def __init__(self):
        super().__init__('listener')
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
    listener = Listener()

    try:
        rclpy.spin(listener)
    except KeyboardInterrupt:
        pass
    finally:
        listener.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating Services

Services provide request-response communication:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request\na: {request.a}, b: {request.b}\nsum: {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()

    try:
        rclpy.spin(minimal_service)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating Actions

Actions handle long-running tasks with feedback:

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Feedback: {feedback_msg.sequence}')

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Result: {result.sequence}')

        return result

def main(args=None):
    rclpy.init(args=args)
    action_server = FibonacciActionServer()

    try:
        rclpy.spin(action_server)
    except KeyboardInterrupt:
        pass
    finally:
        action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Parameter Handling

Nodes can accept parameters for configuration:

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('wheel_radius', 0.05)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.wheel_radius = self.get_parameter('wheel_radius').value

        self.get_logger().info(f'Robot: {self.robot_name}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')
        self.get_logger().info(f'Wheel radius: {self.wheel_radius}')

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

### Timers and Callbacks

Timers allow periodic execution of functions:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class TimerNode(Node):
    def __init__(self):
        super().__init__('timer_node')
        self.publisher = self.create_publisher(Float64, 'timer_data', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        self.count = 0

    def timer_callback(self):
        msg = Float64()
        msg.data = float(self.count)
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')
        self.count += 1

def main(args=None):
    rclpy.init(args=args)
    node = TimerNode()

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

### Working with Custom Message Types

To use custom message types, first create them in a package, then import and use them:

```python
import rclpy
from rclpy.node import Node
# Import custom message (assuming it's in your_package_msgs)
from your_package_msgs.msg import CustomMessage

class CustomMessageNode(Node):
    def __init__(self):
        super().__init__('custom_message_node')
        self.publisher = self.create_publisher(CustomMessage, 'custom_topic', 10)
        self.timer = self.create_timer(1.0, self.publish_custom_message)

    def publish_custom_message(self):
        msg = CustomMessage()
        msg.field1 = 'Hello'
        msg.field2 = 42
        msg.field3 = [1.0, 2.0, 3.0]
        self.publisher.publish(msg)
        self.get_logger().info(f'Published custom message: {msg}')

def main(args=None):
    rclpy.init(args=args)
    node = CustomMessageNode()

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

### Error Handling and Logging

Proper error handling and logging are essential for robust nodes:

```python
import rclpy
from rclpy.node import Node
import traceback

class RobustNode(Node):
    def __init__(self):
        super().__init__('robust_node')
        try:
            # Initialize components
            self.setup_components()
        except Exception as e:
            self.get_logger().error(f'Failed to initialize node: {e}')
            self.get_logger().error(traceback.format_exc())
            raise

    def setup_components(self):
        try:
            self.publisher = self.create_publisher(String, 'chatter', 10)
            self.timer = self.create_timer(0.1, self.safe_timer_callback)
            self.get_logger().info('Components initialized successfully')
        except Exception as e:
            self.get_logger().error(f'Error setting up components: {e}')
            raise

    def safe_timer_callback(self):
        try:
            # Do work that might fail
            msg = String()
            msg.data = 'Safe message'
            self.publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {e}')
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    node = RobustNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted by user')
    except Exception as e:
        node.get_logger().error(f'Unexpected error: {e}')
        node.get_logger().error(traceback.format_exc())
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Creating a Package Structure

To organize your nodes properly, create a ROS 2 package:

```bash
# Create a new package
ros2 pkg create --build-type ament_python my_robot_nodes
cd my_robot_nodes

# Create the package structure
mkdir my_robot_nodes
touch my_robot_nodes/__init__.py
touch my_robot_nodes/talker_node.py
touch my_robot_nodes/listener_node.py
```

In `setup.py`:
```python
from setuptools import find_packages, setup

package_name = 'my_robot_nodes'

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
    description='Examples of ROS 2 nodes in Python',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = my_robot_nodes.talker_node:main',
            'listener = my_robot_nodes.listener_node:main',
        ],
    },
)
```

### Running Your Nodes

After creating your nodes, you can run them using:

```bash
# Build your package
colcon build --packages-select my_robot_nodes

# Source the workspace
source install/setup.bash

# Run the node
ros2 run my_robot_nodes talker
```

Understanding these rclpy fundamentals is crucial for building complex robotic applications. The next section will cover URDF robot description, which is essential for defining robot models in ROS 2.