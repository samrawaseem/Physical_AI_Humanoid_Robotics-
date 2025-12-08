---
title: Nodes, Topics, and Services
sidebar_position: 3
---

# Module 1: The Robotic Nervous System (ROS 2)

## Nodes, Topics, and Services in ROS 2

This section explores the three fundamental communication patterns in ROS 2: nodes for computation, topics for data streaming, and services for request-response interactions.

### Nodes: The Computational Building Blocks

Nodes are processes that perform computation in the ROS 2 system. Each node runs independently and communicates with other nodes through topics, services, or actions.

#### Creating Nodes with rclpy

To create a node in Python using rclpy:

```python
import rclpy
from rclpy.node import Node

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Create publishers, subscribers, services, etc.
        self.velocity_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.odometry_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odometry_callback,
            10
        )

        self.get_logger().info('Robot controller node initialized')

    def odometry_callback(self, msg):
        # Process odometry data
        self.get_logger().info(f'Position: ({msg.pose.pose.position.x}, {msg.pose.pose.position.y})')

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()

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

#### Node Lifecycle

ROS 2 nodes can have a lifecycle with different states:
- Unconfigured → Inactive → Active → Finalized
- This allows for better resource management and coordination

### Topics: Publish-Subscribe Communication

Topics enable asynchronous, one-way communication between nodes using a publish-subscribe pattern.

#### Publisher Example

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class VelocityPublisher(Node):
    def __init__(self):
        super().__init__('velocity_publisher')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.publish_velocity)  # 10 Hz
        self.linear_velocity = 0.5
        self.angular_velocity = 0.0

    def publish_velocity(self):
        msg = Twist()
        msg.linear.x = self.linear_velocity
        msg.angular.z = self.angular_velocity
        self.publisher.publish(msg)
        self.get_logger().info(f'Published velocity: linear={msg.linear.x}, angular={msg.angular.z}')

def main(args=None):
    rclpy.init(args=args)
    node = VelocityPublisher()

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

#### Subscriber Example

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class VelocitySubscriber(Node):
    def __init__(self):
        super().__init__('velocity_subscriber')
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.velocity_callback,
            10
        )
        self.subscription  # Prevent unused variable warning

    def velocity_callback(self, msg):
        self.get_logger().info(f'Received velocity: linear={msg.linear.x}, angular={msg.angular.z}')

def main(args=None):
    rclpy.init(args=args)
    node = VelocitySubscriber()

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

#### Quality of Service (QoS) for Topics

QoS settings allow fine-tuning of topic communication:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Create a QoS profile for reliable communication
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

self.publisher = self.create_publisher(Twist, '/cmd_vel', qos_profile)
```

### Services: Request-Response Communication

Services provide synchronous request-response communication between nodes.

#### Service Server Example

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class AddService(Node):
    def __init__(self):
        super().__init__('add_service')
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_callback
        )

    def add_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning {response.sum}')
        return response

def main(args=None):
    rclpy.init(args=args)
    node = AddService()

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

#### Service Client Example

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class AddClient(Node):
    def __init__(self):
        super().__init__('add_client')
        self.client = self.create_client(AddTwoInts, 'add_two_ints')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.send_request()

    def send_request(self):
        request = AddTwoInts.Request()
        request.a = 41
        request.b = 1
        self.future = self.client.call_async(request)
        self.timer = self.create_timer(0.1, self.check_response)

    def check_response(self):
        if self.future.done():
            try:
                response = self.future.result()
                self.get_logger().info(f'Result: {response.sum}')
            except Exception as e:
                self.get_logger().info(f'Service call failed: {e}')
            finally:
                self.destroy_node()
                rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = AddClient()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
```

### Topic Tools and Commands

ROS 2 provides command-line tools for inspecting and debugging topics:

```bash
# List all topics
ros2 topic list

# Echo messages from a topic
ros2 topic echo /cmd_vel

# Print information about a topic
ros2 topic info /cmd_vel

# Publish a message to a topic
ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear: {x: 1.0}, angular: {z: 0.5}}'

# Show topic statistics
ros2 topic hz /odom
```

### Service Tools and Commands

```bash
# List all services
ros2 service list

# Call a service
ros2 service call /add_two_ints example_interfaces/srv/AddTwoInts '{a: 1, b: 2}'

# Print information about a service
ros2 service info /add_two_ints
```

### Best Practices

1. **Topic Naming**: Use descriptive names with consistent prefixes (e.g., `/robot1/joint_states`, `/robot1/odom`)
2. **Message Types**: Use standard message types when possible for interoperability
3. **Resource Management**: Always properly destroy nodes and clean up resources
4. **Error Handling**: Implement proper exception handling in callbacks
5. **Logging**: Use appropriate log levels (info, warn, error) for debugging
6. **QoS Settings**: Choose appropriate QoS settings based on your application's requirements

### Practical Exercise: Robot Control Node

Create a node that publishes velocity commands to control a simulated robot:

1. Create a publisher for `/cmd_vel` topic
2. Subscribe to `/odom` topic to receive robot position
3. Implement a timer that publishes velocity commands to move the robot in a square pattern
4. Use logging to track the robot's position and movement state

This pattern of combining publishers and subscribers in a single node is common in ROS 2 applications and forms the foundation for more complex robotic behaviors.