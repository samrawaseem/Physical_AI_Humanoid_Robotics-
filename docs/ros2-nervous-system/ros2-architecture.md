---
title: ROS 2 Architecture
sidebar_position: 2
---

# Module 1: The Robotic Nervous System (ROS 2)

## ROS 2 Architecture Fundamentals

ROS 2 (Robot Operating System 2) provides the communication infrastructure that allows different components of a robotic system to work together. Understanding its architecture is crucial for building complex robotic applications.

### Core Architecture Components

ROS 2 follows a distributed computing model where multiple processes (nodes) communicate with each other through a publish-subscribe mechanism. The architecture consists of several key components:

#### Nodes
Nodes are the fundamental building blocks of a ROS 2 system. Each node is a process that performs computation and communicates with other nodes. Nodes can:
- Publish data to topics
- Subscribe to topics to receive data
- Provide services to other nodes
- Call services provided by other nodes
- Send and receive actions

```python
# Example of a simple ROS 2 node
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Minimal node created')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Topics
Topics are named buses over which nodes exchange messages. The publish-subscribe communication pattern allows for loose coupling between nodes. Publishers send messages to topics, and subscribers receive messages from topics.

Key characteristics of topics:
- Unidirectional data flow from publishers to subscribers
- Multiple publishers and subscribers can use the same topic
- Data is sent continuously while the publisher is active
- Message types must be consistent across all publishers and subscribers

#### Services
Services provide a request-response communication pattern. A client sends a request to a service and waits for a response. This is useful for operations that require a specific response or completion confirmation.

Key characteristics of services:
- Synchronous communication
- Request-response pattern
- One client communicates with one service server
- Useful for operations that must complete before proceeding

#### Actions
Actions are used for long-running tasks that may take a significant amount of time to complete. They provide feedback during execution and can be canceled if needed.

Key characteristics of actions:
- Asynchronous communication
- Support for feedback during execution
- Ability to cancel ongoing tasks
- Goal request and result response

### Communication Middleware

ROS 2 uses DDS (Data Distribution Service) as its default middleware. DDS provides:
- Discovery: Nodes automatically find each other
- Data serialization: Efficient conversion of data structures
- Quality of Service (QoS) policies: Configurable reliability and performance settings
- Platform independence: Works across different operating systems

### Quality of Service (QoS) Settings

QoS policies allow you to configure how messages are delivered, balancing between reliability and performance:

- **Reliability**: Whether messages are guaranteed to be delivered
- **Durability**: Whether late-joining subscribers receive previous messages
- **History**: How many messages to store for delivery
- **Deadline**: Maximum time between consecutive messages

### Package Structure

ROS 2 packages organize related functionality and provide a standard structure:

```
my_robot_package/
├── CMakeLists.txt
├── package.xml
├── src/
│   ├── node1.cpp
│   └── node2.cpp
├── include/
│   └── my_robot_package/
├── launch/
│   └── my_launch_file.launch.py
├── config/
│   └── parameters.yaml
└── test/
    └── test_file.cpp
```

### Launch Files

Launch files allow you to start multiple nodes with a single command, along with their configurations and parameters. They are written in Python and provide a convenient way to manage complex robotic systems.

```python
# Example launch file
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='robot_controller',
            parameters=[
                {'param1': 'value1'},
                {'param2': 'value2'}
            ]
        ),
        Node(
            package='my_robot_package',
            executable='sensor_processor',
            name='sensor_processor'
        )
    ])
```

### Parameter Management

Parameters allow you to configure nodes without recompiling. They can be set at launch time, changed during runtime, and stored in YAML files for consistent configuration across runs.

### Best Practices

When designing ROS 2 systems:

1. **Modularity**: Create focused nodes that perform specific functions
2. **Naming Conventions**: Use consistent, descriptive names for topics, services, and nodes
3. **Message Types**: Use appropriate message types for your data
4. **Error Handling**: Implement proper error handling and logging
5. **Resource Management**: Properly initialize and clean up resources

Understanding these architectural concepts is essential for building robust and maintainable robotic systems. The following sections will explore each component in greater detail with practical examples.