---
title: System Integration
sidebar_position: 2
---

# Module 5: Capstone Project — Autonomous Humanoid

## System Integration for Autonomous Humanoid Robots

System integration is the critical process of combining all previously developed components into a cohesive, autonomous humanoid robot system. This involves integrating ROS 2 controllers, perception pipelines, cognitive planning, and various subsystems to create a functional autonomous robot.

### Integration Architecture Overview

The autonomous humanoid system integrates multiple complex subsystems:

- **ROS 2 Communication Layer**: Provides the backbone for all inter-component communication
- **Perception System**: Visual, audio, and tactile processing for environmental understanding
- **Cognitive Planning**: High-level task decomposition and execution planning
- **Navigation System**: Path planning and obstacle avoidance
- **Manipulation System**: Object interaction and manipulation capabilities
- **Voice Interface**: Natural language processing and speech recognition
- **Control System**: Low-level actuator control and safety monitoring

### ROS 2 Integration Framework

The ROS 2 framework serves as the integration platform:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, Imu, JointState
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class SystemComponent:
    """Data structure for system component tracking"""
    name: str
    node_ref: Node
    status: str = "uninitialized"
    last_heartbeat: float = 0.0
    dependencies: List[str] = None

class SystemIntegrationNode(Node):
    def __init__(self):
        super().__init__('system_integration_node')

        # Publishers for system-wide communication
        self.system_status_publisher = self.create_publisher(
            String,
            '/system_status',
            10
        )

        self.integration_command_publisher = self.create_publisher(
            String,
            '/integration_command',
            10
        )

        self.safety_status_publisher = self.create_publisher(
            Bool,
            '/system_safety_ok',
            10
        )

        # Subscribers for component status
        self.component_status_subscribers = {}
        self.component_heartbeats = {}

        # Initialize system components tracking
        self.system_components = {
            'perception': SystemComponent('perception', self),
            'navigation': SystemComponent('navigation', self),
            'manipulation': SystemComponent('manipulation', self),
            'cognitive_planning': SystemComponent('cognitive_planning', self),
            'voice_interface': SystemComponent('voice_interface', self),
            'control_system': SystemComponent('control_system', self)
        }

        # System configuration
        self.system_config = {
            'safety_timeout': 5.0,  # seconds
            'integration_mode': 'autonomous',  # 'autonomous', 'supervised', 'manual'
            'emergency_stop_enabled': True
        }

        # Integration state
        self.integration_ready = False
        self.emergency_stop_active = False
        self.active_tasks = []

        # Start system monitoring
        self.system_monitor_timer = self.create_timer(1.0, self.monitor_system_status)
        self.integration_check_timer = self.create_timer(0.5, self.check_integration_readiness)

        self.get_logger().info('System Integration Node initialized')

    def register_component_subscribers(self):
        """Register subscribers for all system components"""
        component_topics = {
            'perception': '/perception_status',
            'navigation': '/navigation_status',
            'manipulation': '/manipulation_status',
            'cognitive_planning': '/planning_status',
            'voice_interface': '/voice_status',
            'control_system': '/control_status'
        }

        for component_name, topic in component_topics.items():
            self.component_status_subscribers[component_name] = self.create_subscription(
                String,
                topic,
                lambda msg, comp=component_name: self.component_status_callback(msg, comp),
                10
            )

    def component_status_callback(self, msg, component_name):
        """Handle status updates from system components"""
        try:
            status_data = json.loads(msg.data)
            status = status_data.get('status', 'unknown')
            timestamp = status_data.get('timestamp', 0.0)

            # Update component status
            if component_name in self.system_components:
                self.system_components[component_name].status = status
                self.system_components[component_name].last_heartbeat = timestamp

            # Update heartbeat
            self.component_heartbeats[component_name] = self.get_clock().now().nanoseconds / 1e9

        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid JSON in {component_name} status message')

    def monitor_system_status(self):
        """Monitor overall system health and safety"""
        system_healthy = True
        issues = []

        current_time = self.get_clock().now().nanoseconds / 1e9

        # Check component heartbeats
        for component_name, heartbeat_time in self.component_heartbeats.items():
            if current_time - heartbeat_time > self.system_config['safety_timeout']:
                system_healthy = False
                issues.append(f'{component_name} heartbeat timeout')
                self.system_components[component_name].status = 'unresponsive'

        # Check for emergency conditions
        if self.emergency_stop_active:
            system_healthy = False
            issues.append('emergency stop active')

        # Publish system safety status
        safety_msg = Bool()
        safety_msg.data = system_healthy
        self.safety_status_publisher.publish(safety_msg)

        # Publish detailed system status
        system_status = {
            'system_healthy': system_healthy,
            'components': {
                name: {
                    'status': comp.status,
                    'last_heartbeat': comp.last_heartbeat
                }
                for name, comp in self.system_components.items()
            },
            'issues': issues,
            'timestamp': current_time,
            'integration_mode': self.system_config['integration_mode']
        }

        status_msg = String()
        status_msg.data = json.dumps(system_status)
        self.system_status_publisher.publish(status_msg)

        if not system_healthy:
            self.get_logger().warn(f'System issues detected: {issues}')
            self.activate_safety_protocol()

    def check_integration_readiness(self):
        """Check if all components are ready for integration"""
        ready_components = []
        required_components = ['perception', 'navigation', 'control_system']

        for component_name, component in self.system_components.items():
            if component.status == 'operational':
                ready_components.append(component_name)

        # Check if all required components are ready
        all_required_ready = all(
            comp in ready_components for comp in required_components
        )

        self.integration_ready = all_required_ready

        if self.integration_ready:
            self.get_logger().info('System integration ready for autonomous operation')
        else:
            missing = [comp for comp in required_components
                      if comp not in ready_components]
            self.get_logger().info(f'System not ready, missing: {missing}')

    def activate_safety_protocol(self):
        """Activate safety protocols when system issues detected"""
        self.get_logger().warn('Activating safety protocols')

        # Stop all robot motion
        stop_cmd = Twist()
        # This would publish to the robot's velocity command topic
        # self.cmd_vel_publisher.publish(stop_cmd)

        # Publish emergency stop command
        emergency_cmd = String()
        emergency_cmd.data = json.dumps({
            'command': 'emergency_stop',
            'reason': 'system_safety_violation',
            'timestamp': self.get_clock().now().nanoseconds
        })
        self.integration_command_publisher.publish(emergency_cmd)

    def send_integration_command(self, command: str, parameters: Dict = None):
        """Send command to integrated system"""
        if not self.integration_ready and self.system_config['integration_mode'] == 'autonomous':
            self.get_logger().warn('System not ready for autonomous commands')
            return False

        command_msg = String()
        command_msg.data = json.dumps({
            'command': command,
            'parameters': parameters or {},
            'timestamp': self.get_clock().now().nanoseconds,
            'source': 'integration_manager'
        })

        self.integration_command_publisher.publish(command_msg)
        return True

def main(args=None):
    rclpy.init(args=args)
    node = SystemIntegrationNode()
    node.register_component_subscribers()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down System Integration Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Perception System Integration

Integrating the perception system with other components:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, PointCloud2
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import numpy as np
import json
from typing import Dict, List
from scipy.spatial.distance import cdist

class PerceptionIntegrationNode(Node):
    def __init__(self):
        super().__init__('perception_integration_node')

        # Publishers
        self.integrated_perception_publisher = self.create_publisher(
            String,
            '/integrated_perception',
            10
        )

        self.object_map_publisher = self.create_publisher(
            String,
            '/object_map',
            10
        )

        self.environment_model_publisher = self.create_publisher(
            String,
            '/environment_model',
            10
        )

        # Subscribers
        self.vision_subscriber = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.vision_callback,
            10
        )

        self.depth_subscriber = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.system_status_subscriber = self.create_subscription(
            String,
            '/system_status',
            self.system_status_callback,
            10
        )

        # Initialize components
        self.bridge = CvBridge()
        self.latest_vision = None
        self.latest_depth = None
        self.robot_orientation = None
        self.detected_objects = {}
        self.environment_map = {}
        self.system_healthy = True

        # Integration parameters
        self.object_merge_distance = 0.5  # meters
        self.max_object_age = 10.0  # seconds

        # Start perception integration timer
        self.integration_timer = self.create_timer(0.1, self.integrate_perception)

        self.get_logger().info('Perception Integration Node initialized')

    def vision_callback(self, msg):
        """Process vision data"""
        try:
            self.latest_vision = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Vision processing error: {e}')

    def depth_callback(self, msg):
        """Process depth data"""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f'Depth processing error: {e}')

    def imu_callback(self, msg):
        """Process IMU data for orientation"""
        self.robot_orientation = {
            'x': msg.orientation.x,
            'y': msg.orientation.y,
            'z': msg.orientation.z,
            'w': msg.orientation.w
        }

    def system_status_callback(self, msg):
        """Update based on system status"""
        try:
            status_data = json.loads(msg.data)
            self.system_healthy = status_data.get('system_healthy', True)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in system status message')

    def integrate_perception(self):
        """Integrate multiple perception sources"""
        if not self.system_healthy:
            return

        # Process vision and depth data together
        if self.latest_vision is not None and self.latest_depth is not None:
            objects = self.process_vision_depth_fusion()
            self.update_object_map(objects)

        # Publish integrated perception
        integrated_data = {
            'objects': self.detected_objects,
            'environment_map': self.environment_map,
            'timestamp': self.get_clock().now().nanoseconds,
            'system_healthy': self.system_healthy
        }

        integrated_msg = String()
        integrated_msg.data = json.dumps(integrated_data)
        self.integrated_perception_publisher.publish(integrated_msg)

    def process_vision_depth_fusion(self):
        """Process and fuse vision and depth data"""
        # Simple object detection (in real implementation, use deep learning)
        # For demonstration, we'll simulate object detection
        detected_objects = {}

        if self.latest_vision is not None and self.latest_depth is not None:
            # Simulate object detection by finding color-based objects
            hsv = cv2.cvtColor(self.latest_vision, cv2.COLOR_BGR2HSV)

            # Define color ranges for different objects
            color_ranges = [
                (np.array([0, 50, 50]), np.array([10, 255, 255]), 'red_object'),  # Red
                (np.array([40, 50, 50]), np.array([80, 255, 255]), 'green_object'),  # Green
                (np.array([100, 50, 50]), np.array([130, 255, 255]), 'blue_object'),  # Blue
            ]

            for lower, upper, obj_name in color_ranges:
                mask = cv2.inRange(hsv, lower, upper)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for i, contour in enumerate(contours):
                    if cv2.contourArea(contour) > 500:  # Minimum area threshold
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x, center_y = x + w//2, y + h//2

                        # Get depth at center of object
                        if 0 <= center_x < self.latest_depth.shape[1] and 0 <= center_y < self.latest_depth.shape[0]:
                            depth = float(self.latest_depth[center_y, center_x])

                            # Convert pixel coordinates to 3D world coordinates
                            # This is a simplified calculation
                            if self.robot_orientation:  # Would need camera calibration
                                world_x = (center_x - 320) * depth * 0.001  # Simplified
                                world_y = (center_y - 240) * depth * 0.001
                                world_z = depth

                                obj_key = f"{obj_name}_{i}"
                                detected_objects[obj_key] = {
                                    'name': obj_name,
                                    'position': {
                                        'x': world_x,
                                        'y': world_y,
                                        'z': world_z
                                    },
                                    'confidence': 0.8,
                                    'timestamp': self.get_clock().now().nanoseconds / 1e9,
                                    'pixel_location': {'x': center_x, 'y': center_y}
                                }

        return detected_objects

    def update_object_map(self, new_objects: Dict):
        """Update persistent object map with new detections"""
        current_time = self.get_clock().now().nanoseconds / 1e9

        # Merge new objects with existing map
        for obj_id, obj_data in new_objects.items():
            if obj_id in self.detected_objects:
                # Update existing object with new information
                existing_obj = self.detected_objects[obj_id]

                # Average positions if close enough
                existing_pos = existing_obj['position']
                new_pos = obj_data['position']

                # Calculate distance between old and new position
                dist = np.sqrt(
                    (new_pos['x'] - existing_pos['x'])**2 +
                    (new_pos['y'] - existing_pos['y'])**2 +
                    (new_pos['z'] - existing_pos['z'])**2
                )

                if dist < self.object_merge_distance:
                    # Average the positions and update confidence
                    avg_pos = {
                        'x': (existing_pos['x'] + new_pos['x']) / 2,
                        'y': (existing_pos['y'] + new_pos['y']) / 2,
                        'z': (existing_pos['z'] + new_pos['z']) / 2
                    }
                    self.detected_objects[obj_id] = {
                        'name': obj_data['name'],
                        'position': avg_pos,
                        'confidence': max(existing_obj['confidence'], obj_data['confidence']),
                        'timestamp': current_time,
                        'last_seen': current_time
                    }
                else:
                    # Create as new object if too far apart
                    self.detected_objects[obj_id] = obj_data
            else:
                # New object
                self.detected_objects[obj_id] = obj_data

        # Remove old objects
        objects_to_remove = []
        for obj_id, obj_data in self.detected_objects.items():
            age = current_time - obj_data.get('last_seen', current_time)
            if age > self.max_object_age:
                objects_to_remove.append(obj_id)

        for obj_id in objects_to_remove:
            del self.detected_objects[obj_id]

        # Publish updated object map
        object_map_msg = String()
        object_map_msg.data = json.dumps({
            'objects': self.detected_objects,
            'timestamp': current_time
        })
        self.object_map_publisher.publish(object_map_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionIntegrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Perception Integration Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Cognitive Planning Integration

Integrating cognitive planning with the overall system:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
import json
import openai
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class CognitiveTask:
    id: str
    description: str
    task_type: str
    parameters: Dict
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = None
    priority: int = 0  # 0 (low) to 10 (high)

class PlanningIntegrationNode(Node):
    def __init__(self):
        super().__init__('planning_integration_node')

        # Publishers
        self.task_queue_publisher = self.create_publisher(
            String,
            '/task_queue',
            10
        )

        self.planning_status_publisher = self.create_publisher(
            String,
            '/planning_status',
            10
        )

        self.execution_command_publisher = self.create_publisher(
            String,
            '/execution_command',
            10
        )

        # Subscribers
        self.high_level_command_subscriber = self.create_subscription(
            String,
            '/high_level_command',
            self.high_level_command_callback,
            10
        )

        self.perception_subscriber = self.create_subscription(
            String,
            '/integrated_perception',
            self.perception_callback,
            10
        )

        self.action_result_subscriber = self.create_subscription(
            String,
            '/action_result',
            self.action_result_callback,
            10
        )

        # Initialize components
        openai.api_key = self.get_parameter_or('openai_api_key', 'your-api-key-here').value

        self.current_tasks = []
        self.completed_tasks = []
        self.failed_tasks = []
        self.environment_context = {}
        self.robot_capabilities = {
            'navigation': True,
            'manipulation': True,
            'perception': True,
            'speech': True
        }

        # System prompt for cognitive planning
        self.planning_prompt = """
        You are a cognitive planner for an autonomous humanoid robot. Given a high-level goal,
        break it down into specific, executable subtasks. Consider:
        1. The robot's current environment and capabilities
        2. Available objects and their locations
        3. Safety constraints and operational limits
        4. Efficiency and task dependencies

        Available capabilities: {capabilities}
        Current environment: {environment}

        Respond with a JSON array of subtasks, each with:
        - id: unique identifier
        - description: human-readable description
        - task_type: navigation, manipulation, perception, etc.
        - parameters: specific parameters needed
        - priority: 0-10 priority level
        """

        # Start planning integration timer
        self.planning_timer = self.create_timer(0.5, self.execute_planning_cycle)

        self.get_logger().info('Planning Integration Node initialized')

    def high_level_command_callback(self, msg):
        """Process high-level commands and generate plans"""
        try:
            command_data = json.loads(msg.data)
            command_text = command_data.get('command', '')
            command_priority = command_data.get('priority', 5)

            self.get_logger().info(f'Received high-level command: {command_text}')

            # Generate plan using LLM
            plan = self.generate_cognitive_plan(command_text)

            if plan:
                # Add tasks to queue with proper dependencies
                for task_data in plan:
                    task = CognitiveTask(
                        id=task_data['id'],
                        description=task_data['description'],
                        task_type=task_data['task_type'],
                        parameters=task_data['parameters'],
                        priority=task_data.get('priority', 5)
                    )

                    # Add to current tasks
                    self.current_tasks.append(task)

                # Sort tasks by priority
                self.current_tasks.sort(key=lambda x: x.priority, reverse=True)

                # Publish updated task queue
                self.publish_task_queue()

                self.get_logger().info(f'Generated plan with {len(plan)} tasks')

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in high-level command message')

    def perception_callback(self, msg):
        """Update environment context from perception system"""
        try:
            perception_data = json.loads(msg.data)
            self.environment_context = perception_data
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in perception message')

    def action_result_callback(self, msg):
        """Handle action results and update task status"""
        try:
            result_data = json.loads(msg.data)
            task_id = result_data.get('task_id')
            status = result_data.get('status', 'unknown')

            # Find and update the task
            for task in self.current_tasks:
                if task.id == task_id:
                    if status == 'success':
                        task.status = TaskStatus.COMPLETED
                        self.completed_tasks.append(task)
                        self.current_tasks.remove(task)
                    elif status == 'failed':
                        task.status = TaskStatus.FAILED
                        self.failed_tasks.append(task)
                        self.current_tasks.remove(task)

                    break

            # Publish updated status
            self.publish_planning_status()

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in action result message')

    def generate_cognitive_plan(self, command: str):
        """Generate cognitive plan using LLM"""
        try:
            prompt = self.planning_prompt.format(
                capabilities=self.robot_capabilities,
                environment=self.environment_context
            ) + f"\n\nGoal: {command}\n\nPlan:"

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a cognitive planner. Respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            response_text = response.choices[0].message['content'].strip()

            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                return plan

        except Exception as e:
            self.get_logger().error(f'Plan generation error: {e}')

        return []

    def execute_planning_cycle(self):
        """Main planning execution cycle"""
        if not self.current_tasks:
            return

        # Get highest priority task that can be executed
        ready_task = self.get_next_ready_task()

        if ready_task:
            self.execute_task(ready_task)

        # Publish planning status
        self.publish_planning_status()

    def get_next_ready_task(self):
        """Get the next ready task considering dependencies"""
        # For now, return the first pending task
        # In a real implementation, this would consider dependencies
        for task in self.current_tasks:
            if task.status == TaskStatus.PENDING:
                return task
        return None

    def execute_task(self, task: CognitiveTask):
        """Execute a cognitive task"""
        task.status = TaskStatus.IN_PROGRESS

        # Create execution command
        execution_cmd = {
            'task_id': task.id,
            'task_type': task.task_type,
            'parameters': task.parameters,
            'command': task.description
        }

        # Publish execution command
        cmd_msg = String()
        cmd_msg.data = json.dumps(execution_cmd)
        self.execution_command_publisher.publish(cmd_msg)

        self.get_logger().info(f'Executing task: {task.description}')

    def publish_task_queue(self):
        """Publish current task queue"""
        queue_data = []
        for task in self.current_tasks:
            queue_data.append({
                'id': task.id,
                'description': task.description,
                'task_type': task.task_type,
                'status': task.status.value,
                'priority': task.priority
            })

        queue_msg = String()
        queue_msg.data = json.dumps(queue_data)
        self.task_queue_publisher.publish(queue_msg)

    def publish_planning_status(self):
        """Publish planning system status"""
        status_data = {
            'current_tasks': len(self.current_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'system_time': self.get_clock().now().nanoseconds
        }

        status_msg = String()
        status_msg.data = json.dumps(status_data)
        self.planning_status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PlanningIntegrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Planning Integration Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Navigation System Integration

Integrating navigation with perception and planning systems:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Path, OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Bool
import json
import numpy as np
from scipy.spatial.distance import euclidean
from typing import Dict, List, Tuple
import heapq

class NavigationIntegrationNode(Node):
    def __init__(self):
        super().__init__('navigation_integration_node')

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.path_publisher = self.create_publisher(
            Path,
            '/navigation_path',
            10
        )

        self.navigation_goal_publisher = self.create_publisher(
            PoseStamped,
            '/move_base_simple/goal',
            10
        )

        self.navigation_status_publisher = self.create_publisher(
            String,
            '/navigation_status',
            10
        )

        # Subscribers
        self.laser_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.perception_subscriber = self.create_subscription(
            String,
            '/integrated_perception',
            self.perception_callback,
            10
        )

        self.execution_command_subscriber = self.create_subscription(
            String,
            '/execution_command',
            self.execution_command_callback,
            10
        )

        # Initialize navigation components
        self.map_data = None
        self.map_info = None
        self.laser_ranges = None
        self.current_pose = None
        self.current_goal = None
        self.path = []
        self.waypoints = []
        self.obstacle_grid = None

        # Navigation parameters
        self.linear_speed = 0.3
        self.angular_speed = 0.5
        self.arrival_threshold = 0.3
        self.obstacle_threshold = 0.5
        self.path_update_frequency = 10.0  # Hz

        # Start navigation timer
        self.navigation_timer = self.create_timer(0.1, self.execute_navigation)

        self.get_logger().info('Navigation Integration Node initialized')

    def laser_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        self.laser_ranges = np.array(msg.ranges)

        # Update obstacle grid based on laser data
        self.update_obstacle_grid()

    def map_callback(self, msg):
        """Process occupancy grid map"""
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info

    def perception_callback(self, msg):
        """Process perception data for dynamic obstacle avoidance"""
        try:
            perception_data = json.loads(msg.data)
            # In a real implementation, this would update dynamic obstacle information
            # based on object detections from the perception system
            pass
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in perception message')

    def execution_command_callback(self, msg):
        """Process navigation commands from execution system"""
        try:
            command_data = json.loads(msg.data)
            task_type = command_data.get('task_type', '')
            parameters = command_data.get('parameters', {})

            if task_type == 'navigation':
                self.set_navigation_goal(parameters)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in execution command message')

    def set_navigation_goal(self, parameters: Dict):
        """Set navigation goal from parameters"""
        goal_x = parameters.get('x', 0.0)
        goal_y = parameters.get('y', 0.0)
        goal_z = parameters.get('z', 0.0)

        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
        goal_pose.pose.position.z = goal_z
        goal_pose.pose.orientation.w = 1.0

        self.current_goal = goal_pose

        # Plan path to goal
        self.plan_path_to_goal()

        # Publish navigation goal
        self.navigation_goal_publisher.publish(goal_pose)

        self.get_logger().info(f'Set navigation goal to ({goal_x}, {goal_y})')

    def plan_path_to_goal(self):
        """Plan path to current goal using A* algorithm"""
        if self.current_goal is None or self.map_data is None:
            return

        # Convert goal to map coordinates
        goal_map_x = int((self.current_goal.pose.position.x - self.map_info.origin.position.x) / self.map_info.resolution)
        goal_map_y = int((self.current_goal.pose.position.y - self.map_info.origin.position.y) / self.map_info.resolution)

        # Convert current position to map coordinates (simplified)
        current_map_x = int((0 - self.map_info.origin.position.x) / self.map_info.resolution)  # Placeholder
        current_map_y = int((0 - self.map_info.origin.position.y) / self.map_info.resolution)  # Placeholder

        # Plan path using A* (simplified implementation)
        path = self.a_star_pathfinding(
            (current_map_x, current_map_y),
            (goal_map_x, goal_map_y),
            self.map_data
        )

        # Convert path back to world coordinates
        self.path = []
        for map_x, map_y in path:
            world_x = map_x * self.map_info.resolution + self.map_info.origin.position.x
            world_y = map_y * self.map_info.resolution + self.map_info.origin.position.y

            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = world_x
            pose.pose.position.y = world_y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0

            self.path.append(pose)

        # Publish path
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        path_msg.poses = self.path
        self.path_publisher.publish(path_msg)

    def a_star_pathfinding(self, start: Tuple[int, int], goal: Tuple[int, int], grid: np.ndarray) -> List[Tuple[int, int]]:
        """Simple A* pathfinding algorithm"""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def get_neighbors(pos):
            x, y = pos
            neighbors = [
                (x+1, y), (x-1, y), (x, y+1), (x, y-1),  # 4-connectivity
                (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)  # 8-connectivity
            ]
            return [(nx, ny) for nx, ny in neighbors
                   if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0] and grid[ny, nx] < 50]  # Free space

        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in get_neighbors(current):
                tentative_g_score = g_score[current] + 1  # Uniform cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found

    def update_obstacle_grid(self):
        """Update obstacle information based on laser scan"""
        if self.laser_ranges is not None and len(self.laser_ranges) > 0:
            # Convert laser ranges to obstacle positions (simplified)
            # In a real implementation, this would be more sophisticated
            pass

    def execute_navigation(self):
        """Execute navigation to current goal"""
        if self.current_goal is None or not self.path:
            return

        # Check if reached goal
        current_pos = np.array([0, 0])  # Placeholder - would get actual position
        goal_pos = np.array([
            self.current_goal.pose.position.x,
            self.current_goal.pose.position.y
        ])

        distance_to_goal = euclidean(current_pos, goal_pos)

        if distance_to_goal < self.arrival_threshold:
            # Reached goal
            self.get_logger().info('Reached navigation goal')
            self.stop_navigation()
            self.publish_navigation_status('completed')
            self.current_goal = None
            self.path = []
            return

        # Follow path
        self.follow_path()

        # Check for obstacles
        if self.detect_obstacles():
            self.handle_obstacle()

    def follow_path(self):
        """Follow the planned path"""
        if not self.path:
            return

        # Get next waypoint
        next_waypoint = self.path[0].pose.position

        # Calculate direction to waypoint
        current_pos = np.array([0, 0])  # Placeholder
        waypoint_pos = np.array([next_waypoint.x, next_waypoint.y])

        direction = waypoint_pos - current_pos
        distance = np.linalg.norm(direction)

        if distance < self.arrival_threshold:
            # Reached current waypoint, remove it
            self.path.pop(0)
            if not self.path:
                return

        # Calculate velocity commands
        cmd_vel = Twist()

        if distance > 0:
            direction = direction / distance
            cmd_vel.linear.x = min(self.linear_speed, distance * 0.5)
            cmd_vel.angular.z = np.arctan2(direction[1], direction[0]) * self.angular_speed

        # Publish velocity command
        self.cmd_vel_publisher.publish(cmd_vel)

    def detect_obstacles(self) -> bool:
        """Detect obstacles in the robot's path"""
        if self.laser_ranges is None:
            return False

        # Check for obstacles in front of robot
        front_scan = self.laser_ranges[len(self.laser_ranges)//2 - 30 : len(self.laser_ranges)//2 + 30]
        front_distances = [r for r in front_scan if not np.isnan(r) and r > 0]

        if front_distances:
            min_distance = min(front_distances)
            return min_distance < self.obstacle_threshold

        return False

    def handle_obstacle(self):
        """Handle obstacle detection"""
        self.get_logger().warn('Obstacle detected, stopping navigation')
        self.stop_navigation()
        self.publish_navigation_status('obstacle_detected')

    def stop_navigation(self):
        """Stop robot movement"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_vel)

    def publish_navigation_status(self, status: str):
        """Publish navigation status"""
        status_data = {
            'status': status,
            'current_goal': {
                'x': self.current_goal.pose.position.x if self.current_goal else 0.0,
                'y': self.current_goal.pose.position.y if self.current_goal else 0.0
            } if self.current_goal else None,
            'path_remaining': len(self.path),
            'timestamp': self.get_clock().now().nanoseconds
        }

        status_msg = String()
        status_msg.data = json.dumps(status_data)
        self.navigation_status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = NavigationIntegrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Navigation Integration Node')
    finally:
        node.stop_navigation()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### System Health Monitoring and Safety

Implementing comprehensive system health monitoring:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import BatteryState
from geometry_msgs.msg import Twist
import json
from typing import Dict
from datetime import datetime, timedelta
import threading

class HealthMonitoringNode(Node):
    def __init__(self):
        super().__init__('health_monitoring_node')

        # Publishers
        self.safety_status_publisher = self.create_publisher(
            Bool,
            '/system_safety_ok',
            10
        )

        self.health_status_publisher = self.create_publisher(
            String,
            '/system_health_status',
            10
        )

        self.emergency_stop_publisher = self.create_publisher(
            Bool,
            '/emergency_stop',
            10
        )

        self.system_performance_publisher = self.create_publisher(
            Float32,
            '/system_performance_score',
            10
        )

        # Subscribers
        self.system_status_subscriber = self.create_subscription(
            String,
            '/system_status',
            self.system_status_callback,
            10
        )

        self.battery_subscriber = self.create_subscription(
            BatteryState,
            '/battery_status',
            self.battery_callback,
            10
        )

        # Initialize health monitoring
        self.component_health = {}
        self.safety_violations = []
        self.performance_metrics = {
            'uptime': 0.0,
            'response_time_avg': 0.0,
            'success_rate': 1.0
        }

        self.safety_thresholds = {
            'battery_level': 0.2,  # 20% minimum
            'temperature': 80.0,   # 80°C maximum
            'cpu_usage': 90.0,     # 90% maximum
            'memory_usage': 90.0   # 90% maximum
        }

        # Start health monitoring
        self.health_timer = self.create_timer(1.0, self.monitor_health)
        self.safety_check_timer = self.create_timer(0.5, self.check_safety)

        self.get_logger().info('Health Monitoring Node initialized')

    def system_status_callback(self, msg):
        """Update component health from system status"""
        try:
            status_data = json.loads(msg.data)
            self.component_health = status_data.get('components', {})
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in system status message')

    def battery_callback(self, msg):
        """Monitor battery health"""
        battery_level = msg.percentage if msg.percentage is not None else 1.0

        if battery_level < self.safety_thresholds['battery_level']:
            self.add_safety_violation('battery_low', f'Battery level: {battery_level:.2f}')

    def monitor_health(self):
        """Monitor overall system health"""
        current_time = datetime.now()

        # Calculate health metrics
        healthy_components = 0
        total_components = len(self.component_health)

        for comp_name, comp_data in self.component_health.items():
            if comp_data.get('status') == 'operational':
                healthy_components += 1

        health_score = healthy_components / total_components if total_components > 0 else 1.0

        # Update performance metrics
        self.performance_metrics['uptime'] += 1.0  # Increment every second
        self.performance_metrics['success_rate'] = health_score

        # Publish health status
        health_status = {
            'health_score': health_score,
            'healthy_components': healthy_components,
            'total_components': total_components,
            'safety_violations': len(self.safety_violations),
            'performance_metrics': self.performance_metrics,
            'timestamp': current_time.isoformat()
        }

        health_msg = String()
        health_msg.data = json.dumps(health_status)
        self.health_status_publisher.publish(health_msg)

        # Publish performance score
        perf_msg = Float32()
        perf_msg.data = health_score
        self.system_performance_publisher.publish(perf_msg)

    def check_safety(self):
        """Check for safety violations"""
        safety_ok = True
        violations = []

        # Check component health
        for comp_name, comp_data in self.component_health.items():
            if comp_data.get('status') == 'failed':
                violations.append(f'Component {comp_name} failed')

        # Check safety violations
        if self.safety_violations:
            safety_ok = False
            violations.extend(self.safety_violations)

        # Publish safety status
        safety_msg = Bool()
        safety_msg.data = safety_ok
        self.safety_status_publisher.publish(safety_msg)

        if not safety_ok:
            self.get_logger().warn(f'Safety issues detected: {violations}')
            self.trigger_safety_protocol()

        # Keep safety violations manageable
        if len(self.safety_violations) > 50:
            self.safety_violations = self.safety_violations[-25:]

    def add_safety_violation(self, violation_type: str, description: str):
        """Add a safety violation"""
        violation = {
            'type': violation_type,
            'description': description,
            'timestamp': datetime.now().isoformat()
        }
        self.safety_violations.append(violation)

    def trigger_safety_protocol(self):
        """Trigger safety protocols when violations occur"""
        self.get_logger().error('Safety protocol triggered due to violations')

        # Stop all robot motion
        stop_cmd = Twist()
        # self.cmd_vel_publisher.publish(stop_cmd)  # Would need to be set up

        # Publish emergency stop
        emergency_msg = Bool()
        emergency_msg.data = True
        self.emergency_stop_publisher.publish(emergency_msg)

        # Log violations
        for violation in self.safety_violations[-5:]:  # Last 5 violations
            self.get_logger().error(f'Safety violation: {violation}')

def main(args=None):
    rclpy.init(args=args)
    node = HealthMonitoringNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Health Monitoring Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for System Integration

1. **Modular Design**: Keep components loosely coupled with clear interfaces
2. **Error Handling**: Implement comprehensive error handling and recovery
3. **Safety First**: Design with safety protocols and emergency stops
4. **Performance Monitoring**: Continuously monitor system performance
5. **Graceful Degradation**: Systems should continue operating with reduced functionality when components fail
6. **Communication Standards**: Use standard message types and protocols
7. **Testing**: Thoroughly test integrated system behavior
8. **Documentation**: Maintain clear documentation of integration points

System integration is the culmination of all previous modules, combining ROS 2, perception, planning, and control systems into a unified autonomous humanoid robot platform.