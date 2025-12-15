---
title: Cognitive Planning
sidebar_position: 3
---

# Module 4: Vision-Language-Action (VLA)

## Cognitive Planning for Task Execution

Cognitive planning bridges high-level natural language commands with low-level robot actions. It involves task decomposition, world modeling, action selection, and execution planning to transform human instructions into executable robotic behaviors.

### Understanding Cognitive Planning in Robotics

Cognitive planning in robotics encompasses:

- **Task Decomposition**: Breaking complex goals into simpler, executable steps
- **World Modeling**: Maintaining and updating internal representation of the environment
- **State Tracking**: Monitoring the current state of the robot and environment
- **Action Selection**: Choosing appropriate actions based on current state and goals
- **Execution Planning**: Sequencing actions to achieve desired outcomes
- **Recovery Planning**: Handling failures and unexpected situations

### Hierarchical Task Networks (HTN)

HTN provides a framework for decomposing high-level tasks into primitive actions:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    id: str
    name: str
    type: str  # e.g., "primitive", "compound"
    status: TaskStatus = TaskStatus.PENDING
    subtasks: List['Task'] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None

class HTNPlannerNode(Node):
    def __init__(self):
        super().__init__('htn_planner_node')

        # Publishers
        self.task_publisher = self.create_publisher(
            String,
            '/task_queue',
            10
        )

        self.status_publisher = self.create_publisher(
            String,
            '/planner_status',
            10
        )

        # Subscribers
        self.goal_subscriber = self.create_subscription(
            String,
            '/high_level_goal',
            self.goal_callback,
            10
        )

        self.action_result_subscriber = self.create_subscription(
            String,
            '/action_result',
            self.action_result_callback,
            10
        )

        # Planner state
        self.current_task_tree = None
        self.task_queue = []
        self.world_state = {
            'robot_position': None,
            'objects': {},
            'locations': {},
            'robot_status': 'idle'
        }

        # Task decomposition rules
        self.decomposition_rules = {
            'fetch_object': self.decompose_fetch_object,
            'navigate_to_location': self.decompose_navigate_to_location,
            'manipulate_object': self.decompose_manipulate_object,
            'assemble_object': self.decompose_assemble_object
        }

        self.get_logger().info('HTN Planner Node initialized')

    def goal_callback(self, msg):
        """Process high-level goals and create task tree"""
        try:
            goal_data = json.loads(msg.data)
            goal_type = goal_data.get('type')
            goal_parameters = goal_data.get('parameters', {})

            self.get_logger().info(f'Received goal: {goal_type} with params {goal_parameters}')

            # Create root task
            root_task = Task(
                id=f"goal_{self.get_clock().now().nanoseconds}",
                name=f"Goal: {goal_type}",
                type="compound",
                parameters=goal_parameters
            )

            # Decompose the goal
            if goal_type in self.decomposition_rules:
                subtasks = self.decomposition_rules[goal_type](goal_parameters)
                root_task.subtasks = subtasks

                # Flatten task tree into executable queue
                self.task_queue = self.flatten_task_tree(root_task)
                self.current_task_tree = root_task

                # Publish tasks to queue
                self.publish_task_queue()

                self.get_logger().info(f'Created task tree with {len(self.task_queue)} tasks')

                # Update status
                status_msg = String()
                status_msg.data = json.dumps({
                    'status': 'planning_completed',
                    'task_count': len(self.task_queue),
                    'root_task': root_task.name
                })
                self.status_publisher.publish(status_msg)
            else:
                self.get_logger().error(f'Unknown goal type: {goal_type}')

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in goal message')

    def decompose_fetch_object(self, params):
        """Decompose fetch object goal into subtasks"""
        object_name = params.get('object_name', 'unknown')
        destination = params.get('destination', 'current_position')

        subtasks = [
            Task(
                id=f"locate_{object_name}_{self.get_clock().now().nanoseconds}",
                name=f"Locate {object_name}",
                type="primitive",
                parameters={'object_name': object_name},
                preconditions=['robot_operational'],
                effects=[f'{object_name}_located']
            ),
            Task(
                id=f"navigate_to_{object_name}_{self.get_clock().now().nanoseconds}",
                name=f"Navigate to {object_name}",
                type="primitive",
                parameters={
                    'target_object': object_name,
                    'approach_distance': 0.5
                },
                preconditions=[f'{object_name}_located'],
                effects=['robot_at_object']
            ),
            Task(
                id=f"grasp_{object_name}_{self.get_clock().now().nanoseconds}",
                name=f"Grasp {object_name}",
                type="primitive",
                parameters={'object_name': object_name},
                preconditions=['robot_at_object', f'{object_name}_reachable'],
                effects=[f'{object_name}_grasped', 'gripper_occupied']
            ),
            Task(
                id=f"navigate_to_{destination}_{self.get_clock().now().nanoseconds}",
                name=f"Navigate to {destination}",
                type="primitive",
                parameters={'destination': destination},
                preconditions=[f'{object_name}_grasped'],
                effects=['robot_at_destination']
            ),
            Task(
                id=f"place_{object_name}_{self.get_clock().now().nanoseconds}",
                name=f"Place {object_name}",
                type="primitive",
                parameters={
                    'object_name': object_name,
                    'destination': destination
                },
                preconditions=['robot_at_destination', f'{object_name}_grasped'],
                effects=[f'{object_name}_placed', 'gripper_free']
            )
        ]

        return subtasks

    def decompose_navigate_to_location(self, params):
        """Decompose navigation goal into subtasks"""
        target_location = params.get('location', 'unknown')

        subtasks = [
            Task(
                id=f"get_path_to_{target_location}_{self.get_clock().now().nanoseconds}",
                name=f"Get path to {target_location}",
                type="primitive",
                parameters={'target_location': target_location},
                preconditions=['map_available'],
                effects=['path_calculated']
            ),
            Task(
                id=f"navigate_to_{target_location}_{self.get_clock().now().nanoseconds}",
                name=f"Navigate to {target_location}",
                type="primitive",
                parameters={
                    'target_location': target_location,
                    'path_following': True
                },
                preconditions=['path_calculated'],
                effects=['robot_at_location']
            )
        ]

        return subtasks

    def decompose_manipulate_object(self, params):
        """Decompose object manipulation goal into subtasks"""
        object_name = params.get('object_name', 'unknown')
        manipulation_type = params.get('manipulation_type', 'move')

        subtasks = [
            Task(
                id=f"approach_{object_name}_{self.get_clock().now().nanoseconds}",
                name=f"Approach {object_name}",
                type="primitive",
                parameters={
                    'object_name': object_name,
                    'approach_distance': 0.3
                },
                preconditions=['object_located'],
                effects=['robot_in_position']
            )
        ]

        if manipulation_type == 'grasp':
            subtasks.extend([
                Task(
                    id=f"grasp_{object_name}_{self.get_clock().now().nanoseconds}",
                    name=f"Grasp {object_name}",
                    type="primitive",
                    parameters={'object_name': object_name},
                    preconditions=['robot_in_position'],
                    effects=[f'{object_name}_grasped']
                )
            ])
        elif manipulation_type == 'move':
            subtasks.extend([
                Task(
                    id=f"move_{object_name}_{self.get_clock().now().nanoseconds}",
                    name=f"Move {object_name}",
                    type="primitive",
                    parameters={
                        'object_name': object_name,
                        'new_position': params.get('target_position', {})
                    },
                    preconditions=[f'{object_name}_grasped'],
                    effects=[f'{object_name}_moved']
                )
            ])

        return subtasks

    def decompose_assemble_object(self, params):
        """Decompose object assembly goal into subtasks"""
        assembly_name = params.get('assembly_name', 'unknown')
        components = params.get('components', [])

        subtasks = [
            Task(
                id=f"get_assembly_plan_{assembly_name}_{self.get_clock().now().nanoseconds}",
                name=f"Get assembly plan for {assembly_name}",
                type="primitive",
                parameters={'assembly_name': assembly_name},
                preconditions=['assembly_knowledge_available'],
                effects=['assembly_plan_acquired']
            )
        ]

        for i, component in enumerate(components):
            subtasks.extend([
                Task(
                    id=f"fetch_component_{i}_{self.get_clock().now().nanoseconds}",
                    name=f"Fetch component {component}",
                    type="primitive",
                    parameters={'component_name': component},
                    preconditions=['assembly_plan_acquired'],
                    effects=[f'{component}_fetched']
                ),
                Task(
                    id=f"assemble_component_{i}_{self.get_clock().now().nanoseconds}",
                    name=f"Assemble component {component}",
                    type="primitive",
                    parameters={
                        'component_name': component,
                        'assembly_step': i
                    },
                    preconditions=[f'{component}_fetched'],
                    effects=[f'{component}_assembled']
                )
            ])

        return subtasks

    def flatten_task_tree(self, root_task):
        """Convert hierarchical task tree to flat execution queue"""
        queue = []

        def add_task_recursive(task):
            if task.type == "primitive":
                queue.append(task)
            else:
                # For compound tasks, add all subtasks
                for subtask in task.subtasks:
                    subtask.parent_id = task.id
                    add_task_recursive(subtask)

        add_task_recursive(root_task)
        return queue

    def publish_task_queue(self):
        """Publish the current task queue"""
        if not self.task_queue:
            return

        # Convert tasks to JSON
        task_list = []
        for task in self.task_queue:
            task_dict = {
                'id': task.id,
                'name': task.name,
                'type': task.type,
                'parameters': task.parameters,
                'preconditions': task.preconditions,
                'effects': task.effects,
                'parent_id': task.parent_id
            }
            task_list.append(task_dict)

        task_queue_msg = String()
        task_queue_msg.data = json.dumps(task_list)
        self.task_publisher.publish(task_queue_msg)

    def action_result_callback(self, msg):
        """Handle action results and update task status"""
        try:
            result_data = json.loads(msg.data)
            task_id = result_data.get('task_id')
            status = result_data.get('status', 'unknown')

            # Update task status in queue
            for task in self.task_queue:
                if task.id == task_id:
                    task.status = TaskStatus(status)
                    break

            # Check if current task is complete
            if status == 'success':
                self.get_logger().info(f'Task {task_id} completed successfully')
                self.check_task_completion()
            elif status == 'failed':
                self.get_logger().warn(f'Task {task_id} failed')
                self.handle_task_failure(task_id, result_data.get('error', 'unknown'))

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in action result')

    def check_task_completion(self):
        """Check if all tasks in queue are complete"""
        incomplete_tasks = [t for t in self.task_queue if t.status not in [TaskStatus.SUCCESS, TaskStatus.FAILED]]

        if not incomplete_tasks:
            self.get_logger().info('All tasks completed successfully')
            # Publish completion status
            status_msg = String()
            status_msg.data = json.dumps({
                'status': 'all_tasks_completed',
                'completion_time': self.get_clock().now().nanoseconds
            })
            self.status_publisher.publish(status_msg)

    def handle_task_failure(self, task_id, error):
        """Handle task failure with recovery strategies"""
        self.get_logger().error(f'Task {task_id} failed: {error}')

        # Implement recovery strategies
        recovery_strategies = [
            self.retry_task,
            self.use_alternative_method,
            self.request_human_assistance
        ]

        for strategy in recovery_strategies:
            if strategy(task_id, error):
                break

    def retry_task(self, task_id, error):
        """Attempt to retry failed task"""
        # Check if this is a retryable error
        retryable_errors = ['navigation_failed', 'grasp_failed', 'timeout']

        if any(retry_error in error.lower() for retry_error in retryable_errors):
            self.get_logger().info(f'Retrying task {task_id}')
            # In a real implementation, you'd republish the task
            return True
        return False

    def use_alternative_method(self, task_id, error):
        """Try alternative method to achieve same goal"""
        # Check if alternative methods exist for this task type
        # This would involve replanning with different approaches
        self.get_logger().info(f'Using alternative method for task {task_id}')
        return False  # Placeholder - implement based on specific needs

    def request_human_assistance(self, task_id, error):
        """Request human assistance for failed task"""
        self.get_logger().info(f'Requesting human assistance for task {task_id}: {error}')
        # Publish to human assistance topic
        assistance_msg = String()
        assistance_msg.data = json.dumps({
            'task_id': task_id,
            'error': error,
            'request_time': self.get_clock().now().nanoseconds
        })
        # Would need a publisher for human assistance requests
        return True

def main(args=None):
    rclpy.init(args=args)
    node = HTNPlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down HTN Planner Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### State Tracking and World Modeling

Effective cognitive planning requires maintaining accurate state information:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

@dataclass
class ObjectState:
    name: str
    pose: Optional[PoseStamped] = None
    bounding_box: Optional[Dict[str, float]] = None  # x, y, width, height
    confidence: float = 0.0
    last_seen: Optional[datetime] = None
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LocationState:
    name: str
    pose: Optional[PoseStamped] = None
    description: str = ""
    accessibility: float = 1.0  # 0.0 to 1.0, where 1.0 is fully accessible

@dataclass
class RobotState:
    pose: Optional[PoseStamped] = None
    status: str = "idle"  # idle, moving, grasping, etc.
    gripper_status: str = "open"  # open, closed, occupied
    battery_level: float = 100.0
    current_task: Optional[str] = None

class StateTrackerNode(Node):
    def __init__(self):
        super().__init__('state_tracker_node')

        # Publishers
        self.state_publisher = self.create_publisher(
            String,
            '/world_state',
            10
        )

        # Subscribers
        self.robot_pose_subscriber = self.create_subscription(
            PoseStamped,
            '/robot_pose',
            self.robot_pose_callback,
            10
        )

        self.object_detection_subscriber = self.create_subscription(
            String,
            '/object_detections',
            self.object_detection_callback,
            10
        )

        self.location_subscriber = self.create_subscription(
            String,
            '/location_map',
            self.location_callback,
            10
        )

        self.task_status_subscriber = self.create_subscription(
            String,
            '/task_status',
            self.task_status_callback,
            10
        )

        # Initialize components
        self.bridge = CvBridge()
        self.robot_state = RobotState()
        self.object_states: Dict[str, ObjectState] = {}
        self.location_states: Dict[str, LocationState] = {}
        self.state_history = []

        # State update timer
        self.state_update_timer = self.create_timer(1.0, self.publish_state)

        self.get_logger().info('State Tracker Node initialized')

    def robot_pose_callback(self, msg):
        """Update robot pose in state"""
        self.robot_state.pose = msg
        self.robot_state.last_updated = datetime.now()

        # Update robot accessibility based on current pose
        self.update_robot_accessibility()

    def object_detection_callback(self, msg):
        """Update object states from detection results"""
        try:
            detections = json.loads(msg.data)

            for detection in detections.get('objects', []):
                obj_name = detection.get('name', 'unknown')
                confidence = detection.get('confidence', 0.0)

                # Only update if confidence is high enough
                if confidence > 0.5:
                    obj_state = ObjectState(
                        name=obj_name,
                        confidence=confidence,
                        last_seen=datetime.now()
                    )

                    # Set pose if available
                    if 'pose' in detection:
                        pose_msg = PoseStamped()
                        pose_msg.header.stamp = self.get_clock().now().to_msg()
                        pose_msg.header.frame_id = detection['pose'].get('frame_id', 'map')
                        pose_msg.pose.position.x = detection['pose'].get('x', 0.0)
                        pose_msg.pose.position.y = detection['pose'].get('y', 0.0)
                        pose_msg.pose.position.z = detection['pose'].get('z', 0.0)
                        obj_state.pose = pose_msg

                    # Set bounding box if available
                    if 'bbox' in detection:
                        obj_state.bounding_box = detection['bbox']

                    # Set properties
                    if 'properties' in detection:
                        obj_state.properties = detection['properties']

                    self.object_states[obj_name] = obj_state

            self.get_logger().info(f'Updated {len(detections.get("objects", []))} objects in state')

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in object detection message')

    def location_callback(self, msg):
        """Update location states from location map"""
        try:
            locations = json.loads(msg.data)

            for location_name, location_data in locations.items():
                location_state = LocationState(
                    name=location_name,
                    description=location_data.get('description', ''),
                    accessibility=location_data.get('accessibility', 1.0)
                )

                # Set pose if available
                if 'pose' in location_data:
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_msg.header.frame_id = location_data['pose'].get('frame_id', 'map')
                    pose_msg.pose.position.x = location_data['pose'].get('x', 0.0)
                    pose_msg.pose.position.y = location_data['pose'].get('y', 0.0)
                    pose_msg.pose.position.z = location_data['pose'].get('z', 0.0)
                    location_state.pose = pose_msg

                self.location_states[location_name] = location_state

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in location message')

    def task_status_callback(self, msg):
        """Update robot task status"""
        try:
            task_data = json.loads(msg.data)
            self.robot_state.current_task = task_data.get('current_task')
            self.robot_state.status = task_data.get('robot_status', 'idle')

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in task status message')

    def update_robot_accessibility(self):
        """Update accessibility of locations based on robot position"""
        if not self.robot_state.pose:
            return

        robot_pos = np.array([
            self.robot_state.pose.pose.position.x,
            self.robot_state.pose.pose.position.y
        ])

        for location_name, location_state in self.location_states.items():
            if location_state.pose:
                location_pos = np.array([
                    location_state.pose.pose.position.x,
                    location_state.pose.pose.position.y
                ])

                distance = np.linalg.norm(robot_pos - location_pos)

                # Update accessibility based on distance (simplified model)
                if distance < 1.0:  # Within 1 meter
                    location_state.accessibility = 1.0
                elif distance < 5.0:  # Within 5 meters
                    location_state.accessibility = 0.7
                else:  # Far away
                    location_state.accessibility = 0.3

    def get_reachable_objects(self, max_distance=2.0):
        """Get objects that are currently reachable"""
        if not self.robot_state.pose:
            return []

        robot_pos = np.array([
            self.robot_state.pose.pose.position.x,
            self.robot_state.pose.pose.position.y
        ])

        reachable_objects = []
        for obj_name, obj_state in self.object_states.items():
            if obj_state.pose and obj_state.confidence > 0.7:  # High confidence
                obj_pos = np.array([
                    obj_state.pose.pose.position.x,
                    obj_state.pose.pose.position.y
                ])

                distance = np.linalg.norm(robot_pos - obj_pos)
                if distance <= max_distance:
                    reachable_objects.append(obj_name)

        return reachable_objects

    def get_accessible_locations(self, min_accessibility=0.5):
        """Get locations that are accessible"""
        accessible_locations = []
        for loc_name, loc_state in self.location_states.items():
            if loc_state.accessibility >= min_accessibility:
                accessible_locations.append(loc_name)

        return accessible_locations

    def get_object_pose(self, object_name):
        """Get pose of a specific object"""
        obj_state = self.object_states.get(object_name)
        if obj_state and obj_state.pose:
            return obj_state.pose
        return None

    def get_location_pose(self, location_name):
        """Get pose of a specific location"""
        loc_state = self.location_states.get(location_name)
        if loc_state and loc_state.pose:
            return loc_state.pose
        return None

    def publish_state(self):
        """Publish current world state"""
        state_msg = String()
        state_msg.data = json.dumps({
            'timestamp': datetime.now().isoformat(),
            'robot_state': {
                'pose': {
                    'x': self.robot_state.pose.pose.position.x if self.robot_state.pose else 0.0,
                    'y': self.robot_state.pose.pose.position.y if self.robot_state.pose else 0.0,
                    'z': self.robot_state.pose.pose.position.z if self.robot_state.pose else 0.0
                } if self.robot_state.pose else None,
                'status': self.robot_state.status,
                'gripper_status': self.robot_state.gripper_status,
                'battery_level': self.robot_state.battery_level,
                'current_task': self.robot_state.current_task
            },
            'objects': {
                name: {
                    'pose': {
                        'x': obj.pose.pose.position.x if obj.pose else 0.0,
                        'y': obj.pose.pose.position.y if obj.pose else 0.0,
                        'z': obj.pose.pose.position.z if obj.pose else 0.0
                    } if obj.pose else None,
                    'confidence': obj.confidence,
                    'last_seen': obj.last_seen.isoformat() if obj.last_seen else None,
                    'properties': obj.properties
                }
                for name, obj in self.object_states.items()
            },
            'locations': {
                name: {
                    'pose': {
                        'x': loc.pose.pose.position.x if loc.pose else 0.0,
                        'y': loc.pose.pose.position.y if loc.pose else 0.0,
                        'z': loc.pose.pose.position.z if loc.pose else 0.0
                    } if loc.pose else None,
                    'description': loc.description,
                    'accessibility': loc.accessibility
                }
                for name, loc in self.location_states.items()
            },
            'reachable_objects': self.get_reachable_objects(),
            'accessible_locations': self.get_accessible_locations()
        })

        self.state_publisher.publish(state_msg)

    def get_state_summary(self):
        """Get a summary of the current state for planning"""
        return {
            'robot_pose': self.robot_state.pose,
            'robot_status': self.robot_state.status,
            'objects': list(self.object_states.keys()),
            'locations': list(self.location_states.keys()),
            'reachable_objects': self.get_reachable_objects(),
            'accessible_locations': self.get_accessible_locations()
        }

def main(args=None):
    rclpy.init(args=args)
    node = StateTrackerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down State Tracker Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Action Selection and Execution Planning

Selecting appropriate actions based on current state and goals:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import LaserScan
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ActionType(Enum):
    NAVIGATE = "navigate"
    GRASP = "grasp"
    PLACE = "place"
    DETECT = "detect"
    COMMUNICATE = "communicate"
    WAIT = "wait"

@dataclass
class Action:
    type: ActionType
    parameters: Dict[str, any]
    cost: float
    precondition: str
    effect: str

class ActionSelectorNode(Node):
    def __init__(self):
        super().__init__('action_selector_node')

        # Publishers
        self.action_publisher = self.create_publisher(
            String,
            '/selected_action',
            10
        )

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Subscribers
        self.state_subscriber = self.create_subscription(
            String,
            '/world_state',
            self.state_callback,
            10
        )

        self.task_subscriber = self.create_subscription(
            String,
            '/current_task',
            self.task_callback,
            10
        )

        self.laser_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        # Action selection state
        self.current_state = {}
        self.current_task = None
        self.action_history = []
        self.obstacle_map = None

        # Define action templates
        self.action_templates = {
            ActionType.NAVIGATE: self.navigate_action,
            ActionType.GRASP: self.grasp_action,
            ActionType.PLACE: self.place_action,
            ActionType.DETECT: self.detect_action,
            ActionType.COMMUNICATE: self.communicate_action,
            ActionType.WAIT: self.wait_action
        }

        self.get_logger().info('Action Selector Node initialized')

    def state_callback(self, msg):
        """Update current state"""
        try:
            self.current_state = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in state message')

    def task_callback(self, msg):
        """Update current task"""
        try:
            self.current_task = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in task message')

    def laser_callback(self, msg):
        """Update obstacle information"""
        # Store laser scan for navigation planning
        self.obstacle_map = msg

    def select_best_action(self):
        """Select the best action based on current state and task"""
        if not self.current_state or not self.current_task:
            return None

        # Get relevant information
        robot_pose = self.current_state.get('robot_state', {}).get('pose', {})
        reachable_objects = self.current_state.get('reachable_objects', [])
        accessible_locations = self.current_state.get('accessible_locations', [])

        task_type = self.current_task.get('type', '')
        task_parameters = self.current_task.get('parameters', {})

        # Generate possible actions based on task type
        possible_actions = []

        if task_type == 'fetch_object':
            object_name = task_parameters.get('object_name', '')
            if object_name in reachable_objects:
                # Grasp the object
                grasp_action = self.grasp_action(object_name)
                possible_actions.append(grasp_action)
            else:
                # Navigate to object
                if object_name in self.current_state.get('objects', {}):
                    target_pose = self.current_state['objects'][object_name].get('pose', {})
                    if target_pose:
                        navigate_action = self.navigate_action(target_pose)
                        possible_actions.append(navigate_action)

        elif task_type == 'place_object':
            destination = task_parameters.get('destination', '')
            if destination in accessible_locations:
                target_pose = self.current_state['locations'][destination].get('pose', {})
                if target_pose:
                    navigate_action = self.navigate_action(target_pose)
                    place_action = self.place_action(destination)
                    possible_actions.extend([navigate_action, place_action])

        elif task_type == 'navigate_to':
            target_location = task_parameters.get('location', '')
            if target_location in accessible_locations:
                target_pose = self.current_state['locations'][target_location].get('pose', {})
                if target_pose:
                    navigate_action = self.navigate_action(target_pose)
                    possible_actions.append(navigate_action)

        elif task_type == 'detect_object':
            detect_action = self.detect_action(task_parameters.get('object_type', 'any'))
            possible_actions.append(detect_action)

        # Evaluate actions and select best one
        if possible_actions:
            best_action = min(possible_actions, key=lambda x: x.cost)
            return best_action

        return None

    def navigate_action(self, target_pose):
        """Create navigation action"""
        # Calculate distance to target
        robot_x = self.current_state.get('robot_state', {}).get('pose', {}).get('x', 0.0)
        robot_y = self.current_state.get('robot_state', {}).get('pose', {}).get('y', 0.0)
        target_x = target_pose.get('x', 0.0)
        target_y = target_pose.get('y', 0.0)

        distance = np.sqrt((target_x - robot_x)**2 + (target_y - robot_y)**2)

        # Calculate cost (simplified: distance + obstacle consideration)
        base_cost = distance
        if self.obstacle_map:
            # Check for obstacles along path (simplified)
            min_range = min(self.obstacle_map.ranges) if self.obstacle_map.ranges else float('inf')
            if min_range < 0.5:  # Obstacle within 0.5m
                base_cost *= 2.0  # Increase cost for obstacle avoidance

        action = Action(
            type=ActionType.NAVIGATE,
            parameters={
                'target_pose': target_pose,
                'distance': distance
            },
            cost=base_cost,
            precondition='robot_operational and path_clear',
            effect='robot_moved_to_target'
        )

        return action

    def grasp_action(self, object_name):
        """Create grasp action"""
        # Check if gripper is available
        gripper_status = self.current_state.get('robot_state', {}).get('gripper_status', 'open')

        cost = 1.0  # Base cost
        if gripper_status != 'open':
            cost += 2.0  # Higher cost if gripper needs to be opened first

        action = Action(
            type=ActionType.GRASP,
            parameters={
                'object_name': object_name,
                'object_pose': self.current_state.get('objects', {}).get(object_name, {}).get('pose', {})
            },
            cost=cost,
            precondition=f'gripper_available and {object_name}_reachable',
            effect=f'{object_name}_grasped and gripper_occupied'
        )

        return action

    def place_action(self, location_name):
        """Create place action"""
        cost = 1.5  # Base cost for placing

        action = Action(
            type=ActionType.PLACE,
            parameters={
                'location_name': location_name,
                'location_pose': self.current_state.get('locations', {}).get(location_name, {}).get('pose', {})
            },
            cost=cost,
            precondition='object_grasped and location_accessible',
            effect='object_placed and gripper_free'
        )

        return action

    def detect_action(self, object_type):
        """Create detection action"""
        cost = 0.5  # Relatively low cost for detection

        action = Action(
            type=ActionType.DETECT,
            parameters={
                'object_type': object_type,
                'search_area': 'current_view'
            },
            cost=cost,
            precondition='camera_operational',
            effect=f'{object_type}_detected'
        )

        return action

    def communicate_action(self, message):
        """Create communication action"""
        cost = 0.1  # Very low cost for communication

        action = Action(
            type=ActionType.COMMUNICATE,
            parameters={
                'message': message,
                'recipient': 'user'
            },
            cost=cost,
            precondition='communication_system_operational',
            effect='message_delivered'
        )

        return action

    def wait_action(self, duration=1.0):
        """Create wait action"""
        cost = duration * 0.1  # Cost proportional to wait time

        action = Action(
            type=ActionType.WAIT,
            parameters={
                'duration': duration
            },
            cost=cost,
            precondition='robot_idle',
            effect='wait_completed'
        )

        return action

    def execute_action(self, action):
        """Execute the selected action"""
        if not action:
            return

        self.get_logger().info(f'Executing action: {action.type} with parameters {action.parameters}')

        # Publish action for execution
        action_msg = String()
        action_msg.data = json.dumps({
            'type': action.type.value,
            'parameters': action.parameters,
            'cost': action.cost,
            'timestamp': self.get_clock().now().nanoseconds
        })

        self.action_publisher.publish(action_msg)

        # For navigation actions, also publish velocity commands (simplified)
        if action.type == ActionType.NAVIGATE:
            self.execute_navigation(action.parameters)

        # Add to action history
        self.action_history.append({
            'action': action,
            'timestamp': self.get_clock().now().nanoseconds
        })

    def execute_navigation(self, params):
        """Execute navigation by publishing velocity commands"""
        target_pose = params.get('target_pose', {})
        current_pose = self.current_state.get('robot_state', {}).get('pose', {})

        if not target_pose or not current_pose:
            return

        # Calculate direction to target
        dx = target_pose.get('x', 0.0) - current_pose.get('x', 0.0)
        dy = target_pose.get('y', 0.0) - current_pose.get('y', 0.0)
        distance = np.sqrt(dx**2 + dy**2)

        if distance > 0.1:  # If not already at target
            # Simple proportional controller
            cmd_vel = Twist()
            cmd_vel.linear.x = min(0.3, distance * 0.5)  # Forward speed
            cmd_vel.angular.z = np.arctan2(dy, dx) * 0.5  # Turn toward target

            self.cmd_vel_publisher.publish(cmd_vel)

    def plan_and_execute(self):
        """Main planning and execution loop"""
        best_action = self.select_best_action()

        if best_action:
            self.execute_action(best_action)
        else:
            # If no suitable action found, request clarification or wait
            self.get_logger().warn('No suitable action found, waiting for new state/task')
            wait_action = self.wait_action(2.0)
            self.execute_action(wait_action)

def main(args=None):
    rclpy.init(args=args)
    node = ActionSelectorNode()

    # Create timer for planning loop
    node.create_timer(0.5, node.plan_and_execute)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Action Selector Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Handling Ambiguous or Complex Commands

Robots need to handle ambiguous commands gracefully:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cohere
import json
import re
from typing import Dict, List, Optional

class AmbiguityResolverNode(Node):
    def __init__(self):
        super().__init__('ambiguity_resolver_node')

        # Publishers
        self.clarification_publisher = self.create_publisher(
            String,
            '/clarification_request',
            10
        )

        self.resolved_command_publisher = self.create_publisher(
            String,
            '/resolved_command',
            10
        )

        # Subscribers
        self.command_subscriber = self.create_subscription(
            String,
            '/raw_command',
            self.command_callback,
            10
        )

        self.state_subscriber = self.create_subscription(
            String,
            '/world_state',
            self.state_callback,
            10
        )

        # Initialize
        self.cohere_client = cohere.Client(self.get_parameter_or('cohere_api_key', 'your-api-key-here').value)
        self.current_state = {}
        self.pending_resolutions = {}

        # Ambiguity patterns
        self.ambiguity_patterns = {
            'vague_object_reference': [
                r'\bthe (object|item|thing)\b',
                r'\bit\b',
                r'\bthat\b',
                r'\bthis\b'
            ],
            'vague_location_reference': [
                r'\bthere\b',
                r'\bover there\b',
                r'\bthat place\b',
                r'\bthe area\b'
            ],
            'unclear_action': [
                r'\bdo something\b',
                r'\bhelp me\b',
                r'\bwork on\b'
            ]
        }

        self.get_logger().info('Ambiguity Resolver Node initialized')

    def command_callback(self, msg):
        """Process incoming commands for ambiguity"""
        command = msg.data
        self.get_logger().info(f'Processing command for ambiguity: {command}')

        # Check for ambiguity
        ambiguity_issues = self.detect_ambiguity(command)

        if ambiguity_issues:
            # Request clarification
            clarification_request = self.generate_clarification_request(
                command, ambiguity_issues
            )
            self.request_clarification(command, clarification_request, ambiguity_issues)
        else:
            # Command is clear, publish as resolved
            resolved_msg = String()
            resolved_msg.data = json.dumps({
                'original_command': command,
                'resolved_command': command,
                'ambiguity_resolved': True,
                'resolution_type': 'no_ambiguity'
            })
            self.resolved_command_publisher.publish(resolved_msg)

    def state_callback(self, msg):
        """Update current state"""
        try:
            self.current_state = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in state message')

    def detect_ambiguity(self, command):
        """Detect various types of ambiguity in the command"""
        issues = []

        command_lower = command.lower()

        # Check for vague object references
        for pattern in self.ambiguity_patterns['vague_object_reference']:
            if re.search(pattern, command_lower):
                issues.append({
                    'type': 'vague_object_reference',
                    'pattern': pattern,
                    'text': re.search(pattern, command_lower).group()
                })

        # Check for vague location references
        for pattern in self.ambiguity_patterns['vague_location_reference']:
            if re.search(pattern, command_lower):
                issues.append({
                    'type': 'vague_location_reference',
                    'pattern': pattern,
                    'text': re.search(pattern, command_lower).group()
                })

        # Check for unclear actions
        for pattern in self.ambiguity_patterns['unclear_action']:
            if re.search(pattern, command_lower):
                issues.append({
                    'type': 'unclear_action',
                    'pattern': pattern,
                    'text': re.search(pattern, command_lower).group()
                })

        # Additional semantic analysis
        if self.has_ambiguous_pronouns(command):
            issues.append({
                'type': 'ambiguous_pronoun_reference',
                'text': 'ambiguous pronoun usage'
            })

        return issues

    def has_ambiguous_pronouns(self, command):
        """Check for ambiguous pronoun usage"""
        pronouns = ['it', 'that', 'this', 'they', 'them', 'their']
        command_lower = command.lower()

        # Simple check: if pronouns exist without clear antecedents
        for pronoun in pronouns:
            if pronoun in command_lower:
                # In a real implementation, you'd use more sophisticated NLP
                # For now, flag any pronoun usage as potentially ambiguous
                return True
        return False

    def generate_clarification_request(self, command, issues):
        """Generate a clarification request using LLM"""
        # Build context from current state
        context = self.current_state.get('objects', {})
        locations = self.current_state.get('locations', {})

        prompt = f"""
        Command: "{command}"
        Detected Issues: {issues}
        Current Context:
        - Objects: {list(context.keys()) if context else 'none detected'}
        - Locations: {list(locations.keys()) if locations else 'none known'}

        Generate a polite clarification request that addresses the ambiguity.
        Be specific about what information is needed.
        """

        try:
            response = self.cohere_client.chat(
                model="command-r-plus",
                message=prompt,
                preamble="You are a helpful robot assistant. When users give ambiguous commands, ask specific questions to clarify their intent.",
                temperature=0.7
            )

            return response.text

        except Exception as e:
            self.get_logger().error(f'LLM clarification generation error: {e}')
            return self.fallback_clarification_request(command, issues)

    def fallback_clarification_request(self, command, issues):
        """Generate fallback clarification request without LLM"""
        clarification = "I'm not sure I understand your command clearly. "

        issue_types = [issue['type'] for issue in issues]

        if 'vague_object_reference' in issue_types:
            clarification += "Could you specify which object you're referring to? "

        if 'vague_location_reference' in issue_types:
            clarification += "Could you be more specific about the location? "

        if 'unclear_action' in issue_types:
            clarification += "Could you clarify what specific action you'd like me to take? "

        clarification += f"Your command was: '{command}'"

        return clarification

    def request_clarification(self, original_command, clarification_request, issues):
        """Request clarification from user"""
        clarification_msg = String()
        clarification_msg.data = json.dumps({
            'original_command': original_command,
            'clarification_request': clarification_request,
            'detected_issues': issues,
            'request_id': f"clarify_{self.get_clock().now().nanoseconds}",
            'timestamp': self.get_clock().now().nanoseconds
        })

        self.clarification_publisher.publish(clarification_msg)
        self.get_logger().info(f'Requested clarification: {clarification_request}')

        # Store for tracking
        request_id = f"clarify_{self.get_clock().now().nanoseconds}"
        self.pending_resolutions[request_id] = {
            'original_command': original_command,
            'issues': issues,
            'request_time': self.get_clock().now().nanoseconds
        }

    def resolve_with_clarification(self, request_id, user_response):
        """Resolve original command with user's clarification"""
        if request_id not in self.pending_resolutions:
            self.get_logger().warn(f'No pending resolution for request ID: {request_id}')
            return

        original_info = self.pending_resolutions[request_id]
        original_command = original_info['original_command']

        # Use LLM to integrate clarification with original command
        prompt = f"""
        Original ambiguous command: "{original_command}"
        User clarification: "{user_response}"
        Issues that needed clarification: {original_info['issues']}

        Generate a clear, unambiguous command that incorporates the user's clarification.
        """

        try:
            response = self.cohere_client.chat(
                model="command-r-plus",
                message=prompt,
                preamble="You are a command clarifier. Generate a clear, specific command based on the original ambiguous command and user clarification.",
                temperature=0.3
            )

            resolved_command = response.text

            # Publish resolved command
            resolved_msg = String()
            resolved_msg.data = json.dumps({
                'original_command': original_command,
                'user_clarification': user_response,
                'resolved_command': resolved_command,
                'request_id': request_id,
                'ambiguity_resolved': True,
                'resolution_type': 'user_clarification'
            })

            self.resolved_command_publisher.publish(resolved_msg)
            self.get_logger().info(f'Resolved command: {resolved_command}')

            # Remove from pending
            del self.pending_resolutions[request_id]

        except Exception as e:
            self.get_logger().error(f'Command resolution error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = AmbiguityResolverNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Ambiguity Resolver Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for Cognitive Planning

1. **Modularity**: Design planning components to be modular and reusable
2. **Robustness**: Handle unexpected situations and failures gracefully
3. **Efficiency**: Optimize planning algorithms for real-time performance
4. **Context Awareness**: Consider environmental context in planning decisions
5. **Human-Robot Interaction**: Design for natural and intuitive interaction
6. **Safety**: Ensure all planned actions are safe for robot and environment
7. **Verification**: Validate plans before execution

Cognitive planning is essential for creating intelligent robotic systems that can understand high-level commands and execute them reliably in complex environments.