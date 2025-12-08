# Task Execution Framework

## Introduction

Task execution represents the culmination of all integrated systems in our autonomous humanoid robot. This module builds upon the system integration components to execute complex, multi-step commands from natural language inputs to physical actions. The task execution framework orchestrates perception, planning, navigation, and manipulation capabilities to complete autonomous missions.

## Core Task Execution Architecture

The task execution framework operates as a hierarchical state machine that decomposes high-level commands into executable actions. This architecture ensures robust execution while maintaining flexibility for complex mission scenarios.

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point
from action_msgs.msg import GoalStatus
import time

class TaskState(Enum):
    """Enumeration of possible task execution states"""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETED = "completed"

class TaskType(Enum):
    """Enumeration of task types"""
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    COMPOSITE = "composite"

@dataclass
class Task:
    """Data structure for a single task"""
    id: str
    type: TaskType
    description: str
    priority: int = 1
    dependencies: List[str] = None
    parameters: Dict[str, Any] = None
    timeout: float = 30.0
    created_time: float = 0.0

@dataclass
class TaskResult:
    """Data structure for task execution results"""
    task_id: str
    status: TaskState
    success: bool
    message: str
    execution_time: float
    data: Dict[str, Any] = None
```

## Task Planning and Decomposition

The task planning system decomposes complex natural language commands into executable subtasks. This process involves cognitive planning algorithms that understand spatial relationships, object properties, and action sequences.

```python
class TaskPlanner(Node):
    """Node responsible for task planning and decomposition"""

    def __init__(self):
        super().__init__('task_planner')
        self.task_queue = []
        self.completed_tasks = []
        self.failed_tasks = []

        # Publishers and subscribers
        self.command_sub = self.create_subscription(
            String, 'natural_language_command', self.command_callback, 10)
        self.task_pub = self.create_publisher(Task, 'planned_task', 10)
        self.status_pub = self.create_publisher(String, 'task_status', 10)

        # Timer for task processing
        self.timer = self.create_timer(0.1, self.process_tasks)

    def command_callback(self, msg: String):
        """Process natural language commands and decompose into tasks"""
        command = msg.data
        self.get_logger().info(f'Received command: {command}')

        # Parse and decompose command into subtasks
        tasks = self.decompose_command(command)

        # Add tasks to queue with proper dependencies
        for task in tasks:
            self.task_queue.append(task)
            self.task_pub.publish(task)

        self.get_logger().info(f'Planned {len(tasks)} tasks from command')

    def decompose_command(self, command: str) -> List[Task]:
        """Decompose natural language command into executable tasks"""
        tasks = []

        # Example decomposition logic
        if "navigate" in command.lower() or "go to" in command.lower():
            # Extract destination from command
            destination = self.extract_destination(command)
            nav_task = Task(
                id=f"nav_{int(time.time())}",
                type=TaskType.NAVIGATION,
                description=f"Navigate to {destination}",
                parameters={"destination": destination},
                timeout=60.0
            )
            tasks.append(nav_task)

        if "pick up" in command.lower() or "grasp" in command.lower():
            # Extract object from command
            obj = self.extract_object(command)
            manip_task = Task(
                id=f"manip_{int(time.time())}",
                type=TaskType.MANIPULATION,
                description=f"Pick up {obj}",
                parameters={"object": obj},
                timeout=45.0
            )
            tasks.append(manip_task)

        if "look for" in command.lower() or "find" in command.lower():
            # Extract target from command
            target = self.extract_target(command)
            perception_task = Task(
                id=f"perception_{int(time.time())}",
                type=TaskType.PERCEPTION,
                description=f"Look for {target}",
                parameters={"target": target},
                timeout=30.0
            )
            tasks.append(perception_task)

        return tasks

    def extract_destination(self, command: str) -> str:
        """Extract destination from natural language command"""
        # Simple keyword-based extraction (in practice, use NLP)
        if "table" in command:
            return "table"
        elif "kitchen" in command:
            return "kitchen"
        elif "living room" in command:
            return "living_room"
        else:
            return "unknown_location"

    def extract_object(self, command: str) -> str:
        """Extract object to manipulate from natural language command"""
        if "red cube" in command:
            return "red_cube"
        elif "blue cube" in command:
            return "blue_cube"
        elif "green ball" in command:
            return "green_ball"
        else:
            return "unknown_object"

    def extract_target(self, command: str) -> str:
        """Extract target for perception from natural language command"""
        if "person" in command:
            return "person"
        elif "object" in command:
            return "object"
        elif "face" in command:
            return "face"
        else:
            return "unknown_target"

    def process_tasks(self):
        """Process tasks in the queue"""
        if not self.task_queue:
            return

        # Process next task in queue
        current_task = self.task_queue[0]

        # Check dependencies
        if self.check_dependencies(current_task):
            # Execute task
            result = self.execute_task(current_task)
            self.handle_task_result(current_task, result)

            # Remove completed task from queue
            self.task_queue.pop(0)

    def check_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are satisfied"""
        if not task.dependencies:
            return True

        for dep_id in task.dependencies:
            if dep_id not in [t.id for t in self.completed_tasks]:
                return False
        return True

    def execute_task(self, task: Task) -> TaskResult:
        """Execute a single task"""
        start_time = time.time()

        try:
            if task.type == TaskType.NAVIGATION:
                result = self.execute_navigation_task(task)
            elif task.type == TaskType.MANIPULATION:
                result = self.execute_manipulation_task(task)
            elif task.type == TaskType.PERCEPTION:
                result = self.execute_perception_task(task)
            else:
                result = TaskResult(
                    task_id=task.id,
                    status=TaskState.FAILED,
                    success=False,
                    message="Unknown task type",
                    execution_time=time.time() - start_time
                )

        except Exception as e:
            result = TaskResult(
                task_id=task.id,
                status=TaskState.FAILED,
                success=False,
                message=f"Task execution failed: {str(e)}",
                execution_time=time.time() - start_time
            )

        return result

    def execute_navigation_task(self, task: Task) -> TaskResult:
        """Execute navigation task"""
        # Publish navigation goal to navigation system
        # This would integrate with the navigation system created in system-integration.md
        destination = task.parameters.get("destination", "unknown")

        # Simulate navigation execution
        self.get_logger().info(f"Executing navigation to {destination}")

        # In real implementation, this would send goal to Nav2
        # For simulation, we'll just wait and return success
        time.sleep(2.0)  # Simulate navigation time

        return TaskResult(
            task_id=task.id,
            status=TaskState.COMPLETED,
            success=True,
            message=f"Successfully navigated to {destination}",
            execution_time=2.0
        )

    def execute_manipulation_task(self, task: Task) -> TaskResult:
        """Execute manipulation task"""
        obj = task.parameters.get("object", "unknown")

        # Simulate manipulation execution
        self.get_logger().info(f"Executing manipulation of {obj}")

        # In real implementation, this would control robot arms/hands
        # For simulation, we'll just wait and return success
        time.sleep(3.0)  # Simulate manipulation time

        return TaskResult(
            task_id=task.id,
            status=TaskState.COMPLETED,
            success=True,
            message=f"Successfully manipulated {obj}",
            execution_time=3.0
        )

    def execute_perception_task(self, task: Task) -> TaskResult:
        """Execute perception task"""
        target = task.parameters.get("target", "unknown")

        # Simulate perception execution
        self.get_logger().info(f"Executing perception of {target}")

        # In real implementation, this would use perception system
        # For simulation, we'll just wait and return success
        time.sleep(1.5)  # Simulate perception time

        return TaskResult(
            task_id=task.id,
            status=TaskState.COMPLETED,
            success=True,
            message=f"Successfully perceived {target}",
            execution_time=1.5
        )

    def handle_task_result(self, task: Task, result: TaskResult):
        """Handle task execution result"""
        if result.success:
            self.completed_tasks.append(task)
            self.get_logger().info(f"Task {task.id} completed successfully")
        else:
            self.failed_tasks.append(task)
            self.get_logger().error(f"Task {task.id} failed: {result.message}")

        # Update task status
        status_msg = String()
        status_msg.data = f"Task {task.id}: {result.status.value}"
        self.status_pub.publish(status_msg)
```

## Multi-Step Task Execution

Complex commands require coordination of multiple subtasks. The multi-step execution system manages dependencies and ensures proper sequencing of operations.

```python
class MultiStepTaskExecutor(Node):
    """Node for executing complex multi-step tasks"""

    def __init__(self):
        super().__init__('multi_step_executor')
        self.active_mission = None
        self.mission_queue = []

        # Mission control publishers/subscribers
        self.mission_sub = self.create_subscription(
            String, 'mission_command', self.mission_callback, 10)
        self.mission_status_pub = self.create_publisher(String, 'mission_status', 10)

        # Timer for mission processing
        self.mission_timer = self.create_timer(0.5, self.process_missions)

    def mission_callback(self, msg: String):
        """Process multi-step mission commands"""
        command = msg.data
        self.get_logger().info(f'Received mission: {command}')

        # Create mission from command
        mission = self.create_mission_from_command(command)
        self.mission_queue.append(mission)

        if not self.active_mission:
            self.start_next_mission()

    def create_mission_from_command(self, command: str) -> List[Task]:
        """Create mission plan from natural language command"""
        # Example: "Go to the kitchen, find the red cup, pick it up, and bring it to the table"
        mission_tasks = []

        if "go to the kitchen" in command:
            nav_task = Task(
                id=f"mission_nav_{int(time.time())}",
                type=TaskType.NAVIGATION,
                description="Navigate to kitchen",
                parameters={"destination": "kitchen"},
                timeout=60.0
            )
            mission_tasks.append(nav_task)

        if "find the red cup" in command:
            find_task = Task(
                id=f"mission_find_{int(time.time())}",
                type=TaskType.PERCEPTION,
                description="Find red cup",
                parameters={"target": "red_cup"},
                timeout=30.0
            )
            find_task.dependencies = [nav_task.id]  # Depends on navigation
            mission_tasks.append(find_task)

        if "pick it up" in command:
            pick_task = Task(
                id=f"mission_pick_{int(time.time())}",
                type=TaskType.MANIPULATION,
                description="Pick up red cup",
                parameters={"object": "red_cup"},
                timeout=45.0
            )
            pick_task.dependencies = [find_task.id]  # Depends on finding
            mission_tasks.append(pick_task)

        if "bring it to the table" in command:
            return_task = Task(
                id=f"mission_return_{int(time.time())}",
                type=TaskType.NAVIGATION,
                description="Return to table with red cup",
                parameters={"destination": "table"},
                timeout=60.0
            )
            return_task.dependencies = [pick_task.id]  # Depends on picking up
            mission_tasks.append(return_task)

        return mission_tasks

    def start_next_mission(self):
        """Start the next mission in the queue"""
        if not self.mission_queue:
            return

        self.active_mission = self.mission_queue.pop(0)
        self.get_logger().info(f'Starting mission with {len(self.active_mission)} tasks')

        # Publish mission start status
        status_msg = String()
        status_msg.data = f"Starting mission with {len(self.active_mission)} tasks"
        self.mission_status_pub.publish(status_msg)

    def process_missions(self):
        """Process active missions"""
        if not self.active_mission:
            return

        # Check if mission is complete
        completed_count = sum(1 for task in self.active_mission
                            if task.id in [t.id for t in self.completed_tasks])

        if completed_count == len(self.active_mission):
            # Mission complete
            self.get_logger().info('Mission completed successfully')
            status_msg = String()
            status_msg.data = "Mission completed successfully"
            self.mission_status_pub.publish(status_msg)

            # Start next mission if available
            self.active_mission = None
            if self.mission_queue:
                self.start_next_mission()
```

## Error Handling and Recovery

Robust task execution requires comprehensive error handling and recovery mechanisms to handle failures gracefully.

```python
class TaskRecoverySystem(Node):
    """Node for task error handling and recovery"""

    def __init__(self):
        super().__init__('task_recovery_system')
        self.recovery_strategies = {
            "navigation_failure": self.recovery_navigation,
            "manipulation_failure": self.recovery_manipulation,
            "perception_failure": self.recovery_perception
        }

        self.failure_sub = self.create_subscription(
            String, 'task_failure', self.failure_callback, 10)
        self.recovery_pub = self.create_publisher(String, 'recovery_action', 10)

    def failure_callback(self, msg: String):
        """Handle task failures and attempt recovery"""
        failure_info = msg.data
        self.get_logger().error(f'Task failure detected: {failure_info}')

        # Determine failure type and apply recovery strategy
        if "navigation" in failure_info:
            self.attempt_recovery("navigation_failure", failure_info)
        elif "manipulation" in failure_info:
            self.attempt_recovery("manipulation_failure", failure_info)
        elif "perception" in failure_info:
            self.attempt_recovery("perception_failure", failure_info)
        else:
            self.attempt_recovery("general_failure", failure_info)

    def attempt_recovery(self, failure_type: str, failure_info: str):
        """Attempt to recover from a specific type of failure"""
        if failure_type in self.recovery_strategies:
            recovery_func = self.recovery_strategies[failure_type]
            success = recovery_func(failure_info)

            if success:
                self.get_logger().info(f'Recovery successful for {failure_type}')
                recovery_msg = String()
                recovery_msg.data = f"Recovery successful: {failure_type}"
                self.recovery_pub.publish(recovery_msg)
            else:
                self.get_logger().error(f'Recovery failed for {failure_type}')
                # Escalate to human operator or abort mission
                self.escalate_failure(failure_type, failure_info)
        else:
            self.get_logger().error(f'No recovery strategy for {failure_type}')

    def recovery_navigation(self, failure_info: str) -> bool:
        """Recovery strategy for navigation failures"""
        # Try alternative path
        # Retry with different parameters
        # Fall back to manual control if needed
        self.get_logger().info('Attempting navigation recovery')
        return True  # Simulate successful recovery

    def recovery_manipulation(self, failure_info: str) -> bool:
        """Recovery strategy for manipulation failures"""
        # Adjust grasp parameters
        # Retry with different approach angle
        # Request human assistance if needed
        self.get_logger().info('Attempting manipulation recovery')
        return True  # Simulate successful recovery

    def recovery_perception(self, failure_info: str) -> bool:
        """Recovery strategy for perception failures"""
        # Adjust camera parameters
        # Retry with different lighting conditions
        # Use alternative sensors
        self.get_logger().info('Attempting perception recovery')
        return True  # Simulate successful recovery

    def escalate_failure(self, failure_type: str, failure_info: str):
        """Escalate failure to human operator"""
        self.get_logger().warn(f'Escalating {failure_type} to human operator: {failure_info}')
        # Publish to human operator interface
        # Log failure for analysis
```

## Task Execution Monitoring

Continuous monitoring ensures task execution is proceeding as expected and enables rapid response to issues.

```python
class TaskExecutionMonitor(Node):
    """Node for monitoring task execution and performance"""

    def __init__(self):
        super().__init__('task_execution_monitor')
        self.task_performance = {}
        self.mission_metrics = {}

        # Subscribers for task events
        self.task_start_sub = self.create_subscription(
            String, 'task_started', self.task_start_callback, 10)
        self.task_complete_sub = self.create_subscription(
            String, 'task_completed', self.task_complete_callback, 10)
        self.task_status_sub = self.create_subscription(
            String, 'task_status', self.task_status_callback, 10)

        # Publishers for performance metrics
        self.metrics_pub = self.create_publisher(String, 'performance_metrics', 10)

        # Timer for periodic metrics reporting
        self.metrics_timer = self.create_timer(5.0, self.report_metrics)

    def task_start_callback(self, msg: String):
        """Record task start time"""
        task_id = msg.data
        self.task_performance[task_id] = {
            'start_time': time.time(),
            'status': 'executing',
            'attempts': 1
        }

    def task_complete_callback(self, msg: String):
        """Record task completion and calculate metrics"""
        # Parse task result from message
        # Calculate execution time and success rate
        pass

    def task_status_callback(self, msg: String):
        """Update task status"""
        # Update task status in performance tracking
        pass

    def report_metrics(self):
        """Report task execution metrics"""
        # Calculate and publish performance metrics
        metrics_msg = String()
        metrics_msg.data = f"Task execution metrics: {len(self.task_performance)} tasks tracked"
        self.metrics_pub.publish(metrics_msg)
```

## Implementation and Testing

To implement the task execution framework:

1. Create a new ROS 2 package for task execution:
```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python task_execution_framework
```

2. Implement the nodes in the package:
```python
# task_execution_framework/task_execution_framework/__init__.py
from .task_planner import TaskPlanner
from .multi_step_executor import MultiStepTaskExecutor
from .task_recovery_system import TaskRecoverySystem
from .task_execution_monitor import TaskExecutionMonitor

def main(args=None):
    rclpy.init(args=args)

    # Create nodes
    task_planner = TaskPlanner()
    multi_step_executor = MultiStepTaskExecutor()
    recovery_system = TaskRecoverySystem()
    monitor = TaskExecutionMonitor()

    # Spin nodes
    executor = MultiThreadedExecutor()
    executor.add_node(task_planner)
    executor.add_node(multi_step_executor)
    executor.add_node(recovery_system)
    executor.add_node(monitor)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        task_planner.destroy_node()
        multi_step_executor.destroy_node()
        recovery_system.destroy_node()
        monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Testing the Task Execution Framework

Create comprehensive tests to validate the task execution framework:

```python
import unittest
import rclpy
from task_execution_framework.task_planner import TaskPlanner, Task, TaskType

class TestTaskExecutionFramework(unittest.TestCase):

    def setUp(self):
        rclpy.init()
        self.task_planner = TaskPlanner()

    def tearDown(self):
        self.task_planner.destroy_node()
        rclpy.shutdown()

    def test_command_decomposition(self):
        """Test that natural language commands are properly decomposed"""
        command = "Go to the kitchen and pick up the red cup"
        tasks = self.task_planner.decompose_command(command)

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].type, TaskType.NAVIGATION)
        self.assertEqual(tasks[1].type, TaskType.MANIPULATION)

    def test_task_execution(self):
        """Test that tasks execute successfully"""
        nav_task = Task(
            id="test_nav",
            type=TaskType.NAVIGATION,
            description="Test navigation",
            parameters={"destination": "test_location"}
        )

        result = self.task_planner.execute_task(nav_task)
        self.assertTrue(result.success)
        self.assertEqual(result.status.value, "completed")

if __name__ == '__main__':
    unittest.main()
```

## Integration with Existing Systems

The task execution framework integrates with the system integration components created in the previous section. The task planner receives natural language commands from the voice processing system, decomposes them into executable tasks, and coordinates with the perception, navigation, and manipulation systems to execute the mission.

This framework provides the foundation for autonomous task execution in the humanoid robot, enabling complex multi-step missions to be completed from simple voice commands while maintaining robust error handling and performance monitoring.