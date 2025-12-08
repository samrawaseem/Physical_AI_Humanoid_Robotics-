# Debugging and Performance Optimization

## Introduction

Debugging and performance optimization are critical aspects of developing autonomous humanoid robots. This module covers systematic approaches to identify, diagnose, and resolve issues in complex robotic systems, along with strategies for optimizing performance across all subsystems. Effective debugging ensures reliable operation, while performance optimization enables real-time execution of complex tasks.

## Debugging Architecture and Tools

Effective debugging in autonomous humanoid systems requires a multi-layered approach that addresses issues at the system, component, and individual node levels.

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32, Float32
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from rcl_interfaces.msg import Log
import time
import traceback
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class DebugLevel(Enum):
    """Enumeration for debug levels"""
    OFF = 0
    ERROR = 1
    WARN = 2
    INFO = 3
    DEBUG = 4
    VERBOSE = 5

class ComponentStatus(Enum):
    """Enumeration for component status"""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class DebugMessage:
    """Data structure for debug messages"""
    timestamp: float
    component: str
    level: DebugLevel
    message: str
    data: Dict[str, Any] = None

@dataclass
class PerformanceMetric:
    """Data structure for performance metrics"""
    component: str
    metric_name: str
    value: float
    unit: str
    timestamp: float
    threshold: Optional[float] = None
```

## Centralized Debugging System

The centralized debugging system collects, processes, and displays debug information from all components of the humanoid robot system.

```python
class CentralizedDebugSystem(Node):
    """Node for centralized debugging and diagnostic information"""

    def __init__(self):
        super().__init__('centralized_debug_system')

        # Debug configuration
        self.debug_level = DebugLevel.INFO
        self.max_log_entries = 1000
        self.log_buffer = []
        self.component_status = {}

        # Publishers and subscribers
        self.debug_pub = self.create_publisher(String, 'debug_messages', 10)
        self.diag_pub = self.create_publisher(DiagnosticArray, 'diagnostics', 10)
        self.log_sub = self.create_subscription(Log, '/rosout', self.log_callback, 10)

        # Custom debug subscribers for different components
        self.component_subs = {}

        # Timer for periodic diagnostic updates
        self.diag_timer = self.create_timer(1.0, self.publish_diagnostics)

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor(self)

    def log_callback(self, msg: Log):
        """Handle incoming log messages"""
        if self.should_log_level(msg.level):
            debug_msg = DebugMessage(
                timestamp=time.time(),
                component=msg.name,
                level=DebugLevel(msg.level),
                message=msg.msg,
                data={'file': msg.file, 'function': msg.function, 'line': msg.line}
            )
            self.add_debug_message(debug_msg)

    def should_log_level(self, level: int) -> bool:
        """Check if a log level should be processed"""
        return level <= self.debug_level.value

    def add_debug_message(self, debug_msg: DebugMessage):
        """Add a debug message to the buffer"""
        self.log_buffer.append(debug_msg)

        # Maintain buffer size
        if len(self.log_buffer) > self.max_log_entries:
            self.log_buffer = self.log_buffer[-self.max_log_entries:]

        # Publish to debug topic if appropriate level
        if debug_msg.level.value >= self.debug_level.value:
            msg_str = String()
            msg_str.data = f"[{debug_msg.component}] {debug_msg.level.name}: {debug_msg.message}"
            self.debug_pub.publish(msg_str)

    def register_component(self, component_name: str, debug_topic: str = None):
        """Register a component for debugging"""
        self.component_status[component_name] = ComponentStatus.UNKNOWN

        if debug_topic:
            # Create subscriber for component-specific debug messages
            sub = self.create_subscription(
                String, debug_topic,
                lambda msg: self.component_debug_callback(component_name, msg),
                10
            )
            self.component_subs[component_name] = sub

    def component_debug_callback(self, component_name: str, msg: String):
        """Handle component-specific debug messages"""
        debug_msg = DebugMessage(
            timestamp=time.time(),
            component=component_name,
            level=DebugLevel.DEBUG,
            message=msg.data
        )
        self.add_debug_message(debug_msg)

    def set_component_status(self, component: str, status: ComponentStatus):
        """Update the status of a component"""
        self.component_status[component] = status
        self.get_logger().info(f'Component {component} status: {status.value}')

    def publish_diagnostics(self):
        """Publish diagnostic information"""
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()

        for component, status in self.component_status.items():
            status_msg = DiagnosticStatus()
            status_msg.name = component
            status_msg.hardware_id = component

            # Map status to diagnostic level
            if status == ComponentStatus.OK:
                status_msg.level = DiagnosticStatus.OK
                status_msg.message = "Component operational"
            elif status == ComponentStatus.WARNING:
                status_msg.level = DiagnosticStatus.WARN
                status_msg.message = "Component warning"
            elif status == ComponentStatus.ERROR:
                status_msg.level = DiagnosticStatus.ERROR
                status_msg.message = "Component error"
            elif status == ComponentStatus.CRITICAL:
                status_msg.level = DiagnosticStatus.ERROR
                status_msg.message = "Component critical failure"
            else:
                status_msg.level = DiagnosticStatus.OK
                status_msg.message = "Component status unknown"

            diag_array.status.append(status_msg)

        self.diag_pub.publish(diag_array)

    def get_component_status(self, component: str) -> ComponentStatus:
        """Get the status of a specific component"""
        return self.component_status.get(component, ComponentStatus.UNKNOWN)

    def get_recent_logs(self, component: str = None, limit: int = 50) -> List[DebugMessage]:
        """Get recent debug messages, optionally filtered by component"""
        if component:
            return [msg for msg in self.log_buffer
                   if msg.component == component][-limit:]
        else:
            return self.log_buffer[-limit:]

    def clear_logs(self):
        """Clear the debug message buffer"""
        self.log_buffer.clear()
        self.get_logger().info('Debug log buffer cleared')
```

## Performance Monitoring and Analysis

Performance monitoring tracks system metrics to identify bottlenecks and optimize resource utilization.

```python
class PerformanceMonitor:
    """Class for monitoring system performance"""

    def __init__(self, node: Node):
        self.node = node
        self.metrics = {}
        self.performance_history = {}
        self.thresholds = {
            'cpu_usage': 80.0,  # percentage
            'memory_usage': 85.0,  # percentage
            'topic_latency': 0.1,  # seconds
            'loop_rate': 100.0,  # Hz
            'task_execution_time': 5.0  # seconds
        }

    def start_monitoring(self, component: str, metric_name: str):
        """Start monitoring a specific metric"""
        key = f"{component}_{metric_name}"
        self.metrics[key] = {
            'start_time': time.time(),
            'values': [],
            'min': float('inf'),
            'max': float('-inf'),
            'avg': 0.0
        }

    def record_metric(self, component: str, metric_name: str, value: float):
        """Record a performance metric value"""
        key = f"{component}_{metric_name}"

        if key not in self.metrics:
            self.start_monitoring(component, metric_name)

        metric_data = self.metrics[key]
        metric_data['values'].append(value)

        # Update min, max, avg
        metric_data['min'] = min(metric_data['min'], value)
        metric_data['max'] = max(metric_data['max'], value)

        # Calculate running average
        n = len(metric_data['values'])
        if n == 1:
            metric_data['avg'] = value
        else:
            metric_data['avg'] = ((metric_data['avg'] * (n - 1)) + value) / n

        # Check if value exceeds threshold
        threshold_key = f"{metric_name}"
        if threshold_key in self.thresholds and value > self.thresholds[threshold_key]:
            self.node.get_logger().warn(
                f"Performance threshold exceeded: {component}.{metric_name} = {value} (threshold: {self.thresholds[threshold_key]})"
            )

    def get_metric_summary(self, component: str, metric_name: str) -> Dict[str, float]:
        """Get summary statistics for a metric"""
        key = f"{component}_{metric_name}"
        if key in self.metrics:
            return {
                'min': self.metrics[key]['min'],
                'max': self.metrics[key]['max'],
                'avg': self.metrics[key]['avg'],
                'count': len(self.metrics[key]['values'])
            }
        return None

    def get_system_performance_report(self) -> str:
        """Generate a comprehensive system performance report"""
        report = "=== System Performance Report ===\n"
        report += f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        for key, data in self.metrics.items():
            component, metric_name = key.split('_', 1)
            summary = self.get_metric_summary(component, metric_name)

            if summary:
                report += f"{component}.{metric_name}:\n"
                report += f"  Min: {summary['min']:.3f}\n"
                report += f"  Max: {summary['max']:.3f}\n"
                report += f"  Avg: {summary['avg']:.3f}\n"
                report += f"  Count: {summary['count']}\n\n"

        return report

    def monitor_ros_topics(self):
        """Monitor ROS topic performance"""
        # Get list of topics
        topic_names_and_types = self.node.get_topic_names_and_types()

        for topic_name, topic_types in topic_names_and_types:
            # Monitor topic publishing rate and latency
            try:
                # This is a simplified example - in practice, you'd need to
                # subscribe to topics and measure timing
                pass
            except Exception as e:
                self.node.get_logger().error(f"Error monitoring topic {topic_name}: {e}")
```

## Advanced Debugging Techniques

Advanced debugging techniques provide deeper insights into system behavior and help identify complex issues.

```python
class AdvancedDebuggingSystem(Node):
    """Node for advanced debugging techniques"""

    def __init__(self):
        super().__init__('advanced_debugging_system')

        # State tracking
        self.state_history = {}
        self.event_log = []

        # Debug subscribers
        self.debug_trigger_sub = self.create_subscription(
            String, 'debug_trigger', self.debug_trigger_callback, 10)

        # Performance analysis tools
        self.cpu_profiler = CPUProfiler()
        self.memory_profiler = MemoryProfiler()

        # Timer for periodic analysis
        self.analysis_timer = self.create_timer(10.0, self.periodic_analysis)

    def debug_trigger_callback(self, msg: String):
        """Handle debug trigger commands"""
        command = msg.data.lower()

        if command == 'dump_state':
            self.dump_system_state()
        elif command == 'profile_cpu':
            self.cpu_profiler.start_profiling()
        elif command == 'profile_memory':
            self.memory_profiler.start_profiling()
        elif command.startswith('set_debug_level'):
            # Extract level from command
            try:
                level_str = command.split()[-1]
                level = DebugLevel[level_str.upper()]
                self.get_logger().info(f'Debug level set to: {level.name}')
            except (IndexError, KeyError):
                self.get_logger().error(f'Invalid debug level: {command}')

    def dump_system_state(self):
        """Dump current system state for analysis"""
        self.get_logger().info('=== SYSTEM STATE DUMP ===')
        self.get_logger().info(f'Node count: {len(self.get_node_names())}')

        # List all topics and their status
        topic_names_and_types = self.get_topic_names_and_types()
        self.get_logger().info(f'Topic count: {len(topic_names_and_types)}')

        for topic_name, topic_types in topic_names_and_types:
            publishers = self.get_publishers_info_by_topic(topic_name)
            subscribers = self.get_subscriptions_info_by_topic(topic_name)
            self.get_logger().info(f'  {topic_name}: {len(publishers)} pubs, {len(subscribers)} subs')

        # List all services
        service_names_and_types = self.get_service_names_and_types()
        self.get_logger().info(f'Service count: {len(service_names_and_types)}')

        # Add any custom state information
        self.dump_custom_state()

    def dump_custom_state(self):
        """Dump custom state information specific to the robot"""
        # This would include robot-specific state like joint positions,
        # battery levels, sensor readings, etc.
        pass

    def periodic_analysis(self):
        """Perform periodic system analysis"""
        # Check for common issues
        self.check_for_common_issues()

        # Generate performance report
        perf_report = self.generate_performance_report()
        self.get_logger().info(perf_report)

    def check_for_common_issues(self):
        """Check for common system issues"""
        # Check for topic lags
        topic_names_and_types = self.get_topic_names_and_types()

        for topic_name, _ in topic_names_and_types:
            # Check message rate and latency
            try:
                # This is a simplified check - in practice, you'd need more
                # sophisticated monitoring
                pass
            except Exception as e:
                self.get_logger().warn(f'Issue detected with topic {topic_name}: {e}')

    def generate_performance_report(self) -> str:
        """Generate a performance report"""
        report = f"Performance Report - {self.get_clock().now().nanoseconds / 1e9:.2f}s\n"
        report += "=" * 50 + "\n"

        # Add CPU usage
        import psutil
        cpu_percent = psutil.cpu_percent()
        report += f"CPU Usage: {cpu_percent}%\n"

        # Add memory usage
        memory = psutil.virtual_memory()
        report += f"Memory Usage: {memory.percent}%\n"

        # Add disk usage
        disk = psutil.disk_usage('/')
        report += f"Disk Usage: {disk.percent}%\n"

        return report

class CPUProfiler:
    """CPU profiling utility"""

    def __init__(self):
        self.profiling = False
        self.profile_data = {}

    def start_profiling(self):
        """Start CPU profiling"""
        import cProfile
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.profiling = True
        print("CPU Profiling started")

    def stop_profiling(self):
        """Stop CPU profiling and generate report"""
        if self.profiling:
            self.profiler.disable()
            self.profiling = False

            # Print stats
            import pstats
            stats = pstats.Stats(self.profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(10)  # Top 10 functions

class MemoryProfiler:
    """Memory profiling utility"""

    def __init__(self):
        self.profiling = False

    def start_profiling(self):
        """Start memory profiling"""
        try:
            import memory_profiler
            self.profiling = True
            print("Memory Profiling started")
        except ImportError:
            print("Memory profiler not available. Install with: pip install memory-profiler")
```

## Real-Time Performance Optimization

Real-time performance optimization ensures that the humanoid robot can execute tasks within required time constraints.

```python
class RealTimeOptimizer(Node):
    """Node for real-time performance optimization"""

    def __init__(self):
        super().__init__('real_time_optimizer')

        # Performance optimization parameters
        self.target_loop_rate = 50.0  # Hz
        self.current_loop_rate = 0.0
        self.loop_times = []

        # Publishers for optimization commands
        self.optimization_pub = self.create_publisher(String, 'optimization_commands', 10)

        # Timer for performance monitoring
        self.optimization_timer = self.create_timer(1.0, self.optimize_performance)

    def optimize_performance(self):
        """Analyze and optimize system performance"""
        # Calculate current loop rate
        current_time = time.time()
        if hasattr(self, 'last_optimization_time'):
            elapsed = current_time - self.last_optimization_time
            if elapsed > 0:
                self.current_loop_rate = 1.0 / elapsed

        self.last_optimization_time = current_time

        # Store loop time for analysis
        if len(self.loop_times) > 100:
            self.loop_times.pop(0)
        self.loop_times.append(1.0 / self.current_loop_rate if self.current_loop_rate > 0 else 0)

        # Check if performance needs optimization
        avg_loop_time = sum(self.loop_times) / len(self.loop_times) if self.loop_times else 0
        target_loop_time = 1.0 / self.target_loop_rate

        if avg_loop_time > target_loop_time * 1.1:  # 10% over target
            self.get_logger().warn(f'Performance degradation detected: avg loop time {avg_loop_time:.3f}s > target {target_loop_time:.3f}s')
            self.issue_optimization_commands()

    def issue_optimization_commands(self):
        """Issue commands to optimize performance"""
        # Reduce processing load
        cmd = String()
        cmd.data = "reduce_processing_load"
        self.optimization_pub.publish(cmd)

        # Prioritize critical tasks
        cmd.data = "prioritize_critical_tasks"
        self.optimization_pub.publish(cmd)

    def adaptive_qos_adjustment(self, topic_name: str, current_qos: str):
        """Adaptively adjust QoS settings based on performance"""
        # This would dynamically adjust QoS settings (reliability, durability, etc.)
        # based on current system performance and requirements
        pass
```

## Debugging Best Practices

Effective debugging follows established best practices that help identify and resolve issues systematically.

### Systematic Debugging Approach

1. **Reproduce the Issue**: Consistently reproduce the problem before attempting to fix it
2. **Isolate the Problem**: Narrow down which component or subsystem is causing the issue
3. **Check the Logs**: Examine system logs for error messages and warnings
4. **Verify Inputs**: Confirm that all inputs to the system are correct
5. **Test Components**: Test individual components in isolation
6. **Check Dependencies**: Verify that all dependencies are functioning correctly
7. **Monitor Performance**: Check for performance bottlenecks that might cause issues

### Common Debugging Scenarios

```python
def debug_ros_communication_issues():
    """Common debugging for ROS communication issues"""
    # Check if nodes are running
    print("Checking active nodes...")
    # ros2 node list

    # Check topic connections
    print("Checking topic connections...")
    # ros2 topic list
    # ros2 topic info <topic_name>

    # Check service availability
    print("Checking service availability...")
    # ros2 service list
    # ros2 service info <service_name>

def debug_timing_issues():
    """Common debugging for timing issues"""
    # Check system time synchronization
    # Monitor loop rates
    # Check for blocking operations
    # Verify real-time capabilities

def debug_sensor_integration_issues():
    """Common debugging for sensor integration"""
    # Verify sensor connections
    # Check sensor calibration
    # Monitor sensor data rates
    # Validate sensor data quality
```

## Performance Benchmarking

Regular benchmarking helps maintain system performance and identify degradation over time.

```python
class PerformanceBenchmark:
    """Class for performance benchmarking"""

    def __init__(self):
        self.benchmarks = {}
        self.baseline_performance = {}

    def run_comprehensive_benchmark(self):
        """Run comprehensive performance benchmark"""
        results = {}

        # CPU performance test
        results['cpu'] = self.test_cpu_performance()

        # Memory performance test
        results['memory'] = self.test_memory_performance()

        # I/O performance test
        results['io'] = self.test_io_performance()

        # ROS communication test
        results['ros_comm'] = self.test_ros_communication()

        # Task execution test
        results['task_exec'] = self.test_task_execution()

        return results

    def test_cpu_performance(self) -> Dict[str, float]:
        """Test CPU performance"""
        import time
        import math

        start_time = time.time()

        # Perform CPU-intensive operation
        for i in range(1000000):
            math.sqrt(i) * math.sin(i)

        elapsed = time.time() - start_time
        return {'time_seconds': elapsed, 'operations_per_second': 1000000 / elapsed}

    def test_memory_performance(self) -> Dict[str, float]:
        """Test memory performance"""
        import time

        # Allocate and manipulate memory
        start_time = time.time()
        data = [0] * 1000000  # 1M integers

        # Manipulate the data
        for i in range(len(data)):
            data[i] = i * 2

        elapsed = time.time() - start_time
        return {'time_seconds': elapsed, 'size_mb': len(data) * 4 / (1024 * 1024)}

    def test_ros_communication(self) -> Dict[str, float]:
        """Test ROS communication performance"""
        # This would involve setting up test publishers/subscribers
        # and measuring message rates, latency, etc.
        return {'status': 'not_implemented_yet'}

    def test_task_execution(self) -> Dict[str, float]:
        """Test task execution performance"""
        # This would test the actual task execution framework
        return {'status': 'not_implemented_yet'}

    def compare_to_baseline(self, current_results: Dict, baseline_results: Dict) -> Dict[str, float]:
        """Compare current performance to baseline"""
        comparison = {}

        for category, current_data in current_results.items():
            if category in baseline_results:
                baseline_data = baseline_results[category]

                if isinstance(current_data, dict) and isinstance(baseline_data, dict):
                    for key, value in current_data.items():
                        if key in baseline_data and isinstance(value, (int, float)):
                            baseline_value = baseline_data[key]
                            if baseline_value != 0:
                                comparison[f"{category}.{key}_improvement"] = ((value - baseline_value) / baseline_value) * 100

        return comparison
```

## Implementation and Testing

To implement the debugging and performance system:

1. Create a new ROS 2 package:
```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python debugging_performance_system
```

2. Implement the nodes in the package:
```python
# debugging_performance_system/debugging_performance_system/__init__.py
from .centralized_debug_system import CentralizedDebugSystem
from .advanced_debugging_system import AdvancedDebuggingSystem
from .real_time_optimizer import RealTimeOptimizer

def main(args=None):
    rclpy.init(args=args)

    # Create nodes
    debug_system = CentralizedDebugSystem()
    advanced_debug = AdvancedDebuggingSystem()
    optimizer = RealTimeOptimizer()

    # Spin nodes
    executor = MultiThreadedExecutor()
    executor.add_node(debug_system)
    executor.add_node(advanced_debug)
    executor.add_node(optimizer)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        debug_system.destroy_node()
        advanced_debug.destroy_node()
        optimizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Testing the Debugging and Performance System

Create comprehensive tests to validate the debugging and performance system:

```python
import unittest
import rclpy
from debugging_performance_system.centralized_debug_system import CentralizedDebugSystem, DebugLevel, ComponentStatus

class TestDebuggingPerformanceSystem(unittest.TestCase):

    def setUp(self):
        rclpy.init()
        self.debug_system = CentralizedDebugSystem()

    def tearDown(self):
        self.debug_system.destroy_node()
        rclpy.shutdown()

    def test_component_registration(self):
        """Test that components can be registered"""
        self.debug_system.register_component("test_component", "test_debug_topic")

        status = self.debug_system.get_component_status("test_component")
        self.assertEqual(status, ComponentStatus.UNKNOWN)

    def test_debug_message_buffer(self):
        """Test that debug messages are buffered correctly"""
        initial_count = len(self.debug_system.log_buffer)

        # Add a debug message
        from debugging_performance_system.centralized_debug_system import DebugMessage
        debug_msg = DebugMessage(
            timestamp=time.time(),
            component="test",
            level=DebugLevel.INFO,
            message="Test message"
        )
        self.debug_system.add_debug_message(debug_msg)

        self.assertEqual(len(self.debug_system.log_buffer), initial_count + 1)

    def test_performance_monitoring(self):
        """Test that performance metrics are recorded"""
        # Test performance monitoring functionality
        self.debug_system.performance_monitor.record_metric("test_component", "execution_time", 0.05)

        summary = self.debug_system.performance_monitor.get_metric_summary("test_component", "execution_time")
        self.assertIsNotNone(summary)

if __name__ == '__main__':
    unittest.main()
```

## Integration with Existing Systems

The debugging and performance system integrates with all other components of the humanoid robot system, providing comprehensive monitoring and optimization capabilities. It works alongside the system integration and task execution frameworks to ensure reliable and efficient operation.

This system provides the foundation for maintaining high performance and reliability in autonomous humanoid robots, enabling systematic debugging and continuous performance optimization.