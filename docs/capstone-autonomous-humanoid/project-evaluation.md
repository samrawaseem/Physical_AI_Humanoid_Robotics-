# Project Evaluation and Assessment

## Introduction

Project evaluation is a critical component of the capstone autonomous humanoid project, providing systematic methods to assess the performance, reliability, and effectiveness of the integrated robotic system. This module covers comprehensive evaluation frameworks, performance metrics, testing protocols, and validation methodologies to ensure the autonomous humanoid meets the specified requirements and operates reliably in real-world scenarios.

## Evaluation Framework Architecture

The evaluation framework provides a structured approach to assess the autonomous humanoid system across multiple dimensions: functional performance, reliability, safety, and user experience.

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics
import numpy as np

class EvaluationCategory(Enum):
    """Categories for evaluation metrics"""
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    SAFETY = "safety"
    USER_EXPERIENCE = "user_experience"

class TaskStatus(Enum):
    """Status of task execution"""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"

@dataclass
class EvaluationMetric:
    """Data structure for evaluation metrics"""
    name: str
    category: EvaluationCategory
    value: float
    unit: str
    timestamp: float
    threshold: Optional[float] = None
    weight: float = 1.0

@dataclass
class TaskEvaluation:
    """Data structure for task evaluation"""
    task_id: str
    task_description: str
    status: TaskStatus
    execution_time: float
    success_rate: float
    metrics: List[EvaluationMetric]
    timestamp: float

@dataclass
class SystemEvaluation:
    """Data structure for overall system evaluation"""
    evaluation_id: str
    start_time: float
    end_time: float
    overall_score: float
    task_evaluations: List[TaskEvaluation]
    system_metrics: List[EvaluationMetric]
    summary: str
```

## Comprehensive Evaluation System

The comprehensive evaluation system monitors and assesses the autonomous humanoid across all operational aspects.

```python
class ComprehensiveEvaluationSystem(Node):
    """Node for comprehensive system evaluation"""

    def __init__(self):
        super().__init__('comprehensive_evaluation_system')

        # Evaluation configuration
        self.evaluation_metrics = {}
        self.task_evaluations = []
        self.system_evaluations = []
        self.current_evaluation = None

        # Evaluation thresholds
        self.thresholds = {
            'task_success_rate': 0.95,  # 95% success rate
            'average_execution_time': 5.0,  # seconds
            'cpu_usage': 80.0,  # percentage
            'memory_usage': 85.0,  # percentage
            'navigation_accuracy': 0.1,  # meters
            'manipulation_success_rate': 0.90,  # 90% success
            'response_time': 2.0  # seconds for voice commands
        }

        # Publishers and subscribers
        self.evaluation_pub = self.create_publisher(String, 'evaluation_results', 10)
        self.score_pub = self.create_publisher(Float32, 'overall_score', 10)

        # Subscribers for different system components
        self.task_status_sub = self.create_subscription(
            String, 'task_status', self.task_status_callback, 10)
        self.navigation_result_sub = self.create_subscription(
            String, 'navigation_result', self.navigation_result_callback, 10)
        self.manipulation_result_sub = self.create_subscription(
            String, 'manipulation_result', self.manipulation_result_callback, 10)
        self.system_health_sub = self.create_subscription(
            String, 'system_health', self.system_health_callback, 10)

        # Timer for periodic evaluation
        self.evaluation_timer = self.create_timer(5.0, self.periodic_evaluation)

        # Start evaluation session
        self.start_evaluation_session()

    def start_evaluation_session(self):
        """Start a new evaluation session"""
        self.current_evaluation = SystemEvaluation(
            evaluation_id=f"eval_{int(time.time())}",
            start_time=time.time(),
            end_time=0,
            overall_score=0.0,
            task_evaluations=[],
            system_metrics=[],
            summary=""
        )
        self.get_logger().info(f'Started evaluation session: {self.current_evaluation.evaluation_id}')

    def task_status_callback(self, msg: String):
        """Handle task status updates"""
        try:
            status_data = json.loads(msg.data)
            task_id = status_data.get('task_id')
            status = status_data.get('status')
            execution_time = status_data.get('execution_time', 0.0)

            # Create task evaluation
            task_eval = TaskEvaluation(
                task_id=task_id,
                task_description=status_data.get('description', ''),
                status=TaskStatus(status),
                execution_time=execution_time,
                success_rate=1.0 if status == 'success' else 0.0,
                metrics=[],
                timestamp=time.time()
            )

            self.task_evaluations.append(task_eval)

            # Record relevant metrics
            if status == 'success':
                self.record_metric('task_success_rate', 1.0, EvaluationCategory.FUNCTIONAL)
            else:
                self.record_metric('task_success_rate', 0.0, EvaluationCategory.FUNCTIONAL)

            self.record_metric('task_execution_time', execution_time, EvaluationCategory.PERFORMANCE)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in task status message')

    def navigation_result_callback(self, msg: String):
        """Handle navigation result updates"""
        try:
            nav_data = json.loads(msg.data)
            success = nav_data.get('success', False)
            distance_error = nav_data.get('distance_error', 0.0)
            execution_time = nav_data.get('execution_time', 0.0)

            # Record navigation metrics
            self.record_metric('navigation_success_rate', 1.0 if success else 0.0, EvaluationCategory.FUNCTIONAL)
            self.record_metric('navigation_accuracy', distance_error, EvaluationCategory.PERFORMANCE)
            self.record_metric('navigation_execution_time', execution_time, EvaluationCategory.PERFORMANCE)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in navigation result message')

    def manipulation_result_callback(self, msg: String):
        """Handle manipulation result updates"""
        try:
            manip_data = json.loads(msg.data)
            success = manip_data.get('success', False)
            precision_error = manip_data.get('precision_error', 0.0)
            execution_time = manip_data.get('execution_time', 0.0)

            # Record manipulation metrics
            self.record_metric('manipulation_success_rate', 1.0 if success else 0.0, EvaluationCategory.FUNCTIONAL)
            self.record_metric('manipulation_precision', precision_error, EvaluationCategory.PERFORMANCE)
            self.record_metric('manipulation_execution_time', execution_time, EvaluationCategory.PERFORMANCE)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in manipulation result message')

    def system_health_callback(self, msg: String):
        """Handle system health updates"""
        try:
            health_data = json.loads(msg.data)

            # Record system health metrics
            cpu_usage = health_data.get('cpu_usage', 0.0)
            memory_usage = health_data.get('memory_usage', 0.0)
            battery_level = health_data.get('battery_level', 100.0)

            self.record_metric('cpu_usage', cpu_usage, EvaluationCategory.PERFORMANCE)
            self.record_metric('memory_usage', memory_usage, EvaluationCategory.PERFORMANCE)
            self.record_metric('battery_level', battery_level, EvaluationCategory.PERFORMANCE)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in system health message')

    def record_metric(self, name: str, value: float, category: EvaluationCategory, weight: float = 1.0):
        """Record an evaluation metric"""
        metric = EvaluationMetric(
            name=name,
            category=category,
            value=value,
            unit=self.get_unit_for_metric(name),
            timestamp=time.time(),
            threshold=self.thresholds.get(name),
            weight=weight
        )

        # Store in evaluation metrics dictionary
        if name not in self.evaluation_metrics:
            self.evaluation_metrics[name] = []
        self.evaluation_metrics[name].append(metric)

    def get_unit_for_metric(self, name: str) -> str:
        """Get the appropriate unit for a metric"""
        units = {
            'task_success_rate': 'ratio',
            'average_execution_time': 'seconds',
            'cpu_usage': 'percentage',
            'memory_usage': 'percentage',
            'navigation_accuracy': 'meters',
            'manipulation_success_rate': 'ratio',
            'response_time': 'seconds',
            'distance_error': 'meters',
            'precision_error': 'meters',
            'battery_level': 'percentage'
        }
        return units.get(name, 'unitless')

    def periodic_evaluation(self):
        """Perform periodic evaluation and calculate scores"""
        if not self.evaluation_metrics:
            return

        # Calculate overall score
        overall_score = self.calculate_overall_score()
        self.current_evaluation.overall_score = overall_score

        # Publish score
        score_msg = Float32()
        score_msg.data = overall_score
        self.score_pub.publish(score_msg)

        # Check thresholds and report issues
        self.check_thresholds()

        # Log current status
        self.get_logger().info(f'Current evaluation score: {overall_score:.2f}')

    def calculate_overall_score(self) -> float:
        """Calculate overall system score based on all metrics"""
        if not self.evaluation_metrics:
            return 0.0

        total_weighted_score = 0.0
        total_weight = 0.0

        for metric_name, metrics in self.evaluation_metrics.items():
            if not metrics:
                continue

            # Get the most recent value
            latest_metric = metrics[-1]

            # Calculate score for this metric (0-1 scale)
            score = self.calculate_metric_score(latest_metric)

            # Apply weight
            weighted_score = score * latest_metric.weight
            total_weighted_score += weighted_score
            total_weight += latest_metric.weight

        if total_weight > 0:
            return total_weighted_score / total_weight
        else:
            return 0.0

    def calculate_metric_score(self, metric: EvaluationMetric) -> float:
        """Calculate score for a single metric (0-1 scale)"""
        if metric.threshold is None:
            # If no threshold, just return normalized value
            return min(1.0, max(0.0, metric.value))

        # Different metrics have different scoring logic
        if metric.name in ['task_success_rate', 'manipulation_success_rate']:
            # Higher is better, threshold is minimum acceptable
            return min(1.0, metric.value / metric.threshold if metric.threshold > 0 else 1.0)
        elif metric.name in ['cpu_usage', 'memory_usage', 'average_execution_time']:
            # Lower is better, threshold is maximum acceptable
            if metric.value <= metric.threshold:
                return 1.0
            else:
                return max(0.0, 1.0 - (metric.value - metric.threshold) / metric.threshold)
        elif metric.name in ['navigation_accuracy']:
            # Lower is better, threshold is maximum acceptable error
            if metric.value <= metric.threshold:
                return 1.0
            else:
                return max(0.0, 1.0 - (metric.value - metric.threshold) / metric.threshold)
        else:
            # Default: higher is better
            return min(1.0, metric.value / metric.threshold if metric.threshold > 0 else 1.0)

    def check_thresholds(self):
        """Check if any metrics are exceeding thresholds"""
        for metric_name, metrics in self.evaluation_metrics.items():
            if not metrics:
                continue

            latest_metric = metrics[-1]

            if (latest_metric.threshold is not None and
                latest_metric.name in ['cpu_usage', 'memory_usage', 'average_execution_time', 'navigation_accuracy']):
                # These metrics should be below threshold
                if latest_metric.value > latest_metric.threshold:
                    self.get_logger().warn(
                        f'Metric {metric_name} exceeded threshold: {latest_metric.value} > {latest_metric.threshold}'
                    )
            elif latest_metric.threshold is not None:
                # These metrics should be above threshold
                if latest_metric.value < latest_metric.threshold:
                    self.get_logger().warn(
                        f'Metric {metric_name} below threshold: {latest_metric.value} < {latest_metric.threshold}'
                    )

    def generate_evaluation_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        if not self.current_evaluation:
            return "No evaluation data available"

        report = f"=== EVALUATION REPORT ===\n"
        report += f"Evaluation ID: {self.current_evaluation.evaluation_id}\n"
        report += f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.current_evaluation.start_time))}\n"
        report += f"Overall Score: {self.current_evaluation.overall_score:.2f}\n\n"

        # Add metric summaries
        report += "METRIC SUMMARIES:\n"
        for metric_name, metrics in self.evaluation_metrics.items():
            if metrics:
                values = [m.value for m in metrics]
                avg_value = sum(values) / len(values)
                min_value = min(values)
                max_value = max(values)

                report += f"  {metric_name}:\n"
                report += f"    Average: {avg_value:.3f}\n"
                report += f"    Min: {min_value:.3f}\n"
                report += f"    Max: {max_value:.3f}\n"
                report += f"    Count: {len(values)}\n\n"

        # Add task evaluation summary
        report += f"TASK EVALUATIONS: {len(self.task_evaluations)} tasks\n"
        successful_tasks = sum(1 for te in self.task_evaluations if te.status == TaskStatus.SUCCESS)
        report += f"  Successful: {successful_tasks}\n"
        report += f"  Failed: {len(self.task_evaluations) - successful_tasks}\n"
        report += f"  Success Rate: {successful_tasks/len(self.task_evaluations)*100:.1f}%\n"

        return report

    def save_evaluation_results(self, filename: str = None):
        """Save evaluation results to file"""
        if not filename:
            filename = f"evaluation_{self.current_evaluation.evaluation_id}.json"

        evaluation_data = {
            'evaluation_id': self.current_evaluation.evaluation_id,
            'start_time': self.current_evaluation.start_time,
            'end_time': time.time(),
            'overall_score': self.current_evaluation.overall_score,
            'task_evaluations': [
                {
                    'task_id': te.task_id,
                    'task_description': te.task_description,
                    'status': te.status.value,
                    'execution_time': te.execution_time,
                    'success_rate': te.success_rate,
                    'timestamp': te.timestamp
                }
                for te in self.task_evaluations
            ],
            'system_metrics': [
                {
                    'name': m.name,
                    'category': m.category.value,
                    'value': m.value,
                    'unit': m.unit,
                    'timestamp': m.timestamp,
                    'threshold': m.threshold
                }
                for metrics_list in self.evaluation_metrics.values()
                for m in metrics_list
            ]
        }

        with open(filename, 'w') as f:
            json.dump(evaluation_data, f, indent=2)

        self.get_logger().info(f'Evaluation results saved to {filename}')
```

## Performance Testing Framework

The performance testing framework provides systematic methods to test and validate system performance under various conditions.

```python
class PerformanceTestingFramework(Node):
    """Node for systematic performance testing"""

    def __init__(self):
        super().__init__('performance_testing_framework')

        # Testing configuration
        self.test_scenarios = []
        self.test_results = []
        self.current_test = None

        # Publishers and subscribers
        self.test_command_pub = self.create_publisher(String, 'test_commands', 10)
        self.test_result_pub = self.create_publisher(String, 'test_results', 10)

        # Timer for test execution
        self.test_timer = self.create_timer(0.1, self.execute_tests)

    def define_test_scenario(self, name: str, description: str, commands: List[str], expected_results: Dict[str, Any]):
        """Define a test scenario"""
        scenario = {
            'name': name,
            'description': description,
            'commands': commands,
            'expected_results': expected_results,
            'timeout': 60.0  # seconds
        }
        self.test_scenarios.append(scenario)

    def execute_tests(self):
        """Execute defined test scenarios"""
        if not self.test_scenarios or self.current_test:
            return

        # Start next test
        test = self.test_scenarios.pop(0)
        self.current_test = {
            'scenario': test,
            'start_time': time.time(),
            'commands_sent': 0,
            'results_collected': []
        }

        self.get_logger().info(f'Starting test: {test["name"]}')

        # Send test commands
        for command in test['commands']:
            cmd_msg = String()
            cmd_msg.data = command
            self.test_command_pub.publish(cmd_msg)
            self.current_test['commands_sent'] += 1

    def collect_test_result(self, result_data: Dict[str, Any]):
        """Collect test results"""
        if self.current_test:
            self.current_test['results_collected'].append(result_data)

            # Check if test is complete
            elapsed = time.time() - self.current_test['start_time']
            if elapsed > self.current_test['scenario']['timeout']:
                self.complete_test()

    def complete_test(self):
        """Complete the current test and evaluate results"""
        if not self.current_test:
            return

        test_result = {
            'test_name': self.current_test['scenario']['name'],
            'description': self.current_test['scenario']['description'],
            'start_time': self.current_test['start_time'],
            'end_time': time.time(),
            'duration': time.time() - self.current_test['start_time'],
            'results_collected': self.current_test['results_collected'],
            'expected_results': self.current_test['scenario']['expected_results']
        }

        # Evaluate results against expectations
        evaluation = self.evaluate_test_results(test_result)
        test_result['evaluation'] = evaluation

        self.test_results.append(test_result)

        # Publish test result
        result_msg = String()
        result_msg.data = json.dumps(test_result)
        self.test_result_pub.publish(result_msg)

        self.get_logger().info(f'Test completed: {self.current_test["scenario"]["name"]}, Success: {evaluation["success"]}')

        # Clear current test
        self.current_test = None

    def evaluate_test_results(self, test_result: Dict) -> Dict[str, Any]:
        """Evaluate test results against expected outcomes"""
        expected = test_result['expected_results']
        actual = test_result['results_collected']

        evaluation = {
            'success': True,
            'metrics': {},
            'issues': []
        }

        # Compare expected vs actual results
        for expected_key, expected_value in expected.items():
            actual_values = [r.get(expected_key) for r in actual if expected_key in r]
            if actual_values:
                actual_value = actual_values[-1]  # Use most recent value

                # Compare with tolerance for numeric values
                if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                    tolerance = expected_value * 0.1  # 10% tolerance
                    if abs(actual_value - expected_value) > tolerance:
                        evaluation['success'] = False
                        evaluation['issues'].append(
                            f'{expected_key}: expected {expected_value}, got {actual_value}'
                        )
                elif actual_value != expected_value:
                    evaluation['success'] = False
                    evaluation['issues'].append(
                        f'{expected_key}: expected {expected_value}, got {actual_value}'
                    )
            else:
                evaluation['success'] = False
                evaluation['issues'].append(f'{expected_key}: not found in results')

        # Calculate success metrics
        evaluation['metrics']['success_rate'] = 1.0 if evaluation['success'] else 0.0
        evaluation['metrics']['test_duration'] = test_result['duration']

        return evaluation

    def run_standard_performance_tests(self):
        """Run standard performance tests"""
        # Test 1: Task execution performance
        self.define_test_scenario(
            name="task_execution_performance",
            description="Test task execution speed and reliability",
            commands=[
                "execute_task_navigate_to_kitchen",
                "execute_task_find_object",
                "execute_task_pick_up_object",
                "execute_task_place_object"
            ],
            expected_results={
                'success_rate': 1.0,
                'average_execution_time': 5.0  # seconds
            }
        )

        # Test 2: Navigation accuracy
        self.define_test_scenario(
            name="navigation_accuracy",
            description="Test navigation precision and reliability",
            commands=[
                "navigate_to_point_x_1_0_y_2_0",
                "navigate_to_point_x_0_0_y_0_0",
                "navigate_to_point_x_3_0_y_1_5"
            ],
            expected_results={
                'average_position_error': 0.05,  # meters
                'success_rate': 0.95
            }
        )

        # Test 3: Manipulation precision
        self.define_test_scenario(
            name="manipulation_precision",
            description="Test manipulation accuracy and success rate",
            commands=[
                "manipulate_pick_up_cube_red",
                "manipulate_place_cube_red_at_position",
                "manipulate_pick_up_sphere_blue"
            ],
            expected_results={
                'manipulation_success_rate': 0.90,
                'average_precision_error': 0.02  # meters
            }
        )

        # Test 4: System responsiveness
        self.define_test_scenario(
            name="system_responsiveness",
            description="Test system response time to commands",
            commands=[
                "command_response_test_1",
                "command_response_test_2",
                "command_response_test_3"
            ],
            expected_results={
                'average_response_time': 1.0,  # seconds
                'success_rate': 1.0
            }
        )
```

## Safety and Reliability Assessment

Safety and reliability assessment ensures the autonomous humanoid operates safely and reliably in various conditions.

```python
class SafetyReliabilityAssessment(Node):
    """Node for safety and reliability assessment"""

    def __init__(self):
        super().__init__('safety_reliability_assessment')

        # Safety monitoring
        self.safety_metrics = {}
        self.reliability_metrics = {}
        self.risk_assessment = {}

        # Publishers and subscribers
        self.safety_pub = self.create_publisher(String, 'safety_status', 10)
        self.risk_pub = self.create_publisher(String, 'risk_assessment', 10)

        # Safety monitoring subscribers
        self.collision_sub = self.create_subscription(
            Bool, 'collision_detected', self.collision_callback, 10)
        self.emergency_stop_sub = self.create_subscription(
            Bool, 'emergency_stop', self.emergency_stop_callback, 10)
        self.system_error_sub = self.create_subscription(
            String, 'system_error', self.system_error_callback, 10)

        # Timer for safety assessment
        self.safety_timer = self.create_timer(1.0, self.assess_safety)

    def collision_callback(self, msg: Bool):
        """Handle collision detection"""
        if msg.data:
            self.record_safety_incident('collision', time.time())
            self.get_logger().warn('Collision detected!')

    def emergency_stop_callback(self, msg: Bool):
        """Handle emergency stop activation"""
        if msg.data:
            self.record_safety_incident('emergency_stop', time.time())
            self.get_logger().warn('Emergency stop activated!')

    def system_error_callback(self, msg: String):
        """Handle system errors"""
        error_data = msg.data
        self.record_safety_incident('system_error', time.time(), error_data)
        self.get_logger().error(f'System error: {error_data}')

    def record_safety_incident(self, incident_type: str, timestamp: float, details: str = ""):
        """Record a safety incident"""
        if incident_type not in self.safety_metrics:
            self.safety_metrics[incident_type] = []

        incident = {
            'timestamp': timestamp,
            'details': details,
            'resolved': False
        }
        self.safety_metrics[incident_type].append(incident)

    def assess_safety(self):
        """Assess current safety status"""
        safety_status = {
            'timestamp': time.time(),
            'collision_count': len(self.safety_metrics.get('collision', [])),
            'emergency_stop_count': len(self.safety_metrics.get('emergency_stop', [])),
            'system_error_count': len(self.safety_metrics.get('system_error', [])),
            'overall_safety_score': self.calculate_safety_score()
        }

        # Publish safety status
        safety_msg = String()
        safety_msg.data = json.dumps(safety_status)
        self.safety_pub.publish(safety_msg)

        # Perform risk assessment
        self.perform_risk_assessment(safety_status)

    def calculate_safety_score(self) -> float:
        """Calculate overall safety score (0-1 scale)"""
        collision_count = len(self.safety_metrics.get('collision', []))
        emergency_count = len(self.safety_metrics.get('emergency_stop', []))
        error_count = len(self.safety_metrics.get('system_error', []))

        # Calculate risk factors (lower is better)
        collision_risk = min(1.0, collision_count * 0.3)  # Each collision adds 0.3 risk
        emergency_risk = min(1.0, emergency_count * 0.5)  # Each emergency adds 0.5 risk
        error_risk = min(1.0, error_count * 0.1)  # Each error adds 0.1 risk

        total_risk = min(1.0, collision_risk + emergency_risk + error_risk)
        safety_score = 1.0 - total_risk

        return max(0.0, safety_score)

    def perform_risk_assessment(self, safety_status: Dict):
        """Perform comprehensive risk assessment"""
        risk_assessment = {
            'timestamp': time.time(),
            'risk_level': 'low',
            'identified_risks': [],
            'recommended_actions': []
        }

        # Assess different risk categories
        if safety_status['collision_count'] > 5:
            risk_assessment['risk_level'] = 'high'
            risk_assessment['identified_risks'].append('High collision frequency detected')
            risk_assessment['recommended_actions'].append('Review navigation algorithms and obstacle detection')

        if safety_status['emergency_stop_count'] > 3:
            risk_assessment['risk_level'] = 'medium' if risk_assessment['risk_level'] == 'low' else risk_assessment['risk_level']
            risk_assessment['identified_risks'].append('Frequent emergency stops')
            risk_assessment['recommended_actions'].append('Investigate system stability issues')

        if safety_status['system_error_count'] > 10:
            risk_assessment['risk_level'] = 'medium' if risk_assessment['risk_level'] == 'low' else risk_assessment['risk_level']
            risk_assessment['identified_risks'].append('High error frequency')
            risk_assessment['recommended_actions'].append('Review system architecture and error handling')

        self.risk_assessment = risk_assessment

        # Publish risk assessment
        risk_msg = String()
        risk_msg.data = json.dumps(risk_assessment)
        self.risk_pub.publish(risk_msg)

    def calculate_reliability_metrics(self) -> Dict[str, float]:
        """Calculate reliability metrics"""
        if not self.safety_metrics:
            return {'uptime': 1.0, 'mtbf': float('inf'), 'failure_rate': 0.0}

        # Calculate uptime based on error-free periods
        total_time = time.time() - self.get_clock().now().nanoseconds / 1e9
        error_count = sum(len(errors) for errors in self.safety_metrics.values())

        # Simple reliability calculation
        uptime = max(0.0, 1.0 - (error_count * 0.01))  # Each error reduces uptime by 1%
        mtbf = total_time / max(1, error_count)  # Mean time between failures
        failure_rate = error_count / max(1, total_time)  # Failures per second

        reliability_metrics = {
            'uptime': uptime,
            'mtbf_hours': mtbf / 3600,
            'failure_rate_per_hour': failure_rate * 3600,
            'total_incidents': error_count
        }

        return reliability_metrics
```

## Statistical Analysis and Reporting

Statistical analysis provides insights into system performance trends and areas for improvement.

```python
class StatisticalAnalysisSystem(Node):
    """Node for statistical analysis of evaluation data"""

    def __init__(self):
        super().__init__('statistical_analysis_system')

        # Data storage for analysis
        self.historical_data = []
        self.trend_analysis = {}

        # Publishers for analysis results
        self.analysis_pub = self.create_publisher(String, 'statistical_analysis', 10)

        # Timer for periodic analysis
        self.analysis_timer = self.create_timer(30.0, self.perform_analysis)

    def add_evaluation_data(self, evaluation_data: Dict):
        """Add evaluation data for statistical analysis"""
        self.historical_data.append(evaluation_data)

        # Maintain reasonable history size
        if len(self.historical_data) > 1000:
            self.historical_data = self.historical_data[-500:]  # Keep last 500 entries

    def perform_analysis(self):
        """Perform statistical analysis on historical data"""
        if len(self.historical_data) < 2:
            return

        analysis_results = {
            'timestamp': time.time(),
            'summary_statistics': self.calculate_summary_statistics(),
            'trend_analysis': self.calculate_trends(),
            'performance_benchmarks': self.calculate_benchmarks()
        }

        # Publish analysis results
        analysis_msg = String()
        analysis_msg.data = json.dumps(analysis_results)
        self.analysis_pub.publish(analysis_msg)

        self.get_logger().info(f'Statistical analysis completed: {len(self.historical_data)} data points analyzed')

    def calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics for evaluation metrics"""
        if not self.historical_data:
            return {}

        # Extract metric values from historical data
        metrics_summary = {}

        # Get all metric names from recent evaluations
        recent_eval = self.historical_data[-1]
        if 'system_metrics' in recent_eval:
            for metric in recent_eval['system_metrics']:
                metric_name = metric['name']

                # Collect all values for this metric
                values = []
                for eval_data in self.historical_data:
                    for m in eval_data.get('system_metrics', []):
                        if m['name'] == metric_name:
                            values.append(m['value'])

                if values:
                    metrics_summary[metric_name] = {
                        'mean': statistics.mean(values),
                        'median': statistics.median(values),
                        'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }

        return metrics_summary

    def calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends over time"""
        if len(self.historical_data) < 3:
            return {}

        trends = {}

        # Analyze trends for key metrics
        key_metrics = ['overall_score', 'task_success_rate', 'cpu_usage', 'memory_usage']

        for metric_name in key_metrics:
            values = []
            timestamps = []

            for i, eval_data in enumerate(self.historical_data):
                # Extract metric value
                if metric_name == 'overall_score' and 'overall_score' in eval_data:
                    values.append(eval_data['overall_score'])
                    timestamps.append(eval_data.get('start_time', i))
                elif 'system_metrics' in eval_data:
                    for metric in eval_data['system_metrics']:
                        if metric['name'] == metric_name:
                            values.append(metric['value'])
                            timestamps.append(metric['timestamp'])
                            break

            if len(values) >= 3:
                # Calculate trend using linear regression
                if len(set(values)) > 1:  # Only calculate trend if values vary
                    slope, intercept = self.linear_regression(timestamps, values)
                    trend_direction = 'improving' if slope > 0.01 else 'declining' if slope < -0.01 else 'stable'

                    trends[metric_name] = {
                        'slope': slope,
                        'intercept': intercept,
                        'direction': trend_direction,
                        'correlation': self.calculate_correlation(timestamps, values)
                    }

        return trends

    def calculate_benchmarks(self) -> Dict[str, float]:
        """Calculate performance benchmarks from historical data"""
        if not self.historical_data:
            return {}

        benchmarks = {}

        # Calculate 95th percentile for key metrics
        for metric_name in ['overall_score', 'task_success_rate', 'response_time']:
            values = []

            for eval_data in self.historical_data:
                if metric_name == 'overall_score' and 'overall_score' in eval_data:
                    values.append(eval_data['overall_score'])
                elif 'system_metrics' in eval_data:
                    for metric in eval_data['system_metrics']:
                        if metric['name'] == metric_name:
                            values.append(metric['value'])

            if values:
                # Calculate percentiles
                values_sorted = sorted(values)
                n = len(values_sorted)

                benchmarks[f'{metric_name}_p95'] = values_sorted[int(0.95 * n)] if n > 0 else 0
                benchmarks[f'{metric_name}_p50'] = values_sorted[int(0.50 * n)] if n > 0 else 0
                benchmarks[f'{metric_name}_p05'] = values_sorted[int(0.05 * n)] if n > 0 else 0

        return benchmarks

    def linear_regression(self, x_vals: List[float], y_vals: List[float]) -> Tuple[float, float]:
        """Calculate linear regression (slope and intercept)"""
        n = len(x_vals)
        if n < 2:
            return 0.0, 0.0

        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
        sum_x2 = sum(x * x for x in x_vals)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0, sum_y / n

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        return slope, intercept

    def calculate_correlation(self, x_vals: List[float], y_vals: List[float]) -> float:
        """Calculate correlation coefficient"""
        n = len(x_vals)
        if n < 2:
            return 0.0

        mean_x = sum(x_vals) / n
        mean_y = sum(y_vals) / n

        # Calculate standard deviations
        std_x = (sum((x - mean_x) ** 2 for x in x_vals) / n) ** 0.5
        std_y = (sum((y - mean_y) ** 2 for y in y_vals) / n) ** 0.5

        if std_x == 0 or std_y == 0:
            return 0.0

        # Calculate correlation
        correlation = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals)) / (n * std_x * std_y)
        return correlation
```

## Implementation and Testing

To implement the project evaluation system:

1. Create a new ROS 2 package:
```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python project_evaluation_system
```

2. Implement the nodes in the package:
```python
# project_evaluation_system/project_evaluation_system/__init__.py
from .comprehensive_evaluation_system import ComprehensiveEvaluationSystem
from .performance_testing_framework import PerformanceTestingFramework
from .safety_reliability_assessment import SafetyReliabilityAssessment
from .statistical_analysis_system import StatisticalAnalysisSystem

def main(args=None):
    rclpy.init(args=args)

    # Create nodes
    eval_system = ComprehensiveEvaluationSystem()
    perf_test = PerformanceTestingFramework()
    safety_assessment = SafetyReliabilityAssessment()
    stats_analysis = StatisticalAnalysisSystem()

    # Run standard tests
    perf_test.run_standard_performance_tests()

    # Spin nodes
    executor = MultiThreadedExecutor()
    executor.add_node(eval_system)
    executor.add_node(perf_test)
    executor.add_node(safety_assessment)
    executor.add_node(stats_analysis)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        eval_system.destroy_node()
        perf_test.destroy_node()
        safety_assessment.destroy_node()
        stats_analysis.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Testing the Evaluation System

Create comprehensive tests to validate the evaluation system:

```python
import unittest
import rclpy
from project_evaluation_system.comprehensive_evaluation_system import ComprehensiveEvaluationSystem, EvaluationCategory

class TestProjectEvaluationSystem(unittest.TestCase):

    def setUp(self):
        rclpy.init()
        self.eval_system = ComprehensiveEvaluationSystem()

    def tearDown(self):
        self.eval_system.destroy_node()
        rclpy.shutdown()

    def test_metric_recording(self):
        """Test that metrics are recorded correctly"""
        initial_count = len(self.eval_system.evaluation_metrics.get('test_metric', []))

        # Record a metric
        self.eval_system.record_metric('test_metric', 0.8, EvaluationCategory.FUNCTIONAL)

        # Check that metric was recorded
        self.assertEqual(len(self.eval_system.evaluation_metrics.get('test_metric', [])), initial_count + 1)

    def test_score_calculation(self):
        """Test that overall scores are calculated correctly"""
        # Record some metrics
        self.eval_system.record_metric('task_success_rate', 0.95, EvaluationCategory.FUNCTIONAL)
        self.eval_system.record_metric('cpu_usage', 70.0, EvaluationCategory.PERFORMANCE)

        # Calculate score
        score = self.eval_system.calculate_overall_score()

        # Score should be between 0 and 1
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_threshold_checking(self):
        """Test that threshold checking works"""
        # Record a metric that exceeds threshold
        self.eval_system.record_metric('cpu_usage', 90.0, EvaluationCategory.PERFORMANCE)  # Above 80% threshold

        # Check thresholds (this should log a warning)
        self.eval_system.check_thresholds()

if __name__ == '__main__':
    unittest.main()
```

## Integration with Existing Systems

The project evaluation system integrates with all previous components of the autonomous humanoid system, providing comprehensive assessment capabilities. It works alongside the system integration, task execution, and debugging/performance systems to provide a complete picture of system performance and reliability.

This evaluation framework provides the foundation for systematic assessment of autonomous humanoid performance, enabling data-driven improvements and validation of system capabilities.