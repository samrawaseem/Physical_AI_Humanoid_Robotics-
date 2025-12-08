---
title: LLM Integration
sidebar_position: 2
---

# Module 4: Vision-Language-Action (VLA)

## Large Language Model Integration for Robotics

Large Language Models (LLMs) provide the cognitive layer for understanding and reasoning about natural language commands in embodied AI systems. This section explores how to integrate LLMs with robotic systems to create humanoid robots that can understand natural language and execute complex tasks.

### Understanding LLMs in Robotics Context

Large Language Models bring several capabilities to robotics:

- **Natural Language Understanding**: Interpreting human commands and queries
- **Task Planning**: Breaking down complex tasks into executable steps
- **Context Awareness**: Understanding the environment and situation
- **Reasoning**: Making decisions based on incomplete information
- **Adaptation**: Learning from interactions and improving performance

### OpenAI Whisper for Speech Recognition

OpenAI Whisper provides robust speech-to-text capabilities that are essential for voice command processing:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import AudioData
import openai
import speech_recognition as sr
import pyaudio
import wave
import numpy as np
import threading
import queue

class WhisperIntegrationNode(Node):
    def __init__(self):
        super().__init__('whisper_integration_node')

        # Publisher for recognized text
        self.text_publisher = self.create_publisher(
            String,
            '/recognized_speech',
            10
        )

        # Publisher for voice commands
        self.command_publisher = self.create_publisher(
            String,
            '/voice_command',
            10
        )

        # Initialize OpenAI API key (should be set in environment)
        openai.api_key = self.get_parameter_or('openai_api_key', 'your-api-key-here').value

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Audio processing queue
        self.audio_queue = queue.Queue()
        self.listening = True

        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.audio_thread.start()

        # Timer for continuous listening
        self.listen_timer = self.create_timer(0.1, self.check_audio)

        self.get_logger().info('Whisper Integration Node initialized')

    def check_audio(self):
        """Check for audio input"""
        if not self.audio_queue.empty():
            audio_data = self.audio_queue.get()
            self.process_speech(audio_data)

    def process_audio(self):
        """Continuously process audio from microphone"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise

        while self.listening:
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)

                # Add audio to processing queue
                self.audio_queue.put(audio)

            except sr.WaitTimeoutError:
                # No audio detected, continue listening
                continue
            except Exception as e:
                self.get_logger().error(f'Audio processing error: {e}')
                continue

    def process_speech(self, audio):
        """Process speech using Whisper API"""
        try:
            # Save audio to temporary file for Whisper API
            with wave.open('temp_audio.wav', 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)  # 16kHz
                wf.writeframes(audio.get_raw_data())

            # Transcribe using Whisper
            with open('temp_audio.wav', 'rb') as audio_file:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )

            if transcript.strip():
                # Publish recognized text
                text_msg = String()
                text_msg.data = transcript.strip()
                self.text_publisher.publish(text_msg)

                # Check if this is a command
                if self.is_command(transcript):
                    cmd_msg = String()
                    cmd_msg.data = transcript.strip()
                    self.command_publisher.publish(cmd_msg)
                    self.get_logger().info(f'Voice command recognized: {transcript}')

        except Exception as e:
            self.get_logger().error(f'Whisper transcription error: {e}')

    def is_command(self, text):
        """Determine if text is a command"""
        command_keywords = [
            'go to', 'move to', 'pick up', 'place', 'grasp', 'release',
            'turn', 'rotate', 'navigate', 'find', 'locate', 'bring',
            'take', 'put', 'drop', 'stop', 'start', 'continue'
        ]

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in command_keywords)

    def shutdown(self):
        """Clean shutdown"""
        self.listening = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = WhisperIntegrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Whisper Integration Node')
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced LLM Integration with Context Awareness

For more sophisticated integration, we need to maintain context and handle complex interactions:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import openai
import json
from datetime import datetime
from typing import Dict, List, Optional

class ContextAwareLLMNode(Node):
    def __init__(self):
        super().__init__('context_aware_llm_node')

        # Publishers
        self.action_publisher = self.create_publisher(
            String,
            '/llm_action',
            10
        )

        self.query_response_publisher = self.create_publisher(
            String,
            '/llm_response',
            10
        )

        # Subscribers
        self.voice_command_subscriber = self.create_subscription(
            String,
            '/voice_command',
            self.voice_command_callback,
            10
        )

        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        self.robot_pose_subscriber = self.create_subscription(
            PoseStamped,
            '/robot_pose',
            self.pose_callback,
            10
        )

        # Initialize components
        self.bridge = CvBridge()
        openai.api_key = self.get_parameter_or('openai_api_key', 'your-api-key-here').value

        # Context management
        self.context_history = []
        self.current_pose = None
        self.latest_image = None
        self.conversation_history = []

        # System prompt for robotics context
        self.system_prompt = """
        You are an AI assistant controlling a humanoid robot. Your responses should be:
        1. In JSON format with clear action specifications
        2. Based on the robot's current context and environment
        3. Safe and executable by the robot
        4. Appropriate for the given command

        Available actions:
        - navigate_to: Move robot to a location
        - pick_object: Pick up an object
        - place_object: Place an object at a location
        - find_object: Locate an object in the environment
        - answer_question: Respond to a query
        - wait: Wait for further instructions

        Respond with JSON format: {"action": "action_name", "parameters": {...}, "confidence": 0.0-1.0}
        """

        self.get_logger().info('Context-Aware LLM Node initialized')

    def voice_command_callback(self, msg):
        """Process voice commands with context"""
        command = msg.data
        self.get_logger().info(f'Received voice command: {command}')

        # Get current context
        context = self.get_current_context()

        # Prepare prompt with context
        prompt = f"""
        Robot Context:
        - Current Position: {context.get('position', 'unknown')}
        - Environment: {context.get('environment', 'unknown')}
        - Time: {context.get('time', 'unknown')}

        User Command: {command}

        Please provide a specific action in JSON format.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )

            response_text = response.choices[0].message['content'].strip()

            # Extract JSON response
            action_json = self.extract_json_from_response(response_text)

            if action_json:
                # Publish action
                action_msg = String()
                action_msg.data = json.dumps(action_json)
                self.action_publisher.publish(action_msg)

                self.get_logger().info(f'LLM Action: {action_json}')

                # Add to conversation history
                self.conversation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'command': command,
                    'action': action_json
                })

            else:
                self.get_logger().warn(f'Could not extract action from LLM response: {response_text}')

        except Exception as e:
            self.get_logger().error(f'LLM processing error: {e}')

    def image_callback(self, msg):
        """Store latest image for context"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')

    def pose_callback(self, msg):
        """Store current robot pose"""
        self.current_pose = {
            'x': msg.pose.position.x,
            'y': msg.pose.position.y,
            'z': msg.pose.position.z,
            'orientation': {
                'w': msg.pose.orientation.w,
                'x': msg.pose.orientation.x,
                'y': msg.pose.orientation.y,
                'z': msg.pose.orientation.z
            }
        }

    def get_current_context(self):
        """Get current environmental context"""
        context = {
            'position': self.current_pose,
            'time': datetime.now().isoformat(),
            'has_image': self.latest_image is not None
        }

        # Add more context as needed
        if len(self.conversation_history) > 0:
            context['last_command'] = self.conversation_history[-1]

        return context

    def extract_json_from_response(self, response_text):
        """Extract JSON from LLM response"""
        try:
            # Try to parse as JSON directly
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON within the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

        return None

    def query_handler(self, query_msg):
        """Handle general queries (not just commands)"""
        query = query_msg.data

        # Prepare context-aware query
        context = self.get_current_context()
        prompt = f"""
        Robot Context:
        - Current Position: {context.get('position', 'unknown')}
        - Environment: {context.get('environment', 'unknown')}
        - Time: {context.get('time', 'unknown')}

        User Query: {query}

        Please provide a helpful response.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            response_text = response.choices[0].message['content'].strip()

            # Publish response
            response_msg = String()
            response_msg.data = response_text
            self.query_response_publisher.publish(response_msg)

        except Exception as e:
            self.get_logger().error(f'Query processing error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ContextAwareLLMNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Context-Aware LLM Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Cognitive Planning with LLMs

LLMs can be used for higher-level cognitive planning and task decomposition:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import openai
import json
from typing import List, Dict, Any
import re

class CognitivePlanningNode(Node):
    def __init__(self):
        super().__init__('cognitive_planning_node')

        # Publishers
        self.plan_publisher = self.create_publisher(
            String,
            '/cognitive_plan',
            10
        )

        self.subtask_publisher = self.create_publisher(
            String,
            '/subtask_queue',
            10
        )

        # Subscribers
        self.high_level_command_subscriber = self.create_subscription(
            String,
            '/high_level_command',
            self.high_level_command_callback,
            10
        )

        self.action_result_subscriber = self.create_subscription(
            String,
            '/action_result',
            self.action_result_callback,
            10
        )

        # Initialize OpenAI
        openai.api_key = self.get_parameter_or('openai_api_key', 'your-api-key-here').value

        # Planning state
        self.current_plan = []
        self.current_subtask_index = 0
        self.planning_history = []

        # System prompt for planning
        self.planning_prompt = """
        You are a cognitive planner for a humanoid robot. Given a high-level goal,
        break it down into specific, executable subtasks. Each subtask should be:

        1. Specific and actionable
        2. Sequential (order matters)
        3. Achievable with available robot capabilities
        4. Include necessary parameters (locations, objects, etc.)

        Available capabilities:
        - navigation: Move to specific locations
        - object_manipulation: Pick/place objects
        - perception: Detect/identify objects
        - interaction: Communicate with environment

        Format response as JSON with 'subtasks' array, where each subtask has:
        {
            "id": integer,
            "name": "descriptive name",
            "action": "action_type",
            "parameters": {key-value pairs for the action},
            "preconditions": ["what must be true before executing"],
            "effects": ["what will be true after executing"]
        }
        """

        self.get_logger().info('Cognitive Planning Node initialized')

    def high_level_command_callback(self, msg):
        """Process high-level commands and generate plans"""
        command = msg.data
        self.get_logger().info(f'Received high-level command: {command}')

        # Generate plan using LLM
        plan = self.generate_plan(command)

        if plan:
            self.current_plan = plan
            self.current_subtask_index = 0

            # Publish complete plan
            plan_msg = String()
            plan_msg.data = json.dumps(plan)
            self.plan_publisher.publish(plan_msg)

            # Publish first subtask
            if self.current_plan:
                self.publish_next_subtask()

            # Add to history
            self.planning_history.append({
                'command': command,
                'plan': plan,
                'timestamp': self.get_clock().now().nanoseconds
            })

            self.get_logger().info(f'Generated plan with {len(plan)} subtasks')

    def generate_plan(self, command):
        """Generate a detailed plan using LLM"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Using GPT-4 for better reasoning
                messages=[
                    {"role": "system", "content": self.planning_prompt},
                    {"role": "user", "content": f"Goal: {command}"}
                ],
                temperature=0.2,
                max_tokens=1000
            )

            response_text = response.choices[0].message['content'].strip()

            # Extract JSON plan
            plan_json = self.extract_json_from_response(response_text)

            if plan_json and 'subtasks' in plan_json:
                return plan_json['subtasks']
            else:
                self.get_logger().warn(f'Could not extract plan from response: {response_text}')
                return []

        except Exception as e:
            self.get_logger().error(f'Plan generation error: {e}')
            return []

    def publish_next_subtask(self):
        """Publish the next subtask in the plan"""
        if (self.current_subtask_index < len(self.current_plan) and
            self.current_plan):

            subtask = self.current_plan[self.current_subtask_index]

            subtask_msg = String()
            subtask_msg.data = json.dumps(subtask)
            self.subtask_publisher.publish(subtask_msg)

            self.get_logger().info(f'Published subtask {self.current_subtask_index + 1}: {subtask["name"]}')

            self.current_subtask_index += 1

    def action_result_callback(self, msg):
        """Handle action results and continue planning"""
        try:
            result = json.loads(msg.data)
            action_status = result.get('status', 'unknown')
            action_id = result.get('subtask_id', -1)

            if action_status == 'success':
                if self.current_subtask_index < len(self.current_plan):
                    # Continue to next subtask
                    self.publish_next_subtask()
                else:
                    # Plan completed
                    self.get_logger().info('Plan execution completed successfully')
                    self.current_plan = []
                    self.current_subtask_index = 0
            else:
                # Handle failure - could implement recovery strategies
                self.get_logger().warn(f'Subtask {action_id} failed with status: {action_status}')
                # For now, continue to next subtask, but in real implementation
                # you might want to retry or replan
                if self.current_subtask_index < len(self.current_plan):
                    self.publish_next_subtask()

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in action result')

    def extract_json_from_response(self, response_text):
        """Extract JSON from LLM response"""
        try:
            # Try to parse as JSON directly
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find JSON within the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

        return None

    def replan(self, failed_subtask_index, reason):
        """Generate a new plan when current plan fails"""
        if not self.current_plan or failed_subtask_index >= len(self.current_plan):
            return

        # Get the original command that led to this plan
        if self.planning_history:
            original_command = self.planning_history[-1]['command']

            # Generate new plan considering the failure
            replan_prompt = f"""
            Original goal: {original_command}
            Failed at subtask: {failed_subtask_index}
            Reason for failure: {reason}

            Generate a new plan that accounts for this failure and achieves the goal.
            """

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": self.planning_prompt},
                        {"role": "user", "content": replan_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )

                response_text = response.choices[0].message['content'].strip()
                plan_json = self.extract_json_from_response(response_text)

                if plan_json and 'subtasks' in plan_json:
                    self.current_plan = plan_json['subtasks']
                    self.current_subtask_index = 0

                    # Publish new plan
                    plan_msg = String()
                    plan_msg.data = json.dumps(self.current_plan)
                    self.plan_publisher.publish(plan_msg)

                    self.get_logger().info('Replanned successfully')

                    # Start executing new plan
                    self.publish_next_subtask()

            except Exception as e:
                self.get_logger().error(f'Replanning error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = CognitivePlanningNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Cognitive Planning Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Error Handling and Clarification Strategies

LLMs need robust error handling and should be able to ask for clarification:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import openai
import json
from typing import Optional

class ErrorHandlingLLMNode(Node):
    def __init__(self):
        super().__init__('error_handling_llm_node')

        # Publishers
        self.response_publisher = self.create_publisher(
            String,
            '/llm_response',
            10
        )

        self.clarification_publisher = self.create_publisher(
            String,
            '/clarification_request',
            10
        )

        self.status_publisher = self.create_publisher(
            String,
            '/llm_status',
            10
        )

        # Subscribers
        self.command_subscriber = self.create_subscription(
            String,
            '/user_command',
            self.command_callback,
            10
        )

        self.environment_subscriber = self.create_subscription(
            String,
            '/environment_info',
            self.environment_callback,
            10
        )

        # Initialize
        openai.api_key = self.get_parameter_or('openai_api_key', 'your-api-key-here').value
        self.environment_info = {}
        self.pending_clarifications = {}

        self.get_logger().info('Error Handling LLM Node initialized')

    def command_callback(self, msg):
        """Process commands with error handling"""
        command = msg.data
        self.get_logger().info(f'Processing command: {command}')

        # Validate command and environment
        validation_result = self.validate_command_and_environment(command)

        if validation_result['status'] == 'valid':
            # Process the command normally
            response = self.process_command(command, validation_result['context'])
            self.publish_response(response)
        elif validation_result['status'] == 'needs_clarification':
            # Ask for clarification
            clarification = validation_result['clarification']
            self.request_clarification(command, clarification)
        elif validation_result['status'] == 'error':
            # Handle error case
            error_response = validation_result['error']
            self.publish_error_response(error_response)

    def validate_command_and_environment(self, command):
        """Validate command against current environment"""
        # Check if command is clear and actionable
        if not command.strip():
            return {
                'status': 'error',
                'error': 'Empty command received'
            }

        # Check for ambiguous references
        ambiguous_indicators = ['it', 'that', 'there', 'this', 'the object']
        command_lower = command.lower()

        for indicator in ambiguous_indicators:
            if indicator in command_lower:
                # Check if the reference is clear from context
                if not self.has_clear_reference(command, indicator):
                    return {
                        'status': 'needs_clarification',
                        'clarification': f'Could you clarify what "{indicator}" refers to in your command?'
                    }

        # Check if required objects/environment information is available
        if 'pick up' in command_lower or 'grasp' in command_lower:
            if not self.environment_info.get('objects', []):
                return {
                    'status': 'needs_clarification',
                    'clarification': 'I don\'t see any objects to pick up. Could you describe what you want me to pick up?'
                }

        # Check if required locations are known
        if 'go to' in command_lower or 'navigate to' in command_lower:
            if not self.environment_info.get('locations', []):
                return {
                    'status': 'needs_clarification',
                    'clarification': 'I don\'t know the locations in this area. Could you specify where you want me to go?'
                }

        return {
            'status': 'valid',
            'context': {
                'command': command,
                'environment': self.environment_info
            }
        }

    def has_clear_reference(self, command, ambiguous_word):
        """Check if ambiguous reference has clear context"""
        # In a real implementation, this would use more sophisticated NLP
        # For now, return False to demonstrate clarification
        return False

    def process_command(self, command, context):
        """Process validated command"""
        try:
            # Use LLM to process the command
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful robot assistant. Respond with executable actions in JSON format."
                    },
                    {
                        "role": "user",
                        "content": f"Command: {command}\nEnvironment: {context['environment']}\nProvide action in JSON format."
                    }
                ],
                temperature=0.1,
                max_tokens=200
            )

            return response.choices[0].message['content'].strip()

        except Exception as e:
            self.get_logger().error(f'Command processing error: {e}')
            return json.dumps({
                'error': f'Could not process command: {str(e)}',
                'retry_suggestion': 'Please rephrase your command'
            })

    def request_clarification(self, original_command, clarification_request):
        """Request clarification from user"""
        clarification_msg = String()
        clarification_msg.data = json.dumps({
            'original_command': original_command,
            'request': clarification_request,
            'timestamp': self.get_clock().now().nanoseconds
        })

        self.clarification_publisher.publish(clarification_msg)
        self.get_logger().info(f'Clarification requested: {clarification_request}')

        # Store pending clarification
        clarification_id = f"clarify_{self.get_clock().now().nanoseconds}"
        self.pending_clarifications[clarification_id] = {
            'original_command': original_command,
            'request_time': self.get_clock().now().nanoseconds
        }

    def environment_callback(self, msg):
        """Update environment information"""
        try:
            env_data = json.loads(msg.data)
            self.environment_info.update(env_data)
            self.get_logger().info(f'Environment updated: {list(env_data.keys())}')
        except json.JSONDecodeError:
            self.get_logger().error('Invalid environment data format')

    def publish_response(self, response):
        """Publish LLM response"""
        response_msg = String()
        response_msg.data = response
        self.response_publisher.publish(response_msg)

    def publish_error_response(self, error_msg):
        """Publish error response"""
        error_response = json.dumps({
            'error': error_msg,
            'status': 'error'
        })

        response_msg = String()
        response_msg.data = error_response
        self.response_publisher.publish(response_msg)

        status_msg = String()
        status_msg.data = 'error'
        self.status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ErrorHandlingLLMNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Error Handling LLM Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for LLM Integration

1. **Safety First**: Always validate LLM outputs before execution
2. **Context Awareness**: Provide sufficient context for accurate responses
3. **Error Handling**: Implement robust error handling and fallback strategies
4. **Privacy**: Be careful with sensitive information in prompts
5. **Cost Management**: Monitor API usage and implement caching where appropriate
6. **Reliability**: Have fallback mechanisms when LLM services are unavailable

LLM integration provides powerful cognitive capabilities for robotic systems, enabling natural interaction and complex task execution. The next section will cover mapping natural language instructions to ROS 2 actions.