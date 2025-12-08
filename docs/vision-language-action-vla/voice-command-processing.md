---
title: Voice Command Processing
sidebar_position: 5
---

# Module 4: Vision-Language-Action (VLA)

## Voice Command Processing for Human-Robot Interaction

Voice command processing enables natural human-robot interaction by converting spoken language into executable robotic actions. This section covers speech recognition, natural language understanding, and command execution in the context of humanoid robotics.

### Speech Recognition Pipeline

The speech recognition pipeline converts audio signals to text that can be processed by the robot's cognitive systems:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import AudioData
import speech_recognition as sr
import pyaudio
import threading
import queue
import numpy as np
import json
from typing import Optional

class SpeechRecognitionNode(Node):
    def __init__(self):
        super().__init__('speech_recognition_node')

        # Publishers
        self.transcript_publisher = self.create_publisher(
            String,
            '/speech_transcript',
            10
        )

        self.interim_result_publisher = self.create_publisher(
            String,
            '/interim_speech_result',
            10
        )

        self.listening_status_publisher = self.create_publisher(
            Bool,
            '/is_listening',
            10
        )

        # Subscribers
        self.audio_subscriber = self.create_subscription(
            AudioData,
            '/audio',
            self.audio_callback,
            10
        )

        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Set up for ambient noise adjustment
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        # Audio processing components
        self.audio_queue = queue.Queue()
        self.listening_active = True
        self.is_listening = False

        # Speech recognition parameters
        self.energy_threshold = 4000  # Minimum audio energy to consider for recording
        self.dynamic_energy_threshold = True
        self.pause_threshold = 0.8  # Seconds of non-speaking audio before a phrase is considered complete
        self.phrase_threshold = 0.3  # Minimum seconds of speaking audio before we consider the speaking audio a phrase
        self.non_speaking_duration = 0.5  # Seconds of non-speaking audio to keep on both sides of the recording

        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self.process_audio_stream, daemon=True)
        self.audio_thread.start()

        # Start listening status publisher
        self.listening_timer = self.create_timer(0.5, self.publish_listening_status)

        self.get_logger().info('Speech Recognition Node initialized')

    def audio_callback(self, msg):
        """Receive audio data from microphone"""
        try:
            # Convert AudioData message to AudioData object for speech recognition
            audio_data = sr.AudioData(msg.data, msg.info.sample_rate, msg.info.sample_size // 8)

            # Add to processing queue
            self.audio_queue.put(audio_data)

        except Exception as e:
            self.get_logger().error(f'Error processing audio message: {e}')

    def process_audio_stream(self):
        """Continuously process audio stream for speech"""
        while self.listening_active:
            try:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get(timeout=0.1)

                    # Process the audio data
                    self.process_audio_chunk(audio_data)

                else:
                    # Small delay to prevent busy waiting
                    import time
                    time.sleep(0.01)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Audio stream processing error: {e}')

    def process_audio_chunk(self, audio_data):
        """Process a chunk of audio data"""
        try:
            # Convert to text using Google Speech Recognition (or other engine)
            # For offline recognition, you might use pocketsphinx or vosk
            text = self.recognizer.recognize_google(audio_data)

            if text.strip():
                # Publish the recognized text
                transcript_msg = String()
                transcript_msg.data = text.strip()
                self.transcript_publisher.publish(transcript_msg)

                self.get_logger().info(f'Recognized: {text.strip()}')

        except sr.UnknownValueError:
            # Speech was detected but not understood
            self.get_logger().debug('Speech detected but not understood')
        except sr.RequestError as e:
            # API request error
            self.get_logger().error(f'Speech recognition error: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing audio chunk: {e}')

    def publish_listening_status(self):
        """Publish current listening status"""
        status_msg = Bool()
        status_msg.data = self.is_listening
        self.listening_status_publisher.publish(status_msg)

    def start_listening(self):
        """Start active listening"""
        self.is_listening = True
        self.get_logger().info('Started listening for speech')

    def stop_listening(self):
        """Stop active listening"""
        self.is_listening = False
        self.get_logger().info('Stopped listening for speech')

    def shutdown(self):
        """Clean shutdown"""
        self.listening_active = False
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = SpeechRecognitionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Speech Recognition Node')
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Natural Language Understanding for Commands

Converting recognized speech into structured commands requires natural language understanding:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ParsedCommand:
    action: str
    parameters: Dict[str, any]
    confidence: float
    raw_command: str

class NaturalLanguageUnderstandingNode(Node):
    def __init__(self):
        super().__init__('nlu_node')

        # Publishers
        self.parsed_command_publisher = self.create_publisher(
            String,
            '/parsed_command',
            10
        )

        self.intent_publisher = self.create_publisher(
            String,
            '/detected_intent',
            10
        )

        # Subscribers
        self.speech_transcript_subscriber = self.create_subscription(
            String,
            '/speech_transcript',
            self.speech_transcript_callback,
            10
        )

        # Initialize NLU components
        self.command_patterns = {
            'navigation': [
                r'navigate to (.+)',
                r'go to (.+)',
                r'move to (.+)',
                r'travel to (.+)',
                r'go to the (.+)',
                r'head to (.+)'
            ],
            'manipulation': [
                r'pick up (.+)',
                r'grasp (.+)',
                r'grab (.+)',
                r'take (.+)',
                r'pick (.+) up',
                r'lift (.+)',
                r'place (.+) at (.+)',
                r'put (.+) at (.+)'
            ],
            'detection': [
                r'find (.+)',
                r'locate (.+)',
                r'where is (.+)',
                r'show me (.+)',
                r'detect (.+)'
            ],
            'interaction': [
                r'hello',
                r'hi',
                r'hey',
                r'stop',
                r'wait',
                r'continue',
                r'help'
            ]
        }

        # Object and location dictionaries
        self.object_synonyms = {
            'red block': ['red cube', 'red box', 'red object'],
            'blue cylinder': ['blue tube', 'blue can'],
            'green sphere': ['green ball', 'green orb']
        }

        self.location_synonyms = {
            'kitchen': ['cooking area', 'food area'],
            'living room': ['sitting room', 'lounge'],
            'bedroom': ['sleeping room'],
            'table': ['desk', 'counter']
        }

        self.get_logger().info('Natural Language Understanding Node initialized')

    def speech_transcript_callback(self, msg):
        """Process speech transcript for command understanding"""
        transcript = msg.data.lower().strip()
        self.get_logger().info(f'Processing transcript: {transcript}')

        # Parse the command
        parsed_command = self.parse_command(transcript)

        if parsed_command:
            # Publish parsed command
            command_msg = String()
            command_msg.data = json.dumps({
                'action': parsed_command.action,
                'parameters': parsed_command.parameters,
                'confidence': parsed_command.confidence,
                'raw_command': parsed_command.raw_command
            })
            self.parsed_command_publisher.publish(command_msg)

            # Publish intent
            intent_msg = String()
            intent_msg.data = json.dumps({
                'intent': parsed_command.action,
                'confidence': parsed_command.confidence,
                'command': transcript
            })
            self.intent_publisher.publish(intent_msg)

            self.get_logger().info(f'Parsed command: {parsed_command.action} with confidence {parsed_command.confidence:.2f}')

    def parse_command(self, command: str) -> Optional[ParsedCommand]:
        """Parse natural language command into structured format"""
        # Normalize the command
        normalized_command = self.normalize_command(command)

        # Try to match against known patterns
        for action_type, patterns in self.command_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, normalized_command, re.IGNORECASE)
                if match:
                    # Extract parameters based on pattern
                    parameters = self.extract_parameters(action_type, match, normalized_command)

                    # Calculate confidence based on pattern match quality
                    confidence = self.calculate_confidence(action_type, match, normalized_command)

                    return ParsedCommand(
                        action=action_type,
                        parameters=parameters,
                        confidence=confidence,
                        raw_command=command
                    )

        # If no pattern matches, try more sophisticated NLU
        return self.fallback_nlu(command)

    def normalize_command(self, command: str) -> str:
        """Normalize command for better pattern matching"""
        # Remove extra whitespace
        normalized = ' '.join(command.split())

        # Expand contractions (simple version)
        contractions = {
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not",
            "shouldn't": "should not",
            "couldn't": "could not",
            "wouldn't": "would not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "doesn't": "does not",
            "didn't": "did not"
        }

        for contraction, expansion in contractions.items():
            normalized = normalized.replace(contraction, expansion)

        return normalized

    def extract_parameters(self, action_type: str, match, command: str) -> Dict[str, any]:
        """Extract parameters from matched command"""
        parameters = {}

        if action_type == 'navigation':
            # Extract destination
            destination = match.group(1).strip()
            parameters['destination'] = self.resolve_location(destination)

        elif action_type == 'manipulation':
            # Handle different manipulation patterns
            if 'place' in command or 'put' in command:
                # Pattern: place X at Y or put X at Y
                obj = match.group(1).strip()
                location = match.group(2).strip()
                parameters['object'] = self.resolve_object(obj)
                parameters['destination'] = self.resolve_location(location)
            else:
                # Pattern: pick up X, grasp X, etc.
                obj = match.group(1).strip()
                parameters['object'] = self.resolve_object(obj)

        elif action_type == 'detection':
            # Extract object to find
            obj = match.group(1).strip()
            parameters['object'] = self.resolve_object(obj)

        return parameters

    def resolve_location(self, location: str) -> str:
        """Resolve location name to canonical form"""
        # Check for direct match
        if location in self.location_synonyms:
            return location

        # Check synonyms
        for canonical, synonyms in self.location_synonyms.items():
            if location in synonyms or any(syn in location for syn in synonyms):
                return canonical

        # Return as is if no match found
        return location

    def resolve_object(self, obj: str) -> str:
        """Resolve object name to canonical form"""
        # Check for direct match
        if obj in self.object_synonyms:
            return obj

        # Check synonyms
        for canonical, synonyms in self.object_synonyms.items():
            if obj in synonyms or any(syn in obj for syn in synonyms):
                return canonical

        # Return as is if no match found
        return obj

    def calculate_confidence(self, action_type: str, match, command: str) -> float:
        """Calculate confidence in the command parsing"""
        # Base confidence on pattern match
        base_confidence = 0.8

        # Adjust based on command length and complexity
        command_length = len(command.split())
        if command_length > 10:  # Very long commands might be ambiguous
            base_confidence *= 0.8
        elif command_length < 2:  # Very short might be incomplete
            base_confidence *= 0.7

        # Adjust based on match completeness
        match_ratio = len(match.group(0)) / len(command)
        if match_ratio < 0.5:  # Match covers less than half the command
            base_confidence *= 0.6

        return min(1.0, base_confidence)

    def fallback_nlu(self, command: str) -> Optional[ParsedCommand]:
        """Fallback NLU when pattern matching fails"""
        # Simple keyword-based approach
        command_lower = command.lower()

        # Check for keywords that might indicate action type
        navigation_keywords = ['go', 'move', 'navigate', 'travel', 'head', 'to']
        manipulation_keywords = ['pick', 'grasp', 'grab', 'take', 'lift', 'place', 'put']
        detection_keywords = ['find', 'locate', 'where', 'show', 'detect']

        if any(keyword in command_lower for keyword in navigation_keywords):
            return ParsedCommand(
                action='navigation',
                parameters={'destination': 'unknown'},
                confidence=0.3,
                raw_command=command
            )
        elif any(keyword in command_lower for keyword in manipulation_keywords):
            return ParsedCommand(
                action='manipulation',
                parameters={'object': 'unknown'},
                confidence=0.3,
                raw_command=command
            )
        elif any(keyword in command_lower for keyword in detection_keywords):
            return ParsedCommand(
                action='detection',
                parameters={'object': 'unknown'},
                confidence=0.3,
                raw_command=command
            )
        elif any(greeting in command_lower for greeting in ['hello', 'hi', 'hey']):
            return ParsedCommand(
                action='interaction',
                parameters={'type': 'greeting'},
                confidence=0.9,
                raw_command=command
            )

        # If no keywords match, return None
        return None

def main(args=None):
    rclpy.init(args=args)
    node = NaturalLanguageUnderstandingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down NLU Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Command Execution and Action Mapping

Mapping parsed commands to executable robotic actions:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Twist, Point
from sensor_msgs.msg import JointState
import json
from typing import Dict, Any
import time

class CommandExecutionNode(Node):
    def __init__(self):
        super().__init__('command_execution_node')

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.joint_trajectory_publisher = self.create_publisher(
            JointState,
            '/joint_trajectory',
            10
        )

        self.action_status_publisher = self.create_publisher(
            String,
            '/action_status',
            10
        )

        self.navigation_goal_publisher = self.create_publisher(
            Pose,
            '/move_base_simple/goal',
            10
        )

        # Subscribers
        self.parsed_command_subscriber = self.create_subscription(
            String,
            '/parsed_command',
            self.parsed_command_callback,
            10
        )

        # Initialize execution state
        self.current_action = None
        self.action_queue = []
        self.is_executing = False

        # Location mapping (in a real system, this would come from a map)
        self.location_map = {
            'kitchen': Point(x=2.0, y=1.0, z=0.0),
            'living room': Point(x=-1.0, y=2.0, z=0.0),
            'bedroom': Point(x=0.0, y=-2.0, z=0.0),
            'table': Point(x=1.5, y=0.0, z=0.0),
            'desk': Point(x=1.5, y=0.0, z=0.0)  # Synonym for table
        }

        # Object locations (would come from perception system in real implementation)
        self.object_locations = {
            'red block': Point(x=1.0, y=0.5, z=0.0),
            'blue cylinder': Point(x=0.8, y=-0.5, z=0.0),
            'green sphere': Point(x=-0.5, y=1.0, z=0.0)
        }

        self.get_logger().info('Command Execution Node initialized')

    def parsed_command_callback(self, msg):
        """Process parsed command for execution"""
        try:
            command_data = json.loads(msg.data)
            action = command_data['action']
            parameters = command_data['parameters']
            confidence = command_data['confidence']

            self.get_logger().info(f'Received command: {action} with params {parameters} (confidence: {confidence:.2f})')

            # Check confidence threshold
            if confidence < 0.5:
                self.get_logger().warn(f'Command confidence too low ({confidence:.2f}), skipping')
                self.publish_action_status('skipped', 'low_confidence', confidence)
                return

            # Queue the action for execution
            action_item = {
                'action': action,
                'parameters': parameters,
                'confidence': confidence,
                'timestamp': self.get_clock().now().nanoseconds
            }

            self.action_queue.append(action_item)

            # Start execution if not already running
            if not self.is_executing:
                self.execute_next_action()

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in parsed command message')

    def execute_next_action(self):
        """Execute the next action in the queue"""
        if not self.action_queue or self.is_executing:
            return

        self.is_executing = True
        action_item = self.action_queue.pop(0)
        self.current_action = action_item

        action_type = action_item['action']
        parameters = action_item['parameters']

        self.get_logger().info(f'Executing action: {action_type} with params {parameters}')

        # Execute based on action type
        if action_type == 'navigation':
            success = self.execute_navigation(parameters)
        elif action_type == 'manipulation':
            success = self.execute_manipulation(parameters)
        elif action_type == 'detection':
            success = self.execute_detection(parameters)
        elif action_type == 'interaction':
            success = self.execute_interaction(parameters)
        else:
            self.get_logger().warn(f'Unknown action type: {action_type}')
            success = False

        # Publish completion status
        status = 'success' if success else 'failed'
        self.publish_action_status(status, action_type, action_item['confidence'])

        # Mark execution as complete
        self.is_executing = False

        # Process next action if available
        if self.action_queue:
            # Small delay before next action
            time.sleep(0.1)
            self.execute_next_action()

    def execute_navigation(self, parameters: Dict[str, Any]) -> bool:
        """Execute navigation command"""
        destination = parameters.get('destination', '').lower()

        # Look up destination coordinates
        if destination in self.location_map:
            target_point = self.location_map[destination]

            # Create navigation goal
            goal_pose = Pose()
            goal_pose.position = target_point
            goal_pose.orientation.w = 1.0  # No rotation

            # Publish navigation goal
            self.navigation_goal_publisher.publish(goal_pose)

            self.get_logger().info(f'Navigating to {destination} at ({target_point.x}, {target_point.y})')

            # In a real implementation, you'd wait for navigation to complete
            # and return True/False based on success
            return True
        else:
            self.get_logger().warn(f'Unknown destination: {destination}')
            return False

    def execute_manipulation(self, parameters: Dict[str, Any]) -> bool:
        """Execute manipulation command"""
        obj_name = parameters.get('object', '').lower()
        destination = parameters.get('destination')

        # Find object location
        if obj_name in self.object_locations:
            obj_position = self.object_locations[obj_name]
            self.get_logger().info(f'Attempting to manipulate {obj_name} at ({obj_position.x}, {obj_position.y})')

            # In a real implementation, this would:
            # 1. Navigate to object location
            # 2. Detect and approach object
            # 3. Grasp object
            # 4. If destination provided, navigate to destination and place object

            if destination:
                # Navigate to destination and place object
                dest_point = self.location_map.get(destination)
                if dest_point:
                    self.get_logger().info(f'Moving to {destination} to place {obj_name}')
                    # Would execute place action here
                    return True
                else:
                    self.get_logger().warn(f'Unknown destination for placement: {destination}')
                    return False
            else:
                # Just pick up object
                self.get_logger().info(f'Picking up {obj_name}')
                # Would execute pick action here
                return True
        else:
            self.get_logger().warn(f'Object not found: {obj_name}')
            return False

    def execute_detection(self, parameters: Dict[str, Any]) -> bool:
        """Execute detection command"""
        obj_name = parameters.get('object', '').lower()
        self.get_logger().info(f'Searching for {obj_name}')

        # In a real implementation, this would activate perception system
        # to search for the specified object

        if obj_name in self.object_locations:
            obj_position = self.object_locations[obj_name]
            self.get_logger().info(f'Found {obj_name} at ({obj_position.x}, {obj_position.y})')
            return True
        else:
            self.get_logger().info(f'Could not find {obj_name}')
            return False

    def execute_interaction(self, parameters: Dict[str, Any]) -> bool:
        """Execute interaction command"""
        interaction_type = parameters.get('type', 'unknown')

        if interaction_type == 'greeting':
            self.get_logger().info('Greeting detected - robot acknowledges')
            # In a real system, this might trigger audio response
            return True
        else:
            self.get_logger().info(f'Interaction type: {interaction_type}')
            return True

    def publish_action_status(self, status: str, action_type: str, confidence: float):
        """Publish action execution status"""
        status_msg = String()
        status_msg.data = json.dumps({
            'status': status,
            'action_type': action_type,
            'confidence': confidence,
            'timestamp': self.get_clock().now().nanoseconds,
            'current_action': self.current_action
        })
        self.action_status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CommandExecutionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Command Execution Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Voice Command Context Management

Managing context and handling multi-step commands:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

@dataclass
class ConversationContext:
    """Maintains context for ongoing conversations"""
    last_command: Optional[str] = None
    last_entity: Optional[str] = None
    last_location: Optional[str] = None
    last_action: Optional[str] = None
    timestamp: datetime = None
    active_references: List[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.active_references is None:
            self.active_references = []

class ContextAwareCommandNode(Node):
    def __init__(self):
        super().__init__('context_aware_command_node')

        # Publishers
        self.enhanced_command_publisher = self.create_publisher(
            String,
            '/enhanced_command',
            10
        )

        self.context_publisher = self.create_publisher(
            String,
            '/conversation_context',
            10
        )

        # Subscribers
        self.parsed_command_subscriber = self.create_subscription(
            String,
            '/parsed_command',
            self.parsed_command_callback,
            10
        )

        self.user_response_subscriber = self.create_subscription(
            String,
            '/user_response',
            self.user_response_callback,
            10
        )

        # Initialize context management
        self.context = ConversationContext()
        self.context_timeout = timedelta(minutes=5)  # Clear context after 5 minutes
        self.pronoun_resolution = {
            'it': None,
            'that': None,
            'this': None,
            'them': None,
            'there': None
        }

        # Context update timer
        self.context_timer = self.create_timer(1.0, self.check_context_timeout)

        self.get_logger().info('Context-Aware Command Node initialized')

    def parsed_command_callback(self, msg):
        """Process parsed command with context awareness"""
        try:
            command_data = json.loads(msg.data)
            original_action = command_data['action']
            original_params = command_data['parameters']
            confidence = command_data['confidence']
            raw_command = command_data['raw_command']

            self.get_logger().info(f'Processing command with context: {raw_command}')

            # Enhance command with context
            enhanced_command = self.enhance_command_with_context(
                original_action, original_params, raw_command
            )

            # Update context based on command
            self.update_context_from_command(enhanced_command, raw_command)

            # Publish enhanced command
            enhanced_msg = String()
            enhanced_msg.data = json.dumps(enhanced_command)
            self.enhanced_command_publisher.publish(enhanced_msg)

            # Publish updated context
            self.publish_context()

            self.get_logger().info(f'Enhanced command: {enhanced_command["action"]} with params {enhanced_command["parameters"]}')

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in parsed command message')

    def user_response_callback(self, msg):
        """Process user responses that might provide context"""
        response = msg.data.lower().strip()
        self.get_logger().info(f'Received user response for context: {response}')

        # Update context based on response
        self.update_context_from_response(response)

    def enhance_command_with_context(self, action: str, parameters: Dict, raw_command: str) -> Dict:
        """Enhance command with contextual information"""
        enhanced_params = parameters.copy()

        # Resolve pronouns and ambiguous references
        enhanced_params = self.resolve_pronouns(enhanced_params, raw_command)

        # Add contextual defaults
        if action == 'navigation' and 'destination' not in enhanced_params:
            # If no destination specified, might be continuing from previous context
            if self.context.last_location:
                enhanced_params['destination'] = self.context.last_location

        elif action == 'manipulation' and 'object' not in enhanced_params:
            # If no object specified, might refer to last mentioned object
            if self.context.last_entity:
                enhanced_params['object'] = self.context.last_entity

        # Add temporal context
        enhanced_params['timestamp'] = datetime.now().isoformat()
        enhanced_params['context_age'] = (
            datetime.now() - self.context.timestamp
        ).total_seconds()

        return {
            'action': action,
            'parameters': enhanced_params,
            'original_parameters': parameters,
            'context_enhanced': True
        }

    def resolve_pronouns(self, parameters: Dict, raw_command: str) -> Dict:
        """Resolve pronouns and ambiguous references using context"""
        resolved_params = parameters.copy()

        # Check for pronoun usage in parameters
        for key, value in parameters.items():
            if isinstance(value, str):
                resolved_value = self.resolve_pronoun_in_text(value)
                if resolved_value != value:
                    resolved_params[key] = resolved_value

        # Also resolve based on command text
        command_lower = raw_command.lower()

        # Resolve "it", "that", "this" based on last entity
        if self.context.last_entity:
            for pronoun in ['it', 'that', 'this']:
                if pronoun in command_lower:
                    # Replace pronoun with actual entity in parameters
                    for key, value in resolved_params.items():
                        if isinstance(value, str) and pronoun in value.lower():
                            resolved_params[key] = value.lower().replace(pronoun, self.context.last_entity)

        return resolved_params

    def resolve_pronoun_in_text(self, text: str) -> str:
        """Resolve pronouns within a text string"""
        resolved_text = text

        # Simple pronoun resolution based on context
        if 'it' in text.lower() and self.context.last_entity:
            resolved_text = re.sub(r'\bit\b', self.context.last_entity, resolved_text, flags=re.IGNORECASE)

        if 'that' in text.lower() and self.context.last_entity:
            resolved_text = re.sub(r'\bthat\b', self.context.last_entity, resolved_text, flags=re.IGNORECASE)

        if 'this' in text.lower() and self.context.last_entity:
            resolved_text = re.sub(r'\bthis\b', self.context.last_entity, resolved_text, flags=re.IGNORECASE)

        return resolved_text

    def update_context_from_command(self, enhanced_command: Dict, raw_command: str):
        """Update conversation context based on command"""
        action = enhanced_command['action']
        params = enhanced_command['parameters']

        # Update context fields
        self.context.last_command = raw_command
        self.context.timestamp = datetime.now()

        # Update entity references
        if 'object' in params:
            self.context.last_entity = params['object']
            self.pronoun_resolution['it'] = params['object']
            self.pronoun_resolution['that'] = params['object']
            self.pronoun_resolution['this'] = params['object']

        if 'destination' in params:
            self.context.last_location = params['destination']
            self.pronoun_resolution['there'] = params['destination']

        self.context.last_action = action

        # Add to active references
        if 'object' in params:
            self.context.active_references.append(params['object'])

        # Keep active references manageable
        if len(self.context.active_references) > 10:
            self.context.active_references = self.context.active_references[-5:]

        self.get_logger().info(f'Updated context: entity={self.context.last_entity}, location={self.context.last_location}')

    def update_context_from_response(self, response: str):
        """Update context based on user response"""
        response_lower = response.lower()

        # Update pronoun resolution based on response
        if 'yes' in response_lower or 'correct' in response_lower:
            # Affirmative response - keep current context
            pass
        elif 'no' in response_lower or 'wrong' in response_lower:
            # Negative response - might need to reset some context
            pass

        # Update timestamp
        self.context.timestamp = datetime.now()

    def check_context_timeout(self):
        """Check if context has timed out and should be cleared"""
        if datetime.now() - self.context.timestamp > self.context_timeout:
            self.get_logger().info('Context timeout - clearing conversation context')
            self.context = ConversationContext()

    def publish_context(self):
        """Publish current conversation context"""
        context_msg = String()
        context_msg.data = json.dumps({
            'last_command': self.context.last_command,
            'last_entity': self.context.last_entity,
            'last_location': self.context.last_location,
            'last_action': self.context.last_action,
            'timestamp': self.context.timestamp.isoformat(),
            'active_references': self.context.active_references,
            'pronoun_resolution': self.pronoun_resolution
        })
        self.context_publisher.publish(context_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ContextAwareCommandNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Context-Aware Command Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Voice Command Error Handling and Recovery

Handling errors and providing feedback when commands cannot be executed:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import json
from typing import Dict, List
import random

class ErrorHandlingCommandNode(Node):
    def __init__(self):
        super().__init__('error_handling_command_node')

        # Publishers
        self.error_response_publisher = self.create_publisher(
            String,
            '/error_response',
            10
        )

        self.feedback_publisher = self.create_publisher(
            String,
            '/command_feedback',
            10
        )

        self.recovery_command_publisher = self.create_publisher(
            String,
            '/recovery_command',
            10
        )

        # Subscribers
        self.action_status_subscriber = self.create_subscription(
            String,
            '/action_status',
            self.action_status_callback,
            10
        )

        self.enhanced_command_subscriber = self.create_subscription(
            String,
            '/enhanced_command',
            self.enhanced_command_callback,
            10
        )

        # Initialize error handling
        self.error_history = []
        self.recovery_strategies = {
            'navigation_failed': self.recovery_navigation_failed,
            'object_not_found': self.recovery_object_not_found,
            'grasp_failed': self.recovery_grasp_failed,
            'path_blocked': self.recovery_path_blocked
        }

        self.get_logger().info('Error Handling Command Node initialized')

    def action_status_callback(self, msg):
        """Handle action status messages and detect errors"""
        try:
            status_data = json.loads(msg.data)
            status = status_data.get('status', 'unknown')
            action_type = status_data.get('action_type', 'unknown')

            if status == 'failed':
                error_type = self.classify_error(status_data)
                self.handle_error(error_type, status_data)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in action status message')

    def enhanced_command_callback(self, msg):
        """Process enhanced commands and prepare for potential errors"""
        try:
            command_data = json.loads(msg.data)
            # Store command for potential error recovery
            self.last_command = command_data
        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in enhanced command message')

    def classify_error(self, status_data: Dict) -> str:
        """Classify the type of error based on status information"""
        # In a real system, this would analyze more detailed error information
        action_type = status_data.get('action_type', 'unknown')

        if action_type == 'navigation':
            return 'navigation_failed'
        elif action_type == 'manipulation':
            # Would need more specific error information
            return 'grasp_failed'
        else:
            return 'unknown_error'

    def handle_error(self, error_type: str, status_data: Dict):
        """Handle different types of errors"""
        self.get_logger().error(f'Error detected: {error_type}')

        # Add to error history
        error_entry = {
            'type': error_type,
            'timestamp': self.get_clock().now().nanoseconds,
            'status_data': status_data
        }
        self.error_history.append(error_entry)

        # Keep history manageable
        if len(self.error_history) > 50:
            self.error_history = self.error_history[-25:]

        # Try to recover using appropriate strategy
        recovery_func = self.recovery_strategies.get(error_type)
        if recovery_func:
            recovery_result = recovery_func(status_data)
            if recovery_result:
                self.get_logger().info(f'Recovery attempted for {error_type}')
        else:
            # Default error handling
            self.default_error_handling(error_type, status_data)

    def recovery_navigation_failed(self, status_data: Dict) -> bool:
        """Recovery strategy for navigation failures"""
        self.get_logger().info('Attempting navigation recovery')

        # Possible recovery strategies:
        # 1. Retry with different path
        # 2. Ask for alternative destination
        # 3. Report obstacle and wait for clearance

        recovery_options = [
            "I couldn't reach the destination. Would you like me to try a different route?",
            "I encountered an obstacle. Could you clear the path or suggest an alternative location?",
            "Navigation failed. Should I try again or proceed to a different location?"
        ]

        response = random.choice(recovery_options)
        self.publish_error_response(response, 'navigation_recovery')

        return True

    def recovery_object_not_found(self, status_data: Dict) -> bool:
        """Recovery strategy for object detection failures"""
        self.get_logger().info('Attempting object detection recovery')

        recovery_options = [
            "I couldn't find that object. Could you point it out or describe its location?",
            "The object isn't visible to me. Is it in a different location?",
            "Object not found. Would you like me to search in a different area?"
        ]

        response = random.choice(recovery_options)
        self.publish_error_response(response, 'object_recovery')

        return True

    def recovery_grasp_failed(self, status_data: Dict) -> bool:
        """Recovery strategy for manipulation failures"""
        self.get_logger().info('Attempting grasp recovery')

        recovery_options = [
            "I couldn't grasp that object. It might be too heavy or in an awkward position.",
            "Grasp attempt failed. Would you like me to try again or select a different object?",
            "I'm having trouble picking up that item. Could you reposition it?"
        ]

        response = random.choice(recovery_options)
        self.publish_error_response(response, 'grasp_recovery')

        return True

    def recovery_path_blocked(self, status_data: Dict) -> bool:
        """Recovery strategy for blocked path"""
        self.get_logger().info('Attempting path recovery')

        recovery_options = [
            "The path is blocked. Could you clear the obstacle or guide me around it?",
            "I can't proceed due to an obstacle. Please help me navigate around it.",
            "Path blocked. Would you like me to wait or find an alternative route?"
        ]

        response = random.choice(recovery_options)
        self.publish_error_response(response, 'path_recovery')

        return True

    def default_error_handling(self, error_type: str, status_data: Dict):
        """Default error handling when specific strategy not available"""
        response = f"I encountered an issue with the last command: {error_type}. Could you please repeat or rephrase your request?"
        self.publish_error_response(response, 'default_recovery')

    def publish_error_response(self, response: str, recovery_type: str):
        """Publish error response to user"""
        response_msg = String()
        response_msg.data = json.dumps({
            'response': response,
            'recovery_type': recovery_type,
            'timestamp': self.get_clock().now().nanoseconds
        })
        self.error_response_publisher.publish(response_msg)

        # Also publish as feedback
        feedback_msg = String()
        feedback_msg.data = json.dumps({
            'type': 'error_feedback',
            'message': response,
            'recovery_type': recovery_type
        })
        self.feedback_publisher.publish(feedback_msg)

    def provide_success_feedback(self, action_type: str):
        """Provide positive feedback for successful actions"""
        success_messages = {
            'navigation': [
                "I've reached the destination successfully.",
                "Navigation complete. I'm at the specified location.",
                "Successfully navigated to the target location."
            ],
            'manipulation': [
                "Object manipulation completed successfully.",
                "Successfully picked up/placed the object.",
                "Manipulation task completed."
            ],
            'detection': [
                "Object detected successfully.",
                "I found the requested object.",
                "Detection task completed."
            ]
        }

        if action_type in success_messages:
            message = random.choice(success_messages[action_type])
            feedback_msg = String()
            feedback_msg.data = json.dumps({
                'type': 'success_feedback',
                'message': message
            })
            self.feedback_publisher.publish(feedback_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ErrorHandlingCommandNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Error Handling Command Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Voice Command Performance Optimization

Optimizing voice command processing for real-time performance:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from sensor_msgs.msg import AudioData
import json
import time
from collections import deque
import threading

class OptimizedVoiceCommandNode(Node):
    def __init__(self):
        super().__init__('optimized_voice_command_node')

        # Publishers
        self.performance_publisher = self.create_publisher(
            Float32,
            '/command_processing_time',
            10
        )

        self.system_load_publisher = self.create_publisher(
            Float32,
            '/system_load',
            10
        )

        # Subscribers
        self.command_subscriber = self.create_subscription(
            String,
            '/parsed_command',
            self.optimized_command_callback,
            10
        )

        # Initialize optimization components
        self.command_queue = deque(maxlen=10)  # Limit queue size
        self.processing_times = deque(maxlen=50)  # Track processing times
        self.system_load = 0.0

        # Threading for parallel processing
        self.processing_thread = threading.Thread(target=self.process_commands, daemon=True)
        self.command_available = threading.Event()
        self.shutdown_event = threading.Event()

        # Start processing thread
        self.processing_thread.start()

        # Performance monitoring
        self.performance_timer = self.create_timer(1.0, self.publish_performance_metrics)

        self.get_logger().info('Optimized Voice Command Node initialized')

    def optimized_command_callback(self, msg):
        """Optimized command callback with queuing"""
        start_time = time.time()

        try:
            # Add to processing queue
            command_data = json.loads(msg.data)
            command_data['receive_time'] = start_time
            self.command_queue.append(command_data)

            # Signal that command is available
            self.command_available.set()

            # Calculate and store processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in command message')

    def process_commands(self):
        """Process commands in separate thread for better performance"""
        while not self.shutdown_event.is_set():
            try:
                if self.command_queue:
                    command_data = self.command_queue.popleft()
                    self.execute_optimized_command(command_data)
                else:
                    # Wait for command or timeout
                    self.command_available.wait(timeout=0.01)
                    self.command_available.clear()

            except Exception as e:
                self.get_logger().error(f'Command processing error: {e}')

    def execute_optimized_command(self, command_data: Dict):
        """Execute command with optimizations"""
        start_time = time.time()

        try:
            action = command_data['action']
            parameters = command_data['parameters']
            confidence = command_data['confidence']

            # Quick validation
            if confidence < 0.3:
                self.get_logger().debug('Skipping low confidence command')
                return

            # Optimized execution based on action type
            if action == 'navigation':
                self.quick_navigation(parameters)
            elif action == 'interaction':
                self.quick_interaction(parameters)
            else:
                # For complex actions, use standard processing
                self.standard_execution(action, parameters)

            execution_time = time.time() - start_time
            self.processing_times.append(execution_time)

        except Exception as e:
            self.get_logger().error(f'Command execution error: {e}')

    def quick_navigation(self, parameters: Dict):
        """Optimized navigation command execution"""
        # Simplified navigation for high-frequency commands
        destination = parameters.get('destination', 'unknown')
        self.get_logger().debug(f'Quick navigation to {destination}')

    def quick_interaction(self, parameters: Dict):
        """Optimized interaction command execution"""
        interaction_type = parameters.get('type', 'unknown')
        self.get_logger().debug(f'Quick interaction: {interaction_type}')

    def standard_execution(self, action: str, parameters: Dict):
        """Standard command execution for complex actions"""
        # For complex actions, publish to main execution system
        # This would be handled by other nodes in the system
        pass

    def publish_performance_metrics(self):
        """Publish performance metrics"""
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)

            # Publish average processing time
            time_msg = Float32()
            time_msg.data = avg_time
            self.performance_publisher.publish(time_msg)

            # Calculate system load (simplified)
            queue_size = len(self.command_queue)
            max_queue_size = self.command_queue.maxlen or 10
            self.system_load = min(1.0, queue_size / max_queue_size)

            load_msg = Float32()
            load_msg.data = self.system_load
            self.system_load_publisher.publish(load_msg)

            self.get_logger().debug(f'Performance - Avg processing time: {avg_time:.3f}s, System load: {self.system_load:.2f}')

    def shutdown(self):
        """Clean shutdown"""
        self.shutdown_event.set()
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = OptimizedVoiceCommandNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Optimized Voice Command Node')
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for Voice Command Processing

1. **Robust Speech Recognition**: Use multiple recognition engines and handle failures gracefully
2. **Context Awareness**: Maintain conversation context to resolve ambiguous references
3. **Error Recovery**: Implement comprehensive error handling and recovery strategies
4. **Performance Optimization**: Optimize for real-time processing with queuing and threading
5. **User Feedback**: Provide clear feedback about command understanding and execution status
6. **Privacy Considerations**: Handle sensitive information appropriately
7. **Adaptive Learning**: Improve recognition accuracy over time based on user interactions

Voice command processing enables natural and intuitive interaction with humanoid robots, making them more accessible and user-friendly for complex tasks.