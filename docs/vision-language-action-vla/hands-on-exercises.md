# Vision-Language-Action (VLA) - Hands-on Exercises

## Exercise 1: Voice Command Integration

In this exercise, you will integrate OpenAI Whisper for voice-to-text conversion and map natural language commands to ROS 2 actions.

### Objective
- Implement voice command processing
- Map natural language to ROS 2 action sequences
- Execute simple commands through voice input

### Prerequisites
- ROS 2 Humble installed
- OpenAI Whisper API access
- Microphone for voice input
- Basic understanding of ROS 2 actions

### Steps

1. **Setup Voice Processing Node**
   ```python
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   import openai
   import speech_recognition as sr

   class VoiceCommandNode(Node):
       def __init__(self):
           super().__init__('voice_command_node')
           self.publisher = self.create_publisher(String, 'voice_command', 10)
           self.recognizer = sr.Recognizer()
           self.microphone = sr.Microphone()

       def listen_for_command(self):
           with self.microphone as source:
               self.recognizer.adjust_for_ambient_noise(source)
               audio = self.recognizer.listen(source)

           try:
               command = self.recognizer.recognize_whisper_api(audio)
               return command
           except Exception as e:
               self.get_logger().error(f'Error recognizing speech: {e}')
               return None
   ```

2. **Implement Command Mapping**
   ```python
   def map_command_to_action(self, command_text):
       command_text = command_text.lower()

       if 'move forward' in command_text:
           return 'move_forward_action'
       elif 'turn left' in command_text:
           return 'turn_left_action'
       elif 'pick up object' in command_text:
           return 'pick_up_action'
       elif 'go to' in command_text:
           # Extract destination from command
           destination = self.extract_destination(command_text)
           return f'navigation_to_{destination}'
       else:
           return 'unknown_command'
   ```

3. **Test the Integration**
   - Run the voice command node
   - Speak a command like "Move forward 1 meter"
   - Observe the mapped action being published

### Expected Results
- Voice commands are accurately converted to text
- Commands are correctly mapped to appropriate ROS 2 actions
- System responds appropriately to voice input

## Exercise 2: Multi-Modal Perception Integration

In this exercise, you will implement multi-modal perception combining speech, vision, and gesture input.

### Objective
- Integrate multiple sensor inputs
- Fuse data from different modalities
- Create a unified perception system

### Prerequisites
- RGB-D camera (e.g., Intel RealSense)
- Microphone
- Basic computer vision knowledge

### Steps

1. **Setup Multi-Modal Input**
   ```python
   class MultiModalPerceptionNode(Node):
       def __init__(self):
           super().__init__('multi_modal_perception')

           # Publishers and subscribers
           self.image_sub = self.create_subscription(
               Image, '/camera/rgb/image_raw', self.image_callback, 10)
           self.depth_sub = self.create_subscription(
               Image, '/camera/depth/image_raw', self.depth_callback, 10)
           self.audio_sub = self.create_subscription(
               String, 'voice_command', self.audio_callback, 10)

           self.fusion_publisher = self.create_publisher(
               String, 'fused_perception', 10)
   ```

2. **Implement Data Fusion**
   ```python
   def fuse_inputs(self, image_data, depth_data, audio_data):
       # Process visual data
       visual_features = self.extract_visual_features(image_data)

       # Process audio data
       audio_features = self.process_audio_command(audio_data)

       # Combine features based on context
       fused_result = self.combine_features(
           visual_features, audio_features, depth_data)

       return fused_result
   ```

3. **Test Multi-Modal Integration**
   - Run the multi-modal perception node
   - Provide voice command: "Show me the red cube"
   - Verify that the system identifies the red cube in the visual input
   - Test with different combinations of inputs

### Expected Results
- System correctly identifies objects based on voice commands
- Visual and audio inputs are properly fused
- System responds to multi-modal commands

## Exercise 3: Cognitive Planning with LLM Integration

In this exercise, you will implement cognitive planning using large language models to translate high-level goals into action sequences.

### Objective
- Integrate LLM for planning
- Create task decomposition system
- Execute complex multi-step tasks

### Prerequisites
- Cohere API access
- ROS 2 action servers for basic tasks
- Understanding of task planning concepts

### Steps

1. **Setup Planning Node**
   ```python
   class CognitivePlannerNode(Node):
       def __init__(self):
           super().__init__('cognitive_planner')

           # Initialize LLM client
           import cohere
           # Ensure COHERE_API_KEY is available in environment or pass directly
           self.llm_client = cohere.Client(os.getenv('COHERE_API_KEY'))

           # Publishers/subscribers for planning
           self.goal_sub = self.create_subscription(
               String, 'high_level_goal', self.goal_callback, 10)
           self.plan_pub = self.create_publisher(
               String, 'action_plan', 10)

       def generate_plan(self, goal_description):
           prompt = f"""
           Given the following goal: "{goal_description}"

           Generate a sequence of actions that a humanoid robot can execute to achieve this goal.
           Each action should be simple and executable by a ROS 2 system.
           Return the actions in order as a list.

           Example format:
           1. Navigate to location X
           2. Detect object Y
           3. Grasp object Y
           4. Transport object Y to location Z
           5. Release object Y
           """

           response = self.llm_client.chat(
               model="command-r-plus",
               message=prompt,
               preamble="You are a robot planning assistant.",
               temperature=0.3
           )

           return response.text
   ```

2. **Implement Plan Execution**
   ```python
   def execute_plan(self, plan_text):
       # Parse the plan into individual actions
       actions = self.parse_plan_to_actions(plan_text)

       # Execute actions sequentially
       for action in actions:
           success = self.execute_single_action(action)
           if not success:
               self.get_logger().error(f'Action failed: {action}')
               break
   ```

3. **Test Planning and Execution**
   - Provide a complex goal: "Go to the kitchen, find the red cup, pick it up, and bring it to the table"
   - Verify that the system generates an appropriate action plan
   - Test execution of the plan (in simulation)

### Expected Results
- LLM generates appropriate action sequences for complex goals
- System successfully executes multi-step plans
- Planning adapts to environmental changes

## Exercise 4: Autonomous Task Completion

In this exercise, you will integrate all components to create a fully autonomous system that responds to voice commands and executes tasks.

### Objective
- Integrate voice processing, perception, and planning
- Create complete autonomous behavior
- Test end-to-end functionality

### Prerequisites
- All previous exercises completed
- Complete VLA system components
- Simulation environment (Gazebo/NVIDIA Isaac Sim)

### Steps

1. **System Integration**
   ```python
   class VLAMainNode(Node):
       def __init__(self):
           super().__init__('vla_main')

           # Initialize all components
           self.voice_node = VoiceCommandNode()
           self.perception_node = MultiModalPerceptionNode()
           self.planning_node = CognitivePlannerNode()

           # Connect components
           self.voice_node.publisher = self.create_publisher(String, 'user_command', 10)
           self.planning_node.plan_pub = self.create_publisher(String, 'execution_plan', 10)
   ```

2. **End-to-End Testing**
   - Launch complete VLA system
   - Issue voice command: "Robot, please bring me the blue bottle from the shelf"
   - Observe complete task execution:
     - Voice recognition
     - Object detection
     - Path planning
     - Grasping execution
     - Object delivery

3. **Performance Evaluation**
   - Measure success rate of task completion
   - Record execution time
   - Note any failures or errors

### Expected Results
- Complete autonomous task execution from voice command to completion
- High success rate (>80%) for simple tasks
- Reasonable execution time for complex tasks

## Troubleshooting Tips

1. **Voice Recognition Issues**
   - Check microphone permissions
   - Verify internet connection for Whisper API
   - Adjust microphone sensitivity

2. **Visual Processing Problems**
   - Ensure camera is properly calibrated
   - Check lighting conditions
   - Verify camera topic names

3. **Planning Failures**
   - Check LLM API access and quota
   - Verify action server availability
   - Review goal formulation

## Additional Challenges

1. **Multi-Step Tasks**: Try more complex commands requiring multiple sequential actions
2. **Ambiguous Commands**: Test with vague or ambiguous voice commands
3. **Error Recovery**: Test how the system handles failed actions
4. **Real-time Adaptation**: Test system response to changing environments