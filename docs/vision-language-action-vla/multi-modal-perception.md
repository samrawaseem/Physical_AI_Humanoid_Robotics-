---
title: Multi-Modal Perception
sidebar_position: 4
---

# Module 4: Vision-Language-Action (VLA)

## Multi-Modal Sensory Fusion for Robust Perception

Multi-modal perception combines information from multiple sensory modalities (vision, audio, tactile, etc.) to create a more robust and accurate understanding of the environment. This integration is crucial for humanoid robots operating in complex, dynamic environments.

### Understanding Multi-Modal Perception

Multi-modal perception involves:

- **Sensor Fusion**: Combining data from different sensors to improve accuracy
- **Cross-Modal Integration**: Using information from one modality to enhance another
- **Temporal Integration**: Combining sensory information over time
- **Contextual Understanding**: Using multiple modalities to understand context
- **Robustness**: Maintaining performance when individual sensors fail

### Visual Perception Integration

Visual perception forms the foundation of multi-modal systems:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PointStamped, PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import open3d as o3d

class VisualPerceptionNode(Node):
    def __init__(self):
        super().__init__('visual_perception_node')

        # Publishers
        self.object_detection_publisher = self.create_publisher(
            PointStamped,
            '/detected_objects',
            10
        )

        self.segmentation_publisher = self.create_publisher(
            Image,
            '/segmentation_result',
            10
        )

        self.point_cloud_publisher = self.create_publisher(
            PointCloud2,
            '/fused_point_cloud',
            10
        )

        # Subscribers
        self.rgb_subscriber = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.rgb_callback,
            10
        )

        self.depth_subscriber = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            '/camera/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        # Initialize
        self.bridge = CvBridge()
        self.camera_intrinsics = None
        self.latest_rgb = None
        self.latest_depth = None
        self.object_detector = cv2.dnn.readNetFromDarknet(
            'yolov4.cfg', 'yolov4.weights'
        )  # Placeholder - would need actual model files

        # Object detection parameters
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4

        self.get_logger().info('Visual Perception Node initialized')

    def camera_info_callback(self, msg):
        """Store camera intrinsic parameters"""
        self.camera_intrinsics = {
            'fx': msg.k[0],
            'fy': msg.k[4],
            'cx': msg.k[2],
            'cy': msg.k[5],
            'width': msg.width,
            'height': msg.height
        }

    def rgb_callback(self, msg):
        """Process RGB image for object detection and segmentation"""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Perform object detection
            detections = self.detect_objects(self.latest_rgb)

            # Publish detected objects
            for detection in detections:
                point_msg = PointStamped()
                point_msg.header = msg.header
                point_msg.point.x = detection['x']
                point_msg.point.y = detection['y']
                point_msg.point.z = detection['z']  # From depth
                self.object_detection_publisher.publish(point_msg)

            # Perform semantic segmentation
            if self.latest_depth is not None:
                segmentation = self.semantic_segmentation(
                    self.latest_rgb, self.latest_depth
                )

                # Publish segmentation result
                seg_msg = self.bridge.cv2_to_imgmsg(segmentation, encoding='mono8')
                seg_msg.header = msg.header
                self.segmentation_publisher.publish(seg_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {e}')

    def depth_callback(self, msg):
        """Process depth image"""
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def detect_objects(self, image):
        """Detect objects using deep learning model"""
        if self.object_detector is None:
            # Fallback: simple color-based detection
            return self.simple_color_detection(image)

        # Convert image for YOLO
        blob = cv2.dnn.blobFromImage(
            image, 1/255.0, (416, 416), swapRB=True, crop=False
        )
        self.object_detector.setInput(blob)
        outputs = self.object_detector.forward()

        # Process outputs (simplified - in practice, YOLO output processing is more complex)
        detections = []
        height, width = image.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = center_x - w // 2
                    y = center_y - h // 2

                    # Get depth at object center
                    depth = self.get_depth_at_pixel(center_x, center_y)

                    detections.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'confidence': confidence,
                        'class_id': class_id,
                        'z': depth  # Depth value
                    })

        return detections

    def simple_color_detection(self, image):
        """Fallback simple color-based object detection"""
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges (for demo: red objects)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)

                # Get depth at object center
                center_x, center_y = x + w//2, y + h//2
                depth = self.get_depth_at_pixel(center_x, center_y)

                detections.append({
                    'x': float(x),
                    'y': float(y),
                    'width': float(w),
                    'height': float(h),
                    'confidence': 0.8,  # High confidence for simple detection
                    'class_id': 0,  # Red object class
                    'z': depth
                })

        return detections

    def get_depth_at_pixel(self, x, y):
        """Get depth value at specific pixel"""
        if self.latest_depth is not None and 0 <= x < self.latest_depth.shape[1] and 0 <= y < self.latest_depth.shape[0]:
            return float(self.latest_depth[y, x])
        return 0.0  # Default depth

    def semantic_segmentation(self, rgb_image, depth_image):
        """Perform semantic segmentation by combining RGB and depth"""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # Use depth to enhance segmentation
        # Normalize depth for combination
        if depth_image.max() > 0:
            normalized_depth = (depth_image / depth_image.max() * 255).astype(np.uint8)
        else:
            normalized_depth = np.zeros_like(gray)

        # Combine RGB and depth information
        # This is a simplified approach - real segmentation would use deep learning
        combined = cv2.addWeighted(gray, 0.7, normalized_depth, 0.3, 0)

        # Apply threshold to create segmentation mask
        _, segmentation = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)

        return segmentation

def main(args=None):
    rclpy.init(args=args)
    node = VisualPerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Visual Perception Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Audio Processing Integration

Audio processing enables speech recognition and sound-based perception:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import AudioData
from std_msgs.msg import String
import pyaudio
import numpy as np
import librosa
import webrtcvad
from scipy import signal
import threading
import queue

class AudioPerceptionNode(Node):
    def __init__(self):
        super().__init__('audio_perception_node')

        # Publishers
        self.speech_publisher = self.create_publisher(
            String,
            '/detected_speech',
            10
        )

        self.sound_event_publisher = self.create_publisher(
            String,
            '/sound_events',
            10
        )

        self.audio_features_publisher = self.create_publisher(
            String,
            '/audio_features',
            10
        )

        # Subscribers
        self.audio_subscriber = self.create_subscription(
            AudioData,
            '/audio',
            self.audio_callback,
            10
        )

        # Initialize audio processing components
        self.audio_queue = queue.Queue()
        self.vad = webrtcvad.Vad(2)  # Aggressiveness mode 2
        self.is_listening = True

        # Audio processing parameters
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        self.speech_buffer = []
        self.silence_threshold = 0.01

        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.audio_thread.start()

        self.get_logger().info('Audio Perception Node initialized')

    def audio_callback(self, msg):
        """Receive audio data and add to processing queue"""
        try:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

            # Add to processing queue
            self.audio_queue.put(audio_data)

        except Exception as e:
            self.get_logger().error(f'Error processing audio message: {e}')

    def process_audio(self):
        """Continuously process audio from queue"""
        while self.is_listening:
            try:
                # Get audio from queue
                if not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get(timeout=0.1)

                    # Process the audio chunk
                    self.analyze_audio_chunk(audio_chunk)

                else:
                    # Small delay to prevent busy waiting
                    import time
                    time.sleep(0.01)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Audio processing error: {e}')

    def analyze_audio_chunk(self, audio_chunk):
        """Analyze audio chunk for speech and sound events"""
        # Check for voice activity
        is_speech = self.detect_voice_activity(audio_chunk)

        if is_speech:
            # Add to speech buffer
            self.speech_buffer.extend(audio_chunk)

            # If buffer is getting large, process as speech segment
            if len(self.speech_buffer) > self.sample_rate * 2:  # 2 seconds of speech
                speech_segment = np.array(self.speech_buffer)
                self.process_speech_segment(speech_segment)
                self.speech_buffer = []  # Reset buffer
        else:
            # Check if there was accumulated speech to process
            if len(self.speech_buffer) > self.sample_rate * 0.1:  # At least 0.1 seconds
                speech_segment = np.array(self.speech_buffer)
                self.process_speech_segment(speech_segment)
            self.speech_buffer = []  # Reset buffer for non-speech

        # Analyze for sound events
        sound_events = self.detect_sound_events(audio_chunk)
        if sound_events:
            self.publish_sound_events(sound_events)

        # Extract and publish audio features
        features = self.extract_audio_features(audio_chunk)
        self.publish_audio_features(features)

    def detect_voice_activity(self, audio_chunk):
        """Detect voice activity using WebRTC VAD"""
        # Convert to 16-bit PCM for VAD
        audio_int16 = (audio_chunk * 32767).astype(np.int16)

        # Process in frames
        frames = self.frame_generator(self.frame_duration, audio_int16, self.sample_rate)

        voice_frames = 0
        total_frames = 0

        for frame in frames:
            total_frames += 1
            if self.vad.is_speech(frame.tobytes(), self.sample_rate):
                voice_frames += 1

        # Consider speech if more than 30% of frames have voice activity
        return (voice_frames / total_frames) > 0.3 if total_frames > 0 else False

    def frame_generator(self, frame_duration_ms, audio, sample_rate):
        """Generate audio frames from PCM audio data"""
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        while offset + n < len(audio):
            yield audio[offset:offset + n]
            offset += n

    def process_speech_segment(self, speech_segment):
        """Process speech segment for recognition"""
        # In a real implementation, this would use speech recognition
        # For now, we'll just publish a placeholder indicating speech detected
        speech_msg = String()
        speech_msg.data = json.dumps({
            'type': 'speech_detected',
            'duration': len(speech_segment) / self.sample_rate,
            'timestamp': self.get_clock().now().nanoseconds
        })
        self.speech_publisher.publish(speech_msg)

    def detect_sound_events(self, audio_chunk):
        """Detect specific sound events in audio"""
        events = []

        # Calculate energy to detect loud sounds
        energy = np.mean(audio_chunk ** 2)

        if energy > 0.01:  # Threshold for "loud" sound
            events.append({
                'type': 'loud_sound',
                'energy': float(energy),
                'timestamp': self.get_clock().now().nanoseconds
            })

        # Use librosa for more sophisticated analysis
        try:
            # Calculate spectral features
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_chunk, sr=self.sample_rate
            )[0]

            # Detect if spectral centroid is high (indicating high-frequency sounds)
            if np.mean(spectral_centroids) > 2000:  # High frequency threshold
                events.append({
                    'type': 'high_frequency_sound',
                    'spectral_centroid': float(np.mean(spectral_centroids)),
                    'timestamp': self.get_clock().now().nanoseconds
                })

        except Exception as e:
            self.get_logger().warn(f'Error in spectral analysis: {e}')

        return events

    def extract_audio_features(self, audio_chunk):
        """Extract various audio features"""
        features = {}

        # Basic features
        features['rms_energy'] = float(np.sqrt(np.mean(audio_chunk ** 2)))
        features['zero_crossing_rate'] = float(np.mean(
            np.abs(np.diff(np.sign(audio_chunk))) / 2
        ))

        # Spectral features using librosa
        try:
            features['spectral_rolloff'] = float(
                np.mean(librosa.feature.spectral_rolloff(
                    y=audio_chunk, sr=self.sample_rate
                )[0])
            )

            features['spectral_bandwidth'] = float(
                np.mean(librosa.feature.spectral_bandwidth(
                    y=audio_chunk, sr=self.sample_rate
                )[0])
            )

            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_chunk, sr=self.sample_rate, n_mfcc=13)
            features['mfcc_mean'] = [float(np.mean(mfccs[i])) for i in range(mfccs.shape[0])]

        except Exception as e:
            self.get_logger().warn(f'Error extracting spectral features: {e}')

        return features

    def publish_sound_events(self, events):
        """Publish detected sound events"""
        for event in events:
            event_msg = String()
            event_msg.data = json.dumps(event)
            self.sound_event_publisher.publish(event_msg)

    def publish_audio_features(self, features):
        """Publish extracted audio features"""
        features_msg = String()
        features_msg.data = json.dumps(features)
        self.audio_features_publisher.publish(features_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AudioPerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Audio Perception Node')
    finally:
        node.is_listening = False
        if hasattr(node, 'audio_thread'):
            node.audio_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Tactile and IMU Integration

Tactile and inertial sensors provide important feedback for manipulation and navigation:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import WrenchStamped, Vector3
from std_msgs.msg import Float32MultiArray
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque
import json

class TactileIMUPerceptionNode(Node):
    def __init__(self):
        super().__init__('tactile_imu_perception_node')

        # Publishers
        self.pose_publisher = self.create_publisher(
            Vector3,
            '/estimated_pose',
            10
        )

        self.contact_publisher = self.create_publisher(
            WrenchStamped,
            '/contact_force',
            10
        )

        self.motion_publisher = self.create_publisher(
            Vector3,
            '/motion_state',
            10
        )

        # Subscribers
        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.joint_state_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.force_torque_subscriber = self.create_subscription(
            WrenchStamped,
            '/wrench',
            self.force_torque_callback,
            10
        )

        # Initialize state
        self.orientation = R.from_quat([0, 0, 0, 1])  # Identity rotation
        self.linear_velocity = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.position = np.array([0.0, 0.0, 0.0])
        self.acceleration_history = deque(maxlen=10)
        self.gesture_buffer = deque(maxlen=50)  # For gesture recognition

        # IMU calibration
        self.accel_bias = np.array([0.0, 0.0, 0.0])
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        self.calibration_samples = 0
        self.max_calibration_samples = 100

        # Motion detection thresholds
        self.motion_threshold = 0.1
        self.contact_threshold = 0.5

        self.get_logger().info('Tactile IMU Perception Node initialized')

    def imu_callback(self, msg):
        """Process IMU data for orientation and motion estimation"""
        # Extract IMU data
        linear_accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        angular_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Calibration phase - estimate biases
        if self.calibration_samples < self.max_calibration_samples:
            self.accel_bias += linear_accel
            self.gyro_bias += angular_vel
            self.calibration_samples += 1

            if self.calibration_samples == self.max_calibration_samples:
                self.accel_bias /= self.max_calibration_samples
                self.gyro_bias /= self.max_calibration_samples
                self.get_logger().info('IMU calibration completed')

            return

        # Remove biases
        linear_accel -= self.accel_bias
        angular_vel -= self.gyro_bias

        # Integrate angular velocity to get orientation change
        dt = 1.0 / 100.0  # Assuming 100 Hz IMU rate
        delta_angle = angular_vel * dt

        # Update orientation using quaternion integration
        delta_q = self.angle_axis_to_quaternion(delta_angle)
        current_q = self.orientation.as_quat()
        new_q = self.quaternion_multiply(current_q, delta_q)
        self.orientation = R.from_quat(new_q / np.linalg.norm(new_q))

        # Integrate acceleration to get velocity and position
        # Note: This is prone to drift, in practice would use other sensors
        current_orientation_matrix = self.orientation.as_matrix()
        world_accel = current_orientation_matrix @ linear_accel

        self.linear_velocity += world_accel * dt
        self.position += self.linear_velocity * dt

        # Store for history
        self.acceleration_history.append(linear_accel)

        # Detect motion patterns
        motion_detected = np.linalg.norm(linear_accel) > self.motion_threshold
        rotation_detected = np.linalg.norm(angular_vel) > self.motion_threshold

        # Publish estimated pose
        pose_msg = Vector3()
        pose_msg.x = float(self.position[0])
        pose_msg.y = float(self.position[1])
        pose_msg.z = float(self.position[2])
        self.pose_publisher.publish(pose_msg)

        # Publish motion state
        motion_msg = Vector3()
        motion_msg.x = float(np.linalg.norm(linear_accel))
        motion_msg.y = float(np.linalg.norm(angular_vel))
        motion_msg.z = float(1.0 if motion_detected else 0.0)
        self.motion_publisher.publish(motion_msg)

        # Add to gesture buffer for pattern recognition
        gesture_data = {
            'acceleration': linear_accel.tolist(),
            'angular_velocity': angular_vel.tolist(),
            'timestamp': msg.header.stamp.nanosec
        }
        self.gesture_buffer.append(gesture_data)

        # Check for gesture patterns
        detected_gesture = self.recognize_gesture()
        if detected_gesture:
            self.get_logger().info(f'Gesture detected: {detected_gesture}')

    def joint_state_callback(self, msg):
        """Process joint state data for tactile feedback"""
        # In a real implementation, this would process tactile sensors
        # attached to joints or end effectors

        # Example: detect if gripper is grasping based on joint positions
        try:
            # Look for gripper joints (example names)
            gripper_joints = [i for i, name in enumerate(msg.name)
                            if 'gripper' in name.lower() or 'finger' in name.lower()]

            if gripper_joints:
                # Calculate if gripper is closed based on joint positions
                gripper_positions = [msg.position[i] for i in gripper_joints]
                avg_position = sum(gripper_positions) / len(gripper_positions)

                # If gripper is closed, check for contact
                if avg_position < 0.1:  # Threshold for "closed"
                    # This would trigger tactile sensor processing
                    self.check_tactile_feedback()

        except (IndexError, ValueError):
            pass  # Joint not found or invalid data

    def force_torque_callback(self, msg):
        """Process force/torque sensor data"""
        force = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z
        ])

        torque = np.array([
            msg.wrench.torque.x,
            msg.wrench.torque.y,
            msg.wrench.torque.z
        ])

        # Detect contact
        force_magnitude = np.linalg.norm(force)
        torque_magnitude = np.linalg.norm(torque)

        if force_magnitude > self.contact_threshold or torque_magnitude > self.contact_threshold:
            # Contact detected
            contact_msg = WrenchStamped()
            contact_msg.header.stamp = self.get_clock().now().to_msg()
            contact_msg.header.frame_id = msg.header.frame_id
            contact_msg.wrench = msg.wrench
            self.contact_publisher.publish(contact_msg)

            self.get_logger().info(f'Contact detected: force={force_magnitude:.3f}, torque={torque_magnitude:.3f}')

    def check_tactile_feedback(self):
        """Check for tactile feedback from sensors"""
        # In a real implementation, this would process data from tactile sensors
        # For now, we'll just log that tactile processing is happening
        self.get_logger().info('Checking tactile feedback')

    def angle_axis_to_quaternion(self, angle_axis):
        """Convert angle-axis representation to quaternion"""
        angle = np.linalg.norm(angle_axis)
        if angle == 0:
            return np.array([1.0, 0.0, 0.0, 0.0])

        axis = angle_axis / angle
        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)

        return np.array([
            np.cos(half_angle),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half
        ])

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([w, x, y, z])

    def recognize_gesture(self):
        """Recognize gestures from IMU data"""
        if len(self.gesture_buffer) < 10:  # Need enough data
            return None

        # Convert buffer to numpy array for analysis
        accel_data = np.array([item['acceleration'] for item in self.gesture_buffer])
        gyro_data = np.array([item['angular_velocity'] for item in self.gesture_buffer])

        # Simple gesture recognition based on patterns
        avg_accel = np.mean(np.abs(accel_data), axis=0)
        avg_gyro = np.mean(np.abs(gyro_data), axis=0)

        # Detect shake gesture (high acceleration in multiple axes)
        if np.all(avg_accel > 0.5) and np.linalg.norm(avg_accel) > 1.0:
            return 'shake'

        # Detect rotation gesture (high angular velocity)
        if np.linalg.norm(avg_gyro) > 0.5:
            return 'rotate'

        # Detect tap gesture (short burst of acceleration)
        if len(self.acceleration_history) > 1:
            recent_accels = list(self.acceleration_history)[-5:]
            if len(recent_accels) == 5:
                # Check for peak in acceleration
                peak_idx = np.argmax([np.linalg.norm(acc) for acc in recent_accels])
                if peak_idx == 2 and np.linalg.norm(recent_accels[peak_idx]) > 2.0:  # Center peak
                    return 'tap'

        return None

def main(args=None):
    rclpy.init(args=args)
    node = TactileIMUPerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Tactile IMU Perception Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Sensor Fusion and Integration

Combining all sensory modalities into a coherent perception system:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, Imu, JointState
from geometry_msgs.msg import PoseStamped, PointStamped
from cv_bridge import CvBridge
import numpy as np
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import threading
import time

@dataclass
class SensoryInput:
    """Data structure for sensory input with timestamp and confidence"""
    modality: str  # 'vision', 'audio', 'tactile', 'imu', etc.
    data: any
    timestamp: float
    confidence: float = 1.0
    source_frame: str = 'base_link'

class MultiModalFusionNode(Node):
    def __init__(self):
        super().__init__('multi_modal_fusion_node')

        # Publishers
        self.fused_perception_publisher = self.create_publisher(
            String,
            '/fused_perception',
            10
        )

        self.scene_understanding_publisher = self.create_publisher(
            String,
            '/scene_understanding',
            10
        )

        # Subscribers
        self.vision_subscriber = self.create_subscription(
            String,
            '/detected_objects',
            self.vision_callback,
            10
        )

        self.audio_subscriber = self.create_subscription(
            String,
            '/sound_events',
            self.audio_callback,
            10
        )

        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.tactile_subscriber = self.create_subscription(
            String,
            '/contact_force',
            self.tactile_callback,
            10
        )

        # Initialize
        self.bridge = CvBridge()
        self.sensory_buffer = defaultdict(list)  # Store recent sensory inputs
        self.fusion_lock = threading.Lock()
        self.fusion_window = 0.5  # 500ms fusion window

        # Fusion parameters
        self.confidence_weights = {
            'vision': 0.7,
            'audio': 0.3,
            'tactile': 0.9,
            'imu': 0.5
        }

        # Scene understanding state
        self.current_scene = {
            'objects': {},
            'events': [],
            'environment_state': 'normal',
            'timestamp': self.get_clock().now().nanoseconds
        }

        # Start fusion timer
        self.fusion_timer = self.create_timer(0.1, self.perform_fusion)

        self.get_logger().info('Multi-Modal Fusion Node initialized')

    def vision_callback(self, msg):
        """Process visual perception data"""
        try:
            data = json.loads(msg.data)
            input_data = SensoryInput(
                modality='vision',
                data=data,
                timestamp=self.get_clock().now().nanoseconds / 1e9,
                confidence=0.8
            )

            with self.fusion_lock:
                self.sensory_buffer['vision'].append(input_data)

            # Keep only recent data
            self.prune_old_data('vision')

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in vision message')

    def audio_callback(self, msg):
        """Process audio perception data"""
        try:
            data = json.loads(msg.data)
            input_data = SensoryInput(
                modality='audio',
                data=data,
                timestamp=self.get_clock().now().nanoseconds / 1e9,
                confidence=0.6
            )

            with self.fusion_lock:
                self.sensory_buffer['audio'].append(input_data)

            # Keep only recent data
            self.prune_old_data('audio')

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in audio message')

    def imu_callback(self, msg):
        """Process IMU data"""
        data = {
            'linear_acceleration': {
                'x': msg.linear_acceleration.x,
                'y': msg.linear_acceleration.y,
                'z': msg.linear_acceleration.z
            },
            'angular_velocity': {
                'x': msg.angular_velocity.x,
                'y': msg.angular_velocity.y,
                'z': msg.angular_velocity.z
            },
            'orientation': {
                'x': msg.orientation.x,
                'y': msg.orientation.y,
                'z': msg.orientation.z,
                'w': msg.orientation.w
            }
        }

        input_data = SensoryInput(
            modality='imu',
            data=data,
            timestamp=self.get_clock().now().nanoseconds / 1e9,
            confidence=0.9
        )

        with self.fusion_lock:
            self.sensory_buffer['imu'].append(input_data)

        # Keep only recent data
        self.prune_old_data('imu')

    def tactile_callback(self, msg):
        """Process tactile perception data"""
        try:
            data = json.loads(msg.data)
            input_data = SensoryInput(
                modality='tactile',
                data=data,
                timestamp=self.get_clock().now().nanoseconds / 1e9,
                confidence=0.95
            )

            with self.fusion_lock:
                self.sensory_buffer['tactile'].append(input_data)

            # Keep only recent data
            self.prune_old_data('tactile')

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in tactile message')

    def prune_old_data(self, modality):
        """Remove old data from buffer"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        threshold_time = current_time - self.fusion_window

        # Keep only recent data
        self.sensory_buffer[modality] = [
            item for item in self.sensory_buffer[modality]
            if item.timestamp > threshold_time
        ]

    def perform_fusion(self):
        """Perform multi-modal fusion"""
        with self.fusion_lock:
            # Get all recent sensory inputs
            all_inputs = []
            for modality, inputs in self.sensory_buffer.items():
                all_inputs.extend(inputs)

            if not all_inputs:
                return

            # Perform fusion based on temporal proximity and modality
            fused_result = self.fuse_sensory_inputs(all_inputs)

            if fused_result:
                # Publish fused perception
                fused_msg = String()
                fused_msg.data = json.dumps(fused_result)
                self.fused_perception_publisher.publish(fused_msg)

                # Update scene understanding
                self.update_scene_understanding(fused_result)

    def fuse_sensory_inputs(self, inputs: List[SensoryInput]) -> Optional[Dict]:
        """Fuse multiple sensory inputs into coherent understanding"""
        if not inputs:
            return None

        # Group inputs by temporal proximity (within fusion window)
        temporal_groups = self.group_by_temporal_proximity(inputs)

        fused_results = []

        for group in temporal_groups:
            # Group by modality within the temporal group
            modality_groups = defaultdict(list)
            for input_data in group:
                modality_groups[input_data.modality].append(input_data)

            # Fuse within each modality group first
            modality_fusions = {}
            for modality, modality_inputs in modality_groups.items():
                modality_fusions[modality] = self.fuse_modality_data(
                    modality, modality_inputs
                )

            # Then cross-modal fusion
            cross_modal_fusion = self.cross_modal_fusion(modality_fusions)

            if cross_modal_fusion:
                fused_results.append(cross_modal_fusion)

        if fused_results:
            # Combine all fused results
            final_fusion = self.combine_fused_results(fused_results)
            return final_fusion

        return None

    def group_by_temporal_proximity(self, inputs: List[SensoryInput]) -> List[List[SensoryInput]]:
        """Group inputs by temporal proximity"""
        if not inputs:
            return []

        # Sort by timestamp
        sorted_inputs = sorted(inputs, key=lambda x: x.timestamp)

        groups = []
        current_group = [sorted_inputs[0]]

        for input_data in sorted_inputs[1:]:
            time_diff = abs(input_data.timestamp - current_group[-1].timestamp)

            if time_diff <= self.fusion_window:
                current_group.append(input_data)
            else:
                groups.append(current_group)
                current_group = [input_data]

        if current_group:
            groups.append(current_group)

        return groups

    def fuse_modality_data(self, modality: str, inputs: List[SensoryInput]) -> Dict:
        """Fuse data within the same modality"""
        if not inputs:
            return {}

        if modality == 'vision':
            # For vision, combine object detections
            objects = {}
            for input_data in inputs:
                obj_data = input_data.data
                if isinstance(obj_data, dict) and 'objects' in obj_data:
                    for obj_id, obj_info in obj_data['objects'].items():
                        if obj_id not in objects:
                            objects[obj_id] = obj_info
                        else:
                            # Average position and update confidence
                            prev_pos = objects[obj_id].get('position', {})
                            new_pos = obj_info.get('position', {})
                            if prev_pos and new_pos:
                                avg_pos = {
                                    'x': (prev_pos.get('x', 0) + new_pos.get('x', 0)) / 2,
                                    'y': (prev_pos.get('y', 0) + new_pos.get('y', 0)) / 2,
                                    'z': (prev_pos.get('z', 0) + new_pos.get('z', 0)) / 2
                                }
                                objects[obj_id]['position'] = avg_pos

            return {'objects': objects, 'modality': modality}

        elif modality == 'audio':
            # For audio, combine sound events
            events = []
            for input_data in inputs:
                event_data = input_data.data
                if isinstance(event_data, dict):
                    event_data['confidence'] = input_data.confidence
                    events.append(event_data)

            return {'audio_events': events, 'modality': modality}

        elif modality == 'imu':
            # For IMU, average recent readings
            avg_linear_acc = np.array([0.0, 0.0, 0.0])
            avg_angular_vel = np.array([0.0, 0.0, 0.0])
            count = 0

            for input_data in inputs:
                data = input_data.data
                if 'linear_acceleration' in data and 'angular_velocity' in data:
                    acc = data['linear_acceleration']
                    gyro = data['angular_velocity']
                    avg_linear_acc += np.array([acc['x'], acc['y'], acc['z']])
                    avg_angular_vel += np.array([gyro['x'], gyro['y'], gyro['z']])
                    count += 1

            if count > 0:
                avg_linear_acc /= count
                avg_angular_vel /= count

            return {
                'modality': modality,
                'linear_acceleration': {
                    'x': float(avg_linear_acc[0]),
                    'y': float(avg_linear_acc[1]),
                    'z': float(avg_linear_acc[2])
                },
                'angular_velocity': {
                    'x': float(avg_angular_vel[0]),
                    'y': float(avg_angular_vel[1]),
                    'z': float(avg_angular_vel[2])
                }
            }

        elif modality == 'tactile':
            # For tactile, combine contact forces
            contacts = []
            for input_data in inputs:
                contacts.append(input_data.data)

            return {'tactile_contacts': contacts, 'modality': modality}

        return {}

    def cross_modal_fusion(self, modality_fusions: Dict[str, Dict]) -> Optional[Dict]:
        """Perform cross-modal fusion between different sensory modalities"""
        fusion_result = {
            'timestamp': self.get_clock().now().nanoseconds,
            'confidence': 0.0,
            'modalities_present': list(modality_fusions.keys())
        }

        # Example fusion rules:
        # 1. Audio + Vision: Sound source localization
        if 'audio' in modality_fusions and 'vision' in modality_fusions:
            audio_events = modality_fusions['audio'].get('audio_events', [])
            vision_objects = modality_fusions['vision'].get('objects', {})

            if audio_events and vision_objects:
                # Try to associate audio events with visual objects
                associated_events = self.associate_audio_visual(
                    audio_events, vision_objects
                )
                fusion_result['associated_events'] = associated_events

        # 2. Tactile + Vision: Object manipulation confirmation
        if 'tactile' in modality_fusions and 'vision' in modality_fusions:
            tactile_contacts = modality_fusions['tactile'].get('tactile_contacts', [])
            vision_objects = modality_fusions['vision'].get('objects', {})

            if tactile_contacts and vision_objects:
                # Confirm that visual objects are being contacted
                confirmed_objects = self.confirm_object_contact(
                    tactile_contacts, vision_objects
                )
                fusion_result['confirmed_contacts'] = confirmed_objects

        # 3. IMU + Vision: Motion-aware object tracking
        if 'imu' in modality_fusions and 'vision' in modality_fusions:
            imu_data = modality_fusions['imu']
            vision_objects = modality_fusions['vision'].get('objects', {})

            if imu_data and vision_objects:
                # Adjust object positions based on robot motion
                adjusted_objects = self.adjust_objects_for_motion(
                    vision_objects, imu_data
                )
                fusion_result['motion_adjusted_objects'] = adjusted_objects

        # Calculate overall confidence based on modalities present
        confidence = 0.0
        for modality in modality_fusions:
            weight = self.confidence_weights.get(modality, 0.5)
            confidence += weight

        fusion_result['confidence'] = min(1.0, confidence / len(modality_fusions))

        return fusion_result if any(fusion_result.values()) else None

    def associate_audio_visual(self, audio_events, vision_objects):
        """Associate audio events with visual objects"""
        associations = []

        for audio_event in audio_events:
            # In a real implementation, this would use direction of arrival
            # and visual object positions to make associations
            for obj_id, obj_info in vision_objects.items():
                # Simple association based on timing and general area
                association = {
                    'audio_event': audio_event,
                    'visual_object': obj_id,
                    'confidence': 0.7  # Placeholder confidence
                }
                associations.append(association)

        return associations

    def confirm_object_contact(self, tactile_contacts, vision_objects):
        """Confirm object contact using tactile and visual data"""
        confirmations = []

        for contact in tactile_contacts:
            # In a real implementation, this would match contact location
            # with visual object positions
            for obj_id, obj_info in vision_objects.items():
                confirmation = {
                    'object_id': obj_id,
                    'contact_data': contact,
                    'confidence': 0.9
                }
                confirmations.append(confirmation)

        return confirmations

    def adjust_objects_for_motion(self, vision_objects, imu_data):
        """Adjust object positions based on robot motion"""
        adjusted_objects = {}

        # Get motion from IMU
        linear_acc = imu_data.get('linear_acceleration', {})
        angular_vel = imu_data.get('angular_velocity', {})

        motion_vector = np.array([
            linear_acc.get('x', 0),
            linear_acc.get('y', 0),
            linear_acc.get('z', 0)
        ])

        for obj_id, obj_info in vision_objects.items():
            original_pos = obj_info.get('position', {})
            if original_pos:
                # Adjust position based on robot motion
                # (This is simplified - real implementation would use proper transforms)
                adjusted_pos = {
                    'x': original_pos.get('x', 0) - motion_vector[0] * 0.01,  # Scale factor
                    'y': original_pos.get('y', 0) - motion_vector[1] * 0.01,
                    'z': original_pos.get('z', 0) - motion_vector[2] * 0.01
                }
                obj_info['adjusted_position'] = adjusted_pos

            adjusted_objects[obj_id] = obj_info

        return adjusted_objects

    def combine_fused_results(self, fused_results: List[Dict]) -> Dict:
        """Combine multiple fused results"""
        combined = {
            'timestamp': self.get_clock().now().nanoseconds,
            'individual_fusions': fused_results,
            'overall_confidence': np.mean([r.get('confidence', 0.0) for r in fused_results]),
            'fusion_count': len(fused_results)
        }

        # Merge common fields
        all_modalities = set()
        all_associated_events = []
        all_confirmed_contacts = []

        for result in fused_results:
            all_modalities.update(result.get('modalities_present', []))
            all_associated_events.extend(result.get('associated_events', []))
            all_confirmed_contacts.extend(result.get('confirmed_contacts', []))

        combined['all_modalities'] = list(all_modalities)
        if all_associated_events:
            combined['associated_events'] = all_associated_events
        if all_confirmed_contacts:
            combined['confirmed_contacts'] = all_confirmed_contacts

        return combined

    def update_scene_understanding(self, fused_result):
        """Update scene understanding based on fused perception"""
        # Update current scene with new information
        self.current_scene['timestamp'] = fused_result.get('timestamp', self.current_scene['timestamp'])
        self.current_scene['fusion_confidence'] = fused_result.get('overall_confidence', 0.0)

        # Update objects if present in fusion
        if 'motion_adjusted_objects' in fused_result:
            self.current_scene['objects'] = fused_result['motion_adjusted_objects']
        elif 'individual_fusions' in fused_result:
            for fusion in fused_result['individual_fusions']:
                if 'objects' in fusion:
                    self.current_scene['objects'].update(fusion['objects'])

        # Add events
        if 'associated_events' in fused_result:
            self.current_scene['events'].extend(fused_result['associated_events'])

        # Publish scene understanding
        scene_msg = String()
        scene_msg.data = json.dumps(self.current_scene)
        self.scene_understanding_publisher.publish(scene_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MultiModalFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Multi-Modal Fusion Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Robustness and Error Handling

Implementing robust perception systems that handle sensor failures gracefully:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, Imu, JointState
import json
from enum import Enum
from typing import Dict, Optional
import time
from collections import defaultdict

class SensorStatus(Enum):
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"

class RobustPerceptionNode(Node):
    def __init__(self):
        super().__init__('robust_perception_node')

        # Publishers
        self.perception_publisher = self.create_publisher(
            String,
            '/robust_perception',
            10
        )

        self.diagnostic_publisher = self.create_publisher(
            String,
            '/perception_diagnostics',
            10
        )

        # Subscribers
        self.vision_subscriber = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.vision_callback,
            10
        )

        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Initialize sensor monitoring
        self.sensor_status = {
            'vision': SensorStatus.OPERATIONAL,
            'audio': SensorStatus.OPERATIONAL,
            'imu': SensorStatus.OPERATIONAL,
            'tactile': SensorStatus.OPERATIONAL
        }

        self.sensor_health_history = defaultdict(list)
        self.last_sensor_update = {}
        self.sensor_timeout_threshold = 5.0  # seconds

        # Recovery strategies
        self.recovery_strategies = {
            'vision': self.recover_vision,
            'audio': self.recover_audio,
            'imu': self.recover_imu,
            'tactile': self.recover_tactile
        }

        # Start health monitoring
        self.health_timer = self.create_timer(1.0, self.monitor_sensor_health)

        self.get_logger().info('Robust Perception Node initialized')

    def vision_callback(self, msg):
        """Process vision data and update sensor status"""
        try:
            # Process vision data (simplified)
            self.last_sensor_update['vision'] = time.time()

            # Check for common vision issues
            if self.detect_vision_issues(msg):
                self.update_sensor_status('vision', SensorStatus.DEGRADED)
            else:
                self.update_sensor_status('vision', SensorStatus.OPERATIONAL)

        except Exception as e:
            self.get_logger().error(f'Vision processing error: {e}')
            self.update_sensor_status('vision', SensorStatus.FAILED)

    def imu_callback(self, msg):
        """Process IMU data and update sensor status"""
        try:
            # Check for IMU issues (e.g., extreme values that indicate malfunction)
            linear_acc = msg.linear_acceleration
            angular_vel = msg.angular_velocity

            # Check for impossible values
            if (abs(linear_acc.x) > 100 or abs(linear_acc.y) > 100 or abs(linear_acc.z) > 100 or
                abs(angular_vel.x) > 100 or abs(angular_vel.y) > 100 or abs(angular_vel.z) > 100):
                self.update_sensor_status('imu', SensorStatus.FAILED)
            else:
                self.last_sensor_update['imu'] = time.time()
                self.update_sensor_status('imu', SensorStatus.OPERATIONAL)

        except Exception as e:
            self.get_logger().error(f'IMU processing error: {e}')
            self.update_sensor_status('imu', SensorStatus.FAILED)

    def detect_vision_issues(self, image_msg):
        """Detect common vision sensor issues"""
        # In a real implementation, this would analyze the image
        # for issues like blur, overexposure, etc.

        # Placeholder: assume issues if image timestamp is too old
        current_time = time.time()
        image_time = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec / 1e9

        if current_time - image_time > 1.0:  # More than 1 second old
            return True

        return False

    def update_sensor_status(self, sensor_name, new_status):
        """Update sensor status and trigger recovery if needed"""
        old_status = self.sensor_status[sensor_name]

        if old_status != new_status:
            self.get_logger().info(f'Sensor {sensor_name} status changed: {old_status.value} -> {new_status.value}')

            # Add to health history
            self.sensor_health_history[sensor_name].append({
                'timestamp': time.time(),
                'status': new_status.value,
                'duration': time.time() - self.last_sensor_update.get(sensor_name, time.time())
            })

            # Keep history manageable
            if len(self.sensor_health_history[sensor_name]) > 100:
                self.sensor_health_history[sensor_name] = self.sensor_health_history[sensor_name][-50:]

            # Trigger recovery if sensor failed
            if new_status == SensorStatus.FAILED:
                self.trigger_recovery(sensor_name)

            # Update status
            self.sensor_status[sensor_name] = new_status

    def monitor_sensor_health(self):
        """Monitor sensor health and detect timeouts"""
        current_time = time.time()

        for sensor_name in self.sensor_status:
            last_update = self.last_sensor_update.get(sensor_name, 0)

            if current_time - last_update > self.sensor_timeout_threshold:
                if self.sensor_status[sensor_name] != SensorStatus.FAILED:
                    self.update_sensor_status(sensor_name, SensorStatus.FAILED)

        # Publish diagnostic information
        self.publish_diagnostics()

    def trigger_recovery(self, sensor_name):
        """Trigger recovery procedure for failed sensor"""
        self.get_logger().info(f'Triggering recovery for {sensor_name}')

        # Mark as recovering
        self.update_sensor_status(sensor_name, SensorStatus.RECOVERING)

        # Execute recovery strategy
        recovery_func = self.recovery_strategies.get(sensor_name)
        if recovery_func:
            try:
                success = recovery_func()
                if success:
                    self.update_sensor_status(sensor_name, SensorStatus.OPERATIONAL)
                    self.get_logger().info(f'Recovery successful for {sensor_name}')
                else:
                    self.get_logger().warn(f'Recovery failed for {sensor_name}')
                    # Could implement escalation procedures here
            except Exception as e:
                self.get_logger().error(f'Recovery function error for {sensor_name}: {e}')
                self.update_sensor_status(sensor_name, SensorStatus.FAILED)

    def recover_vision(self):
        """Recovery strategy for vision sensor"""
        # In a real implementation, this might:
        # - Restart camera driver
        # - Adjust camera parameters
        # - Switch to backup camera
        # - Use alternative visual processing

        self.get_logger().info('Attempting vision sensor recovery')

        # Simulate recovery attempt
        time.sleep(0.5)  # Simulate recovery time

        # For demo purposes, assume recovery successful
        return True

    def recover_audio(self):
        """Recovery strategy for audio sensor"""
        self.get_logger().info('Attempting audio sensor recovery')

        # Simulate audio recovery
        time.sleep(0.2)

        return True

    def recover_imu(self):
        """Recovery strategy for IMU sensor"""
        self.get_logger().info('Attempting IMU sensor recovery')

        # Simulate IMU recovery (e.g., recalibration)
        time.sleep(0.3)

        return True

    def recover_tactile(self):
        """Recovery strategy for tactile sensors"""
        self.get_logger().info('Attempting tactile sensor recovery')

        # Simulate tactile sensor recovery
        time.sleep(0.1)

        return True

    def publish_diagnostics(self):
        """Publish sensor diagnostic information"""
        diagnostic_info = {
            'timestamp': time.time(),
            'sensor_status': {name: status.value for name, status in self.sensor_status.items()},
            'last_updates': self.last_sensor_update,
            'health_history': {
                name: history[-10:] for name, history in self.sensor_health_history.items()  # Last 10 events
            }
        }

        diag_msg = String()
        diag_msg.data = json.dumps(diagnostic_info)
        self.diagnostic_publisher.publish(diag_msg)

    def get_available_sensors(self):
        """Get list of currently available sensors"""
        available = []
        for sensor_name, status in self.sensor_status.items():
            if status in [SensorStatus.OPERATIONAL, SensorStatus.DEGRADED]:
                available.append(sensor_name)
        return available

    def is_perception_reliable(self):
        """Check if perception system is reliable enough for operation"""
        # Define minimum required sensors
        critical_sensors = ['imu']  # IMU is critical for safety
        required_operational = 1  # At least 1 critical sensor must be operational

        operational_count = 0
        for sensor in critical_sensors:
            if self.sensor_status.get(sensor) == SensorStatus.OPERATIONAL:
                operational_count += 1

        return operational_count >= required_operational

def main(args=None):
    rclpy.init(args=args)
    node = RobustPerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Robust Perception Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for Multi-Modal Perception

1. **Timing Alignment**: Ensure proper temporal synchronization between sensors
2. **Calibration**: Maintain accurate calibration between different sensor frames
3. **Redundancy**: Design with sensor redundancy for critical functions
4. **Real-time Performance**: Optimize algorithms for real-time operation
5. **Robustness**: Handle sensor failures and degraded performance gracefully
6. **Data Association**: Properly associate data from different modalities
7. **Uncertainty Management**: Track and propagate uncertainty through fusion
8. **Validation**: Continuously validate fused results against expectations

Multi-modal perception is fundamental to creating robust, intelligent robotic systems that can operate effectively in complex, real-world environments.