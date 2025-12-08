---
title: Sim-to-Real Transfer
sidebar_position: 5
---

# Module 3: The AI-Robot Brain (NVIDIA Isaac™)

## Sim-to-Real Transfer for Robotics

Sim-to-Real transfer is the process of taking models, algorithms, and behaviors developed in simulation and successfully deploying them on physical robots. This is one of the key challenges in robotics, as the reality gap between simulated and real environments can significantly impact performance.

### Understanding the Reality Gap

The reality gap consists of differences between simulation and reality that can affect robot performance:

- **Visual Differences**: Lighting, textures, colors, and visual artifacts
- **Physics Differences**: Friction, mass, inertia, and contact dynamics
- **Sensor Differences**: Noise, latency, and accuracy variations
- **Actuator Differences**: Response time, precision, and power limitations
- **Environmental Differences**: Unmodeled objects, dynamics, and disturbances

### Domain Randomization

Domain randomization is a technique to make models robust to domain shift by training in diverse simulated environments:

```python
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import random

class DomainRandomizationNode(Node):
    def __init__(self):
        super().__init__('domain_randomization_node')

        # Randomization parameters
        self.lighting_params = {
            'brightness_range': (0.5, 1.5),
            'contrast_range': (0.8, 1.2),
            'saturation_range': (0.8, 1.2),
            'hue_range': (-10, 10)
        }

        self.texture_params = {
            'texture_library': [
                'wood', 'metal', 'fabric', 'concrete', 'plastic'
            ],
            'roughness_range': (0.1, 0.9),
            'specular_range': (0.1, 0.9)
        }

        self.physics_params = {
            'friction_range': (0.1, 0.8),
            'restitution_range': (0.0, 0.3),
            'mass_variance': 0.1  # 10% variance
        }

    def randomize_lighting(self, image):
        """Apply random lighting conditions to image"""
        # Random brightness
        brightness_factor = random.uniform(
            self.lighting_params['brightness_range'][0],
            self.lighting_params['brightness_range'][1]
        )
        image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

        # Random contrast
        contrast_factor = random.uniform(
            self.lighting_params['contrast_range'][0],
            self.lighting_params['contrast_range'][1]
        )
        image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=128*(1-contrast_factor))

        # Random saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        saturation_factor = random.uniform(
            self.lighting_params['saturation_range'][0],
            self.lighting_params['saturation_range'][1]
        )
        hsv[:,:,1] *= saturation_factor
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return image

    def randomize_textures(self, scene_description):
        """Randomize textures in the scene"""
        randomized_scene = scene_description.copy()

        # Randomly assign textures to objects
        for obj_name, obj_props in randomized_scene.items():
            if 'material' in obj_props:
                new_texture = random.choice(self.texture_params['texture_library'])
                obj_props['material']['type'] = new_texture
                obj_props['material']['roughness'] = random.uniform(
                    self.texture_params['roughness_range'][0],
                    self.texture_params['roughness_range'][1]
                )
                obj_props['material']['specular'] = random.uniform(
                    self.texture_params['specular_range'][0],
                    self.texture_params['specular_range'][1]
                )

        return randomized_scene

    def randomize_physics(self, robot_properties):
        """Randomize physics properties"""
        randomized_robot = robot_properties.copy()

        # Randomize friction coefficients
        for joint_name, joint_props in randomized_robot.get('joints', {}).items():
            joint_props['friction'] = random.uniform(
                self.physics_params['friction_range'][0],
                self.physics_params['friction_range'][1]
            )
            joint_props['restitution'] = random.uniform(
                self.physics_params['restitution_range'][0],
                self.physics_params['restitution_range'][1]
            )

        # Randomize masses with variance
        for link_name, link_props in randomized_robot.get('links', {}).items():
            original_mass = link_props.get('mass', 1.0)
            mass_variance = self.physics_params['mass_variance']
            variance_factor = random.uniform(1 - mass_variance, 1 + mass_variance)
            link_props['mass'] = original_mass * variance_factor

        return randomized_robot

    def generate_randomized_training_data(self, base_scene, base_robot, num_variants=1000):
        """Generate training data with domain randomization"""
        training_data = []

        for i in range(num_variants):
            # Randomize scene
            randomized_scene = self.randomize_textures(base_scene)

            # Randomize robot physics
            randomized_robot = self.randomize_physics(base_robot)

            # Simulate and collect data
            simulation_result = self.run_simulation(
                randomized_scene,
                randomized_robot
            )

            training_data.append({
                'scene': randomized_scene,
                'robot': randomized_robot,
                'observations': simulation_result['observations'],
                'actions': simulation_result['actions'],
                'rewards': simulation_result['rewards']
            })

        return training_data

    def run_simulation(self, scene, robot):
        """Run simulation with given scene and robot configuration"""
        # This would interface with Isaac Sim or other simulator
        # For demonstration, return mock data
        return {
            'observations': np.random.random(10).astype(np.float32),
            'actions': np.random.random(6).astype(np.float32),
            'rewards': np.random.random(1).astype(np.float32)[0]
        }

def main(args=None):
    rclpy.init(args=args)
    dr_node = DomainRandomizationNode()

    # Example usage
    base_scene = {
        'table': {'material': {'type': 'wood', 'roughness': 0.5, 'specular': 0.3}},
        'floor': {'material': {'type': 'concrete', 'roughness': 0.2, 'specular': 0.1}}
    }

    base_robot = {
        'links': {
            'base': {'mass': 5.0},
            'arm_link': {'mass': 1.0}
        },
        'joints': {
            'arm_joint': {'friction': 0.1, 'restitution': 0.05}
        }
    }

    training_data = dr_node.generate_randomized_training_data(
        base_scene, base_robot, num_variants=10
    )

    print(f"Generated {len(training_data)} training samples with domain randomization")

    dr_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Synthetic Data Generation

Synthetic data generation creates diverse training datasets in simulation:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
import os

@dataclass
class SyntheticDataSample:
    rgb_image: np.ndarray
    depth_image: np.ndarray
    segmentation: np.ndarray
    camera_pose: np.ndarray
    object_poses: Dict[str, np.ndarray]
    timestamp: float

class SyntheticDataGenerator(Node):
    def __init__(self):
        super().__init__('synthetic_data_generator')

        # Publishers for synthetic data
        self.rgb_publisher = self.create_publisher(
            Image, '/synthetic_rgb', 10
        )
        self.depth_publisher = self.create_publisher(
            Image, '/synthetic_depth', 10
        )
        self.seg_publisher = self.create_publisher(
            Image, '/synthetic_segmentation', 10
        )

        self.bridge = CvBridge()

        # Object database for synthetic generation
        self.object_library = [
            {'name': 'cube', 'size': (0.1, 0.1, 0.1), 'color': (255, 0, 0)},
            {'name': 'sphere', 'size': (0.08, 0.08, 0.08), 'color': (0, 255, 0)},
            {'name': 'cylinder', 'size': (0.05, 0.1, 0.05), 'color': (0, 0, 255)},
            {'name': 'cone', 'size': (0.06, 0.12, 0.06), 'color': (255, 255, 0)},
        ]

        # Camera parameters
        self.camera_intrinsics = {
            'fx': 554.0, 'fy': 554.0,
            'cx': 320.0, 'cy': 240.0,
            'width': 640, 'height': 480
        }

        # Data storage
        self.data_samples = []
        self.save_directory = '/tmp/synthetic_data'

    def generate_scene(self, num_objects=5):
        """Generate a random scene with objects"""
        scene = {
            'objects': [],
            'camera_pose': self.random_camera_pose(),
            'lighting': self.random_lighting()
        }

        # Add random objects to scene
        for i in range(num_objects):
            obj = random.choice(self.object_library)
            obj_instance = {
                'name': f"{obj['name']}_{i}",
                'type': obj['name'],
                'size': obj['size'],
                'color': obj['color'],
                'position': self.random_position(),
                'orientation': self.random_orientation()
            }
            scene['objects'].append(obj_instance)

        return scene

    def random_position(self):
        """Generate random position within workspace"""
        return [
            random.uniform(-1.0, 1.0),  # x
            random.uniform(-1.0, 1.0),  # y
            random.uniform(0.1, 2.0)    # z
        ]

    def random_orientation(self):
        """Generate random orientation (Euler angles)"""
        return [
            random.uniform(0, 2 * np.pi),  # roll
            random.uniform(0, 2 * np.pi),  # pitch
            random.uniform(0, 2 * np.pi)   # yaw
        ]

    def random_camera_pose(self):
        """Generate random camera pose"""
        return {
            'position': [
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
                random.uniform(0.5, 1.5)
            ],
            'orientation': self.random_orientation()
        }

    def random_lighting(self):
        """Generate random lighting conditions"""
        return {
            'intensity': random.uniform(500, 2000),
            'color': (random.random(), random.random(), random.random()),
            'direction': [
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 1)
            ]
        }

    def render_scene(self, scene):
        """Render scene to generate synthetic data"""
        width, height = self.camera_intrinsics['width'], self.camera_intrinsics['height']

        # Create blank images
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        depth_image = np.ones((height, width), dtype=np.float32) * 10.0  # Initialize with far distance
        segmentation = np.zeros((height, width), dtype=np.uint8)

        # Render each object
        for obj_idx, obj in enumerate(scene['objects']):
            # For simplicity, we'll draw basic shapes
            # In real implementation, this would use Isaac Sim rendering
            center_x = int((obj['position'][0] + 1) * width / 2)
            center_y = int((obj['position'][1] + 1) * height / 2)
            depth = obj['position'][2]

            if 0 <= center_x < width and 0 <= center_y < height:
                # Draw object based on type
                if obj['type'] == 'cube':
                    size_px = int(obj['size'][0] * width / 2)
                    cv2.rectangle(
                        rgb_image,
                        (center_x - size_px//2, center_y - size_px//2),
                        (center_x + size_px//2, center_y + size_px//2),
                        obj['color'], -1
                    )
                    cv2.rectangle(
                        segmentation,
                        (center_x - size_px//2, center_y - size_px//2),
                        (center_x + size_px//2, center_y + size_px//2),
                        obj_idx + 1, -1
                    )
                elif obj['type'] == 'sphere':
                    radius_px = int(obj['size'][0] * width / 2)
                    cv2.circle(
                        rgb_image,
                        (center_x, center_y),
                        radius_px, obj['color'], -1
                    )
                    cv2.circle(
                        segmentation,
                        (center_x, center_y),
                        radius_px, obj_idx + 1, -1
                    )

                # Update depth image
                if obj['type'] in ['cube', 'sphere']:
                    mask = segmentation == (obj_idx + 1)
                    depth_image[mask] = depth

        return rgb_image, depth_image, segmentation

    def generate_synthetic_dataset(self, num_samples=1000):
        """Generate synthetic dataset"""
        self.get_logger().info(f'Generating {num_samples} synthetic data samples')

        for i in range(num_samples):
            # Generate random scene
            scene = self.generate_scene(num_objects=random.randint(3, 8))

            # Render scene
            rgb, depth, seg = self.render_scene(scene)

            # Create synthetic data sample
            sample = SyntheticDataSample(
                rgb_image=rgb,
                depth_image=depth,
                segmentation=seg,
                camera_pose=np.array(scene['camera_pose']['position'] + scene['camera_pose']['orientation']),
                object_poses={obj['name']: np.array(obj['position'] + obj['orientation']) for obj in scene['objects']},
                timestamp=self.get_clock().now().nanoseconds
            )

            self.data_samples.append(sample)

            # Publish sample
            self.publish_sample(sample)

            if i % 100 == 0:
                self.get_logger().info(f'Generated {i}/{num_samples} samples')

        self.get_logger().info(f'Completed generating {len(self.data_samples)} synthetic samples')

    def publish_sample(self, sample):
        """Publish synthetic data sample"""
        # Publish RGB image
        rgb_msg = self.bridge.cv2_to_imgmsg(sample.rgb_image, encoding='bgr8')
        rgb_msg.header.stamp = self.get_clock().now().to_msg()
        rgb_msg.header.frame_id = 'camera_rgb_optical_frame'
        self.rgb_publisher.publish(rgb_msg)

        # Publish depth image
        depth_msg = self.bridge.cv2_to_imgmsg(sample.depth_image, encoding='32FC1')
        depth_msg.header.stamp = rgb_msg.header.stamp
        depth_msg.header.frame_id = 'camera_depth_optical_frame'
        self.depth_publisher.publish(depth_msg)

        # Publish segmentation
        seg_msg = self.bridge.cv2_to_imgmsg(sample.segmentation, encoding='mono8')
        seg_msg.header.stamp = rgb_msg.header.stamp
        seg_msg.header.frame_id = 'camera_rgb_optical_frame'
        self.seg_publisher.publish(seg_msg)

    def save_dataset(self, save_path=None):
        """Save generated dataset to disk"""
        if save_path is None:
            save_path = self.save_directory

        os.makedirs(save_path, exist_ok=True)

        for i, sample in enumerate(self.data_samples):
            # Save RGB image
            cv2.imwrite(f'{save_path}/rgb_{i:06d}.png', sample.rgb_image)

            # Save depth image
            cv2.imwrite(f'{save_path}/depth_{i:06d}.png',
                       (sample.depth_image * 1000).astype(np.uint16))  # Scale for 16-bit storage

            # Save segmentation
            cv2.imwrite(f'{save_path}/seg_{i:06d}.png', sample.segmentation)

            # Save metadata
            metadata = {
                'camera_pose': sample.camera_pose.tolist(),
                'object_poses': {k: v.tolist() for k, v in sample.object_poses.items()},
                'timestamp': sample.timestamp
            }

            import json
            with open(f'{save_path}/metadata_{i:06d}.json', 'w') as f:
                json.dump(metadata, f)

        self.get_logger().info(f'Saved dataset to {save_path}')

def main(args=None):
    rclpy.init(args=args)
    gen_node = SyntheticDataGenerator()

    # Generate synthetic dataset
    gen_node.generate_synthetic_dataset(num_samples=50)  # Smaller number for demo

    # Save dataset
    gen_node.save_dataset()

    gen_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Model Adaptation Techniques

#### Domain Adaptation with Neural Networks

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

class DomainAdaptationNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super(DomainAdaptationNetwork, self).__init__()

        # Feature extractor (shared between domains)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Label classifier (task-specific)
        self.label_classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Domain classifier (domain-specific)
        self.domain_classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # 2 domains: sim and real
        )

    def forward(self, x, alpha=0.0):
        # Extract features
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)

        # Reverse gradient for domain classification (for training)
        reverse_features = self.gradient_reverse_layer(features, alpha)

        # Get predictions
        label_pred = self.label_classifier(features)
        domain_pred = self.domain_classifier(reverse_features)

        return label_pred, domain_pred

    def gradient_reverse_layer(self, x, alpha):
        """Gradient reversal layer for domain adaptation"""
        return GradientReverseFunction.apply(x, alpha)

class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        return input

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DomainAdaptationNode(Node):
    def __init__(self):
        super().__init__('domain_adaptation_node')

        # Create subscriber for camera images
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        # Publisher for adapted predictions
        self.prediction_publisher = self.create_publisher(
            Float32MultiArray,
            '/adapted_predictions',
            10
        )

        self.bridge = CvBridge()

        # Initialize domain adaptation network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DomainAdaptationNetwork(num_classes=5).to(self.device)

        # Transformation for input images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Training state
        self.is_training = False
        self.sim_data_buffer = []
        self.real_data_buffer = []

    def image_callback(self, msg):
        """Process incoming images"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Preprocess image
            input_tensor = self.transform(cv_image).unsqueeze(0).to(self.device)

            if self.is_training:
                # During training, we might want to collect real data
                self.collect_real_data(input_tensor)
            else:
                # Inference mode - get adapted prediction
                with torch.no_grad():
                    alpha = 0.0  # No gradient reversal during inference
                    label_pred, domain_pred = self.model(input_tensor, alpha)

                    # Publish prediction
                    pred_msg = Float32MultiArray()
                    pred_msg.data = label_pred.cpu().numpy()[0].tolist()
                    self.prediction_publisher.publish(pred_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def collect_real_data(self, image_tensor):
        """Collect real data for adaptation"""
        self.real_data_buffer.append(image_tensor)

        # Keep buffer size manageable
        if len(self.real_data_buffer) > 100:
            self.real_data_buffer.pop(0)

    def train_adaptation(self, sim_loader, num_epochs=10):
        """Train domain adaptation model"""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            total_loss = 0.0
            total_domain_loss = 0.0

            for batch_idx, (sim_data, sim_labels) in enumerate(sim_loader):
                sim_data, sim_labels = sim_data.to(self.device), sim_labels.to(self.device)

                # Prepare real data batch (if available)
                if self.real_data_buffer:
                    real_batch_size = min(len(self.real_data_buffer), sim_data.size(0))
                    real_data = torch.cat(self.real_data_buffer[:real_batch_size], dim=0)
                else:
                    # If no real data, create dummy batch with different style
                    real_data = sim_data + 0.1 * torch.randn_like(sim_data)

                # Combine sim and real data
                combined_data = torch.cat([sim_data, real_data], dim=0)

                # Create domain labels (0 for sim, 1 for real)
                domain_labels = torch.cat([
                    torch.zeros(sim_data.size(0), dtype=torch.long),
                    torch.ones(real_data.size(0), dtype=torch.long)
                ]).to(self.device)

                # Set up alpha for gradient reversal (increase during training)
                p = float(batch_idx + epoch * len(sim_loader)) / (num_epochs * len(sim_loader))
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                # Forward pass
                label_pred, domain_pred = self.model(combined_data, alpha)

                # Calculate losses
                sim_label_pred = label_pred[:sim_data.size(0)]
                label_loss = criterion(sim_label_pred, sim_labels)

                domain_loss = domain_criterion(domain_pred, domain_labels)

                # Total loss
                total_loss_batch = label_loss + domain_loss
                total_loss += total_loss_batch.item()
                total_domain_loss += domain_loss.item()

                # Backward pass
                optimizer.zero_grad()
                total_loss_batch.backward()
                optimizer.step()

            self.get_logger().info(
                f'Epoch {epoch+1}/{num_epochs}, '
                f'Label Loss: {total_loss/len(sim_loader):.4f}, '
                f'Domain Loss: {total_domain_loss/len(sim_loader):.4f}'
            )

    def adapt_to_real(self):
        """Adapt model to real domain"""
        if not self.real_data_buffer:
            self.get_logger().warn('No real data available for adaptation')
            return

        # Create real data loader
        real_tensor = torch.cat(self.real_data_buffer, dim=0)
        real_dataset = TensorDataset(real_tensor)
        real_loader = DataLoader(real_dataset, batch_size=8, shuffle=True)

        # Switch to training mode
        self.is_training = True

        # Perform adaptation (in practice, this would involve more sophisticated techniques)
        self.get_logger().info('Starting domain adaptation to real data...')

        # Fine-tune with real data
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        for epoch in range(5):  # Few epochs for fine-tuning
            for real_batch in real_loader:
                real_data = real_batch[0].to(self.device)

                # Forward pass
                label_pred, domain_pred = self.model(real_data, alpha=0.0)

                # In real adaptation, we might not have labels
                # So we could use self-supervised or unsupervised techniques
                # For demonstration, we'll just minimize domain classifier output for real data
                real_domain_pred = domain_pred[len(real_data)//2:]  # Assuming half are real
                adaptation_loss = F.cross_entropy(
                    real_domain_pred,
                    torch.zeros(real_domain_pred.size(0), dtype=torch.long).to(self.device)
                )

                optimizer.zero_grad()
                adaptation_loss.backward()
                optimizer.step()

        self.is_training = False
        self.get_logger().info('Domain adaptation completed')

def main(args=None):
    rclpy.init(args=args)
    da_node = DomainAdaptationNode()

    # Example: adapt to real domain after collecting some data
    def adapt_callback():
        da_node.adapt_to_real()

    # Adapt after 10 seconds
    timer = da_node.create_timer(10.0, adapt_callback)

    try:
        rclpy.spin(da_node)
    except KeyboardInterrupt:
        pass
    finally:
        da_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Jetson Edge Deployment

#### Optimizing Models for Edge Deployment

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import numpy as np
import torch
import torch.nn as nn
import torch_tensorrt
from torchvision import transforms
import time

class EdgeOptimizedModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # Optimized architecture for edge devices
        self.backbone = nn.Sequential(
            # First conv block - keep channels low for edge
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # Stride 2 for downsampling
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # Another stride 2
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((2, 2))  # Global average pooling
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256 * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class JetsonDeploymentNode(Node):
    def __init__(self):
        super().__init__('jetson_deployment_node')

        # Create subscriber
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.image_callback,
            10
        )

        # Publisher for predictions
        self.prediction_publisher = self.create_publisher(
            Float32MultiArray,
            '/edge_predictions',
            10
        )

        self.bridge = CvBridge()

        # Check for Jetson-specific optimizations
        self.is_jetson = self.detect_jetson()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = EdgeOptimizedModel(num_classes=5)

        # Load pre-trained weights (sim-to-real adapted model)
        # self.model.load_state_dict(torch.load('adapted_model.pth'))

        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Optimize model for Jetson if available
        if self.is_jetson:
            self.optimize_for_jetson()

        # Performance monitoring
        self.inference_times = []
        self.frame_count = 0

        # Transformation for input images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),  # Smaller size for edge efficiency
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def detect_jetson(self):
        """Detect if running on Jetson platform"""
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip().decode('utf-8')
                if 'Jetson' in model:
                    self.get_logger().info(f'Detected Jetson platform: {model}')
                    return True
        except:
            pass

        # Alternative detection method
        import platform
        if 'aarch64' in platform.machine():
            self.get_logger().info('Detected ARM64 architecture (likely Jetson)')
            return True

        return False

    def optimize_for_jetson(self):
        """Apply Jetson-specific optimizations"""
        self.get_logger().info('Applying Jetson-specific optimizations...')

        # Convert to TensorRT if available
        if hasattr(torch_tensorrt, 'compile'):
            try:
                # Create example input for TensorRT
                example_input = torch.randn(1, 3, 64, 64).to(self.device)

                # Compile with TensorRT
                self.model = torch_tensorrt.compile(
                    self.model,
                    inputs=[example_input],
                    enabled_precisions={torch.float}
                )
                self.get_logger().info('Model compiled with TensorRT')
            except Exception as e:
                self.get_logger().warn(f'TensorRT compilation failed: {e}')

        # Apply quantization if available
        try:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
            )
            self.get_logger().info('Model quantized for edge deployment')
        except Exception as e:
            self.get_logger().warn(f'Quantization failed: {e}')

    def image_callback(self, msg):
        """Process incoming images on Jetson edge device"""
        start_time = time.time()

        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Preprocess image
            input_tensor = self.transform(cv_image).unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                if self.is_jetson and hasattr(torch, 'inference_mode'):
                    with torch.inference_mode():
                        predictions = self.model(input_tensor)
                else:
                    predictions = self.model(input_tensor)

                # Apply softmax to get probabilities
                probabilities = torch.softmax(predictions, dim=1)

                # Get top prediction
                top_prob, top_class = torch.max(probabilities, dim=1)

                # Publish results
                pred_msg = Float32MultiArray()
                pred_msg.data = probabilities.cpu().numpy()[0].tolist()
                self.prediction_publisher.publish(pred_msg)

                # Performance monitoring
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                self.frame_count += 1

                if self.frame_count % 30 == 0:  # Log every 30 frames
                    avg_time = np.mean(self.inference_times[-30:])
                    fps = 1.0 / avg_time if avg_time > 0 else 0
                    self.get_logger().info(
                        f'Jetson inference: {avg_time*1000:.1f}ms ({fps:.1f} FPS)'
                    )

        except Exception as e:
            self.get_logger().error(f'Error in Jetson inference: {e}')

    def get_model_summary(self):
        """Get model information for deployment"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Approximate size
            'is_quantized': hasattr(self.model, 'activation_post_process'),
            'platform': 'Jetson' if self.is_jetson else 'Generic'
        }

        return info

def main(args=None):
    rclpy.init(args=args)
    jetson_node = JetsonDeploymentNode()

    # Print model summary
    summary = jetson_node.get_model_summary()
    jetson_node.get_logger().info(f'Model Summary: {summary}')

    try:
        rclpy.spin(jetson_node)
    except KeyboardInterrupt:
        pass
    finally:
        jetson_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Validation and Testing

#### Sim-to-Real Performance Validation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from collections import deque
import statistics

class SimRealValidationNode(Node):
    def __init__(self):
        super().__init__('sim_real_validation')

        # Subscribers for both sim and real data
        self.sim_pose_subscriber = self.create_subscription(
            PoseStamped,
            '/sim_robot_pose',
            self.sim_pose_callback,
            10
        )

        self.real_pose_subscriber = self.create_subscription(
            PoseStamped,
            '/real_robot_pose',
            self.real_pose_callback,
            10
        )

        self.sim_image_subscriber = self.create_subscription(
            Image,
            '/sim_camera/image_raw',
            self.sim_image_callback,
            10
        )

        self.real_image_subscriber = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.real_image_callback,
            10
        )

        # Publishers for validation metrics
        self.pose_error_publisher = self.create_publisher(
            Float32,
            '/pose_error',
            10
        )

        self.image_similarity_publisher = self.create_publisher(
            Float32,
            '/image_similarity',
            10
        )

        self.validation_status_publisher = self.create_publisher(
            Bool,
            '/validation_status',
            10
        )

        self.bridge = CvBridge()

        # Storage for pose comparison
        self.sim_poses = deque(maxlen=100)
        self.real_poses = deque(maxlen=100)
        self.pose_timestamps = deque(maxlen=100)

        # Storage for image comparison
        self.sim_images = deque(maxlen=10)
        self.real_images = deque(maxlen=10)

        # Validation thresholds
        self.position_threshold = 0.1  # 10cm tolerance
        self.orientation_threshold = 0.2  # 0.2 rad tolerance
        self.image_similarity_threshold = 0.7  # 70% similarity

        # Performance tracking
        self.validation_results = {
            'pose_errors': [],
            'image_similarities': [],
            'success_rate': 0.0
        }

    def sim_pose_callback(self, msg):
        """Store simulated pose"""
        self.sim_poses.append({
            'position': np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            'orientation': np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                   msg.pose.orientation.z, msg.pose.orientation.w]),
            'timestamp': msg.header.stamp.nanosec
        })

    def real_pose_callback(self, msg):
        """Store real pose and validate"""
        real_pose = {
            'position': np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            'orientation': np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                                   msg.pose.orientation.z, msg.pose.orientation.w]),
            'timestamp': msg.header.stamp.nanosec
        }
        self.real_poses.append(real_pose)

        # Match with closest sim pose in time
        if len(self.sim_poses) > 0:
            # Find closest sim pose by timestamp
            time_diffs = [abs(real_pose['timestamp'] - sim_pose['timestamp'])
                         for sim_pose in self.sim_poses]
            closest_idx = np.argmin(time_diffs)
            sim_pose = self.sim_poses[closest_idx]

            # Calculate pose error
            pos_error = euclidean(real_pose['position'], sim_pose['position'])
            orient_error = self.quaternion_distance(
                real_pose['orientation'], sim_pose['orientation']
            )

            # Publish error metrics
            pos_error_msg = Float32()
            pos_error_msg.data = pos_error
            self.pose_error_publisher.publish(pos_error_msg)

            # Store for statistics
            self.validation_results['pose_errors'].append(pos_error)

            # Check if validation passes
            is_valid = (pos_error < self.position_threshold and
                       orient_error < self.orientation_threshold)

            status_msg = Bool()
            status_msg.data = is_valid
            self.validation_status_publisher.publish(status_msg)

            self.get_logger().info(
                f'Pose validation - Position error: {pos_error:.3f}m, '
                f'Orientation error: {orient_error:.3f}rad, '
                f'Valid: {is_valid}'
            )

    def quaternion_distance(self, q1, q2):
        """Calculate angular distance between two quaternions"""
        # Convert to rotation vectors for comparison
        from scipy.spatial.transform import Rotation as R
        r1 = R.from_quat(q1[[1, 2, 3, 0]])  # Convert from xyzw to wxyz
        r2 = R.from_quat(q2[[1, 2, 3, 0]])  # Convert from xyzw to wxyz

        # Calculate relative rotation
        relative_r = r1.inv() * r2
        angle = relative_r.magnitude()  # This gives the rotation angle

        return angle

    def sim_image_callback(self, msg):
        """Store simulated image"""
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.sim_images.append(cv_image)

    def real_image_callback(self, msg):
        """Store real image and validate"""
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.real_images.append(cv_image)

        # Compare with stored sim images
        if len(self.sim_images) > 0:
            sim_img = self.sim_images[-1]  # Most recent sim image

            # Calculate structural similarity (SSIM) or other metrics
            similarity = self.calculate_image_similarity(sim_img, cv_image)

            # Publish similarity
            sim_msg = Float32()
            sim_msg.data = similarity
            self.image_similarity_publisher.publish(sim_msg)

            # Store for statistics
            self.validation_results['image_similarities'].append(similarity)

            # Log result
            self.get_logger().info(f'Image similarity: {similarity:.3f}')

    def calculate_image_similarity(self, img1, img2):
        """Calculate similarity between two images"""
        # Resize images to same size if different
        if img1.shape != img2.shape:
            min_h, min_w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (min_w, min_h))
            img2 = cv2.resize(img2, (min_w, min_h))

        # Convert to grayscale for comparison
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Calculate normalized cross-correlation
        correlation = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]

        # Ensure correlation is between 0 and 1
        correlation = max(0, min(1, (correlation + 1) / 2))

        return correlation

    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        report = {
            'total_pose_comparisons': len(self.validation_results['pose_errors']),
            'total_image_comparisons': len(self.validation_results['image_similarities'])
        }

        if self.validation_results['pose_errors']:
            report['avg_position_error'] = statistics.mean(self.validation_results['pose_errors'])
            report['max_position_error'] = max(self.validation_results['pose_errors'])
            report['pose_accuracy'] = sum(
                1 for err in self.validation_results['pose_errors']
                if err < self.position_threshold
            ) / len(self.validation_results['pose_errors'])

        if self.validation_results['image_similarities']:
            report['avg_image_similarity'] = statistics.mean(self.validation_results['image_similarities'])
            report['image_quality_score'] = sum(
                1 for sim in self.validation_results['image_similarities']
                if sim > self.image_similarity_threshold
            ) / len(self.validation_results['image_similarities'])

        return report

    def print_validation_summary(self):
        """Print validation summary to console"""
        report = self.generate_validation_report()

        self.get_logger().info('=== Sim-to-Real Validation Summary ===')
        for key, value in report.items():
            self.get_logger().info(f'{key}: {value}')
        self.get_logger().info('=====================================')

def main(args=None):
    rclpy.init(args=args)
    validation_node = SimRealValidationNode()

    # Print summary every 30 seconds
    def print_summary():
        validation_node.print_validation_summary()

    summary_timer = validation_node.create_timer(30.0, print_summary)

    try:
        rclpy.spin(validation_node)
    except KeyboardInterrupt:
        validation_node.print_validation_summary()
    finally:
        validation_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for Sim-to-Real Transfer

1. **Progressive Domain Adaptation**: Start with small domain shifts and gradually increase complexity
2. **Systematic Randomization**: Randomize all possible parameters that might differ between sim and real
3. **Validation Pipeline**: Implement comprehensive validation to measure sim-to-real gap
4. **Model Complexity**: Balance model performance with edge deployment requirements
5. **Sensor Modeling**: Accurately model sensor noise and limitations in simulation
6. **Physics Tuning**: Carefully tune physics parameters to match real-world behavior
7. **Incremental Deployment**: Deploy and test components incrementally rather than all at once

Sim-to-Real transfer is a critical capability for deploying simulation-trained models on physical robots, requiring careful attention to domain differences and systematic validation approaches.