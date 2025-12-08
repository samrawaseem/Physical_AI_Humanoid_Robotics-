---
title: Sensor Simulation
sidebar_position: 4
---

# Module 2: The Digital Twin (Gazebo & Unity)

## Sensor Simulation in Robotics

Sensor simulation is a critical component of digital twin systems, enabling robots to perceive their environment in simulation. Accurate sensor simulation allows for testing perception algorithms, navigation systems, and control strategies before deployment on physical hardware.

### Types of Sensors in Robotics

Robotic systems typically use various types of sensors:

- **Vision Sensors**: Cameras, depth sensors, stereo cameras
- **Range Sensors**: LiDAR, sonar, infrared distance sensors
- **Inertial Sensors**: IMUs, accelerometers, gyroscopes
- **Force/Torque Sensors**: For manipulation and contact detection
- **Proprioceptive Sensors**: Joint encoders, motor current sensors

### Camera Simulation

Cameras provide visual information about the environment:

#### RGB Camera Configuration

```xml
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="head_camera">
      <horizontal_fov>1.089</horizontal_fov>  <!-- ~62.4 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/image_raw:=/camera/image_raw</remapping>
        <remapping>~/camera_info:=/camera/camera_info</remapping>
      </ros>
      <camera_name>camera</camera_name>
      <frame_name>camera_optical_frame</frame_name>
      <hack_baseline>0.07</hack_baseline>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
    </plugin>
  </sensor>
</gazebo>
```

#### Depth Camera Configuration

```xml
<gazebo reference="depth_camera_link">
  <sensor name="depth_camera" type="depth">
    <update_rate>30</update_rate>
    <camera name="depth_head_camera">
      <horizontal_fov>1.089</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10</far>
      </clip>
    </camera>
    <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/image_raw:=/camera/depth/image_raw</remapping>
        <remapping>~/camera_info:=/camera/depth/camera_info</remapping>
        <remapping>~/depth/image_raw:=/camera/depth/depth_image</remapping>
      </ros>
      <camera_name>depth_camera</camera_name>
      <frame_name>depth_camera_optical_frame</frame_name>
      <baseline>0.2</baseline>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
    </plugin>
  </sensor>
</gazebo>
```

#### Stereo Camera Configuration

```xml
<gazebo reference="stereo_camera_left">
  <sensor name="stereo_left" type="camera">
    <update_rate>30</update_rate>
    <camera name="left_cam">
      <horizontal_fov>1.089</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="stereo_left_camera" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>/my_robot/stereo</namespace>
        <remapping>~/image_raw:=/left/image_raw</remapping>
        <remapping>~/camera_info:=/left/camera_info</remapping>
      </ros>
      <camera_name>stereo_left</camera_name>
      <frame_name>stereo_left_optical_frame</frame_name>
    </plugin>
  </sensor>
</gazebo>

<gazebo reference="stereo_camera_right">
  <sensor name="stereo_right" type="camera">
    <update_rate>30</update_rate>
    <camera name="right_cam">
      <horizontal_fov>1.089</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="stereo_right_camera" filename="libgazebo_ros_camera.so">
      <ros>
        <namespace>/my_robot/stereo</namespace>
        <remapping>~/image_raw:=/right/image_raw</remapping>
        <remapping>~/camera_info:=/right/camera_info</remapping>
      </ros>
      <camera_name>stereo_right</camera_name>
      <frame_name>stereo_right_optical_frame</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR Simulation

LiDAR sensors provide 3D point cloud data for mapping and navigation:

#### 2D LiDAR Configuration

```xml
<gazebo reference="lidar_link">
  <sensor name="laser" type="gpu_lidar">
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="laser_controller" filename="libgazebo_ros_gpu_laser.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/out:=/scan</remapping>
      </ros>
      <frame_name>laser_frame</frame_name>
      <topic_name>scan</topic_name>
    </plugin>
  </sensor>
</gazebo>
```

#### 3D LiDAR Configuration (Velodyne-style)

```xml
<gazebo reference="velodyne_link">
  <sensor name="velodyne" type="gpu_lidar">
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>800</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
        <vertical>
          <samples>32</samples>
          <resolution>1</resolution>
          <min_angle>-0.5236</min_angle>  <!-- -30 degrees -->
          <max_angle>0.2618</max_angle>    <!-- 15 degrees -->
        </vertical>
      </scan>
      <range>
        <min>0.1</min>
        <max>100.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="velodyne_controller" filename="libgazebo_ros_velodyne_gpu_lidar.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/out:=/velodyne_points</remapping>
      </ros>
      <frame_name>velodyne_frame</frame_name>
      <topic_name>velodyne_points</topic_name>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Simulation

IMUs (Inertial Measurement Units) provide acceleration, angular velocity, and orientation data:

```xml
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/out:=/imu</remapping>
      </ros>
      <frame_name>imu_link</frame_name>
      <body_name>imu_link</body_name>
      <update_rate>100</update_rate>
    </plugin>
  </sensor>
</gazebo>
```

### Force/Torque Sensor Simulation

Force/torque sensors are crucial for manipulation tasks:

```xml
<gazebo reference="wrist_link">
  <sensor name="ft_sensor" type="force_torque">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <force_torque>
      <frame>sensor</frame>
      <measure_direction>child_to_parent</measure_direction>
    </force_torque>
    <plugin name="ft_sensor_plugin" filename="libgazebo_ros_ft_sensor.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/wrench:=/wrist_ft_sensor</remapping>
      </ros>
      <frame_name>wrist_link</frame_name>
      <topic_name>wrist_ft_sensor</topic_name>
    </plugin>
  </sensor>
</gazebo>
```

### GPS Simulation

For outdoor robots, GPS simulation can be useful:

```xml
<gazebo reference="gps_link">
  <sensor name="gps_sensor" type="gps">
    <always_on>true</always_on>
    <update_rate>1</update_rate>
    <visualize>false</visualize>
    <plugin name="gps_plugin" filename="libgazebo_ros_gps.so">
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/out:=/gps/fix</remapping>
      </ros>
      <frame_name>gps_link</frame_name>
      <topic_name>gps/fix</topic_name>
      <update_rate>1</update_rate>
    </plugin>
  </sensor>
</gazebo>
```

### Sensor Integration with ROS 2

Sensors in Gazebo publish data to ROS 2 topics. Here's how to work with sensor data:

#### Reading Camera Data

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/my_robot/camera/image_raw',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Process the image (example: display it)
        cv2.imshow('Camera View', cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = CameraSubscriber()

    try:
        rclpy.spin(camera_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        camera_subscriber.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Reading LiDAR Data

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math

class LidarSubscriber(Node):
    def __init__(self):
        super().__init__('lidar_subscriber')
        self.subscription = self.create_subscription(
            LaserScan,
            '/my_robot/scan',
            self.scan_callback,
            10
        )

    def scan_callback(self, msg):
        # Process LiDAR data
        ranges = msg.ranges
        min_distance = min(ranges) if ranges else float('inf')

        self.get_logger().info(f'Min distance: {min_distance:.2f}m')

        # Example: Find obstacles in front of robot
        front_range = ranges[len(ranges)//2]  # Front reading
        if front_range < 1.0:  # Obstacle within 1 meter
            self.get_logger().warn('Obstacle detected in front!')

def main(args=None):
    rclpy.init(args=args)
    lidar_subscriber = LidarSubscriber()

    try:
        rclpy.spin(lidar_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        lidar_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Sensor Fusion

Combine data from multiple sensors for better perception:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from geometry_msgs.msg import Vector3
import numpy as np

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Subscribe to multiple sensors
        self.imu_subscription = self.create_subscription(
            Imu, '/my_robot/imu', self.imu_callback, 10
        )

        self.lidar_subscription = self.create_subscription(
            LaserScan, '/my_robot/scan', self.lidar_callback, 10
        )

        # Publisher for fused data
        self.fused_data_publisher = self.create_publisher(
            Vector3, '/my_robot/environment_state', 10
        )

        self.latest_imu = None
        self.latest_lidar = None

    def imu_callback(self, msg):
        self.latest_imu = msg
        self.fuse_sensors()

    def lidar_callback(self, msg):
        self.latest_lidar = msg
        self.fuse_sensors()

    def fuse_sensors(self):
        if self.latest_imu and self.latest_lidar:
            # Example fusion: combine orientation and obstacle detection
            roll, pitch, yaw = self.quaternion_to_euler(
                self.latest_imu.orientation
            )

            # Find minimum distance from LiDAR
            min_distance = min(self.latest_lidar.ranges) if self.latest_lidar.ranges else float('inf')

            # Publish fused state
            fused_msg = Vector3()
            fused_msg.x = roll  # Orientation
            fused_msg.y = pitch
            fused_msg.z = min_distance  # Obstacle distance

            self.fused_data_publisher.publish(fused_msg)

    def quaternion_to_euler(self, quaternion):
        # Convert quaternion to Euler angles
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp) if abs(sinp) <= 1 else math.copysign(math.pi/2, sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)
    sensor_fusion = SensorFusionNode()

    try:
        rclpy.spin(sensor_fusion)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_fusion.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Sensor Calibration and Validation

#### Camera Calibration

Camera simulation should include realistic distortion parameters:

```xml
<plugin name="camera_controller" filename="libgazebo_ros_camera.so">
  <!-- ... other parameters ... -->
  <distortion_k1>-0.1253</distortion_k1>
  <distortion_k2>0.0943</distortion_k2>
  <distortion_k3>-0.0043</distortion_k3>
  <distortion_t1>-0.0008</distortion_t1>
  <distortion_t2>0.0000</distortion_t2>
</plugin>
```

#### Noise Modeling

Real sensors have noise characteristics that should be modeled:

```xml
<noise>
  <type>gaussian</type>
  <mean>0.0</mean>
  <stddev>0.01</stddev>  <!-- Adjust based on real sensor specs -->
  <bias_mean>0.0</bias_mean>
  <bias_stddev>0.001</bias_stddev>
</noise>
```

### Performance Considerations

#### Sensor Update Rates

Balance accuracy with performance:

- **Cameras**: 10-30 Hz (high computational cost)
- **LiDAR**: 5-20 Hz (depends on resolution)
- **IMU**: 100-1000 Hz (low computational cost)
- **GPS**: 1-10 Hz (low frequency updates)

#### GPU vs CPU Sensors

- **GPU sensors**: Faster rendering, higher frame rates, but require GPU
- **CPU sensors**: More compatible, but slower

```xml
<!-- GPU LiDAR -->
<sensor name="lidar" type="gpu_lidar">
  <!-- ... configuration ... -->
</sensor>

<!-- CPU LiDAR -->
<sensor name="lidar" type="ray">
  <!-- ... configuration ... -->
</sensor>
```

### Best Practices

1. **Use realistic noise models** based on actual sensor specifications
2. **Match update rates** to real hardware capabilities
3. **Validate sensor data** by comparing with real sensor output
4. **Optimize for performance** by reducing unnecessary sensor complexity
5. **Test sensor fusion algorithms** with simulated data before real hardware
6. **Consider sensor limitations** like field of view, range, and resolution

Sensor simulation is essential for developing robust perception and navigation systems. Properly configured sensors enable thorough testing of robotic algorithms in a safe, controlled environment before deployment on physical hardware.