# Troubleshooting Guide

## Introduction

This troubleshooting guide provides solutions for common issues encountered when working with ROS 2, Gazebo, NVIDIA Isaac, and the autonomous humanoid robot system. The guide is organized by system components and includes diagnostic procedures, common error messages, and step-by-step resolution procedures.

## ROS 2 Troubleshooting

### Common ROS 2 Issues and Solutions

#### 1. ROS 2 Nodes Not Communicating

**Symptoms:**
- Nodes cannot see each other
- Topics/services are not being published/subscribed
- `ros2 topic list` shows no topics

**Diagnosis:**
```bash
# Check if ROS 2 domain ID is consistent
echo $ROS_DOMAIN_ID

# Check network configuration
ip addr show
```

**Solutions:**
```bash
# Reset ROS 2 domain (default is 0)
export ROS_DOMAIN_ID=0

# Check if daemon is running
ros2 daemon status

# Restart daemon if needed
ros2 daemon stop
ros2 daemon start

# Verify network settings
export ROS_LOCALHOST_ONLY=0  # For multi-machine communication
```

#### 2. Permission Denied Errors

**Symptoms:**
- Error: "Permission denied" when running ROS 2 commands
- Cannot access ROS 2 directories

**Solutions:**
```bash
# Check ROS 2 installation permissions
ls -la /opt/ros/humble/

# Fix permissions if needed
sudo chown -R $USER:$USER /opt/ros/humble/lib/python3.*/site-packages/
```

#### 3. Package Not Found Errors

**Symptoms:**
- Error: "Package 'package_name' not found"
- `ament_cmake` errors during build

**Solutions:**
```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Check if package is installed
apt list --installed | grep ros-humble-package-name

# Install missing package
sudo apt install ros-humble-package-name

# In workspace, check if package is built
cd ~/robotics_ws
colcon build --packages-select package_name
source install/setup.bash
```

### ROS 2 Performance Issues

#### 1. High CPU Usage

**Diagnosis:**
```bash
# Monitor ROS 2 processes
htop
# Look for rclcpp/rclpy processes with high CPU

# Check topic rates
ros2 topic hz /topic_name

# Monitor node activity
ros2 node info node_name
```

**Solutions:**
```bash
# Reduce publishing rates in your nodes
# Example: change timer period from 0.01 to 0.1 seconds

# Use Quality of Service (QoS) settings appropriately
# Reduce reliability requirements if not critical
```

#### 2. Memory Leaks

**Diagnosis:**
```bash
# Monitor memory usage
watch -n 1 'free -h'

# Check specific ROS 2 processes
ps aux --sort=-%mem | grep ros
```

**Solutions:**
```python
# Properly destroy nodes and publishers
def destroy_node(self):
    self.publisher_.destroy()
    self.timer.destroy()
    super().destroy_node()
```

## Gazebo Troubleshooting

### Common Gazebo Issues

#### 1. Gazebo Won't Launch

**Symptoms:**
- Gazebo crashes immediately on startup
- OpenGL errors in console
- Segmentation fault

**Diagnosis:**
```bash
# Check OpenGL support
glxinfo | grep -i opengl
glxinfo | grep -i "direct rendering"

# Check graphics drivers
nvidia-smi  # For NVIDIA
```

**Solutions:**
```bash
# Install graphics libraries
sudo apt install mesa-utils libgl1-mesa-glx libgl1-mesa-dri

# For NVIDIA systems, ensure proper drivers
sudo apt install nvidia-driver-535 nvidia-prime

# Launch with software rendering (slower but more compatible)
export LIBGL_ALWAYS_SOFTWARE=1
gz sim
```

#### 2. Models Not Loading in Gazebo

**Symptoms:**
- Robot models appear as boxes or don't appear
- Error: "Model not found" or "URDF parsing error"

**Solutions:**
```bash
# Check Gazebo model paths
echo $GAZEBO_MODEL_PATH

# Add custom model paths
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/robotics_ws/src/my_robot/models

# Verify URDF files
check_urdf /path/to/robot.urdf

# Launch with verbose output
gz sim -v 4
```

#### 3. Physics Simulation Issues

**Symptoms:**
- Robot falls through ground
- Unstable joint movements
- Objects pass through each other

**Solutions:**
```xml
<!-- In URDF, ensure proper collision and visual elements -->
<link name="link_name">
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </collision>
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </visual>
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
  </inertial>
</link>
```

### Gazebo Performance Optimization

#### 1. Slow Simulation Speed

**Diagnosis:**
```bash
# Check real-time factor
gz topic -e -t /statistics

# Monitor system resources
htop
nvidia-smi  # For GPU usage
```

**Solutions:**
```bash
# Reduce physics update rate in world file
# <physics><max_step_size>0.001</max_step_size></physics>

# Simplify collision geometries
# Use boxes instead of meshes for collision elements

# Reduce number of objects in simulation
```

## NVIDIA Isaac Troubleshooting

### Isaac Sim Common Issues

#### 1. Isaac Sim Won't Launch

**Symptoms:**
- Isaac Sim crashes on startup
- CUDA-related errors
- "Failed to initialize Omniverse" errors

**Diagnosis:**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Check Isaac Sim logs
cat ~/.nvidia-omniverse/logs/isaac-sim/*.log
```

**Solutions:**
```bash
# Ensure compatible CUDA version
# Isaac Sim typically requires CUDA 11.8

# Install required NVIDIA drivers
sudo apt install nvidia-driver-535 nvidia-utils-535

# Check system requirements
# Ensure sufficient VRAM (8GB+ recommended)
```

#### 2. Isaac ROS Bridge Issues

**Symptoms:**
- ROS 2 topics not appearing in Isaac Sim
- Isaac Sim topics not available in ROS 2
- Bridge node crashes

**Solutions:**
```bash
# Check Isaac ROS bridge installation
ros2 pkg list | grep isaac_ros

# Launch bridge with proper configuration
ros2 launch isaac_ros_bridge isaac_ros_bridge.launch.py

# Verify ROS 2 environment in Isaac Sim
source /opt/ros/humble/setup.bash
```

### Isaac Performance Issues

#### 1. High GPU Memory Usage

**Diagnosis:**
```bash
# Monitor GPU memory
nvidia-smi -l 1

# Check Isaac Sim settings
# In Isaac Sim UI: Window > Compute Graph > Stats
```

**Solutions:**
```bash
# Reduce rendering quality in Isaac Sim
# Go to Window > Rendering > Render Settings
# Lower resolution, disable MSAA, reduce ray tracing

# Limit physics complexity
# Simplify collision meshes
# Reduce number of rigid bodies
```

## Vision and Perception Troubleshooting

### Camera and Sensor Issues

#### 1. Camera Not Publishing Data

**Symptoms:**
- Camera topics show no data
- Image viewers remain black
- Sensor data streams are empty

**Diagnosis:**
```bash
# Check camera topics
ros2 topic list | grep camera
ros2 topic echo /camera/image_raw

# Check camera node status
ros2 node list | grep camera
ros2 lifecycle list camera_node_name
```

**Solutions:**
```bash
# Check camera permissions (Linux)
sudo chmod 666 /dev/video0

# Verify camera configuration
# Check camera intrinsics and extrinsics
# Ensure proper calibration files exist

# Test camera directly
v4l2-ctl --list-devices
```

#### 2. OpenCV Integration Issues

**Symptoms:**
- cv2 bridge errors
- Image format conversion failures
- Segmentation faults with image processing

**Solutions:**
```python
# Proper cv2 bridge usage
from cv_bridge import CvBridge

def image_callback(self, msg):
    try:
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        self.get_logger().error(f'cv2 conversion error: {e}')
        return
```

## Audio and Voice Processing Troubleshooting

### Microphone and Audio Issues

#### 1. Audio Input Not Working

**Symptoms:**
- Voice commands not detected
- Microphone shows no input
- Audio processing nodes fail

**Diagnosis:**
```bash
# Check audio devices
arecord -l
pactl list sources short

# Test audio input
arecord -D hw:0,0 -f cd test.wav
```

**Solutions:**
```bash
# Set default audio device
pactl set-default-source alsa_input.pci-0000_00_1f.3.analog-stereo

# Check audio permissions
sudo usermod -a -G audio $USER

# Test with simple recording
python3 -c "import speech_recognition as sr; r=sr.Recognizer(); m=sr.Microphone(); print(r.listen(m, timeout=5))"
```

#### 2. Speech Recognition Failures

**Solutions:**
```python
# Robust speech recognition setup
import speech_recognition as sr

def recognize_speech_with_fallback():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        # Adjust for ambient noise
        r.adjust_for_ambient_noise(source, duration=1)

        try:
            # Listen with timeout
            audio = r.listen(source, timeout=5, phrase_time_limit=10)

            # Try multiple recognition services
            try:
                text = r.recognize_google(audio)
            except sr.UnknownValueError:
                text = r.recognize_sphinx(audio)  # Fallback to offline recognition

        except sr.WaitTimeoutError:
            print("No speech detected")
            return None

    return text
```

## Network and Communication Troubleshooting

### Multi-Machine Communication Issues

#### 1. ROS 2 Multi-Machine Setup

**Symptoms:**
- Nodes on different machines cannot communicate
- Topics appear on one machine but not another
- High latency in communication

**Diagnosis:**
```bash
# Check network connectivity
ping other_machine_ip

# Check ROS 2 configuration
echo $ROS_DOMAIN_ID
echo $ROS_LOCALHOST_ONLY
echo $RMW_IMPLEMENTATION

# Monitor network traffic
netstat -tulpn | grep 11711  # Default DDS port
```

**Solutions:**
```bash
# Ensure same domain ID on all machines
export ROS_DOMAIN_ID=0

# Configure firewall
sudo ufw allow 11711:11720/udp
sudo ufw allow 11711:11720/tcp

# Set proper ROS environment
export ROS_LOCALHOST_ONLY=0
export ROS_IP=your_machine_ip
```

## System Integration Troubleshooting

### Complex System Issues

#### 1. System Integration Failures

**Symptoms:**
- Multiple nodes fail simultaneously
- System crashes during complex tasks
- Resource exhaustion (CPU, memory, GPU)

**Diagnosis:**
```bash
# Monitor system resources
htop
iotop
nvidia-smi

# Check system logs
journalctl -f
dmesg | tail -20

# Monitor ROS 2 system
ros2 doctor
```

**Solutions:**
```bash
# Implement resource monitoring in your nodes
import psutil
import os

def check_system_resources():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage('/').percent

    if cpu_percent > 90 or memory_percent > 90:
        print("Warning: High resource usage detected")
        # Implement resource management strategies
```

#### 2. Real-time Performance Issues

**Solutions:**
```bash
# Configure real-time settings (for critical applications)
# Add to /etc/security/limits.conf:
# * - rtprio 99
# * - memlock unlimited

# Run critical nodes with real-time priority
chrt -f 99 ros2 run package_name node_name
```

## Debugging Tools and Techniques

### ROS 2 Debugging

#### 1. Using rqt Tools

```bash
# Install rqt tools
sudo apt install ros-humble-rqt ros-humble-rqt-common-plugins

# Launch various debug tools
rqt_graph  # Visualize node connections
rqt_plot   # Plot numeric values
rqt_console # View log messages
```

#### 2. Logging and Monitoring

```python
# Comprehensive logging in ROS 2 nodes
import rclpy
from rclpy.node import Node

class DebugNode(Node):
    def __init__(self):
        super().__init__('debug_node')

        # Set log level
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)

        # Log different severity levels
        self.get_logger().debug('Debug message')
        self.get_logger().info('Info message')
        self.get_logger().warn('Warning message')
        self.get_logger().error('Error message')
        self.get_logger().fatal('Fatal message')
```

### Performance Profiling

#### 1. CPU Profiling

```bash
# Profile ROS 2 nodes
py-spy record -o profile.svg --pid <process_id>

# System-wide profiling
perf record -g -p <process_id>
perf report
```

#### 2. Memory Profiling

```python
# Memory profiling in Python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Your code here
    large_list = [i for i in range(1000000)]
    return large_list
```

## Recovery Procedures

### System Recovery

#### 1. Emergency Stop Procedures

```python
# Implement emergency stop in all nodes
def emergency_stop_callback(self, msg):
    if msg.data:  # Emergency stop activated
        # Stop all motors
        self.stop_all_motors()

        # Cancel all goals
        self.cancel_all_goals()

        # Log emergency event
        self.get_logger().fatal('EMERGENCY STOP ACTIVATED')

        # Shutdown safely
        rclpy.shutdown()
```

#### 2. Safe Shutdown Procedures

```bash
# Graceful shutdown of ROS 2 system
# First, send shutdown signal to all nodes
ros2 lifecycle set node_name shutdown

# Then, kill any remaining processes
pkill -f ros2
pkill -f gz
pkill -f python
```

## Quick Reference Commands

### Common Troubleshooting Commands

```bash
# ROS 2 diagnostics
ros2 doctor                    # Check ROS 2 installation
ros2 lifecycle list node_name  # Check node state
ros2 param list node_name      # Check node parameters

# System monitoring
htop                           # CPU/Memory monitoring
iotop                          # I/O monitoring
nvidia-smi                     # GPU monitoring
netstat -tulpn                 # Network connections

# Gazebo debugging
gz topic -l                    # List Gazebo topics
gz service -l                  # List Gazebo services
gz sim -v 4                    # Verbose Gazebo startup

# File permissions
sudo chown -R $USER:$USER ~/robotics_ws
chmod +x ~/robotics_ws/install/setup.bash
```

## Getting Help

When encountering issues not covered in this guide:

1. **Check ROS 2 Documentation**: https://docs.ros.org
2. **Search ROS Answers**: https://answers.ros.org
3. **Check GitHub Issues**: For Isaac Sim, Gazebo, and other tools
4. **Course Forums**: Post specific questions about course content
5. **System Logs**: Always check `/var/log/` and `~/.ros/log/` for detailed error information

Remember to document any unique solutions you discover, as they may help other students facing similar issues.