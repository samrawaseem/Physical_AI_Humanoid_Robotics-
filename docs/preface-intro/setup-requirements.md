---
title: Setup Requirements
sidebar_position: 3
---

# Module 0: Preface & Introduction

## Software Installation Requirements

This section outlines the complete software setup required for the Physical AI & Humanoid Robotics course. Follow these steps carefully to ensure all components work together seamlessly.

### Prerequisites

Before beginning the installation process, ensure your system meets the minimum requirements:

- **Operating System**: Ubuntu 22.04 LTS (recommended) or Windows 10/11 with WSL2
- **Python Version**: Python 3.8 or higher
- **Disk Space**: At least 50GB of free space for all components
- **Internet Connection**: Stable connection for downloading large packages

### ROS 2 Humble Hawksbill Installation

ROS 2 (Robot Operating System 2) serves as the communication backbone for all robotic applications in this course.

```bash
# Setup locale
sudo locale-gen en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Add ROS 2 apt repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# Initialize rosdep
sudo rosdep init
rosdep update

# Source ROS 2 environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Gazebo Garden Installation

Gazebo provides the physics simulation environment for testing robotic algorithms.

```bash
# Install Gazebo Garden
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-gazebo-plugins

# Verify installation
gz sim --version
```

### NVIDIA Isaac Sim Setup

NVIDIA Isaac Sim provides high-fidelity simulation and synthetic data generation capabilities.

1. **NVIDIA Developer Account**: Register at [NVIDIA Developer Portal](https://developer.nvidia.com) to access Isaac Sim
2. **System Requirements**: Ensure you have a compatible NVIDIA GPU (RTX series recommended)
3. **Download Isaac Sim**: Download from the NVIDIA Developer website
4. **Installation**: Follow the installation guide for your operating system
5. **GPU Drivers**: Ensure you have the latest NVIDIA drivers installed (520+ recommended)

```bash
# Verify GPU compatibility
nvidia-smi

# Check Isaac Sim installation (if installed)
isaac-sim --version
```

### Python Environment Setup

Create a dedicated Python environment for the course:

```bash
# Install Python virtual environment
sudo apt install python3-venv

# Create virtual environment
python3 -m venv ~/physical_ai_env
source ~/physical_ai_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install essential packages
pip install numpy scipy matplotlib pandas jupyter

# Install OpenAI API client for LLM integration
pip install openai

# Install speech recognition packages
pip install SpeechRecognition pyaudio

# Install ROS 2 Python client library
pip install rclpy
```

### Unity Installation (Optional)

For enhanced visualization and human-robot interaction:

1. **Unity Hub**: Download and install Unity Hub from [Unity's website](https://unity.com/)
2. **Unity Version**: Install Unity 2022.3 LTS or later
3. **Unity Robotics Hub**: Install the Unity Robotics Hub package for ROS 2 integration
4. **ROS TCP Connector**: Install the ROS TCP Connector package for communication

### Development Tools

Install essential development tools for the course:

```bash
# Install Git for version control
sudo apt install git

# Install VS Code or preferred IDE
sudo snap install code --classic

# Install additional development tools
sudo apt install build-essential cmake pkg-config
sudo apt install libusb-1.0-0-dev libtbb-dev libblas-dev liblapack-dev
```

### Environment Configuration

Create a workspace for the course projects:

```bash
# Create ROS 2 workspace
mkdir -p ~/physical_ai_ws/src
cd ~/physical_ai_ws

# Source ROS 2 and build workspace
source /opt/ros/humble/setup.bash
colcon build

# Add workspace to bashrc
echo "source ~/physical_ai_ws/install/setup.bash" >> ~/.bashrc
```

### Verification Steps

After completing the installation, verify all components work together:

```bash
# Verify ROS 2 installation
ros2 --version

# Verify Python environment
python3 -c "import rclpy; print('rclpy imported successfully')"

# Test basic ROS 2 functionality
ros2 topic list

# Verify Gazebo
gz sim --version
```

### Troubleshooting Common Issues

**Issue**: ROS 2 packages not found
- **Solution**: Ensure you've sourced the ROS 2 environment: `source /opt/ros/humble/setup.bash`

**Issue**: Gazebo fails to start
- **Solution**: Check GPU drivers and ensure you have proper graphics support

**Issue**: Isaac Sim installation fails
- **Solution**: Verify GPU compatibility and ensure you have sufficient disk space

**Issue**: Python packages fail to install
- **Solution**: Ensure your virtual environment is activated and pip is updated

### Next Steps

Once your environment is properly set up, proceed to the next module where you'll learn about ROS 2 fundamentals and create your first ROS 2 nodes. The setup completed here will be used throughout the entire course.