# Software Installation Guide

## Introduction

This appendix provides comprehensive instructions for setting up the software environment required for the Physical AI & Humanoid Robotics course. The installation process includes ROS 2, Gazebo, NVIDIA Isaac, Unity, and all necessary dependencies. Following these instructions will ensure a consistent and functional development environment across all students.

## System Requirements

Before beginning the installation, ensure your system meets the following requirements:

### Minimum System Requirements:
- **Operating System**: Ubuntu 22.04 LTS (recommended) or Windows 10/11 with WSL2
- **Processor**: Intel i7 or AMD Ryzen 7 (8+ cores recommended)
- **Memory**: 16GB RAM (32GB recommended)
- **Storage**: 100GB free space (SSD recommended)
- **Graphics**: Dedicated GPU with CUDA support (NVIDIA RTX 2060 or better)

### Recommended System Requirements:
- **Operating System**: Ubuntu 22.04 LTS
- **Processor**: Intel i9 or AMD Ryzen 9 (12+ cores)
- **Memory**: 32GB RAM
- **Storage**: 500GB NVMe SSD
- **Graphics**: NVIDIA RTX 3080 or better

## Ubuntu Installation (Recommended)

### 1. Install Ubuntu 22.04 LTS

If not already installed, download Ubuntu 22.04 LTS from https://ubuntu.com/download/desktop and install it on your system or dual-boot setup.

### 2. Update System Packages

Open a terminal and run the following commands:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt autoremove -y
```

### 3. Install ROS 2 Humble Hawksbill

#### Set up locale
```bash
locale  # check for UTF-8
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

#### Set up sources
```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
```

#### Add ROS 2 GPG key
```bash
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
```

#### Add ROS 2 repository
```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

#### Install ROS 2 packages
```bash
sudo apt update
sudo apt install ros-humble-desktop-full
sudo apt install python3-colcon-common-extensions
```

#### Setup environment
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 4. Install Gazebo Garden

#### Add Gazebo repository
```bash
curl -sSL http://get.gazebosim.org | sh
```

#### Install Gazebo Garden
```bash
sudo apt install gz-garden
```

### 5. Install NVIDIA Isaac Sim

#### Prerequisites
```bash
sudo apt install python3-dev python3-pip python3-venv
pip3 install --upgrade pip
```

#### Install Isaac Sim dependencies
```bash
sudo apt install libglib2.0-dev libgstreamer-plugins-base1.0-dev libopencv-dev libeigen3-dev
sudo apt install nvidia-prime nvidia-utils-535 nvidia-driver-535
```

#### Install Isaac ROS packages
```bash
sudo apt install ros-humble-isaac-ros-dev
sudo apt install ros-humble-isaac-ros-perceptor
sudo apt install ros-humble-isaac-ros-visual- slam
```

### 6. Install Unity Hub and Robotics Package

#### Install Unity Hub
```bash
# Download Unity Hub from https://unity.com/download
# Or install via Snap:
sudo snap install unity-hub
```

#### Install Unity version for robotics
- Launch Unity Hub
- Sign in with Unity account
- Install Unity 2022.3 LTS
- Install the Robotics package from the Package Manager

### 7. Install Python Dependencies

#### Create Python virtual environment
```bash
python3 -m venv ~/robotics_env
source ~/robotics_env/bin/activate
```

#### Install required Python packages
```bash
pip install numpy scipy matplotlib pandas
pip install openai whisper speech-recognition pyaudio
pip install opencv-python transforms3d
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 8. Install Additional Tools

#### Git and version control
```bash
sudo apt install git git-lfs
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

#### Development tools
```bash
sudo apt install build-essential cmake pkg-config
sudo apt install python3-dev python3-pip python3-setuptools
sudo apt install terminator htop iotop
```

## Windows Installation with WSL2 (Alternative)

### 1. Install WSL2 with Ubuntu 22.04

Open PowerShell as Administrator and run:
```powershell
wsl --install -d Ubuntu-22.04
```

### 2. Configure WSL2 for GUI applications

Install VcXsrv or X410 X-server for Windows:
```bash
# In WSL2 terminal:
export DISPLAY=:0
export LIBGL_ALWAYS_INDIRECT=1
```

### 3. Follow Ubuntu Installation Steps

Proceed with the Ubuntu installation steps from section 3 onwards.

## NVIDIA Jetson Setup

### 1. Prepare Jetson Device

For Jetson Nano, Xavier NX, or Orin:
- Flash the device with JetPack SDK
- Connect to internet
- Update the system

### 2. Install ROS 2 on Jetson

```bash
# Add ROS 2 repository
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository universe

# Install ROS 2
sudo apt update
sudo apt install ros-humble-ros-base
sudo apt install python3-colcon-common-extensions

# Setup environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

### 3. Install Isaac ROS on Jetson

```bash
# Install Isaac ROS packages
sudo apt install ros-humble-isaac-ros-dev
sudo apt install ros-humble-isaac-ros-perceptor
```

## Environment Verification

### 1. Test ROS 2 Installation

```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Test basic ROS 2 functionality
ros2 topic list
ros2 node list
```

### 2. Test Gazebo Installation

```bash
gz sim
```

### 3. Test Python Environment

```bash
# Activate virtual environment
source ~/robotics_env/bin/activate

# Test Python packages
python3 -c "import rclpy; print('ROS 2 Python client OK')"
python3 -c "import cv2; print('OpenCV OK')"
python3 -c "import torch; print('PyTorch OK')"
```

## Troubleshooting Common Issues

### 1. ROS 2 Installation Issues

If you encounter issues with ROS 2 installation:

```bash
# Clean package cache
sudo apt clean
sudo apt autoclean

# Remove and reinstall ROS 2 keys
sudo rm /usr/share/keyrings/ros-archive-keyring.gpg
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

# Update and reinstall
sudo apt update
sudo apt install ros-humble-desktop-full
```

### 2. Gazebo Not Launching

If Gazebo fails to launch:

```bash
# Check OpenGL support
glxinfo | grep "OpenGL version"

# Install additional graphics libraries
sudo apt install mesa-utils libgl1-mesa-glx libgl1-mesa-dri
```

### 3. Python Package Issues

If Python packages fail to install:

```bash
# Upgrade pip
pip3 install --upgrade pip setuptools wheel

# Install packages individually if needed
pip3 install numpy
pip3 install scipy
pip3 install matplotlib
```

## Post-Installation Verification

Create a test workspace to verify all components work together:

```bash
# Create workspace
mkdir -p ~/robotics_ws/src
cd ~/robotics_ws

# Build workspace
colcon build

# Source the workspace
source install/setup.bash

# Test with a simple publisher
ros2 run demo_nodes_cpp talker
```

## Optional: Docker Setup for Isolated Development

For advanced users, Docker can provide isolated development environments:

```bash
# Install Docker
sudo apt install docker.io
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-v2

# Create Docker group and restart
newgrp docker
```

## Next Steps

After successful installation:

1. **Configure your development environment** with the appropriate IDE (VS Code with ROS extensions recommended)
2. **Set up your first ROS 2 workspace** following the tutorials
3. **Test the basic ROS 2 concepts** with simple publisher/subscriber examples
4. **Explore Gazebo simulation** with basic robot models
5. **Begin Module 1** of the course curriculum

## References

- [ROS 2 Humble Installation Guide](https://docs.ros.org/en/humble/Installation.html)
- [Gazebo Installation Guide](https://gazebosim.org/docs/garden/install)
- [NVIDIA Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/released/index.html)
- [Ubuntu Installation Guide](https://ubuntu.com/tutorials/install-ubuntu-desktop)

Remember to restart your system after completing the installation to ensure all services start correctly. If you encounter issues not covered in the troubleshooting section, consult the course forums or contact the teaching staff for assistance.