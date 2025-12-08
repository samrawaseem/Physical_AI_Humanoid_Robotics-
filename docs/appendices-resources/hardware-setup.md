---
title: Hardware Setup
sidebar_position: 1
---

# Module 6: Appendices & Resources

## Hardware Setup Guide

This section provides comprehensive instructions for setting up the hardware components required for the Physical AI & Humanoid Robotics course, including Digital Twin workstations, Jetson Edge kits, sensors, and actuators.

## Learning Objectives

By the end of this module, you will be able to:
1. Guide students in setting up their workstations, Edge kits, and software environments
2. Ensure students understand hardware requirements and configurations
3. Provide clear instructions for installing ROS 2, Gazebo, Unity, NVIDIA Isaac, Python packages, and dependencies
4. Offer resources for further learning in Physical AI and humanoid robotics
5. Include a glossary to standardize terminology and concepts across the course

## Digital Twin Workstation Setup

### Minimum Requirements
- CPU: Intel i7 or AMD Ryzen 7 (8+ cores)
- RAM: 32GB or more
- GPU: NVIDIA RTX 3060 or higher (for Isaac Sim)
- Storage: 500GB SSD (1TB+ recommended)
- OS: Ubuntu 22.04 LTS or Windows 10/11 with WSL2

### Recommended Configuration
- CPU: Intel i9 or AMD Ryzen 9
- RAM: 64GB
- GPU: NVIDIA RTX 4080 or higher
- Storage: 1TB+ NVMe SSD

## Jetson Edge Kit Setup

### Jetson Orin Configuration
- Jetson AGX Orin Developer Kit (recommended)
- Power adapter (150W for AGX Orin)
- MicroSD card (64GB+ UHS-I)
- Camera module (Arducam or official NVIDIA camera)

### Initial Setup Process
1. Flash Jetson with appropriate SDK Manager
2. Configure network settings
3. Install required packages for ROS 2 communication
4. Test sensor and actuator connectivity

```bash
# Install ROS 2 packages on Jetson
sudo apt update
sudo apt install ros-humble-ros-base
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
```

## Sensor Configuration

### RealSense Depth Camera
- Intel RealSense D435i (recommended)
- USB 3.0 connection required
- Calibration procedures for depth accuracy

### IMU Setup
- Bosch BNO055 or similar 9-axis IMU
- I2C or UART communication
- Integration with ROS 2 sensor interface

### Additional Sensors
- LiDAR units for mapping (optional)
- Force/torque sensors for manipulation (optional)

## Software Installation Guide

### ROS 2 Humble Hawksbill
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
```

### Gazebo Installation
```bash
sudo apt install gazebo libgazebo-dev
```

### NVIDIA Isaac Setup
- Download Isaac Sim from NVIDIA Developer portal
- Follow installation instructions for your OS
- Verify GPU compatibility and drivers

## Cloud-Based Lab Setup (Optional)

### AWS RoboMaker
- AWS account with appropriate permissions
- RoboMaker service access
- S3 buckets for model storage

### Omniverse Cloud
- NVIDIA Developer account
- Cloud instance configuration
- Network requirements for streaming

## Troubleshooting Common Issues

### ROS 2 Network Issues
```bash
# Check ROS domain
echo $ROS_DOMAIN_ID

# Verify network connectivity
ping other-ros-node
```

### Gazebo Performance Issues
- Ensure adequate GPU resources
- Check for driver compatibility
- Verify physics engine settings

### Isaac Sim Problems
- Check GPU compute capability
- Verify CUDA installation
- Confirm license validity

## Glossary of Terms

- **URDF**: Unified Robot Description Format - XML format for representing robot models
- **VSLAM**: Visual Simultaneous Localization and Mapping
- **ROS 2 Nodes**: Processes that perform computation in the Robot Operating System
- **Topics**: Asynchronous communication channels in ROS 2
- **Actions**: Long-running tasks with feedback in ROS 2
- **LLMs**: Large Language Models for AI reasoning
- **VLA**: Vision-Language-Action framework for embodied AI
- **Cognitive Planning**: High-level reasoning for task execution

## References and Further Reading

- ROS 2 Documentation: https://docs.ros.org
- Isaac Sim User Guide: https://docs.omniverse.nvidia.com/isaacsim
- Gazebo Tutorials: http://gazebosim.org/tutorials
- Research papers on Physical AI and embodied robotics

This module provides all the supplementary materials, setup instructions, and references needed to support your journey through the Physical AI & Humanoid Robotics course.