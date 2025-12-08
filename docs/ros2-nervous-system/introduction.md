---
title: The Robotic Nervous System (ROS 2)
sidebar_position: 1
---

# Module 1: The Robotic Nervous System (ROS 2)

## Introduction to ROS 2 Architecture

ROS 2 (Robot Operating System 2) serves as the nervous system for robotic applications, providing the communication infrastructure that allows different components of a robot to work together. In this module, we'll explore the core concepts that make ROS 2 the foundation for modern robotics development.

## Learning Objectives

By the end of this module, you will be able to:
1. Explain ROS 2 architecture: Nodes, Topics, Services, Actions
2. Build and launch ROS 2 packages with Python
3. Connect Python agents to ROS 2 nodes for control of simulated robots
4. Describe URDF structure and its role in humanoid robot modeling
5. Implement a simple humanoid control example in simulation

## Core ROS 2 Concepts

### Nodes
Nodes are processes that perform computation. In a robotic system, each component (sensors, actuators, controllers) typically runs as a separate node. Nodes can be written in different programming languages and can run on different machines.

### Topics and Message Passing
Topics enable asynchronous communication between nodes through a publish/subscribe pattern. Publishers send messages to topics, and subscribers receive messages from topics. This decouples nodes from each other.

### Services
Services provide synchronous request/response communication. A client sends a request and waits for a response from a service server. This is useful for operations that need acknowledgment or return specific results.

### Actions
Actions are for long-running tasks that provide feedback during execution. They support goals, results, and continuous feedback, making them ideal for navigation and manipulation tasks.

## Python Integration with rclpy

The `rclpy` library provides Python bindings for ROS 2, allowing you to create ROS 2 nodes using Python. This is particularly valuable for AI applications where Python's rich ecosystem of libraries can be leveraged.

## URDF: Unified Robot Description Format

URDF (Unified Robot Description Format) is an XML format for representing a robot model. It defines the physical and visual properties of a robot, including:
- Links (rigid bodies)
- Joints (connections between links)
- Sensors
- Materials

## Hands-on Exercise: Simple Humanoid Controller

In this exercise, you'll create a simple ROS 2 package with Python nodes that control a simulated humanoid joint. You'll implement both publisher/subscriber patterns for motor control and service servers for robot commands.

### Prerequisites
- ROS 2 Humble Hawksbill installed
- Python 3.8 or higher
- Basic understanding of Python programming

### Setup
First, create a new ROS 2 package:
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python simple_humanoid_controller
```

This module provides the foundation for all subsequent robotics work in this course. Understanding these concepts is crucial for developing more complex robotic systems in later modules.