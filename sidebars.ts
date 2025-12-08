import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Define the textbook sidebar structure
  textbookSidebar: [
    {
      type: 'category',
      label: 'Preface & Introduction',
      link: {type: 'doc', id: 'preface-intro/introduction'},
      items: [
        'preface-intro/course-overview',
        'preface-intro/setup-requirements',
        'preface-intro/learning-objectives'
      ],
    },
    {
      type: 'category',
      label: 'The Robotic Nervous System (ROS 2)',
      link: {type: 'doc', id: 'ros2-nervous-system/introduction'},
      items: [
        'ros2-nervous-system/ros2-architecture',
        'ros2-nervous-system/nodes-topics-services',
        'ros2-nervous-system/rclpy-python-nodes',
        'ros2-nervous-system/urdf-robot-description',
        'ros2-nervous-system/hands-on-exercises'
      ],
    },
    {
      type: 'category',
      label: 'The Digital Twin (Gazebo & Unity)',
      link: {type: 'doc', id: 'digital-twin-gazebo-unity/introduction'},
      items: [
        'digital-twin-gazebo-unity/gazebo-simulation',
        'digital-twin-gazebo-unity/physics-engine-basics',
        'digital-twin-gazebo-unity/sensor-simulation',
        'digital-twin-gazebo-unity/unity-visualization',
        'digital-twin-gazebo-unity/hands-on-exercises'
      ],
    },
    {
      type: 'category',
      label: 'The AI-Robot Brain (NVIDIA Isaac™)',
      link: {type: 'doc', id: 'ai-robot-brain-isaac/introduction'},
      items: [
        'ai-robot-brain-isaac/isaac-sim-environment',
        'ai-robot-brain-isaac/isaac-ros-integration',
        'ai-robot-brain-isaac/visual-slam-navigation',
        'ai-robot-brain-isaac/sim-to-real-transfer',
        'ai-robot-brain-isaac/hands-on-exercises'
      ],
    },
    {
      type: 'category',
      label: 'Vision-Language-Action (VLA)',
      link: {type: 'doc', id: 'vision-language-action-vla/introduction'},
      items: [
        'vision-language-action-vla/llm-integration',
        'vision-language-action-vla/cognitive-planning',
        'vision-language-action-vla/multi-modal-perception',
        'vision-language-action-vla/voice-command-processing',
        'vision-language-action-vla/hands-on-exercises'
      ],
    },
    {
      type: 'category',
      label: 'Capstone Project — Autonomous Humanoid',
      link: {type: 'doc', id: 'capstone-autonomous-humanoid/introduction'},
      items: [
        'capstone-autonomous-humanoid/system-integration',
        'capstone-autonomous-humanoid/task-execution',
        'capstone-autonomous-humanoid/debugging-performance',
        'capstone-autonomous-humanoid/project-evaluation'
      ],
    },
    {
      type: 'category',
      label: 'Appendices & Resources',
      link: {type: 'doc', id: 'appendices/software-installation'},
      items: [
        'appendices/software-installation',
        'appendices/troubleshooting',
        'appendices/glossary',
        'appendices/references'
      ],
    }
  ],

  // Keep tutorialSidebar for any tutorial content
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Tutorial',
      items: ['tutorial-basics/create-a-document'],
    },
  ],
};

export default sidebars;