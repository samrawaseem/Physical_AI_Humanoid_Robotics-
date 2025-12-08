---
title: Physics Engine Basics
sidebar_position: 3
---

# Module 2: The Digital Twin (Gazebo & Unity)

## Physics Engine Fundamentals in Robotics Simulation

Physics engines are crucial for creating realistic robot simulations. They calculate the motion and interactions of objects based on physical laws, enabling accurate testing of robotic algorithms before deployment on real hardware.

### Understanding Physics Simulation

Physics engines in robotics simulation handle:

- **Rigid Body Dynamics**: Movement and interaction of solid objects
- **Collision Detection**: Identifying when objects intersect
- **Contact Physics**: Calculating forces when objects touch
- **Constraints**: Maintaining relationships between objects (joints)
- **Integration**: Calculating motion over time

### Physics Engine Components in Gazebo

Gazebo supports multiple physics engines, with Ignition Physics being the current standard:

#### Key Physics Parameters

```xml
<!-- In a world file -->
<physics type="ignition-physics_6_0-plugin">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <solver>
    <type>quick</type>
    <iters>10</iters>
    <sor>1.3</sor>
  </solver>
  <constraints>
    <cfm>0</cfm>
    <erp>0.2</erp>
    <contact_max_correcting_vel>100</contact_max_correcting_vel>
    <contact_surface_layer>0.001</contact_surface_layer>
  </constraints>
</physics>
```

#### Parameter Explanations:

- **max_step_size**: Maximum time step for physics calculations (smaller = more accurate but slower)
- **real_time_factor**: Target simulation speed relative to real time (1.0 = real-time)
- **real_time_update_rate**: Updates per second (higher = more accurate but more CPU intensive)
- **gravity**: Gravitational acceleration vector (typically [0, 0, -9.8] m/s²)
- **solver iterations**: Number of iterations for constraint solving (more = more stable but slower)
- **CFM (Constraint Force Mixing)**: Softness of constraints (0 = hard constraints)
- **ERP (Error Reduction Parameter)**: How strongly to correct constraint violations

### Rigid Body Dynamics

Rigid body dynamics govern how objects move under forces and torques:

#### Inertial Properties

For accurate physics simulation, each link must have proper inertial properties:

```xml
<link name="link_name">
  <inertial>
    <mass value="1.0"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia
      ixx="0.1" ixy="0.0" ixz="0.0"
      iyy="0.1" iyz="0.0"
      izz="0.1"/>
  </inertial>
</link>
```

The inertia matrix represents how mass is distributed in the object:
- **Diagonal elements (ixx, iyy, izz)**: Resistance to rotation about each axis
- **Off-diagonal elements (ixy, ixz, iyz)**: Coupling between rotation axes

For common shapes, the inertia values are:

**Box** (width w, depth d, height h, mass m):
```
ixx = m * (d² + h²) / 12
iyy = m * (w² + h²) / 12
izz = m * (w² + d²) / 12
```

**Cylinder** (radius r, height h, mass m):
```
ixx = m * (3*r² + h²) / 12
iyy = m * (3*r² + h²) / 12
izz = m * r² / 2
```

**Sphere** (radius r, mass m):
```
ixx = iyy = izz = 2 * m * r² / 5
```

### Collision Detection

Collision detection is critical for realistic simulation:

#### Collision vs Visual Geometry

```xml
<link name="example_link">
  <!-- Visual geometry (for rendering) -->
  <visual name="visual">
    <geometry>
      <mesh filename="package://my_robot/meshes/complex_shape.stl"/>
    </geometry>
  </visual>

  <!-- Collision geometry (for physics, often simpler) -->
  <collision name="collision">
    <geometry>
      <!-- Simplified geometry for performance -->
      <cylinder radius="0.1" length="0.2"/>
    </geometry>
  </collision>
</link>
```

### Joint Physics

Joints connect links and constrain their relative motion:

#### Joint Friction and Damping

```xml
<joint name="joint_with_friction" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <axis xyz="0 0 1">
    <dynamics damping="0.1" friction="0.05"/>
  </axis>
  <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
</joint>
```

- **Damping**: Velocity-dependent resistance (like viscous fluid)
- **Friction**: Static and dynamic friction between joint surfaces

### Contact Physics

When objects touch, contact physics determines the resulting forces:

#### Contact Properties

```xml
<gazebo reference="link_name">
  <collision name="collision">
    <surface>
      <friction>
        <ode>
          <mu>1.0</mu>        <!-- Coefficient of friction -->
          <mu2>1.0</mu2>      <!-- Secondary friction coefficient -->
          <fdir1>0 0 1</fdir1> <!-- Friction direction -->
        </ode>
      </friction>
      <bounce>
        <restitution_coefficient>0.2</restitution_coefficient> <!-- Bounciness -->
        <threshold>100000</threshold> <!-- Velocity threshold for bouncing -->
      </bounce>
      <contact>
        <ode>
          <kp>1e+6</kp>      <!-- Spring stiffness -->
          <kd>100</kd>       <!-- Damping coefficient -->
          <max_vel>100</max_vel> <!-- Maximum contact correction velocity -->
          <min_depth>0.001</min_depth> <!-- Penetration depth before contact force -->
        </ode>
      </contact>
    </surface>
  </collision>
</gazebo>
```

### Physics Tuning for Humanoid Robots

Humanoid robots have specific physics requirements:

#### Center of Mass Considerations

For stable humanoid walking, the center of mass should be properly positioned:

```xml
<link name="torso">
  <inertial>
    <!-- Position center of mass lower for stability -->
    <mass value="5.0"/>
    <origin xyz="0 0 -0.1" rpy="0 0 0"/>
    <inertia
      ixx="0.2" ixy="0.0" ixz="0.0"
      iyy="0.3" iyz="0.0"
      izz="0.25"/>
  </inertial>
</link>
```

#### Foot Contact for Walking

```xml
<gazebo reference="left_foot">
  <collision name="foot_collision">
    <surface>
      <friction>
        <ode>
          <mu>1.0</mu>  <!-- High friction for stable stance -->
          <mu2>1.0</mu2>
        </ode>
      </friction>
      <contact>
        <ode>
          <kp>1e+7</kp>    <!-- High stiffness for solid contact -->
          <kd>1000</kd>    <!-- Adequate damping -->
          <max_vel>100</max_vel>
          <min_depth>0.0001</min_depth> <!-- Minimal penetration -->
        </ode>
      </contact>
    </surface>
  </collision>
</gazebo>
```

### Common Physics Issues and Solutions

#### Robot Instability

**Issue**: Robot wobbles or falls over unexpectedly
**Solutions**:
- Verify mass and inertia properties are realistic
- Increase solver iterations
- Reduce physics step size
- Check joint limits and positions

#### Penetration Issues

**Issue**: Objects pass through each other
**Solutions**:
- Increase contact stiffness (kp)
- Reduce minimum contact depth
- Use more accurate collision geometries
- Decrease physics step size

#### Performance Problems

**Issue**: Simulation runs slowly
**Solutions**:
- Increase physics step size (reduces accuracy)
- Reduce solver iterations (reduces stability)
- Simplify collision geometries
- Reduce update rates

### Physics Debugging

#### Visualizing Physics Properties

```xml
<gazebo reference="link_name">
  <!-- Visualize collision geometry -->
  <visual name="collision_visual" type="collision">
    <material>
      <ambient>1 0 0 0.5</ambient>
      <diffuse>1 0 0 0.5</diffuse>
    </material>
  </visual>
</gazebo>
```

#### Checking Inertial Properties

Use the `check_urdf` tool to validate your URDF:
```bash
check_urdf my_robot.urdf
```

### Physics Performance Optimization

#### Simplified Collision Models

For complex robots, use simplified collision geometries:

```xml
<!-- Instead of complex mesh collisions -->
<collision>
  <geometry>
    <mesh filename="complex_robot.stl"/>
  </geometry>
</collision>

<!-- Use simplified shapes -->
<collision>
  <geometry>
    <cylinder radius="0.1" length="0.5"/>
  </geometry>
</collision>
```

#### Hierarchical Collision Detection

For complex links, break them into multiple collision elements:

```xml
<link name="complex_link">
  <collision name="collision_1">
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.2 0.1 0.1"/>
    </geometry>
  </collision>
  <collision name="collision_2">
    <origin xyz="-0.1 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.2 0.1 0.1"/>
    </geometry>
  </collision>
</link>
```

### Physics Integration with ROS 2

Physics simulation integrates with ROS 2 through various interfaces:

#### Joint State Feedback

Physics engines provide joint state information that can be published via ROS 2:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        self.publisher = self.create_publisher(JointState, 'joint_states', 10)
        self.timer = self.create_timer(0.01, self.publish_joint_states)  # 100 Hz

    def publish_joint_states(self):
        msg = JointState()
        msg.name = ['joint1', 'joint2', 'joint3']
        msg.position = [math.sin(self.get_clock().now().nanoseconds * 1e-9),
                       math.cos(self.get_clock().now().nanoseconds * 1e-9),
                       0.0]
        msg.velocity = [0.0, 0.0, 0.0]
        msg.effort = [0.0, 0.0, 0.0]

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Understanding physics engine fundamentals is essential for creating stable and realistic robot simulations. Proper physics configuration ensures that algorithms tested in simulation will perform similarly on real hardware.