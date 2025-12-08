---
title: Unity Visualization
sidebar_position: 5
---

# Module 2: The Digital Twin (Gazebo & Unity)

## Unity Integration for High-Fidelity Visualization

Unity provides high-fidelity visualization capabilities that complement Gazebo's physics simulation. While Gazebo excels at physics-based simulation, Unity offers photorealistic rendering and advanced visualization features that enhance human-robot interaction and system understanding.

### Unity Robotics Setup

Unity provides specialized tools for robotics integration:

#### Unity Robotics Hub

The Unity Robotics Hub is a collection of tools and packages that facilitate robotics development:

1. **Unity-Robotics-Hub**: GitHub repository with samples and documentation
2. **ROS-TCP-Connector**: Enables communication between Unity and ROS 2
3. **Unity-Robotics-Demo**: Example scenes and implementations
4. **URDF-Importer**: Imports URDF files directly into Unity

#### Installation Process

1. **Install Unity Hub**: Download from Unity's website
2. **Install Unity Editor**: Version 2022.3 LTS or later recommended
3. **Install Unity Robotics Packages**:
   - ROS-TCP-Connector
   - URDF-Importer
   - Visualization tools

### ROS-TCP-Connector Integration

The ROS-TCP-Connector enables bidirectional communication between Unity and ROS 2:

#### Unity Side Setup

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

public class RobotController : MonoBehaviour
{
    ROSConnection ros;
    string robotName = "my_robot";

    // Start is called before the first frame update
    void Start()
    {
        // Get the ROS connection static instance
        ros = ROSConnection.instance;

        // Register a callback for a topic
        ros.Subscribe<UInt8Msg>("/unity_command", CommandCallback);
    }

    void CommandCallback(UInt8Msg command)
    {
        // Process command from ROS
        Debug.Log("Received command: " + command.data);

        // Execute robot movement based on command
        switch(command.data)
        {
            case 1:
                MoveForward();
                break;
            case 2:
                MoveBackward();
                break;
            case 3:
                TurnLeft();
                break;
            case 4:
                TurnRight();
                break;
        }
    }

    void MoveForward()
    {
        transform.Translate(Vector3.forward * Time.deltaTime);
    }

    void MoveBackward()
    {
        transform.Translate(Vector3.back * Time.deltaTime);
    }

    void TurnLeft()
    {
        transform.Rotate(Vector3.up, -90 * Time.deltaTime);
    }

    void TurnRight()
    {
        transform.Rotate(Vector3.up, 90 * Time.deltaTime);
    }

    // Update is called once per frame
    void Update()
    {
        // Send robot position to ROS
        var positionMsg = new RosMessageTypes.Geometry.PointMsg();
        positionMsg.x = transform.position.x;
        positionMsg.y = transform.position.y;
        positionMsg.z = transform.position.z;

        ros.Send("unity_robot_position", positionMsg);
    }
}
```

#### ROS 2 Side Setup

Create a ROS 2 node to communicate with Unity:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8
from geometry_msgs.msg import Point
import socket
import json

class UnityBridge(Node):
    def __init__(self):
        super().__init__('unity_bridge')

        # ROS publishers and subscribers
        self.command_publisher = self.create_publisher(UInt8, '/unity_command', 10)
        self.position_subscriber = self.create_subscription(
            Point, 'unity_robot_position', self.position_callback, 10
        )

        # Unity communication setup
        self.unity_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.unity_socket.connect(('localhost', 10000))

        # Timer for sending commands to Unity
        self.timer = self.create_timer(0.1, self.send_commands)

        self.robot_position = Point()

    def position_callback(self, msg):
        self.robot_position = msg
        self.get_logger().info(f'Robot position: ({msg.x}, {msg.y}, {msg.z})')

    def send_commands(self):
        # Example: Send a movement command to Unity
        command = UInt8()
        command.data = 1  # Move forward
        self.command_publisher.publish(command)

def main(args=None):
    rclpy.init(args=args)
    unity_bridge = UnityBridge()

    try:
        rclpy.spin(unity_bridge)
    except KeyboardInterrupt:
        pass
    finally:
        unity_bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### URDF Importer for Unity

The URDF Importer allows you to directly import ROS robot models into Unity:

#### Import Process

1. **Prepare URDF Files**: Ensure all meshes and materials are properly referenced
2. **Import URDF**: Use the URDF Importer tool in Unity
3. **Configure Joints**: Set up joint constraints and limits
4. **Add Physics**: Configure collision and rigid body properties

#### Example URDF Import Script

```csharp
using UnityEngine;
using Unity.Robotics.URDFImport;

public class RobotImporter : MonoBehaviour
{
    public string urdfPath;
    public GameObject robotPrefab;

    void Start()
    {
        if (!string.IsNullOrEmpty(urdfPath))
        {
            // Import robot from URDF
            robotPrefab = URDFRobotExtensions.LoadRobot(urdfPath);

            // Position the robot in the scene
            robotPrefab.transform.position = Vector3.zero;

            // Add joint controllers
            ConfigureJoints(robotPrefab);
        }
    }

    void ConfigureJoints(GameObject robot)
    {
        // Find all joint components and configure them
        var jointComponents = robot.GetComponentsInChildren<Unity.Robotics.ROSTCPConnector.JointController>();

        foreach (var joint in jointComponents)
        {
            // Configure joint limits and properties
            joint.minAngle = -90f;
            joint.maxAngle = 90f;
            joint.speed = 30f; // degrees per second
        }
    }
}
```

### Creating Visualization Scenes

#### Basic Scene Setup

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class RobotVisualizationScene : MonoBehaviour
{
    public GameObject robot;
    public Camera mainCamera;
    public Light mainLight;

    void Start()
    {
        // Initialize ROS connection
        ROSConnection.instance = gameObject.AddComponent<ROSConnection>();
        ROSConnection.instance.rosIPAddress = "127.0.0.1";
        ROSConnection.instance.rosPort = 10000;

        // Load robot model
        LoadRobotModel();

        // Set up camera
        SetupCamera();

        // Initialize lighting
        SetupLighting();
    }

    void LoadRobotModel()
    {
        // This could load from URDF or instantiate a prefab
        robot = GameObject.CreatePrimitive(PrimitiveType.Capsule);
        robot.name = "Robot";
        robot.transform.position = Vector3.zero;
    }

    void SetupCamera()
    {
        if (mainCamera == null)
        {
            mainCamera = Camera.main;
        }

        // Position camera for optimal viewing
        mainCamera.transform.position = new Vector3(5, 3, -5);
        mainCamera.transform.LookAt(robot.transform);
    }

    void SetupLighting()
    {
        if (mainLight == null)
        {
            mainLight = Light.main;
        }

        // Configure lighting for good visibility
        mainLight.type = LightType.Directional;
        mainLight.intensity = 1.0f;
        mainLight.transform.rotation = Quaternion.Euler(50, -30, 0);
    }
}
```

### Advanced Visualization Features

#### Real-time Sensor Visualization

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class SensorVisualizer : MonoBehaviour
{
    public GameObject robot;
    public GameObject[] sensorVisualizers; // Visual representations for each sensor
    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.instance;

        // Subscribe to sensor topics
        ros.Subscribe<LaserScanMsg>("/scan", OnLaserScanReceived);
        ros.Subscribe<ImageMsg>("/camera/image_raw", OnImageReceived);
        ros.Subscribe<ImuMsg>("/imu", OnImuReceived);
    }

    void OnLaserScanReceived(LaserScanMsg scan)
    {
        // Visualize LiDAR data as points or lines
        VisualizeLidarData(scan);
    }

    void OnImageReceived(ImageMsg image)
    {
        // Process and display camera image
        DisplayCameraImage(image);
    }

    void OnImuReceived(ImuMsg imu)
    {
        // Visualize IMU data (orientation, acceleration)
        VisualizeImuData(imu);
    }

    void VisualizeLidarData(LaserScanMsg scan)
    {
        // Create visualization of LiDAR points
        for (int i = 0; i < scan.ranges.Length; i++)
        {
            float angle = scan.angle_min + i * scan.angle_increment;
            float distance = scan.ranges[i];

            if (distance < scan.range_max && distance > scan.range_min)
            {
                Vector3 point = new Vector3(
                    distance * Mathf.Cos(angle),
                    0,
                    distance * Mathf.Sin(angle)
                );

                // Create or update visualization point
                CreateLidarPoint(point, robot.transform.position);
            }
        }
    }

    void CreateLidarPoint(Vector3 localPosition, Vector3 robotPosition)
    {
        GameObject point = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        point.transform.localScale = Vector3.one * 0.05f;
        point.transform.position = robotPosition + localPosition;
        point.GetComponent<Renderer>().material.color = Color.red;

        // Destroy after a few seconds to prevent clutter
        Destroy(point, 2.0f);
    }

    void DisplayCameraImage(ImageMsg image)
    {
        // This would typically update a texture in the scene
        // Implementation depends on how you want to display the image
    }

    void VisualizeImuData(ImuMsg imu)
    {
        // Visualize orientation as a 3D orientation indicator
        // Convert quaternion to rotation
        Quaternion rotation = new Quaternion(
            imu.orientation.x,
            imu.orientation.y,
            imu.orientation.z,
            imu.orientation.w
        );

        // Apply to a visualization object
        robot.transform.rotation = rotation;
    }
}
```

#### Interactive Controls

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class RobotControllerUI : MonoBehaviour
{
    public GameObject robot;
    private ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.instance;
    }

    void Update()
    {
        // Handle keyboard input for direct control
        if (Input.GetKeyDown(KeyCode.W))
        {
            SendVelocityCommand(1.0f, 0.0f); // Move forward
        }
        if (Input.GetKeyDown(KeyCode.S))
        {
            SendVelocityCommand(-1.0f, 0.0f); // Move backward
        }
        if (Input.GetKeyDown(KeyCode.A))
        {
            SendVelocityCommand(0.0f, 1.0f); // Turn left
        }
        if (Input.GetKeyDown(KeyCode.D))
        {
            SendVelocityCommand(0.0f, -1.0f); // Turn right
        }
    }

    void SendVelocityCommand(float linear, float angular)
    {
        var twistMsg = new TwistMsg();
        twistMsg.linear = new Vector3Msg(linear, 0, 0);
        twistMsg.angular = new Vector3Msg(0, 0, angular);

        ros.Send("cmd_vel", twistMsg);
    }

    // UI button callbacks
    public void MoveForward()
    {
        SendVelocityCommand(0.5f, 0.0f);
    }

    public void MoveBackward()
    {
        SendVelocityCommand(-0.5f, 0.0f);
    }

    public void TurnLeft()
    {
        SendVelocityCommand(0.0f, 0.5f);
    }

    public void TurnRight()
    {
        SendVelocityCommand(0.0f, -0.5f);
    }

    public void Stop()
    {
        SendVelocityCommand(0.0f, 0.0f);
    }
}
```

### Unity-ROS 2 Communication Patterns

#### Publisher Pattern

```csharp
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class UnityPublisher : MonoBehaviour
{
    ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.instance;
    }

    void Update()
    {
        // Publish robot state at regular intervals
        if (Time.frameCount % 60 == 0) // Every 60 frames (approx 1 Hz if 60 FPS)
        {
            var poseMsg = new PoseMsg();
            poseMsg.position = new Vector3Msg(
                transform.position.x,
                transform.position.y,
                transform.position.z
            );

            poseMsg.orientation = new QuaternionMsg(
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w
            );

            ros.Send("unity_robot_pose", poseMsg);
        }
    }
}
```

#### Service Client Pattern

```csharp
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;

public class UnityServiceClient : MonoBehaviour
{
    ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.instance;
    }

    public void CallService()
    {
        // Create service request
        var request = new TriggerMsg();

        // Send service call and handle response
        ros.SendServiceMessage<TriggerSrvResponse>(
            "/trigger_service",
            request,
            OnServiceResponse
        );
    }

    void OnServiceResponse(TriggerSrvResponse response)
    {
        Debug.Log($"Service response: {response.success}, {response.message}");
    }
}
```

### Performance Optimization

#### Level of Detail (LOD) System

```csharp
using UnityEngine;

public class RobotLOD : MonoBehaviour
{
    public Transform[] lodLevels;
    public float[] lodDistances;

    Transform cameraTransform;
    float lastDistance = -1f;

    void Start()
    {
        cameraTransform = Camera.main.transform;
    }

    void Update()
    {
        float distance = Vector3.Distance(cameraTransform.position, transform.position);

        // Only update LOD when distance changes significantly
        if (Mathf.Abs(distance - lastDistance) > 1f)
        {
            UpdateLOD(distance);
            lastDistance = distance;
        }
    }

    void UpdateLOD(float distance)
    {
        for (int i = 0; i < lodDistances.Length; i++)
        {
            if (distance <= lodDistances[i])
            {
                ActivateLOD(i);
                return;
            }
        }

        // Use highest detail if within minimum distance
        ActivateLOD(0);
    }

    void ActivateLOD(int level)
    {
        for (int i = 0; i < lodLevels.Length; i++)
        {
            lodLevels[i].gameObject.SetActive(i == level);
        }
    }
}
```

#### Occlusion Culling

```csharp
using UnityEngine;

public class OcclusionCullingManager : MonoBehaviour
{
    public float updateInterval = 0.5f;
    private float lastUpdateTime = 0f;

    void Update()
    {
        if (Time.time - lastUpdateTime > updateInterval)
        {
            UpdateOcclusion();
            lastUpdateTime = Time.time;
        }
    }

    void UpdateOcclusion()
    {
        // Update visibility of robot components based on camera view
        // This can be done using Unity's built-in occlusion culling
        // or custom implementation
    }
}
```

### Best Practices for Unity Integration

1. **Separate Visualization from Logic**: Keep Unity visualization separate from ROS 2 control logic
2. **Optimize for Performance**: Use LOD systems and efficient rendering techniques
3. **Maintain Synchronization**: Ensure Unity visualization matches ROS 2 state
4. **Handle Connection Failures**: Implement robust connection handling
5. **Use Appropriate Update Rates**: Balance visualization quality with performance
6. **Validate Data Types**: Ensure ROS message types match Unity expectations

### Integration Workflow

1. **Design Robot in URDF**: Create your robot model using URDF
2. **Import to Unity**: Use URDF Importer to bring model into Unity
3. **Add Visualization Components**: Enhance with Unity-specific visual elements
4. **Implement ROS Communication**: Add TCP connection for data exchange
5. **Test Integration**: Verify that Unity visualization reflects ROS state
6. **Optimize Performance**: Fine-tune for smooth real-time operation

Unity integration provides powerful visualization capabilities that enhance the digital twin experience, allowing for photorealistic rendering and intuitive human-robot interaction interfaces that complement the physics-based simulation of Gazebo.