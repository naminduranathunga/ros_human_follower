# Human Following Robot

This robot follows human and keeps a minimum distance.



## Prerequities

- ros2
- pyyaml
- opencv-python
- mediapipe

> Note: use `pip install "numpy<2" mediapipe==0.10.14` to install media pipe library. Newer version not supported.


## Project Structure

This was originally written for ROS2 Jemmy. 

```text
/human_detection
    /detect_human.py      -- Uses RGB Sensor output to detect a person, and gestures
    /follow_controller.py -- Handles Robot Movements.
```

## Configurations

1. Copy the code to <workspace_dir>/src

2. Create virtual env or install dependancis on global package
```sh
sudo apt install python3-venv
cd ~/ros2_ws
python3 -m venv venv
source venv/bin/activate
# source ~/ros2_ws/venv/bin/activate
```

Then install dependancies.


3. Build

```sh
cd ~/ros2_ws
colcon build
source install/setup.bash
```

4. Start Robot and Camera nodes

5. Launch nodes

```sh
ros2 run human_follower person_detector
ros2 run human_follower follow_controller
```

OR 

```sh
ros2 launch human_follower follower.launch.py
```

- Human Detection:
    - Set Threashold values 
        ```python
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        ```
    - Set Image node to "/camera/rgb/image_raw"
        ```python
        self.rgb_sub = self.create_subscription(Image, "/camera/rgb/image_raw", self.rgb_callback, 10)
        ```

- Movements:
    - Adjust the parameters as needed for calibration:
        ```python
        self.declare_parameter("min_distance", 1.0)             # Minimum distance to maintain
        self.declare_parameter("linear_kp", 0.6)                # kP of PID Controller for smoothing
        self.declare_parameter("max_linear_speed", 0.2)         # Maximum speed

        self.declare_parameter("offset_left_threshold", -0.1)   # Person's offset to the left (Always < 0)
        self.declare_parameter("offset_right_threshold", 0.1)   # Person's offset to the rigth (Always > 0)
        self.declare_parameter("angular_speed", 0.25)           # Angular speed for turning
        ```
    - Change the publisher if robot is not moving.
        ```python
        self.cmd_pub = self.create_publisher(Twist, "/mobile_base/commands/velocity", 10)
        ```
        
## Event–Condition–Action Logics

```text
Event: LEFT_HAND gesture
Condition: State = IDLE
Action: Set State = FOLLOW

Event: Person detected
Condition: State = FOLLOW AND DistanceToPerson > MinDistance
Action: MoveForward()

Event: Person detected
Condition: State = FOLLOW AND DistanceToPerson < MinDistance
Action: StopRobot()

Event: Person detected
Condition: State = FOLLOW AND PersonOffset < MinOffsetLeft
Action: DeltaTurnRobotLeft()

Event: Person detected
Condition: State = FOLLOW AND PersonOffset > MinOffsetRight
Action: DeltaTurnRobotRight()

Event: LEFT_HAND gesture
Condition: State = FOLLOW
Action: Set State = IDLE
```

