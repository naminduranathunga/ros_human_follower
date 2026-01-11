# Human Follower Robot - Competition System

## 🏆 Competition-Grade Human Following Robot

A complete ROS 2 system for autonomous human following using RGB-D camera with gesture-based control.

## 📋 Features

### Task A - Person Detection & Tracking ✅

- YOLOv8 for robust person detection
- IoU-based tracking with velocity prediction
- Handles partial occlusions using temporal tracking

### Task B - Depth-Based Distance Control ✅

- Real-time depth processing from Astra camera
- Maintains 1.5m safety zone
- Robust depth sampling with noise filtering

### Task C - Robot Navigation & Integration ✅

- Smooth velocity control with PID-like behavior
- Automatic speed adjustment based on distance
- Obstacle avoidance using depth zones

### Task D - Hand Signal Recognition ✅

- MediaPipe-based hand detection
- Position-based left/right determination (no flickering)
- Debounced gesture recognition

## 🔧 System Requirements

- Ubuntu 22.04 (or compatible)
- ROS 2 Humble
- Python 3.10+
- Kobuki TurtleBot with ROS 2 driver
- Astra/Xtion RGB-D camera

### Python Dependencies

```bash
pip install ultralytics mediapipe opencv-python numpy
```

## 📁 Package Structure

```
human_follower_ws/
└── src/
    └── human_follower/
        ├── package.xml
        ├── setup.py
        ├── setup.cfg
        ├── config/
        │   └── params.yaml          # Configuration parameters
        ├── launch/
        │   └── follower_launch.py   # Main launch file
        ├── resource/
        │   └── human_follower
        └── human_follower/
            ├── __init__.py
            ├── person_detector_node.py    # YOLO person detection
            ├── depth_tracker_node.py      # Depth processing
            ├── gesture_detector_node.py   # Hand gesture recognition
            ├── robot_controller_node.py   # Velocity control
            ├── obstacle_avoider_node.py   # Obstacle detection
            └── follower_main_node.py      # System coordinator
```

## 🚀 Installation & Setup

### Step 1: Create Workspace (Already Done)

The workspace is already created at `human_follower_ws/`

### Step 2: Copy to Your Linux System

Copy the `human_follower_ws` folder to your Linux system where ROS 2 is installed.

### Step 3: Build the Package

```bash
cd ~/human_follower_ws
colcon build --packages-select human_follower
source install/setup.bash
```

### Step 4: Download YOLO Model (First Run)

The YOLOv8 model will auto-download on first run, or manually:

```bash
pip install ultralytics
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## 🎮 Running the System

### Terminal 1: Start Kobuki Robot

```bash
source ~/kobuki_ws/install/setup.bash
ros2 launch kobuki_node kobuki_node-launch.py
```

### Terminal 2: Start Astra Camera

```bash
source ~/astra_ws/install/setup.bash
ros2 launch astra_camera astra.launch.xml
```

### Terminal 3: Start Human Follower

```bash
source ~/human_follower_ws/install/setup.bash
ros2 launch human_follower follower_launch.py
```

### Terminal 4: View Debug Images (Optional)

```bash
ros2 run rqt_image_view rqt_image_view
# Select topics:
# - /human_follower/debug_image (person detection)
# - /human_follower/depth_debug (depth visualization)
# - /human_follower/gesture_debug (gesture recognition)
# - /human_follower/obstacle_debug (obstacle zones)
```

## 🖐️ Control Gestures

| Gesture              | Action          |
| -------------------- | --------------- |
| **Raise RIGHT hand** | START following |
| **Raise LEFT hand**  | STOP following  |

## 📊 ROS 2 Topics

### Published by System

| Topic                             | Type    | Description             |
| --------------------------------- | ------- | ----------------------- |
| `/commands/velocity`              | Twist   | Robot velocity commands |
| `/human_follower/person_detected` | Bool    | Person detection status |
| `/human_follower/person_distance` | Float32 | Distance to person (m)  |
| `/human_follower/gesture_command` | String  | START/STOP/NONE         |
| `/human_follower/robot_state`     | String  | IDLE/FOLLOWING/STOPPED  |
| `/human_follower/system_status`   | String  | Full system status      |

### Subscribed Topics

| Topic                       | Type       | Source           |
| --------------------------- | ---------- | ---------------- |
| `/camera/color/image_raw`   | Image      | Astra RGB        |
| `/camera/depth/image_raw`   | Image      | Astra Depth      |
| `/camera/depth/camera_info` | CameraInfo | Astra intrinsics |

## ⚙️ Configuration

Edit `config/params.yaml` to tune:

```yaml
robot:
  target_distance: 1.5 # Following distance (m)
  min_safe_distance: 0.8 # Emergency stop distance
  max_linear_velocity: 0.5 # Max speed (m/s)

detection:
  confidence_threshold: 0.5 # YOLO confidence
  frame_skip: 2 # Process every Nth frame

gesture:
  debounce_frames: 3 # Frames to confirm gesture
  fingers_required: 3 # Min fingers for raised hand
```

## 🛡️ Safety Features

1. **Emergency Stop**: Robot stops if obstacle < 0.5m
2. **Lost Person**: Auto-stop after 3 seconds without detection
3. **Distance Limits**: Won't follow if person > 4m away
4. **Smooth Control**: Velocity ramping prevents jerky motion
5. **Gesture Confirmation**: Debouncing prevents accidental commands

## 🔍 Troubleshooting

### Camera not detected

```bash
# Check if camera topics exist
ros2 topic list | grep camera
```

### Robot not moving

```bash
# Check velocity commands
ros2 topic echo /commands/velocity

# Check robot state
ros2 topic echo /human_follower/robot_state
```

### Poor detection

- Ensure good lighting
- Stand 1-3m from camera
- Raise hand clearly above shoulder

## 📈 Performance Optimization

- **GPU**: Set `enable_gpu: True` if CUDA available
- **Frame Skip**: Increase `frame_skip` for slower systems
- **Input Size**: Reduce `input_size` (320 vs 416) for speed

## 🏁 Competition Tips

1. **Test the gesture system** before the run
2. **Calibrate distance** - adjust `target_distance` for your venue
3. **Practice smooth movements** - jerky movements can lose tracking
4. **Good lighting** is essential for reliable detection
5. **Stand facing the robot** for initial gesture recognition

---

Good luck in your competition! 🤖🏆
