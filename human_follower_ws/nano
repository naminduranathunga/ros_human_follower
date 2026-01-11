#!/bin/bash

# 1. Source the ROS 2 environments
source /opt/ros/humble/setup.bash
source ~/turtlebot2_ws/install/setup.bash

# 2. Define a cleanup function to kill all background jobs
cleanup() {
    echo ""
    echo "Caught Ctrl+C. Terminating all nodes..."
    
    # Kill the specific process IDs (PIDs) captured below
    kill $PID1 $PID2 $PID3
    
    # Wait for them to actually exit
    wait $PID1 $PID2 $PID3 2>/dev/null
    
    echo "All processes terminated."
    exit
}

# 3. Trap the SIGINT signal (Ctrl+C) and run the cleanup function
trap cleanup SIGINT

# 4. Run commands in the background using '&' and save their PIDs
echo "Starting Kobuki Node..."
ros2 launch kobuki_node kobuki_node-launch.py &
PID1=$!

echo "Starting Astra Camera..."
ros2 launch astra_camera astra.launch.xml &
PID2=$!

echo "Starting RQT Image View..."
ros2 run rqt_image_view rqt_image_view &
PID3=$!

# 5. Keep the script running to listen for Ctrl+C
wait
