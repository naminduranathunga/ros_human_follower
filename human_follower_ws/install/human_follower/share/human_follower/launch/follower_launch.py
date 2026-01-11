#!/usr/bin/env python3
"""
Launch file for Human Follower System
Launches all nodes required for human-following robot.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, GroupAction, TimerAction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('human_follower')
    
    # Load params.yaml
    params_file = os.path.join(pkg_dir, 'config', 'params.yaml')
    
    # Declare launch arguments
    use_sim = DeclareLaunchArgument(
        'use_sim',
        default_value='false',
        description='Use simulation time'
    )
    
    debug = DeclareLaunchArgument(
        'debug',
        default_value='true',
        description='Enable debug output'
    )
    
    # Parameters
    common_params = {
        'use_sim_time': LaunchConfiguration('use_sim'),
    }
    
    # Person Detector Node - YOLOv3-tiny (Optimized for VM)
    person_detector_node = Node(
        package='human_follower',
        executable='person_detector',
        name='person_detector',
        output='screen',
        parameters=[
            common_params,
            {
                'confidence_threshold': 0.4,
                'nms_threshold': 0.4,
                'frame_skip': 4,
                'input_size': 320,  # 320 for speed, 416 for accuracy
                'tracking_timeout': 3.0,
                'iou_threshold': 0.25,
                'use_tiny': True,  # Use YOLOv3-tiny (much faster)
            }
        ],
        remappings=[
            ('/camera/color/image_raw', '/camera/color/image_raw'),
        ]
    )
    
    # Depth Tracker Node
    depth_tracker_node = Node(
        package='human_follower',
        executable='depth_tracker',
        name='depth_tracker',
        output='screen',
        parameters=[
            common_params,
            {
                'depth_scale': 0.001,  # Astra uses mm
                'min_valid_depth': 0.3,
                'max_valid_depth': 8.0,
                'roi_width_ratio': 0.3,
                'roi_height_ratio': 0.3,
                'depth_percentile': 30,
                'temporal_smoothing': 0.4,
            }
        ],
        remappings=[
            ('/camera/depth/image_raw', '/camera/depth/image_raw'),
            ('/camera/depth/camera_info', '/camera/depth/camera_info'),
        ]
    )
    
    # Gesture Detector Node
    gesture_detector_node = Node(
        package='human_follower',
        executable='gesture_detector',
        name='gesture_detector',
        output='screen',
        parameters=[
            common_params,
            {
                'min_detection_confidence': 0.6,
                'min_tracking_confidence': 0.5,
                'model_complexity': 0,  # 0=lite for VM speed
                'hand_raise_threshold': 0.45,
                'fingers_required': 2,
                'debounce_frames': 2,
                'use_pose_reference': False,  # Disabled for VM speed
                'flip_frame': True,
                'frame_skip': 3,  # Skip frames for VM performance
            }
        ],
        remappings=[
            ('/camera/color/image_raw', '/camera/color/image_raw'),
        ]
    )
    
    # Obstacle Avoider Node
    obstacle_avoider_node = Node(
        package='human_follower',
        executable='obstacle_avoider',
        name='obstacle_avoider',
        output='screen',
        parameters=[
            common_params,
            {
                'depth_scale': 0.001,
                'stop_distance': 0.5,
                'slow_distance': 1.0,
                'min_valid_depth': 0.3,
                'max_valid_depth': 4.0,
            }
        ],
        remappings=[
            ('/camera/depth/image_raw', '/camera/depth/image_raw'),
        ]
    )
    
    # Robot Controller Node
    robot_controller_node = Node(
        package='human_follower',
        executable='robot_controller',
        name='robot_controller',
        output='screen',
        parameters=[
            common_params,
            {
                'target_distance': 1.5,
                'min_distance': 0.8,
                'max_tracking_distance': 4.0,
                'max_linear_velocity': 0.20,
                'min_linear_velocity': 0.03,
                'max_angular_velocity': 0.3,
                'linear_kp': 0.3,
                'angular_kp': 0.8,
                'velocity_smoothing': 0.15,
                'lost_timeout': 3.0,
                'image_width': 640,
                # Smooth speed control (set to False for simple proportional control)
                'smooth_speed_enabled': True,
                'max_linear_accel': 0.05,
                'max_linear_decel': 0.025,
                'max_angular_accel': 0.1,
                'max_angular_decel': 0.06,
                'decel_distance': 0.5,
            }
        ],
        remappings=[
            ('/commands/velocity', '/commands/velocity'),
        ]
    )
    
    # Main Follower Node (coordinator)
    follower_main_node = Node(
        package='human_follower',
        executable='follower_main',
        name='follower_main',
        output='screen',
        parameters=[common_params],
    )
    
    # Create launch description
    return LaunchDescription([
        use_sim,
        debug,
        
        # Launch perception nodes first
        person_detector_node,
        depth_tracker_node,
        gesture_detector_node,
        obstacle_avoider_node,
        
        # Launch control nodes with slight delay
        TimerAction(
            period=2.0,
            actions=[robot_controller_node]
        ),
        
        # Launch main coordinator
        TimerAction(
            period=3.0,
            actions=[follower_main_node]
        ),
    ])
