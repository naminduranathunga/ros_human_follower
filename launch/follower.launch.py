from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='human_follower',
            executable='person_detector',
            name='person_detector'
        ),
        Node(
            package='human_follower',
            executable='follow_controller',
            name='follow_controller'
        ),
    ])
