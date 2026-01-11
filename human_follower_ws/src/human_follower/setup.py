from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'human_follower'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Include config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='student@university.edu',
    description='Competition-grade human following robot with gesture control',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'person_detector = human_follower.person_detector_node:main',
            'depth_tracker = human_follower.depth_tracker_node:main',
            'robot_controller = human_follower.robot_controller_node:main',
            'follower_main = human_follower.follower_main_node:main',
            'gesture_detector = human_follower.gesture_detector_node:main',
            'obstacle_avoider = human_follower.obstacle_avoider_node:main',
        ],
    },
)
