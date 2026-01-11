#!/usr/bin/env python3
"""
Main Follower Node - Competition Grade
Central coordinator that combines all modules into a complete system.
Provides unified control and status monitoring with combined debug output.

This node coordinates:
- Person detection
- Depth tracking
- Gesture recognition
- Robot control
- Obstacle avoidance

Published Topics:
    /human_follower/system_status (String) - Overall system status
    /human_follower/combined_debug (Image) - Combined debug visualization

Subscribed Topics:
    All status topics from other nodes
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from enum import Enum


class SystemState(Enum):
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    ACTIVE = "ACTIVE"
    ERROR = "ERROR"


class FollowerMainNode(Node):
    def __init__(self):
        super().__init__('follower_main')
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # System state
        self.system_state = SystemState.INITIALIZING
        
        # Component status
        self.person_detected = False
        self.person_distance = 0.0
        self.gesture_command = "NONE"
        self.robot_state = "IDLE"
        self.obstacle_detected = False
        
        # Debug images storage
        self.person_debug_image = None
        self.depth_debug_image = None
        self.gesture_debug_image = None
        self.obstacle_debug_image = None
        
        # Health monitoring
        self.last_person_update = 0.0
        self.last_depth_update = 0.0
        self.last_gesture_update = 0.0
        
        # QoS for image topics
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers for monitoring
        self.detected_sub = self.create_subscription(
            Bool, '/human_follower/person_detected', self.detected_cb, 10)
        
        self.distance_sub = self.create_subscription(
            Float32, '/human_follower/person_distance', self.distance_cb, 10)
        
        self.gesture_sub = self.create_subscription(
            String, '/human_follower/gesture_command', self.gesture_cb, 10)
        
        self.robot_state_sub = self.create_subscription(
            String, '/human_follower/robot_state', self.robot_state_cb, 10)
        
        self.obstacle_sub = self.create_subscription(
            Bool, '/human_follower/obstacle_detected', self.obstacle_cb, 10)
        
        # Debug image subscribers
        self.person_debug_sub = self.create_subscription(
            Image, '/human_follower/debug_image', self.person_debug_cb, image_qos)
        
        self.depth_debug_sub = self.create_subscription(
            Image, '/human_follower/depth_debug', self.depth_debug_cb, image_qos)
        
        self.gesture_debug_sub = self.create_subscription(
            Image, '/human_follower/gesture_debug', self.gesture_debug_cb, image_qos)
        
        self.obstacle_debug_sub = self.create_subscription(
            Image, '/human_follower/obstacle_debug', self.obstacle_debug_cb, image_qos)
        
        # Publishers
        self.status_pub = self.create_publisher(
            String, '/human_follower/system_status', 10)
        
        self.combined_debug_pub = self.create_publisher(
            Image, '/human_follower/combined_debug', 10)
        
        # Combined debug output timer (15 Hz)
        self.debug_timer = self.create_timer(0.067, self.publish_combined_debug)
        
        # Status update timer
        self.status_timer = self.create_timer(0.5, self.status_update)
        
        # Print startup banner
        self.print_banner()
        
        self.system_state = SystemState.READY
        self.get_logger().info('System READY - Waiting for START gesture')
    
    def print_banner(self):
        """Print startup banner."""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║           HUMAN FOLLOWER ROBOT - COMPETITION SYSTEM          ║
╠══════════════════════════════════════════════════════════════╣
║  CONTROLS:                                                   ║
║    🖐️  Raise RIGHT hand  →  START following                  ║
║    🖐️  Raise LEFT hand   →  STOP following                   ║
╠══════════════════════════════════════════════════════════════╣
║  SAFETY:                                                     ║
║    • Target distance: 1.5m                                   ║
║    • Emergency stop if obstacle < 0.5m                       ║
║    • Auto-stop if person lost for > 3 seconds                ║
╠══════════════════════════════════════════════════════════════╣
║  DEBUG:                                                      ║
║    /human_follower/combined_debug - All views in one         ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def detected_cb(self, msg: Bool):
        self.person_detected = msg.data
        self.last_person_update = time.time()
    
    def distance_cb(self, msg: Float32):
        self.person_distance = msg.data
        self.last_depth_update = time.time()
    
    def gesture_cb(self, msg: String):
        if msg.data != "NONE":
            self.gesture_command = msg.data
        self.last_gesture_update = time.time()
    
    def robot_state_cb(self, msg: String):
        self.robot_state = msg.data
    
    def obstacle_cb(self, msg: Bool):
        self.obstacle_detected = msg.data
    
    def person_debug_cb(self, msg: Image):
        try:
            self.person_debug_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            pass
    
    def depth_debug_cb(self, msg: Image):
        try:
            self.depth_debug_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            pass
    
    def gesture_debug_cb(self, msg: Image):
        try:
            self.gesture_debug_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            pass
    
    def obstacle_debug_cb(self, msg: Image):
        try:
            self.obstacle_debug_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            pass
    
    def publish_combined_debug(self):
        """Combine all debug images into a single output."""
        target_h, target_w = 240, 320
        
        # Create placeholder for missing images
        def get_image_or_placeholder(img, label):
            if img is not None:
                return cv2.resize(img, (target_w, target_h))
            placeholder = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            cv2.putText(placeholder, label, (10, target_h // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            cv2.putText(placeholder, "No Data", (10, target_h // 2 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            return placeholder
        
        # Get all images
        person_img = get_image_or_placeholder(self.person_debug_image, "Person Detection")
        depth_img = get_image_or_placeholder(self.depth_debug_image, "Depth Tracking")
        gesture_img = get_image_or_placeholder(self.gesture_debug_image, "Gesture Detection")
        obstacle_img = get_image_or_placeholder(self.obstacle_debug_image, "Obstacle Detection")
        
        # Add labels to images
        label_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        
        def add_label(img, text):
            cv2.rectangle(img, (0, 0), (len(text) * 10 + 10, 22), bg_color, -1)
            cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)
            return img
        
        person_img = add_label(person_img, "PERSON (YOLO)")
        depth_img = add_label(depth_img, "DEPTH")
        gesture_img = add_label(gesture_img, "GESTURE")
        obstacle_img = add_label(obstacle_img, "OBSTACLE")
        
        # Combine into 2x2 grid
        top_row = np.hstack([person_img, depth_img])
        bottom_row = np.hstack([gesture_img, obstacle_img])
        combined = np.vstack([top_row, bottom_row])
        
        # Add status bar at bottom
        status_bar = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
        
        # Status text
        state_color = (0, 255, 0) if self.robot_state == "FOLLOWING" else (0, 255, 255) if self.robot_state == "IDLE" else (0, 0, 255)
        person_color = (0, 255, 0) if self.person_detected else (0, 0, 255)
        
        status_text = f"State: {self.robot_state} | Person: {'YES' if self.person_detected else 'NO'} | Dist: {self.person_distance:.2f}m | Gesture: {self.gesture_command}"
        cv2.putText(status_bar, status_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        combined = np.vstack([combined, status_bar])
        
        # Publish combined image
        try:
            combined_msg = self.bridge.cv2_to_imgmsg(combined, 'bgr8')
            self.combined_debug_pub.publish(combined_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish combined debug: {e}')
    
    def status_update(self):
        """Publish comprehensive system status."""
        current_time = time.time()
        
        # Check component health
        person_ok = (current_time - self.last_person_update) < 1.0
        depth_ok = (current_time - self.last_depth_update) < 1.0
        gesture_ok = (current_time - self.last_gesture_update) < 1.0
        
        # Update system state
        if person_ok and depth_ok and gesture_ok:
            if self.robot_state == "FOLLOWING":
                self.system_state = SystemState.ACTIVE
            else:
                self.system_state = SystemState.READY
        else:
            self.system_state = SystemState.ERROR
        
        # Build status message
        status_parts = [
            f"STATE:{self.robot_state}",
            f"PERSON:{'Yes' if self.person_detected else 'No'}",
            f"DIST:{self.person_distance:.2f}m",
            f"GESTURE:{self.gesture_command}",
            f"OBSTACLE:{'Yes' if self.obstacle_detected else 'No'}"
        ]
        
        status_msg = String()
        status_msg.data = " | ".join(status_parts)
        self.status_pub.publish(status_msg)
        
        # Log periodic status
        if int(current_time) % 5 == 0:  # Every 5 seconds
            self.get_logger().info(
                f'Status: {self.robot_state} | '
                f'Person: {self.person_detected} | '
                f'Distance: {self.person_distance:.2f}m | '
                f'Obstacle: {self.obstacle_detected}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = FollowerMainNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
