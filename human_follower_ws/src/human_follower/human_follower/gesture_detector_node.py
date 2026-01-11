#!/usr/bin/env python3
"""
Gesture Detector Node - Competition Grade
Hand gesture recognition for start/stop control using MediaPipe Pose.
Uses pose landmarks for accurate and reliable left/right hand detection.

Published Topics:
    /human_follower/gesture_command (String) - START, STOP, or NONE
    /human_follower/gesture_debug (Image) - Debug visualization

Subscribed Topics:
    /camera/color/image_raw (Image) - RGB camera input
    /human_follower/person_detected (Bool) - Person detection status
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple


class GestureDetectorNode(Node):
    def __init__(self):
        super().__init__('gesture_detector')
        
        # Declare parameters
        self.declare_parameter('min_detection_confidence', 0.7)
        self.declare_parameter('min_tracking_confidence', 0.7)
        self.declare_parameter('debounce_frames', 3)
        self.declare_parameter('flip_frame', True)
        self.declare_parameter('frame_skip', 2)  # Skip frames for VM performance
        
        # Get parameters
        self.debounce_frames = self.get_parameter('debounce_frames').value
        self.flip_frame = self.get_parameter('flip_frame').value
        self.frame_skip = self.get_parameter('frame_skip').value
        
        # Frame counter for skipping
        self.frame_count = 0
        self.last_frame = None
        
        # Initialize MediaPipe Pose for gesture detection
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_style = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=self.get_parameter('min_detection_confidence').value,
            min_tracking_confidence=self.get_parameter('min_tracking_confidence').value
        )
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # State
        self.person_detected = False
        
        # Gesture debouncing
        self.pending_gesture = "NONE"
        self.gesture_confirm_count = 0
        self.current_gesture = "NONE"
        
        # QoS for camera topics
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.rgb_callback,
            camera_qos
        )
        
        self.detected_sub = self.create_subscription(
            Bool,
            '/human_follower/person_detected',
            self.detected_callback,
            10
        )
        
        # Publishers
        self.gesture_pub = self.create_publisher(
            String,
            '/human_follower/gesture_command',
            10
        )
        
        self.debug_pub = self.create_publisher(
            Image,
            '/human_follower/gesture_debug',
            10
        )
        
        self.get_logger().info('Gesture Detector Node initialized')
    
    def detected_callback(self, msg: Bool):
        """Receive person detection status."""
        self.person_detected = msg.data
    
    def detect_gesture_from_pose(self, pose_landmarks) -> Tuple[str, bool, bool]:
        """
        Detect raised hand gesture using pose landmarks.
        Simple and reliable: compare wrist.y to shoulder.y
        
        Returns: (gesture_command, left_raised, right_raised)
        """
        lm = pose_landmarks.landmark
        
        # Get landmark positions
        left_wrist_y = lm[self.mp_pose.PoseLandmark.LEFT_WRIST].y
        left_shoulder_y = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_wrist_y = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST].y
        right_shoulder_y = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        
        # Simple check: is wrist above shoulder?
        left_raised = left_wrist_y < left_shoulder_y
        right_raised = right_wrist_y < right_shoulder_y
        
        # Determine gesture command
        # Note: In camera view, person's left appears on right side of image
        # When flip_frame is True, the image is mirrored so it matches natural view
        if self.flip_frame:
            # Mirrored view: person sees themselves naturally
            # Their right hand = START, left hand = STOP
            if right_raised and not left_raised:
                gesture = "START"
            elif left_raised and not right_raised:
                gesture = "STOP"
            else:
                gesture = "NONE"
        else:
            # Non-mirrored view
            if right_raised and not left_raised:
                gesture = "START"
            elif left_raised and not right_raised:
                gesture = "STOP"
            else:
                gesture = "NONE"
        
        return gesture, left_raised, right_raised
    
    def debounce_gesture(self, gesture: str) -> str:
        """Apply debouncing to prevent gesture flickering."""
        if gesture == "NONE":
            self.gesture_confirm_count = 0
            return "NONE"
        
        if gesture == self.pending_gesture:
            self.gesture_confirm_count += 1
        else:
            self.pending_gesture = gesture
            self.gesture_confirm_count = 1
        
        if self.gesture_confirm_count >= self.debounce_frames:
            return gesture
        
        return "NONE"
    
    def rgb_callback(self, msg: Image):
        """Process RGB image for gesture detection using pose landmarks."""
        # Frame skipping for VM performance
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            # Publish last processed frame if available
            if self.last_frame is not None:
                debug_msg = self.bridge.cv2_to_imgmsg(self.last_frame, 'bgr8')
                debug_msg.header = msg.header
                self.debug_pub.publish(debug_msg)
            return
        
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return
        
        # Resize for faster processing
        h, w = frame.shape[:2]
        scale = 0.5 if w > 640 else 1.0
        if scale < 1.0:
            small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        else:
            small_frame = frame.copy()
        
        # Flip for mirror behavior
        if self.flip_frame:
            small_frame = cv2.flip(small_frame, 1)
        
        sh, sw = small_frame.shape[:2]
        
        gesture_command = "NONE"
        left_raised = False
        right_raised = False
        
        # Only process pose if YOLO has detected a person first
        # This prevents false positives from chairs, objects, etc.
        if self.person_detected:
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Process pose for gesture detection
            pose_results = self.pose.process(rgb_frame)
            
            if pose_results.pose_landmarks:
                # Draw pose landmarks
                self.mp_draw.draw_landmarks(
                    small_frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_style.get_default_pose_landmarks_style()
                )
                
                # Detect gesture using pose landmarks
                gesture_command, left_raised, right_raised = self.detect_gesture_from_pose(
                    pose_results.pose_landmarks
                )
                
                # Draw hand status indicators
                lm = pose_results.pose_landmarks.landmark
                
                # Left wrist indicator
                left_wrist = lm[self.mp_pose.PoseLandmark.LEFT_WRIST]
                lw_px = (int(left_wrist.x * sw), int(left_wrist.y * sh))
                left_color = (0, 255, 0) if left_raised else (0, 0, 255)
                left_text = "LEFT UP" if left_raised else "left"
                cv2.putText(small_frame, left_text, (lw_px[0] - 30, lw_px[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)
                
                # Right wrist indicator
                right_wrist = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                rw_px = (int(right_wrist.x * sw), int(right_wrist.y * sh))
                right_color = (0, 255, 0) if right_raised else (0, 0, 255)
                right_text = "RIGHT UP" if right_raised else "right"
                cv2.putText(small_frame, right_text, (rw_px[0] - 30, rw_px[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)
        else:
            # No person detected by YOLO - skip pose detection
            cv2.putText(small_frame, "Waiting for person detection...", (10, sh // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
        
        # Apply debouncing
        confirmed_gesture = self.debounce_gesture(gesture_command)
        
        if confirmed_gesture != "NONE":
            self.current_gesture = confirmed_gesture
        
        # Publish gesture
        gesture_msg = String()
        gesture_msg.data = confirmed_gesture
        self.gesture_pub.publish(gesture_msg)
        
        # Draw debug info
        self.draw_debug(small_frame, gesture_command, confirmed_gesture, sh, sw)
        
        # Save and publish debug image
        self.last_frame = small_frame.copy()
        debug_msg = self.bridge.cv2_to_imgmsg(small_frame, 'bgr8')
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)
    
    def draw_debug(self, frame, gesture: str, confirmed: str, h: int, w: int):
        """Draw debug information on frame."""
        # Current gesture status
        if self.current_gesture == "START":
            color = (0, 255, 0)
        elif self.current_gesture == "STOP":
            color = (0, 0, 255)
        else:
            color = (200, 200, 200)
        
        cv2.putText(frame, f"Command: {self.current_gesture}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Show pending gesture
        if gesture != "NONE":
            text = f"Detecting: {gesture} ({self.gesture_confirm_count}/{self.debounce_frames})"
            cv2.putText(frame, text, (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Right hand up = START | Left hand up = STOP", (10, h - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def main(args=None):
    rclpy.init(args=args)
    node = GestureDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
