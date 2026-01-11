#!/usr/bin/env python3
"""
Depth Tracker Node - Competition Grade
Processes depth images to calculate real distance to tracked person.
Uses robust depth sampling with noise filtering.

Published Topics:
    /human_follower/person_distance (Float32) - Distance to person in meters
    /human_follower/depth_debug (Image) - Depth visualization

Subscribed Topics:
    /camera/depth/image_raw (Image) - Depth camera input
    /human_follower/person_bbox (Image) - Person bounding box
    /human_follower/person_detected (Bool) - Detection status
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import Optional, Tuple
import time


class DepthTrackerNode(Node):
    def __init__(self):
        super().__init__('depth_tracker')
        
        # Declare parameters
        self.declare_parameter('depth_scale', 0.001)  # mm to meters for Astra
        self.declare_parameter('min_valid_depth', 0.3)
        self.declare_parameter('max_valid_depth', 8.0)
        self.declare_parameter('roi_width_ratio', 0.3)
        self.declare_parameter('roi_height_ratio', 0.3)
        self.declare_parameter('depth_percentile', 30)
        self.declare_parameter('temporal_smoothing', 0.4)
        
        # Get parameters
        self.depth_scale = self.get_parameter('depth_scale').value
        self.min_valid_depth = self.get_parameter('min_valid_depth').value
        self.max_valid_depth = self.get_parameter('max_valid_depth').value
        self.roi_width_ratio = self.get_parameter('roi_width_ratio').value
        self.roi_height_ratio = self.get_parameter('roi_height_ratio').value
        self.depth_percentile = self.get_parameter('depth_percentile').value
        self.temporal_smoothing = self.get_parameter('temporal_smoothing').value
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # State
        self.person_bbox: Optional[Tuple[int, int, int, int]] = None
        self.person_detected = False
        self.current_distance = 0.0
        self.smoothed_distance = 0.0
        self.last_depth_image: Optional[np.ndarray] = None
        self.camera_info: Optional[CameraInfo] = None
        
        # QoS for camera topics
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            camera_qos
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/depth/camera_info',
            self.camera_info_callback,
            10
        )
        
        self.bbox_sub = self.create_subscription(
            Image,
            '/human_follower/person_bbox',
            self.bbox_callback,
            10
        )
        
        self.detected_sub = self.create_subscription(
            Bool,
            '/human_follower/person_detected',
            self.detected_callback,
            10
        )
        
        # Publishers
        self.distance_pub = self.create_publisher(
            Float32,
            '/human_follower/person_distance',
            10
        )
        
        self.depth_point_pub = self.create_publisher(
            Point,
            '/human_follower/person_3d_position',
            10
        )
        
        self.debug_pub = self.create_publisher(
            Image,
            '/human_follower/depth_debug',
            10
        )
        
        self.get_logger().info('Depth Tracker Node initialized')
    
    def camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsic parameters."""
        self.camera_info = msg
    
    def bbox_callback(self, msg: Image):
        """Receive person bounding box."""
        try:
            # Decode bbox from image data
            bbox_data = np.frombuffer(msg.data, dtype=np.uint16)
            if len(bbox_data) == 4:
                self.person_bbox = tuple(bbox_data)
        except Exception as e:
            self.get_logger().error(f'Bbox decode error: {e}')
    
    def detected_callback(self, msg: Bool):
        """Receive detection status."""
        self.person_detected = msg.data
        if not self.person_detected:
            self.person_bbox = None
    
    def calculate_depth_in_roi(self, depth_image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        Calculate robust depth measurement within person bounding box.
        Uses center ROI and percentile filtering for noise rejection.
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_image.shape[:2]
        
        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Calculate center ROI (chest area is most reliable)
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        # Focus on torso area (upper-center of bounding box)
        roi_w = int(bbox_w * self.roi_width_ratio)
        roi_h = int(bbox_h * self.roi_height_ratio)
        
        cx = (x1 + x2) // 2
        cy = y1 + int(bbox_h * 0.35)  # Upper torso area
        
        roi_x1 = max(0, cx - roi_w // 2)
        roi_y1 = max(0, cy - roi_h // 2)
        roi_x2 = min(w, cx + roi_w // 2)
        roi_y2 = min(h, cy + roi_h // 2)
        
        # Extract ROI
        roi = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if roi.size == 0:
            return 0.0
        
        # Convert to meters
        roi_meters = roi.astype(np.float32) * self.depth_scale
        
        # Filter valid depths
        valid_mask = (roi_meters > self.min_valid_depth) & (roi_meters < self.max_valid_depth)
        valid_depths = roi_meters[valid_mask]
        
        if len(valid_depths) == 0:
            return 0.0
        
        # Use percentile for robust estimation (handles outliers)
        distance = np.percentile(valid_depths, self.depth_percentile)
        
        return float(distance)
    
    def pixel_to_3d(self, u: int, v: int, depth: float) -> Tuple[float, float, float]:
        """Convert pixel coordinates and depth to 3D point."""
        if self.camera_info is None:
            # Default values if no camera info
            fx, fy = 525.0, 525.0  # Typical values for Astra
            cx, cy = 320.0, 240.0
        else:
            fx = self.camera_info.k[0]
            fy = self.camera_info.k[4]
            cx = self.camera_info.k[2]
            cy = self.camera_info.k[5]
        
        # Convert to 3D
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        return x, y, z
    
    def depth_callback(self, msg: Image):
        """Process depth image."""
        try:
            # Handle different depth encodings
            if msg.encoding == '16UC1':
                depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')
            elif msg.encoding == '32FC1':
                depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')
                depth_image = (depth_image * 1000).astype(np.uint16)  # Convert to mm
            else:
                depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth conversion error: {e}')
            return
        
        self.last_depth_image = depth_image
        
        # Calculate distance if person is detected
        distance_msg = Float32()
        
        if self.person_detected and self.person_bbox is not None:
            self.current_distance = self.calculate_depth_in_roi(depth_image, self.person_bbox)
            
            # Apply temporal smoothing
            if self.smoothed_distance > 0:
                alpha = self.temporal_smoothing
                self.smoothed_distance = alpha * self.current_distance + (1 - alpha) * self.smoothed_distance
            else:
                self.smoothed_distance = self.current_distance
            
            distance_msg.data = self.smoothed_distance
            
            # Calculate 3D position
            if self.smoothed_distance > 0:
                x1, y1, x2, y2 = self.person_bbox
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                x3d, y3d, z3d = self.pixel_to_3d(cx, cy, self.smoothed_distance)
                
                point_msg = Point()
                point_msg.x = z3d   # Forward (depth axis)
                point_msg.y = -x3d  # Left (ROS convention)
                point_msg.z = -y3d  # Up
                self.depth_point_pub.publish(point_msg)
        else:
            distance_msg.data = 0.0
            self.smoothed_distance = 0.0
        
        self.distance_pub.publish(distance_msg)
        
        # Publish debug visualization
        self.publish_debug(depth_image, msg.header)
    
    def publish_debug(self, depth_image: np.ndarray, header):
        """Create and publish depth debug visualization."""
        # Normalize depth for visualization
        depth_display = depth_image.astype(np.float32) * self.depth_scale
        depth_display = np.clip(depth_display, 0, 5)  # Clip to 5m
        depth_display = (depth_display / 5 * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        
        # Draw ROI if tracking
        if self.person_detected and self.person_bbox is not None:
            x1, y1, x2, y2 = self.person_bbox
            
            # Draw bounding box
            cv2.rectangle(depth_color, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
            # Draw center ROI
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            roi_w = int(bbox_w * self.roi_width_ratio)
            roi_h = int(bbox_h * self.roi_height_ratio)
            cx = (x1 + x2) // 2
            cy = y1 + int(bbox_h * 0.35)
            
            cv2.rectangle(depth_color,
                         (cx - roi_w // 2, cy - roi_h // 2),
                         (cx + roi_w // 2, cy + roi_h // 2),
                         (0, 255, 0), 2)
            
            # Draw distance
            text = f'Distance: {self.smoothed_distance:.2f}m'
            cv2.putText(depth_color, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Publish
        debug_msg = self.bridge.cv2_to_imgmsg(depth_color, 'bgr8')
        debug_msg.header = header
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthTrackerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
