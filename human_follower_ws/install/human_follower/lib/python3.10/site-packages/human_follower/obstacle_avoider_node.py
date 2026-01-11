#!/usr/bin/env python3
"""
Obstacle Avoider Node - Competition Grade (Enhanced)
Depth-based obstacle detection for safe navigation.
Monitors multiple zones around the robot for obstacles.
Enhanced with edge detection for shallow objects like chairs.

Published Topics:
    /human_follower/obstacle_detected (Bool) - Obstacle in critical zone
    /human_follower/obstacle_zones (String) - Zone status
    /human_follower/obstacle_info (String) - Detailed obstacle info for smart avoidance

Subscribed Topics:
    /camera/depth/image_raw (Image) - Depth camera input
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import Tuple, Dict
import json


class ObstacleAvoiderNode(Node):
    def __init__(self):
        super().__init__('obstacle_avoider')
        
        # Declare parameters
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('stop_distance', 0.5)
        self.declare_parameter('slow_distance', 1.0)
        self.declare_parameter('front_sector_angle', 30)
        self.declare_parameter('side_sector_angle', 60)
        self.declare_parameter('min_valid_depth', 0.3)
        self.declare_parameter('max_valid_depth', 4.0)
        self.declare_parameter('edge_detection_enabled', True)
        self.declare_parameter('depth_gradient_threshold', 0.3)  # meters per pixel
        
        # Get parameters
        self.depth_scale = self.get_parameter('depth_scale').value
        self.stop_distance = self.get_parameter('stop_distance').value
        self.slow_distance = self.get_parameter('slow_distance').value
        self.min_valid_depth = self.get_parameter('min_valid_depth').value
        self.max_valid_depth = self.get_parameter('max_valid_depth').value
        self.edge_detection_enabled = self.get_parameter('edge_detection_enabled').value
        self.depth_gradient_threshold = self.get_parameter('depth_gradient_threshold').value
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Zone definitions (as image column ranges)
        self.zones: Dict[str, Tuple[int, int]] = {}
        self.zone_distances: Dict[str, float] = {}
        self.zone_has_obstacle: Dict[str, bool] = {}
        
        # Image dimensions
        self.image_height = 0
        self.image_width = 0
        self.roi_y_start = 0
        self.roi_y_end = 0
        self.roi_lower_y_start = 0
        self.roi_lower_y_end = 0
        self.roi_mid_y_start = 0
        self.roi_mid_y_end = 0
        
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
        
        # Publishers
        self.obstacle_pub = self.create_publisher(
            Bool,
            '/human_follower/obstacle_detected',
            10
        )
        
        self.zones_pub = self.create_publisher(
            String,
            '/human_follower/obstacle_zones',
            10
        )
        
        # New publisher for detailed obstacle info (for intelligent avoidance)
        self.obstacle_info_pub = self.create_publisher(
            String,
            '/human_follower/obstacle_info',
            10
        )
        
        self.debug_pub = self.create_publisher(
            Image,
            '/human_follower/obstacle_debug',
            10
        )
        
        self.get_logger().info('Obstacle Avoider Node initialized (Enhanced with edge detection)')
    
    def setup_zones(self, width: int, height: int):
        """Setup detection zones based on image dimensions."""
        # Front center: middle 40% (wider for better coverage)
        front_width = int(width * 0.4)
        front_start = (width - front_width) // 2
        
        # Side zones: 30% each
        side_width = int(width * 0.3)
        left_end = side_width
        right_start = width - side_width
        
        self.zones = {
            'left': (0, left_end),
            'front': (front_start, front_start + front_width),
            'right': (right_start, width),
        }
        
        # Multiple vertical ROIs for better detection
        # Lower region: ground obstacles and chair legs
        self.roi_lower_y_start = int(height * 0.6)
        self.roi_lower_y_end = height
        
        # Middle region: chair seats, table edges
        self.roi_mid_y_start = int(height * 0.35)
        self.roi_mid_y_end = int(height * 0.65)
        
        # Full region for general obstacles
        self.roi_y_start = height // 3
        self.roi_y_end = height
        
        self.image_height = height
        self.image_width = width
    
    def detect_depth_edges(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Detect edges in depth image to find shallow objects like chair legs.
        Returns a mask of detected edges/discontinuities.
        """
        depth_meters = depth_image.astype(np.float32) * self.depth_scale
        
        # Replace invalid depths with max value
        depth_meters[depth_meters < self.min_valid_depth] = self.max_valid_depth
        depth_meters[depth_meters > self.max_valid_depth] = self.max_valid_depth
        
        # Calculate depth gradients (Sobel)
        grad_x = cv2.Sobel(depth_meters, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_meters, cv2.CV_32F, 0, 1, ksize=3)
        
        # Gradient magnitude
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold to find significant depth discontinuities
        edge_mask = (gradient_mag > self.depth_gradient_threshold).astype(np.uint8) * 255
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, kernel)
        
        return edge_mask
    
    def detect_shallow_objects(self, depth_image: np.ndarray, zone: Tuple[int, int]) -> Tuple[bool, float]:
        """
        Detect shallow objects like chair legs using multiple techniques.
        Returns: (has_shallow_obstacle, estimated_distance)
        """
        x_start, x_end = zone
        
        # Extract lower ROI (where chair legs would be)
        roi_lower = depth_image[self.roi_lower_y_start:self.roi_lower_y_end, x_start:x_end]
        roi_meters = roi_lower.astype(np.float32) * self.depth_scale
        
        valid_mask = (roi_meters > self.min_valid_depth) & (roi_meters < self.max_valid_depth)
        
        if not np.any(valid_mask):
            return False, self.max_valid_depth
        
        # Look for close isolated regions (chair legs pattern)
        close_mask = (roi_meters < self.stop_distance) & valid_mask
        close_pixels = np.sum(close_mask)
        total_pixels = roi_lower.size
        
        close_ratio = close_pixels / total_pixels if total_pixels > 0 else 0
        
        # Check for depth variance in close regions (indicates thin objects)
        if close_pixels > 10:
            close_depths = roi_meters[close_mask]
            depth_variance = np.var(close_depths)
            
            # Chair legs have low variance but are isolated
            if 0.01 < close_ratio < 0.3 and depth_variance < 0.1:
                min_dist = np.min(close_depths)
                return True, float(min_dist)
        
        # Edge detection for thin vertical structures
        if self.edge_detection_enabled:
            edge_mask = self.detect_depth_edges(depth_image)
            edge_roi = edge_mask[self.roi_lower_y_start:self.roi_lower_y_end, x_start:x_end]
            
            edge_with_close = edge_roi & (close_mask.astype(np.uint8) * 255)
            edge_close_count = np.sum(edge_with_close > 0)
            
            if edge_close_count > 50:
                valid_depths = roi_meters[valid_mask]
                min_dist = np.percentile(valid_depths, 5)
                if min_dist < self.slow_distance:
                    return True, float(min_dist)
        
        return False, self.max_valid_depth
    
    def calculate_zone_distance(self, depth_image: np.ndarray, zone: Tuple[int, int]) -> Tuple[float, bool]:
        """
        Calculate minimum distance in a zone using multiple ROI heights.
        Returns: (min_distance, has_obstacle)
        """
        x_start, x_end = zone
        distances = []
        
        # Check full ROI
        roi_full = depth_image[self.roi_y_start:self.roi_y_end, x_start:x_end]
        roi_meters = roi_full.astype(np.float32) * self.depth_scale
        valid_mask = (roi_meters > self.min_valid_depth) & (roi_meters < self.max_valid_depth)
        
        if np.any(valid_mask):
            dist_full = float(np.percentile(roi_meters[valid_mask], 10))
            distances.append(dist_full)
        
        # Check lower ROI (ground obstacles)
        roi_lower = depth_image[self.roi_lower_y_start:self.roi_lower_y_end, x_start:x_end]
        roi_lower_meters = roi_lower.astype(np.float32) * self.depth_scale
        valid_lower = (roi_lower_meters > self.min_valid_depth) & (roi_lower_meters < self.max_valid_depth)
        
        if np.any(valid_lower):
            dist_lower = float(np.percentile(roi_lower_meters[valid_lower], 10))
            distances.append(dist_lower)
        
        # Check middle ROI (table edges, chair seats)
        roi_mid = depth_image[self.roi_mid_y_start:self.roi_mid_y_end, x_start:x_end]
        roi_mid_meters = roi_mid.astype(np.float32) * self.depth_scale
        valid_mid = (roi_mid_meters > self.min_valid_depth) & (roi_mid_meters < self.max_valid_depth)
        
        if np.any(valid_mid):
            dist_mid = float(np.percentile(roi_mid_meters[valid_mid], 10))
            distances.append(dist_mid)
        
        # Get minimum across all ROIs
        min_distance = min(distances) if distances else self.max_valid_depth
        
        # Check for shallow objects
        has_shallow, shallow_dist = self.detect_shallow_objects(depth_image, zone)
        
        if has_shallow and shallow_dist < min_distance:
            min_distance = shallow_dist
        
        has_obstacle = min_distance < self.stop_distance or has_shallow
        
        return min_distance, has_obstacle
    
    def _get_best_direction(self) -> str:
        """Determine the best direction to avoid obstacle."""
        left_dist = self.zone_distances.get('left', 0)
        right_dist = self.zone_distances.get('right', 0)
        front_dist = self.zone_distances.get('front', 0)
        
        # If front is clear, go straight
        if front_dist > self.slow_distance:
            return 'front'
        
        # Compare left and right
        if left_dist > right_dist and left_dist > self.stop_distance:
            return 'left'
        elif right_dist > left_dist and right_dist > self.stop_distance:
            return 'right'
        elif left_dist > self.stop_distance:
            return 'left'
        elif right_dist > self.stop_distance:
            return 'right'
        else:
            return 'blocked'
    
    def depth_callback(self, msg: Image):
        """Process depth image for obstacle detection."""
        try:
            if msg.encoding == '16UC1':
                depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')
            elif msg.encoding == '32FC1':
                depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')
                depth_image = (depth_image * 1000).astype(np.uint16)
            else:
                depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth conversion error: {e}')
            return
        
        h, w = depth_image.shape[:2]
        
        # Setup zones on first frame
        if not self.zones:
            self.setup_zones(w, h)
        
        # Calculate distances for each zone
        obstacle_in_front = False
        zone_status = []
        
        for zone_name, zone_range in self.zones.items():
            distance, has_obstacle = self.calculate_zone_distance(depth_image, zone_range)
            self.zone_distances[zone_name] = distance
            self.zone_has_obstacle[zone_name] = has_obstacle
            
            if zone_name == 'front' and (distance < self.stop_distance or has_obstacle):
                obstacle_in_front = True
            
            status = 'CLEAR'
            if distance < self.stop_distance or has_obstacle:
                status = 'STOP'
            elif distance < self.slow_distance:
                status = 'SLOW'
            
            zone_status.append(f'{zone_name}:{distance:.2f}m({status})')
        
        # Publish obstacle status
        obstacle_msg = Bool()
        obstacle_msg.data = obstacle_in_front
        self.obstacle_pub.publish(obstacle_msg)
        
        # Publish zone status
        zones_msg = String()
        zones_msg.data = ' | '.join(zone_status)
        self.zones_pub.publish(zones_msg)
        
        # Publish detailed obstacle info for intelligent avoidance
        obstacle_info = {
            'front_blocked': obstacle_in_front,
            'left_distance': self.zone_distances.get('left', self.max_valid_depth),
            'front_distance': self.zone_distances.get('front', self.max_valid_depth),
            'right_distance': self.zone_distances.get('right', self.max_valid_depth),
            'left_clear': self.zone_distances.get('left', 0) > self.slow_distance,
            'right_clear': self.zone_distances.get('right', 0) > self.slow_distance,
            'best_direction': self._get_best_direction(),
            'stop_distance': self.stop_distance,
            'slow_distance': self.slow_distance,
        }
        
        info_msg = String()
        info_msg.data = json.dumps(obstacle_info)
        self.obstacle_info_pub.publish(info_msg)
        
        # Publish debug visualization
        self.publish_debug(depth_image, msg.header)
    
    def publish_debug(self, depth_image: np.ndarray, header):
        """Create and publish obstacle debug visualization."""
        depth_display = depth_image.astype(np.float32) * self.depth_scale
        depth_display = np.clip(depth_display, 0, 4)
        depth_display = (depth_display / 4 * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        
        h, w = depth_color.shape[:2]
        
        # Draw zones
        for zone_name, (x_start, x_end) in self.zones.items():
            distance = self.zone_distances.get(zone_name, 0)
            has_obstacle = self.zone_has_obstacle.get(zone_name, False)
            
            if distance < self.stop_distance or has_obstacle:
                color = (0, 0, 255)  # Red
            elif distance < self.slow_distance:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green
            
            # Draw zone rectangle (full ROI)
            cv2.rectangle(depth_color,
                         (x_start, self.roi_y_start),
                         (x_end, self.roi_y_end),
                         color, 2)
            
            # Draw lower ROI (for shallow objects)
            cv2.rectangle(depth_color,
                         (x_start, self.roi_lower_y_start),
                         (x_end, self.roi_lower_y_end),
                         color, 1)
            
            label = f'{zone_name}: {distance:.2f}m'
            if has_obstacle:
                label += ' [OBS]'
            cv2.putText(depth_color, label,
                       (x_start + 5, self.roi_y_start + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw best direction indicator
        best_dir = self._get_best_direction()
        cv2.putText(depth_color, f'Best: {best_dir.upper()}',
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(depth_color, f'Stop: <{self.stop_distance}m | Slow: <{self.slow_distance}m',
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        debug_msg = self.bridge.cv2_to_imgmsg(depth_color, 'bgr8')
        debug_msg.header = header
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoiderNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
