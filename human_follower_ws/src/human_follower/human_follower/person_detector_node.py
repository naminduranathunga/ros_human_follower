#!/usr/bin/env python3
"""
Person Detector Node - Competition Grade
Uses YOLOv3-tiny with OpenCV DNN for lightweight person detection.
Optimized for CPU/VM performance.

Published Topics:
    /human_follower/person_detection (Detection2D) - Detected person info
    /human_follower/debug_image (Image) - Debug visualization

Subscribed Topics:
    /camera/color/image_raw (Image) - RGB camera input
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Header, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class TrackedPerson:
    """Tracked person with temporal smoothing."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    confidence: float
    last_seen: float
    track_id: int
    velocity: Tuple[float, float] = (0.0, 0.0)


class PersonDetectorNode(Node):
    def __init__(self):
        super().__init__('person_detector')
        
        # Declare parameters
        self.declare_parameter('confidence_threshold', 0.4)
        self.declare_parameter('nms_threshold', 0.4)
        self.declare_parameter('frame_skip', 4)
        self.declare_parameter('input_size', 320)  # 320 or 416
        self.declare_parameter('tracking_timeout', 3.0)
        self.declare_parameter('iou_threshold', 0.25)
        self.declare_parameter('use_tiny', True)  # Use YOLOv3-tiny (faster)
        
        # Get parameters
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.nms_threshold = self.get_parameter('nms_threshold').value
        self.frame_skip = self.get_parameter('frame_skip').value
        self.input_size = self.get_parameter('input_size').value
        self.tracking_timeout = self.get_parameter('tracking_timeout').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        self.use_tiny = self.get_parameter('use_tiny').value
        
        # Initialize YOLOv3
        self.net = None
        self.output_layers = None
        self.load_yolov3()
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Tracking state
        self.tracked_person: Optional[TrackedPerson] = None
        self.frame_count = 0
        self.next_track_id = 0
        
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
        
        # Publishers
        self.detection_pub = self.create_publisher(
            Image,
            '/human_follower/person_bbox',
            10
        )
        
        self.center_pub = self.create_publisher(
            Point,
            '/human_follower/person_center',
            10
        )
        
        self.detected_pub = self.create_publisher(
            Bool,
            '/human_follower/person_detected',
            10
        )
        
        self.debug_pub = self.create_publisher(
            Image,
            '/human_follower/debug_image',
            10
        )
        
        # Performance tracking
        self.fps_history = []
        self.last_time = time.time()
        
        self.get_logger().info('Person Detector Node initialized (YOLOv3)')
    
    def download_file(self, url: str, filepath: str):
        """Download file with progress."""
        self.get_logger().info(f'Downloading: {url}')
        urllib.request.urlretrieve(url, filepath)
        self.get_logger().info(f'Downloaded: {filepath}')
    
    def load_yolov3(self):
        """Load YOLOv3 or YOLOv3-tiny model."""
        home_dir = os.path.expanduser('~')
        model_dir = os.path.join(home_dir, '.yolov3_models')
        os.makedirs(model_dir, exist_ok=True)
        
        if self.use_tiny:
            weights_file = os.path.join(model_dir, 'yolov3-tiny.weights')
            cfg_file = os.path.join(model_dir, 'yolov3-tiny.cfg')
            weights_url = 'https://pjreddie.com/media/files/yolov3-tiny.weights'
            cfg_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg'
            model_name = 'YOLOv3-tiny'
        else:
            weights_file = os.path.join(model_dir, 'yolov3.weights')
            cfg_file = os.path.join(model_dir, 'yolov3.cfg')
            weights_url = 'https://pjreddie.com/media/files/yolov3.weights'
            cfg_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
            model_name = 'YOLOv3'
        
        # Download if not exists
        if not os.path.exists(weights_file):
            self.get_logger().info(f'Downloading {model_name} weights...')
            self.download_file(weights_url, weights_file)
        
        if not os.path.exists(cfg_file):
            self.get_logger().info(f'Downloading {model_name} config...')
            self.download_file(cfg_url, cfg_file)
        
        # Load network
        self.get_logger().info(f'Loading {model_name}...')
        self.net = cv2.dnn.readNet(weights_file, cfg_file)
        
        # Use CPU (faster on VM than trying GPU without CUDA)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        self.get_logger().info(f'✓ {model_name} loaded successfully')
    
    def detect_persons(self, frame: np.ndarray) -> List[Tuple]:
        """Detect persons using YOLOv3."""
        h, w = frame.shape[:2]
        
        # Create blob
        blob = cv2.dnn.blobFromImage(
            frame, 
            1/255.0, 
            (self.input_size, self.input_size), 
            swapRB=True, 
            crop=False
        )
        
        # Forward pass
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Process outputs
        boxes = []
        confidences = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Class 0 is 'person' in COCO
                if class_id == 0 and confidence > self.confidence_threshold:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    box_w = int(detection[2] * w)
                    box_h = int(detection[3] * h)
                    
                    x1 = int(center_x - box_w / 2)
                    y1 = int(center_y - box_h / 2)
                    
                    boxes.append([x1, y1, box_w, box_h])
                    confidences.append(float(confidence))
        
        # Apply NMS
        detections = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
            
            for i in indices:
                idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                x, y, bw, bh = boxes[idx]
                detections.append((x, y, x + bw, y + bh, confidences[idx]))
        
        return detections
    
    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def predict_position(self, person: TrackedPerson, dt: float) -> Tuple[int, int, int, int]:
        """Predict person position based on velocity."""
        vx, vy = person.velocity
        x1, y1, x2, y2 = person.bbox
        
        dx = int(vx * dt)
        dy = int(vy * dt)
        
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
    
    def update_tracking(self, detections: List[Tuple], current_time: float, frame_shape: Tuple):
        """Update tracking with new detections, handling occlusions."""
        h, w = frame_shape[:2]
        
        if not detections:
            # No detections - check if we should keep tracking (occlusion handling)
            if self.tracked_person is not None:
                time_since_seen = current_time - self.tracked_person.last_seen
                
                if time_since_seen < self.tracking_timeout:
                    # Predict position based on velocity (handle temporary occlusion)
                    predicted_bbox = self.predict_position(self.tracked_person, time_since_seen)
                    
                    # Check if prediction is still in frame
                    if (predicted_bbox[0] >= 0 and predicted_bbox[2] <= w and
                        predicted_bbox[1] >= 0 and predicted_bbox[3] <= h):
                        self.tracked_person.bbox = predicted_bbox
                        self.tracked_person.center = (
                            (predicted_bbox[0] + predicted_bbox[2]) // 2,
                            (predicted_bbox[1] + predicted_bbox[3]) // 2
                        )
                        self.tracked_person.confidence *= 0.9  # Decay confidence
                        return
                
                # Lost tracking
                self.tracked_person = None
            return
        
        # Find best matching detection
        best_detection = None
        best_iou = 0
        
        if self.tracked_person is not None:
            predicted_bbox = self.predict_position(
                self.tracked_person,
                current_time - self.tracked_person.last_seen
            )
            
            for det in detections:
                iou = self.calculate_iou(predicted_bbox, det[:4])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_detection = det
        
        if best_detection is not None:
            # Update existing track
            x1, y1, x2, y2, conf = best_detection
            
            # Calculate velocity
            dt = current_time - self.tracked_person.last_seen
            if dt > 0:
                old_cx, old_cy = self.tracked_person.center
                new_cx = (x1 + x2) // 2
                new_cy = (y1 + y2) // 2
                
                # Smooth velocity update
                alpha = 0.3
                vx = alpha * (new_cx - old_cx) / dt + (1 - alpha) * self.tracked_person.velocity[0]
                vy = alpha * (new_cy - old_cy) / dt + (1 - alpha) * self.tracked_person.velocity[1]
                self.tracked_person.velocity = (vx, vy)
            
            self.tracked_person.bbox = (x1, y1, x2, y2)
            self.tracked_person.center = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.tracked_person.confidence = conf
            self.tracked_person.last_seen = current_time
            
        else:
            # Start new track with highest confidence detection
            best_det = max(detections, key=lambda x: x[4])
            x1, y1, x2, y2, conf = best_det
            
            self.tracked_person = TrackedPerson(
                bbox=(x1, y1, x2, y2),
                center=((x1 + x2) // 2, (y1 + y2) // 2),
                confidence=conf,
                last_seen=current_time,
                track_id=self.next_track_id,
                velocity=(0.0, 0.0)
            )
            self.next_track_id += 1
    
    def rgb_callback(self, msg: Image):
        """Process RGB image for person detection."""
        self.frame_count += 1
        current_time = time.time()
        
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return
        
        h, w = frame.shape[:2]
        detections = []
        
        # Run YOLOv3 detection (with frame skipping)
        if self.frame_count % self.frame_skip == 0:
            detections = self.detect_persons(frame)
        
        # Update tracking
        self.update_tracking(detections, current_time, frame.shape)
        
        # Publish results
        detected_msg = Bool()
        detected_msg.data = self.tracked_person is not None
        self.detected_pub.publish(detected_msg)
        
        if self.tracked_person is not None:
            # Publish center point
            center_msg = Point()
            center_msg.x = float(self.tracked_person.center[0])
            center_msg.y = float(self.tracked_person.center[1])
            center_msg.z = float(self.tracked_person.confidence)
            self.center_pub.publish(center_msg)
            
            # Publish bbox as image metadata (x1, y1, x2, y2 encoded)
            bbox_msg = Image()
            bbox_msg.header = msg.header
            bbox_msg.height = 1
            bbox_msg.width = 4
            bbox_msg.encoding = 'mono16'
            bbox_msg.step = 8
            x1, y1, x2, y2 = self.tracked_person.bbox
            bbox_msg.data = np.array([x1, y1, x2, y2], dtype=np.uint16).tobytes()
            self.detection_pub.publish(bbox_msg)
        
        # Debug visualization
        debug_frame = self.draw_debug(frame, current_time)
        debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, 'bgr8')
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)
    
    def draw_debug(self, frame: np.ndarray, current_time: float) -> np.ndarray:
        """Draw debug visualization."""
        debug = frame.copy()
        
        # Calculate FPS
        fps = 1.0 / (current_time - self.last_time + 1e-6)
        self.last_time = current_time
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        # Draw FPS
        cv2.putText(debug, f'FPS: {avg_fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw tracked person
        if self.tracked_person is not None:
            x1, y1, x2, y2 = self.tracked_person.bbox
            conf = self.tracked_person.confidence
            track_id = self.tracked_person.track_id
            
            # Draw bounding box
            color = (0, 255, 0) if conf > 0.5 else (0, 255, 255)
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
            
            # Draw center
            cx, cy = self.tracked_person.center
            cv2.circle(debug, (cx, cy), 5, (0, 0, 255), -1)
            
            # Draw info
            text = f'Person #{track_id} ({conf:.2f})'
            cv2.putText(debug, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw velocity vector
            vx, vy = self.tracked_person.velocity
            end_x = int(cx + vx * 0.5)
            end_y = int(cy + vy * 0.5)
            cv2.arrowedLine(debug, (cx, cy), (end_x, end_y), (255, 0, 0), 2)
        else:
            cv2.putText(debug, 'No person detected', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return debug


def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
