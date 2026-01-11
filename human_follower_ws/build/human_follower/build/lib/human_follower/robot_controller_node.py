#!/usr/bin/env python3
"""
Robot Controller Node - Competition Grade
Controls Kobuki TurtleBot velocity based on person position and distance.
Implements smooth following behavior with PID-like control.

Movement Logic:
- Turn left/right based on person's horizontal offset from image center
- Move forward to maintain target distance (but okay to be closer than min distance)
- Stop if person is lost or gesture commands STOP

Published Topics:
    /commands/velocity (Twist) - Robot velocity commands

Subscribed Topics:
    /human_follower/person_distance (Float32) - Distance to person
    /human_follower/person_center (Point) - Person center in image
    /human_follower/person_3d_position (Point) - 3D position
    /human_follower/person_detected (Bool) - Detection status
    /human_follower/gesture_command (String) - Gesture commands
    /human_follower/obstacle_detected (Bool) - Obstacle status
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Float32, Bool, String
import math
import time
import signal
import json
from enum import Enum
from typing import Optional, Dict


class RobotState(Enum):
    IDLE = "IDLE"
    WAITING = "WAITING"
    FOLLOWING = "FOLLOWING"
    STOPPED = "STOPPED"
    OBSTACLE_AVOID = "OBSTACLE_AVOID"
    OBSTACLE_DRIVE_THROUGH = "OBSTACLE_DRIVE_THROUGH"  # Drive past obstacle before resuming
    LOST = "LOST"
    BACKING = "BACKING"


class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Declare parameters
        self.declare_parameter('target_distance', 1.0)        # Desired following distance (meters)
        self.declare_parameter('min_distance', 0.5)           # Minimum acceptable distance (won't back up if closer)
        self.declare_parameter('max_tracking_distance', 4.0)  # Max distance to track
        self.declare_parameter('max_linear_velocity', 0.15)   # Max forward speed (reduced for safety)
        self.declare_parameter('max_angular_velocity', 0.3)   # Max turning speed (reduced for smoothness)
        self.declare_parameter('linear_kp', 0.3)              # Forward speed proportional gain (reduced)
        self.declare_parameter('angular_kp', 0.8)             # Turn speed proportional gain (reduced)
        self.declare_parameter('velocity_smoothing', 0.15)    # Smoothing factor (0-1, lower = smoother)
        self.declare_parameter('lost_timeout', 3.0)           # Seconds before declaring person lost
        self.declare_parameter('image_width', 640)            # Camera image width
        self.declare_parameter('center_deadzone', 0.1)        # Deadzone for centering (fraction of image width)
        self.declare_parameter('cmd_timeout', 0.5)            # Safety watchdog timeout
        self.declare_parameter('avoidance_turn_speed', 0.25)  # Turn speed during obstacle avoidance
        self.declare_parameter('avoidance_timeout', 5.0)      # Max time in avoidance mode
        
        # Multi-phase avoidance parameters
        self.declare_parameter('avoidance_turn_duration', 1.0)   # Seconds to turn before driving
        self.declare_parameter('avoidance_drive_duration', 1.5)  # Seconds to drive forward after turning
        self.declare_parameter('avoidance_drive_speed', 0.12)    # Forward speed during drive-through
        
        # Smooth speed control parameters (can be toggled on/off)
        self.declare_parameter('smooth_speed_enabled', True)  # Enable advanced smooth speed control
        self.declare_parameter('min_linear_velocity', 0.03)   # Min speed to overcome friction
        self.declare_parameter('max_linear_accel', 0.05)      # Max acceleration (m/s per cycle)
        self.declare_parameter('max_linear_decel', 0.025)     # Max deceleration - gentler than accel
        self.declare_parameter('max_angular_accel', 0.1)      # Max angular acceleration
        self.declare_parameter('max_angular_decel', 0.06)     # Max angular deceleration - gentler
        self.declare_parameter('decel_distance', 0.5)         # Start decelerating this far from target
        
        # Get parameters
        self.target_distance = self.get_parameter('target_distance').value
        self.min_distance = self.get_parameter('min_distance').value
        self.max_tracking_distance = self.get_parameter('max_tracking_distance').value
        self.max_linear_vel = self.get_parameter('max_linear_velocity').value
        self.max_angular_vel = self.get_parameter('max_angular_velocity').value
        self.linear_kp = self.get_parameter('linear_kp').value
        self.angular_kp = self.get_parameter('angular_kp').value
        self.velocity_smoothing = self.get_parameter('velocity_smoothing').value
        self.lost_timeout = self.get_parameter('lost_timeout').value
        self.image_width = self.get_parameter('image_width').value
        self.center_deadzone = self.get_parameter('center_deadzone').value
        self.cmd_timeout = self.get_parameter('cmd_timeout').value
        self.avoidance_turn_speed = self.get_parameter('avoidance_turn_speed').value
        self.avoidance_timeout = self.get_parameter('avoidance_timeout').value
        
        # Multi-phase avoidance parameters
        self.avoidance_turn_duration = self.get_parameter('avoidance_turn_duration').value
        self.avoidance_drive_duration = self.get_parameter('avoidance_drive_duration').value
        self.avoidance_drive_speed = self.get_parameter('avoidance_drive_speed').value
        
        # Smooth speed control parameters
        self.smooth_speed_enabled = self.get_parameter('smooth_speed_enabled').value
        self.min_linear_vel = self.get_parameter('min_linear_velocity').value
        self.max_linear_accel = self.get_parameter('max_linear_accel').value
        self.max_linear_decel = self.get_parameter('max_linear_decel').value
        self.max_angular_accel = self.get_parameter('max_angular_accel').value
        self.max_angular_decel = self.get_parameter('max_angular_decel').value
        self.decel_distance = self.get_parameter('decel_distance').value
        
        # State
        self.state = RobotState.IDLE
        self.person_detected = False
        self.person_distance = 0.0
        self.person_center_x = self.image_width / 2
        self.person_3d_position: Optional[Point] = None
        self.obstacle_detected = False
        self.obstacle_info: Dict = {}  # Detailed obstacle info
        self.avoidance_direction = 'left'  # Direction to turn when avoiding
        self.avoidance_start_time = 0.0  # When avoidance started
        self.avoidance_turn_direction = 1.0  # Store turn direction (1.0=left, -1.0=right)
        self.drive_through_start_time = 0.0  # When drive-through phase started
        self.last_detection_time = 0.0
        self.last_cmd_time = time.time()
        
        # Shutdown flag for safety
        self.is_shutting_down = False
        
        # Velocity smoothing
        self.current_linear = 0.0
        self.current_angular = 0.0
        
        # Subscribers
        self.distance_sub = self.create_subscription(
            Float32,
            '/human_follower/person_distance',
            self.distance_callback,
            10
        )
        
        self.center_sub = self.create_subscription(
            Point,
            '/human_follower/person_center',
            self.center_callback,
            10
        )
        
        self.position_sub = self.create_subscription(
            Point,
            '/human_follower/person_3d_position',
            self.position_callback,
            10
        )
        
        self.detected_sub = self.create_subscription(
            Bool,
            '/human_follower/person_detected',
            self.detected_callback,
            10
        )
        
        self.gesture_sub = self.create_subscription(
            String,
            '/human_follower/gesture_command',
            self.gesture_callback,
            10
        )
        
        self.obstacle_sub = self.create_subscription(
            Bool,
            '/human_follower/obstacle_detected',
            self.obstacle_callback,
            10
        )
        
        # Subscribe to detailed obstacle info for intelligent avoidance
        self.obstacle_info_sub = self.create_subscription(
            String,
            '/human_follower/obstacle_info',
            self.obstacle_info_callback,
            10
        )
        
        # Publisher
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/commands/velocity',
            10
        )
        
        self.state_pub = self.create_publisher(
            String,
            '/human_follower/robot_state',
            10
        )
        
        # Control loop timer (20Hz)
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        # Register shutdown callback for safety
        self.context.on_shutdown(self.on_shutdown)
        
        self.get_logger().info('Robot Controller Node initialized')
        self.get_logger().info(f'Target distance: {self.target_distance}m, Min distance: {self.min_distance}m')
        self.get_logger().info('Waiting for START gesture (raise RIGHT hand)...')
    
    def distance_callback(self, msg: Float32):
        """Receive distance to person."""
        self.person_distance = msg.data
    
    def center_callback(self, msg: Point):
        """Receive person center position."""
        self.person_center_x = msg.x
        self.last_detection_time = time.time()
    
    def position_callback(self, msg: Point):
        """Receive 3D position of person."""
        self.person_3d_position = msg
    
    def detected_callback(self, msg: Bool):
        """Receive detection status."""
        self.person_detected = msg.data
        if msg.data:
            self.last_detection_time = time.time()
            self.last_cmd_time = time.time()  # Reset watchdog on valid detection
    
    def gesture_callback(self, msg: String):
        """Handle gesture commands."""
        gesture = msg.data
        
        if gesture == "START":
            if self.state in [RobotState.IDLE, RobotState.STOPPED, RobotState.WAITING]:
                self.state = RobotState.FOLLOWING
                self.get_logger().info('Gesture: START - Beginning to follow')
        
        elif gesture == "STOP":
            if self.state == RobotState.FOLLOWING:
                self.state = RobotState.STOPPED
                self.get_logger().info('Gesture: STOP - Stopping')
                self.stop_robot()
    
    def obstacle_callback(self, msg: Bool):
        """Receive obstacle detection status."""
        prev_obstacle = self.obstacle_detected
        self.obstacle_detected = msg.data
        
        # Debug logging for obstacle state changes
        if msg.data and not prev_obstacle:
            self.get_logger().warn('OBSTACLE DETECTED - Initiating avoidance!')
        elif not msg.data and prev_obstacle:
            self.get_logger().info('Obstacle cleared')
    
    def obstacle_info_callback(self, msg: String):
        """Receive detailed obstacle information for intelligent avoidance."""
        try:
            self.obstacle_info = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error('Failed to parse obstacle info')
    
    def _determine_avoidance_direction(self):
        """Determine the best direction to avoid obstacle based on obstacle info and person position."""
        # Get distances from obstacle info
        left_dist = self.obstacle_info.get('left_distance', 0)
        right_dist = self.obstacle_info.get('right_distance', 0)
        best_dir = self.obstacle_info.get('best_direction', 'left')
        
        # If person is visible, prefer turning toward the person
        if self.person_detected and self.person_3d_position is not None:
            person_y = self.person_3d_position.y  # Positive = person on left
            
            # If person is on left and left is clear, go left
            if person_y > 0 and left_dist > self.obstacle_info.get('stop_distance', 0.5):
                self.avoidance_direction = 'left'
            # If person is on right and right is clear, go right
            elif person_y < 0 and right_dist > self.obstacle_info.get('stop_distance', 0.5):
                self.avoidance_direction = 'right'
            else:
                # Use best direction from obstacle detector
                self.avoidance_direction = best_dir if best_dir != 'blocked' else 'left'
        else:
            # No person visible, use obstacle detector's recommendation
            self.avoidance_direction = best_dir if best_dir not in ['blocked', 'front'] else 'left'
        
        self.get_logger().info(f'Avoidance direction set to: {self.avoidance_direction} (left:{left_dist:.2f}m, right:{right_dist:.2f}m)')
    
    def _calculate_avoidance_turn(self) -> float:
        """
        Calculate turn direction for obstacle avoidance while keeping person in view.
        Returns: positive for left turn, negative for right turn
        """
        # Get obstacle distances
        left_dist = self.obstacle_info.get('left_distance', self.max_tracking_distance)
        right_dist = self.obstacle_info.get('right_distance', self.max_tracking_distance)
        stop_dist = self.obstacle_info.get('stop_distance', 0.5)
        
        # Default to predetermined avoidance direction
        turn_direction = 1.0 if self.avoidance_direction == 'left' else -1.0
        
        # If person is visible, try to turn toward them if that direction is clear
        if self.person_detected:
            if self.person_3d_position is not None:
                person_y = self.person_3d_position.y
                
                # Person is on left (positive y) and left is clear
                if person_y > 0.1 and left_dist > stop_dist:
                    turn_direction = 1.0  # Turn left
                # Person is on right (negative y) and right is clear
                elif person_y < -0.1 and right_dist > stop_dist:
                    turn_direction = -1.0  # Turn right
            else:
                # Use image center to determine person position
                image_center = self.image_width / 2
                if self.person_center_x < image_center - 50 and left_dist > stop_dist:
                    turn_direction = 1.0  # Person on left, turn left
                elif self.person_center_x > image_center + 50 and right_dist > stop_dist:
                    turn_direction = -1.0  # Person on right, turn right
        
        return turn_direction

    def on_shutdown(self):
        """Called when ROS context is shutting down."""
        self.get_logger().info('Shutdown requested - stopping robot')
        self.is_shutting_down = True
        self.emergency_stop()
    
    def emergency_stop(self):
        """Send multiple stop commands to ensure robot stops."""
        self.get_logger().warn('EMERGENCY STOP - Sending multiple stop commands')
        cmd = Twist()  # All zeros
        # Send multiple times to ensure delivery
        for i in range(10):
            try:
                self.cmd_vel_pub.publish(cmd)
                time.sleep(0.02)  # Small delay between publishes
            except Exception as e:
                self.get_logger().error(f'Failed to publish stop command: {e}')
                break
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.get_logger().info('Emergency stop commands sent')
    
    def stop_robot(self):
        """Send stop command."""
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        self.current_linear = 0.0
        self.current_angular = 0.0
    
    def calculate_angular_velocity(self) -> float:
        """
        Calculate angular velocity to center person in frame.
        Turns left/right based on person's horizontal offset.
        Uses 3D position if available, otherwise image center.
        """
        if self.person_3d_position is not None:
            # Use 3D position (y is lateral offset in robot frame)
            # In ROS convention: positive y is LEFT
            # We want: person on left -> turn left (positive angular.z)
            lateral_offset = self.person_3d_position.y
            angular = self.angular_kp * lateral_offset
        else:
            # Use image center - calculate offset from center
            image_center = self.image_width / 2
            offset = (image_center - self.person_center_x) / image_center  # Normalized -1 to 1
            
            # Apply deadzone to prevent jitter when person is centered
            if abs(offset) < self.center_deadzone:
                return 0.0
            
            # Positive offset = person is to the left = turn left (positive angular)
            angular = self.angular_kp * offset
        
        # Clamp
        angular = max(-self.max_angular_vel, min(self.max_angular_vel, angular))
        
        return angular
    
    def calculate_linear_velocity(self) -> float:
        """
        Calculate linear velocity based on distance to person.
        Uses either simple proportional control or advanced distance-based profile.
        """
        if self.person_distance <= 0:
            return 0.0
        
        # Person is closer than minimum - stop
        if self.person_distance < self.min_distance:
            return 0.0
        
        # Person is at or within target distance - no need to move
        if self.person_distance <= self.target_distance:
            return 0.0
        
        # Person is too far - stop tracking
        if self.person_distance > self.max_tracking_distance:
            return 0.0
        
        if self.smooth_speed_enabled:
            # Advanced: distance-based speed with anticipatory deceleration
            return self._calculate_smooth_linear_velocity()
        else:
            # Simple: proportional control (original behavior)
            error = self.person_distance - self.target_distance
            linear = self.linear_kp * error
            return min(linear, self.max_linear_vel)
    
    def _calculate_smooth_linear_velocity(self) -> float:
        """
        Calculate speed using smooth distance-based profile with anticipatory deceleration.
        """
        # Calculate base speed from distance profile
        base_speed = self._calculate_distance_based_speed(self.person_distance)
        
        # Apply anticipatory deceleration when approaching target
        distance_to_target = self.person_distance - self.target_distance
        
        if distance_to_target < self.decel_distance and distance_to_target > 0:
            # We're in the deceleration zone - reduce speed smoothly
            # Use quadratic curve for natural deceleration feel
            decel_factor = (distance_to_target / self.decel_distance) ** 2
            base_speed = base_speed * decel_factor
            
            # Ensure minimum speed to keep moving (avoid stalling)
            if base_speed > 0 and base_speed < self.min_linear_vel:
                if distance_to_target > 0.1:
                    base_speed = self.min_linear_vel
                else:
                    base_speed = 0.0  # Close enough, stop
        
        return base_speed
    
    def _calculate_distance_based_speed(self, distance: float) -> float:
        """
        Calculate speed based on distance using smooth non-linear profile.
        
        Speed Profile (sqrt curve):
        - At target_distance: 0 (no movement needed)
        - At target + 0.5m: ~35% of max speed (gentle start)
        - At target + 1.0m: ~50% of max speed (moderate)
        - At target + 2.0m+: ~80-100% of max speed (catch up)
        """
        error = distance - self.target_distance
        
        if error <= 0:
            return 0.0
        
        # Distance range for speed scaling
        speed_ramp_distance = self.max_tracking_distance - self.target_distance
        
        if speed_ramp_distance <= 0:
            return 0.0
        
        # Normalize error to 0-1 range
        normalized_error = min(error / speed_ramp_distance, 1.0)
        
        # Apply sqrt profile: gentle acceleration close, faster when far
        speed_factor = math.sqrt(normalized_error)
        
        # Calculate speed from profile
        base_speed = speed_factor * self.max_linear_vel
        
        # Apply minimum speed threshold
        if base_speed > 0 and base_speed < self.min_linear_vel:
            base_speed = self.min_linear_vel
        
        return min(base_speed, self.max_linear_vel)
    
    def smooth_velocity(self, target_linear: float, target_angular: float) -> tuple:
        """
        Apply smoothing with optional asymmetric acceleration/deceleration limits.
        Deceleration is gentler for smoother stops.
        """
        alpha = self.velocity_smoothing
        
        # First apply low-pass filter for overall smoothing
        smoothed_linear = alpha * target_linear + (1 - alpha) * self.current_linear
        smoothed_angular = alpha * target_angular + (1 - alpha) * self.current_angular
        
        if self.smooth_speed_enabled:
            # Advanced: asymmetric rate limiting (different for accel vs decel)
            smoothed_linear = self._apply_asymmetric_rate_limit(
                smoothed_linear, self.current_linear,
                self.max_linear_accel, self.max_linear_decel
            )
            smoothed_angular = self._apply_angular_rate_limit(
                smoothed_angular, self.current_angular
            )
        else:
            # Simple: symmetric acceleration limiting
            linear_diff = smoothed_linear - self.current_linear
            if abs(linear_diff) > self.max_linear_accel:
                smoothed_linear = self.current_linear + self.max_linear_accel * (1 if linear_diff > 0 else -1)
            
            angular_diff = smoothed_angular - self.current_angular
            if abs(angular_diff) > self.max_angular_accel:
                smoothed_angular = self.current_angular + self.max_angular_accel * (1 if angular_diff > 0 else -1)
        
        # Update current velocities
        self.current_linear = smoothed_linear
        self.current_angular = smoothed_angular
        
        # Dead zone for very small values (prevents motor whine)
        if abs(self.current_linear) < 0.01:
            self.current_linear = 0.0
        if abs(self.current_angular) < 0.01:
            self.current_angular = 0.0
        
        return self.current_linear, self.current_angular
    
    def _apply_asymmetric_rate_limit(self, target: float, current: float,
                                      max_accel: float, max_decel: float) -> float:
        """
        Apply rate limiting with different limits for acceleration vs deceleration.
        """
        diff = target - current
        
        if diff > 0:
            # Accelerating (speeding up)
            max_change = max_accel
        else:
            # Decelerating (slowing down) - use gentler limit
            max_change = max_decel
        
        if abs(diff) > max_change:
            return current + max_change * (1 if diff > 0 else -1)
        return target
    
    def _apply_angular_rate_limit(self, target: float, current: float) -> float:
        """
        Apply angular rate limiting with asymmetric accel/decel.
        """
        diff = target - current
        
        # Determine if we're accelerating or decelerating the rotation
        # Accelerating = increasing magnitude of rotation
        # Decelerating = decreasing magnitude or changing direction
        if abs(target) > abs(current):
            # Increasing rotation speed
            max_change = self.max_angular_accel
        else:
            # Decreasing rotation speed or stopping
            max_change = self.max_angular_decel
        
        if abs(diff) > max_change:
            return current + max_change * (1 if diff > 0 else -1)
        return target
        
        # Dead zone for small values
        if abs(self.current_linear) < 0.01:
            self.current_linear = 0.0
        if abs(self.current_angular) < 0.01:
            self.current_angular = 0.0
        
        return self.current_linear, self.current_angular
    
    def control_loop(self):
        """Main control loop - runs at 20Hz."""
        # Safety check: if shutting down, stop immediately
        if self.is_shutting_down:
            self.stop_robot()
            return
        
        current_time = time.time()
        
        # Safety watchdog: stop if no recent detection updates (sensor may have died)
        time_since_last_cmd = current_time - self.last_cmd_time
        if self.state == RobotState.FOLLOWING and time_since_last_cmd > self.cmd_timeout:
            self.get_logger().warn(f'Safety watchdog: No detection for {time_since_last_cmd:.1f}s - stopping')
            self.stop_robot()
        
        # Update last command time
        self.last_cmd_time = current_time
        
        # Publish state
        state_msg = String()
        state_msg.data = self.state.value
        self.state_pub.publish(state_msg)
        
        # State machine
        if self.state == RobotState.IDLE:
            self.stop_robot()
            return
        
        if self.state == RobotState.STOPPED:
            self.stop_robot()
            return
        
        if self.state == RobotState.FOLLOWING:
            # Check for obstacles FIRST (highest priority for safety)
            if self.obstacle_detected:
                # Determine best avoidance direction based on obstacle info
                self._determine_avoidance_direction()
                self.avoidance_start_time = current_time
                self.state = RobotState.OBSTACLE_AVOID
                self.get_logger().warn(f'Obstacle detected - avoiding to the {self.avoidance_direction}')
                return
            
            # Check for person loss
            time_since_detection = current_time - self.last_detection_time
            
            if not self.person_detected or time_since_detection > self.lost_timeout:
                self.state = RobotState.LOST
                self.get_logger().warn('Lost person - searching...')
                self.stop_robot()
                return
            
            # Check if person is too far
            if self.person_distance > self.max_tracking_distance:
                self.get_logger().warn(f'Person too far ({self.person_distance:.2f}m) - stopping forward motion')
                # Still turn to track, but don't move forward
                angular = self.calculate_angular_velocity()
                angular, _ = self.smooth_velocity(0.0, angular)
                
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = angular
                self.cmd_vel_pub.publish(cmd)
                return
            
            # Calculate velocities
            linear = self.calculate_linear_velocity()
            angular = self.calculate_angular_velocity()
            
            # Apply smoothing
            linear, angular = self.smooth_velocity(linear, angular)
            
            # Reduce linear speed when turning sharply (prioritize turning)
            if abs(angular) > 0.1:
                turn_reduction = 1.0 - min(abs(angular) / self.max_angular_vel * 0.7, 0.7)
                linear *= turn_reduction
            
            # Publish command
            cmd = Twist()
            cmd.linear.x = linear
            cmd.angular.z = angular
            self.cmd_vel_pub.publish(cmd)
        
        elif self.state == RobotState.LOST:
            # Try to reacquire
            if self.person_detected:
                self.state = RobotState.FOLLOWING
                self.get_logger().info('Person reacquired - following')
            else:
                # Could implement search behavior here (e.g., slow rotation)
                self.stop_robot()
        
        elif self.state == RobotState.OBSTACLE_AVOID:
            # Phase 1: Turn to avoid obstacle
            avoidance_duration = current_time - self.avoidance_start_time
            
            # Check timeout
            if avoidance_duration > self.avoidance_timeout:
                self.get_logger().warn('Avoidance timeout - stopping')
                self.stop_robot()
                self.state = RobotState.STOPPED
                return
            
            # Get best direction from obstacle info
            best_direction = self.obstacle_info.get('best_direction', 'blocked')
            
            if best_direction == 'blocked':
                # All directions blocked - stop and wait
                self.stop_robot()
                if int(current_time * 10) % 20 == 0:
                    self.get_logger().warn('All directions blocked - waiting')
                return
            
            # Calculate and store turn direction
            if self.person_detected:
                self.avoidance_turn_direction = self._calculate_avoidance_turn()
            else:
                self.avoidance_turn_direction = 1.0 if self.avoidance_direction == 'left' else -1.0
            
            # Turn in place
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = self.avoidance_turn_direction * self.avoidance_turn_speed
            _, cmd.angular.z = self.smooth_velocity(0.0, cmd.angular.z)
            self.cmd_vel_pub.publish(cmd)
            
            if int(current_time * 10) % 10 == 0:
                turn_dir_str = 'left' if self.avoidance_turn_direction > 0 else 'right'
                self.get_logger().info(f'Avoiding obstacle - turning {turn_dir_str} ({avoidance_duration:.1f}s)')
            
            # After turning for minimum duration AND obstacle cleared, move to drive-through
            if avoidance_duration >= self.avoidance_turn_duration and not self.obstacle_detected:
                self.state = RobotState.OBSTACLE_DRIVE_THROUGH
                self.drive_through_start_time = current_time
                self.get_logger().info('Obstacle cleared - driving through before resuming')
        
        elif self.state == RobotState.OBSTACLE_DRIVE_THROUGH:
            # Phase 2: Drive forward a bit to get past the obstacle
            drive_duration = current_time - self.drive_through_start_time
            
            # If obstacle detected again, go back to avoidance
            if self.obstacle_detected:
                self.state = RobotState.OBSTACLE_AVOID
                self.avoidance_start_time = current_time
                self._determine_avoidance_direction()
                self.get_logger().warn('Obstacle detected during drive-through - re-avoiding')
                return
            
            # Drive forward with slight turn in same direction (arc around obstacle)
            cmd = Twist()
            cmd.linear.x = self.avoidance_drive_speed
            # Maintain slight turn in same direction to arc around obstacle
            cmd.angular.z = self.avoidance_turn_direction * self.avoidance_turn_speed * 0.3
            
            cmd.linear.x, cmd.angular.z = self.smooth_velocity(cmd.linear.x, cmd.angular.z)
            self.cmd_vel_pub.publish(cmd)
            
            if int(current_time * 10) % 10 == 0:
                self.get_logger().info(f'Drive-through: {drive_duration:.1f}s / {self.avoidance_drive_duration:.1f}s')
            
            # After driving for set duration, resume following
            if drive_duration >= self.avoidance_drive_duration:
                self.state = RobotState.FOLLOWING
                self.get_logger().info('Drive-through complete - resuming normal following')


def main(args=None):
    rclpy.init(args=args)
    node = RobotControllerNode()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        node.get_logger().warn(f'Received signal {sig} - initiating emergency stop')
        node.is_shutting_down = True
        node.emergency_stop()
        raise SystemExit
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, ExternalShutdownException):
        pass
    except Exception as e:
        node.get_logger().error(f'Unexpected error: {e}')
    finally:
        # Ensure robot is stopped on any exit
        node.get_logger().info('Shutting down - ensuring robot is stopped')
        try:
            if not node.is_shutting_down:
                node.emergency_stop()
        except Exception:
            pass
        
        try:
            node.destroy_node()
        except Exception:
            pass
        
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
