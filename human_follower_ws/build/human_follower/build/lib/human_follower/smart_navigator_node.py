#!/usr/bin/env python3
"""
Smart Navigator Node - Nav2 Integration for Human Following
Uses Nav2's navigation stack for intelligent obstacle avoidance and path planning.

When an obstacle is detected, this node:
1. Gets the current person position as the goal
2. Uses Nav2 to compute an optimal path around obstacles
3. Follows the path while continuously tracking the person
4. Handles small obstacles using local costmap with high resolution

Published Topics:
    /human_follower/nav_status (String) - Navigation status
    /human_follower/planned_path (Path) - Current planned path for visualization
    /human_follower/nav_cmd_vel (Twist) - Navigation velocity commands

Subscribed Topics:
    /human_follower/person_3d_position (Point) - Person's 3D position
    /human_follower/person_detected (Bool) - Detection status
    /human_follower/obstacle_detected (Bool) - Obstacle detection
    /human_follower/obstacle_info (String) - Detailed obstacle information
    /odom (Odometry) - Robot odometry
    /scan (LaserScan) - Laser scan for costmap (if available)
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Twist, Point, PoseStamped, Pose, TransformStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String, Bool, Float32
from cv_bridge import CvBridge

# Nav2 imports - optional, will use local planning if not available
try:
    from nav2_msgs.action import NavigateToPose, ComputePathToPose, FollowPath
    from nav2_msgs.srv import ClearEntireCostmap
    NAV2_AVAILABLE = True
except ImportError:
    NAV2_AVAILABLE = False
    print("Warning: nav2_msgs not available. Smart navigation will use local planning only.")

import numpy as np
import math
import json
import time
from enum import Enum
from typing import Optional, Tuple, List
from dataclasses import dataclass
import tf2_ros


class NavState(Enum):
    IDLE = "IDLE"
    DIRECT_FOLLOW = "DIRECT_FOLLOW"  # Simple following (no obstacle)
    PLANNING = "PLANNING"  # Computing path around obstacle
    NAVIGATING = "NAVIGATING"  # Following computed path
    RECOVERY = "RECOVERY"  # Stuck recovery behavior
    WAITING = "WAITING"  # Waiting for obstacle to clear


@dataclass
class NavigationGoal:
    """Represents a navigation goal point."""
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    timestamp: float = 0.0


class SmartNavigatorNode(Node):
    def __init__(self):
        super().__init__('smart_navigator')
        
        # Callback groups for async operations
        self.action_cb_group = ReentrantCallbackGroup()
        self.service_cb_group = MutuallyExclusiveCallbackGroup()
        
        # Declare parameters
        self._declare_parameters()
        
        # Get parameters
        self._load_parameters()
        
        # CV Bridge for depth processing
        self.bridge = CvBridge()
        
        # TF2 for coordinate transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # State
        self.nav_state = NavState.IDLE
        self.smart_nav_enabled = self.enabled
        self.person_position: Optional[Point] = None
        self.person_detected = False
        self.obstacle_detected = False
        self.obstacle_info = {}
        self.current_pose: Optional[Pose] = None
        self.current_path: Optional[Path] = None
        self.current_goal: Optional[NavigationGoal] = None
        
        # Timing
        self.last_replan_time = 0.0
        self.last_goal_update_time = 0.0
        self.navigation_start_time = 0.0
        self.stuck_start_time = 0.0
        self.recovery_attempts = 0
        
        # Path following state
        self.path_index = 0
        self.is_navigating = False
        
        # Local costmap from depth
        self.local_costmap: Optional[np.ndarray] = None
        self.costmap_origin = (0.0, 0.0)
        self.costmap_size = (100, 100)  # cells
        
        # QoS profiles
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribers
        self.position_sub = self.create_subscription(
            Point,
            '/human_follower/person_3d_position',
            self.person_position_callback,
            10
        )
        
        self.detected_sub = self.create_subscription(
            Bool,
            '/human_follower/person_detected',
            self.person_detected_callback,
            10
        )
        
        self.obstacle_sub = self.create_subscription(
            Bool,
            '/human_follower/obstacle_detected',
            self.obstacle_callback,
            10
        )
        
        self.obstacle_info_sub = self.create_subscription(
            String,
            '/human_follower/obstacle_info',
            self.obstacle_info_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            sensor_qos
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            sensor_qos
        )
        
        # Publishers
        self.nav_status_pub = self.create_publisher(
            String,
            '/human_follower/nav_status',
            10
        )
        
        self.nav_cmd_pub = self.create_publisher(
            Twist,
            '/human_follower/nav_cmd_vel',
            10
        )
        
        self.path_pub = self.create_publisher(
            Path,
            '/human_follower/planned_path',
            10
        )
        
        self.costmap_pub = self.create_publisher(
            OccupancyGrid,
            '/human_follower/local_costmap',
            10
        )
        
        self.nav_active_pub = self.create_publisher(
            Bool,
            '/human_follower/nav_active',
            10
        )
        
        # Nav2 Action Clients (only if Nav2 is available)
        self.nav_to_pose_client = None
        self.compute_path_client = None
        self.clear_costmap_client = None
        
        if NAV2_AVAILABLE:
            self.nav_to_pose_client = ActionClient(
                self,
                NavigateToPose,
                'navigate_to_pose',
                callback_group=self.action_cb_group
            )
            
            self.compute_path_client = ActionClient(
                self,
                ComputePathToPose,
                'compute_path_to_pose',
                callback_group=self.action_cb_group
            )
            
            # Service clients
            self.clear_costmap_client = self.create_client(
                ClearEntireCostmap,
                '/local_costmap/clear_entirely_local_costmap',
                callback_group=self.service_cb_group
            )
        else:
            self.get_logger().warn('Nav2 not available - using local planning only')
            self.use_nav2_planner = False
        
        # Control loop timer (20Hz)
        self.control_timer = self.create_timer(0.05, self.control_loop)
        
        # Path planning timer (slower rate)
        self.planning_timer = self.create_timer(self.replan_interval, self.planning_loop)
        
        # Status publishing timer
        self.status_timer = self.create_timer(0.1, self.publish_status)
        
        self.get_logger().info('Smart Navigator Node initialized')
        self.get_logger().info(f'Smart navigation enabled: {self.smart_nav_enabled}')
    
    def _declare_parameters(self):
        """Declare all parameters."""
        # Main enable/disable
        self.declare_parameter('enabled', True)
        self.declare_parameter('use_nav2_planner', True)
        self.declare_parameter('replan_on_obstacle', True)
        
        # Costmap settings
        self.declare_parameter('costmap_resolution', 0.05)
        self.declare_parameter('obstacle_range', 2.5)
        self.declare_parameter('raytrace_range', 3.0)
        self.declare_parameter('inflation_radius', 0.35)
        self.declare_parameter('robot_radius', 0.18)
        
        # Recovery settings
        self.declare_parameter('recovery_enabled', True)
        self.declare_parameter('recovery_spin_dist', 1.57)
        self.declare_parameter('recovery_backup_dist', 0.3)
        self.declare_parameter('max_recovery_attempts', 3)
        
        # Path following settings
        self.declare_parameter('goal_tolerance_xy', 0.25)
        self.declare_parameter('goal_tolerance_yaw', 0.2)
        self.declare_parameter('lookahead_distance', 0.6)
        
        # Timing settings
        self.declare_parameter('replan_interval', 0.5)
        self.declare_parameter('min_obstacle_size', 0.05)
        self.declare_parameter('planning_timeout', 2.0)
        self.declare_parameter('stuck_timeout', 5.0)
        
        # Velocity limits
        self.declare_parameter('max_linear_velocity', 0.2)
        self.declare_parameter('max_angular_velocity', 0.5)
        self.declare_parameter('nav_linear_velocity', 0.15)
        self.declare_parameter('nav_angular_velocity', 0.4)
    
    def _load_parameters(self):
        """Load all parameters."""
        self.enabled = self.get_parameter('enabled').value
        self.use_nav2_planner = self.get_parameter('use_nav2_planner').value
        self.replan_on_obstacle = self.get_parameter('replan_on_obstacle').value
        
        self.costmap_resolution = self.get_parameter('costmap_resolution').value
        self.obstacle_range = self.get_parameter('obstacle_range').value
        self.raytrace_range = self.get_parameter('raytrace_range').value
        self.inflation_radius = self.get_parameter('inflation_radius').value
        self.robot_radius = self.get_parameter('robot_radius').value
        
        self.recovery_enabled = self.get_parameter('recovery_enabled').value
        self.recovery_spin_dist = self.get_parameter('recovery_spin_dist').value
        self.recovery_backup_dist = self.get_parameter('recovery_backup_dist').value
        self.max_recovery_attempts = self.get_parameter('max_recovery_attempts').value
        
        self.goal_tolerance_xy = self.get_parameter('goal_tolerance_xy').value
        self.goal_tolerance_yaw = self.get_parameter('goal_tolerance_yaw').value
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        
        self.replan_interval = self.get_parameter('replan_interval').value
        self.min_obstacle_size = self.get_parameter('min_obstacle_size').value
        self.planning_timeout = self.get_parameter('planning_timeout').value
        self.stuck_timeout = self.get_parameter('stuck_timeout').value
        
        self.max_linear_velocity = self.get_parameter('max_linear_velocity').value
        self.max_angular_velocity = self.get_parameter('max_angular_velocity').value
        self.nav_linear_velocity = self.get_parameter('nav_linear_velocity').value
        self.nav_angular_velocity = self.get_parameter('nav_angular_velocity').value
    
    def person_position_callback(self, msg: Point):
        """Update person's 3D position."""
        self.person_position = msg
        self.last_goal_update_time = time.time()
    
    def person_detected_callback(self, msg: Bool):
        """Update person detection status."""
        self.person_detected = msg.data
    
    def obstacle_callback(self, msg: Bool):
        """Handle obstacle detection."""
        prev_obstacle = self.obstacle_detected
        self.obstacle_detected = msg.data
        
        # Trigger replanning when obstacle newly detected
        if msg.data and not prev_obstacle and self.smart_nav_enabled:
            if self.replan_on_obstacle:
                self.get_logger().info('Obstacle detected - triggering smart navigation')
                self.trigger_replan()
    
    def obstacle_info_callback(self, msg: String):
        """Receive detailed obstacle information."""
        try:
            self.obstacle_info = json.loads(msg.data)
        except json.JSONDecodeError:
            pass
    
    def odom_callback(self, msg: Odometry):
        """Update robot's current pose from odometry."""
        self.current_pose = msg.pose.pose
    
    def depth_callback(self, msg: Image):
        """Process depth image to update local costmap for small obstacle detection."""
        if not self.smart_nav_enabled:
            return
        
        try:
            if msg.encoding == '16UC1':
                depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')
            elif msg.encoding == '32FC1':
                depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')
                depth_image = (depth_image * 1000).astype(np.uint16)
            else:
                depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            
            # Update local costmap from depth
            self._update_costmap_from_depth(depth_image)
            
        except Exception as e:
            self.get_logger().error(f'Depth processing error: {e}')
    
    def _update_costmap_from_depth(self, depth_image: np.ndarray):
        """
        Create/update local costmap from depth image.
        This helps detect small obstacles that might be missed by laser scanners.
        """
        h, w = depth_image.shape[:2]
        
        # Convert to meters
        depth_meters = depth_image.astype(np.float32) * 0.001  # mm to m
        
        # Initialize costmap if needed
        costmap_cells = int(self.obstacle_range * 2 / self.costmap_resolution)
        if self.local_costmap is None or self.local_costmap.shape[0] != costmap_cells:
            self.local_costmap = np.zeros((costmap_cells, costmap_cells), dtype=np.int8)
            self.costmap_size = (costmap_cells, costmap_cells)
        
        # Clear costmap
        self.local_costmap.fill(0)
        
        # Camera parameters (typical values for depth camera)
        fx, fy = 525.0, 525.0
        cx, cy = w / 2, h / 2
        
        # Process depth to find obstacles
        # Focus on lower half of image (ground-level obstacles)
        roi_start = int(h * 0.4)
        roi_end = h
        
        for v in range(roi_start, roi_end, 4):  # Skip pixels for performance
            for u in range(0, w, 4):
                z = depth_meters[v, u]
                
                # Valid depth range
                if z < 0.3 or z > self.obstacle_range:
                    continue
                
                # Project to 3D (camera frame)
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                
                # Transform to robot frame (assuming camera is roughly aligned)
                # x forward, y left in robot frame
                robot_x = z
                robot_y = -x
                
                # Convert to costmap coordinates
                cell_x = int((robot_x + self.obstacle_range) / self.costmap_resolution)
                cell_y = int((robot_y + self.obstacle_range) / self.costmap_resolution)
                
                # Mark as obstacle
                if 0 <= cell_x < self.costmap_size[0] and 0 <= cell_y < self.costmap_size[1]:
                    self.local_costmap[cell_y, cell_x] = 100  # Occupied
        
        # Apply inflation
        self._inflate_costmap()
        
        # Publish costmap for visualization
        self._publish_costmap()
    
    def _inflate_costmap(self):
        """Apply inflation to obstacles in costmap."""
        if self.local_costmap is None:
            return
        
        inflation_cells = int(self.inflation_radius / self.costmap_resolution)
        
        # Create kernel for inflation
        kernel_size = inflation_cells * 2 + 1
        y, x = np.ogrid[-inflation_cells:inflation_cells+1, -inflation_cells:inflation_cells+1]
        kernel = (x*x + y*y <= inflation_cells*inflation_cells).astype(np.float32)
        
        # Apply dilation
        from scipy import ndimage
        try:
            inflated = ndimage.maximum_filter(self.local_costmap.astype(np.float32), 
                                               footprint=kernel)
            self.local_costmap = inflated.astype(np.int8)
        except ImportError:
            # Fallback if scipy not available
            pass
    
    def _publish_costmap(self):
        """Publish local costmap as OccupancyGrid."""
        if self.local_costmap is None:
            return
        
        grid = OccupancyGrid()
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.header.frame_id = 'base_link'
        
        grid.info.resolution = self.costmap_resolution
        grid.info.width = self.costmap_size[0]
        grid.info.height = self.costmap_size[1]
        grid.info.origin.position.x = -self.obstacle_range
        grid.info.origin.position.y = -self.obstacle_range
        grid.info.origin.position.z = 0.0
        
        grid.data = self.local_costmap.flatten().tolist()
        
        self.costmap_pub.publish(grid)
    
    def trigger_replan(self):
        """Trigger path replanning."""
        if not self.smart_nav_enabled or not self.person_detected:
            return
        
        current_time = time.time()
        
        # Rate limit replanning
        if current_time - self.last_replan_time < self.replan_interval:
            return
        
        self.last_replan_time = current_time
        self.nav_state = NavState.PLANNING
        
        # Compute new goal based on person position
        if self.person_position is not None and self.current_pose is not None:
            goal = self._compute_goal_from_person()
            if goal is not None:
                self.current_goal = goal
                self._request_path_to_goal(goal)
    
    def _compute_goal_from_person(self) -> Optional[NavigationGoal]:
        """
        Compute navigation goal based on person's position.
        Goal is set at a safe following distance from the person.
        """
        if self.person_position is None or self.current_pose is None:
            return None
        
        # Person position is in camera/robot frame
        person_x = self.person_position.x  # Distance forward
        person_y = self.person_position.y  # Lateral offset
        person_z = self.person_position.z  # Height (not used for nav)
        
        # Get robot position
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y
        robot_yaw = self._quaternion_to_yaw(self.current_pose.orientation)
        
        # Transform person position to world frame
        world_person_x = robot_x + person_x * math.cos(robot_yaw) - person_y * math.sin(robot_yaw)
        world_person_y = robot_y + person_x * math.sin(robot_yaw) + person_y * math.cos(robot_yaw)
        
        # Compute goal position (1m behind the person along the line from robot to person)
        distance_to_person = math.sqrt(person_x**2 + person_y**2)
        
        if distance_to_person < 0.1:
            return None
        
        # Target distance behind person
        target_offset = min(1.0, distance_to_person - 0.5)  # Stay 0.5m+ from person
        
        # Direction from robot to person
        dx = world_person_x - robot_x
        dy = world_person_y - robot_y
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < 0.1:
            return None
        
        # Goal is along the direction to person, but closer
        goal_x = robot_x + dx * (1.0 - target_offset / dist)
        goal_y = robot_y + dy * (1.0 - target_offset / dist)
        goal_yaw = math.atan2(dy, dx)
        
        return NavigationGoal(
            x=goal_x,
            y=goal_y,
            yaw=goal_yaw,
            timestamp=time.time()
        )
    
    def _quaternion_to_yaw(self, q) -> float:
        """Convert quaternion to yaw angle."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def _yaw_to_quaternion(self, yaw: float):
        """Convert yaw angle to quaternion."""
        from geometry_msgs.msg import Quaternion
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q
    
    def _request_path_to_goal(self, goal: NavigationGoal):
        """Request path computation from Nav2."""
        if not self.use_nav2_planner:
            # Use simple local planning instead
            self._compute_local_path(goal)
            return
        
        # Check if Nav2 action server is available
        if not self.compute_path_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn('Nav2 compute_path server not available, using local planning')
            self._compute_local_path(goal)
            return
        
        # Create goal message
        goal_msg = ComputePathToPose.Goal()
        goal_msg.goal.header.frame_id = 'odom'
        goal_msg.goal.header.stamp = self.get_clock().now().to_msg()
        goal_msg.goal.pose.position.x = goal.x
        goal_msg.goal.pose.position.y = goal.y
        goal_msg.goal.pose.position.z = 0.0
        goal_msg.goal.pose.orientation = self._yaw_to_quaternion(goal.yaw)
        
        # Start pose
        if self.current_pose is not None:
            goal_msg.start.header.frame_id = 'odom'
            goal_msg.start.header.stamp = self.get_clock().now().to_msg()
            goal_msg.start.pose = self.current_pose
        
        # Send goal asynchronously
        self.get_logger().info(f'Requesting path to ({goal.x:.2f}, {goal.y:.2f})')
        
        future = self.compute_path_client.send_goal_async(goal_msg)
        future.add_done_callback(self._path_goal_response_callback)
    
    def _path_goal_response_callback(self, future):
        """Handle path computation goal response."""
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().warn('Path computation rejected')
            self.nav_state = NavState.DIRECT_FOLLOW
            return
        
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._path_result_callback)
    
    def _path_result_callback(self, future):
        """Handle path computation result."""
        result = future.result().result
        
        if result.path.poses:
            self.current_path = result.path
            self.path_index = 0
            self.nav_state = NavState.NAVIGATING
            self.navigation_start_time = time.time()
            self.is_navigating = True
            
            self.get_logger().info(f'Path computed with {len(result.path.poses)} waypoints')
            
            # Publish path for visualization
            self.path_pub.publish(self.current_path)
        else:
            self.get_logger().warn('No valid path found')
            self.nav_state = NavState.DIRECT_FOLLOW
    
    def _compute_local_path(self, goal: NavigationGoal):
        """
        Compute a simple local path when Nav2 is not available.
        Uses local costmap to find obstacle-free waypoints.
        """
        if self.current_pose is None:
            return
        
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = self.get_clock().now().to_msg()
        
        start_x = self.current_pose.position.x
        start_y = self.current_pose.position.y
        
        # Simple path: try to go around obstacle
        # Check which side is clearer from obstacle info
        left_dist = self.obstacle_info.get('left_distance', 2.0)
        right_dist = self.obstacle_info.get('right_distance', 2.0)
        
        # Create waypoints
        num_waypoints = 5
        
        for i in range(num_waypoints):
            t = (i + 1) / num_waypoints
            
            # Interpolate toward goal with lateral offset to avoid obstacle
            waypoint_x = start_x + t * (goal.x - start_x)
            waypoint_y = start_y + t * (goal.y - start_y)
            
            # Add lateral offset in the middle of the path
            if 0.2 < t < 0.8:
                offset = 0.3 if left_dist > right_dist else -0.3
                angle_to_goal = math.atan2(goal.y - start_y, goal.x - start_x)
                waypoint_x += offset * math.sin(angle_to_goal)
                waypoint_y -= offset * math.cos(angle_to_goal)
            
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = waypoint_x
            pose.pose.position.y = waypoint_y
            pose.pose.orientation = self._yaw_to_quaternion(
                math.atan2(goal.y - waypoint_y, goal.x - waypoint_x)
            )
            path.poses.append(pose)
        
        self.current_path = path
        self.path_index = 0
        self.nav_state = NavState.NAVIGATING
        self.navigation_start_time = time.time()
        self.is_navigating = True
        
        self.path_pub.publish(path)
        self.get_logger().info(f'Local path computed with {len(path.poses)} waypoints')
    
    def planning_loop(self):
        """Periodic planning update."""
        if not self.smart_nav_enabled:
            return
        
        # Update path if navigating and person moved significantly
        if self.nav_state == NavState.NAVIGATING and self.person_detected:
            if self.current_goal is not None:
                new_goal = self._compute_goal_from_person()
                if new_goal is not None:
                    # Check if person moved significantly
                    dist = math.sqrt(
                        (new_goal.x - self.current_goal.x)**2 +
                        (new_goal.y - self.current_goal.y)**2
                    )
                    if dist > 0.5:  # Person moved more than 0.5m
                        self.get_logger().info('Person moved - replanning')
                        self.current_goal = new_goal
                        self._request_path_to_goal(new_goal)
    
    def control_loop(self):
        """Main control loop for path following."""
        current_time = time.time()
        
        # Publish navigation active status
        nav_active_msg = Bool()
        nav_active_msg.data = self.is_navigating and self.smart_nav_enabled
        self.nav_active_pub.publish(nav_active_msg)
        
        if not self.smart_nav_enabled:
            self.nav_state = NavState.IDLE
            return
        
        # State machine
        if self.nav_state == NavState.IDLE:
            if self.obstacle_detected and self.person_detected:
                self.trigger_replan()
        
        elif self.nav_state == NavState.DIRECT_FOLLOW:
            # Not navigating - let robot_controller handle it
            self.is_navigating = False
            if self.obstacle_detected:
                self.trigger_replan()
        
        elif self.nav_state == NavState.PLANNING:
            # Waiting for path - check timeout
            if current_time - self.last_replan_time > self.planning_timeout:
                self.get_logger().warn('Planning timeout')
                self.nav_state = NavState.DIRECT_FOLLOW
        
        elif self.nav_state == NavState.NAVIGATING:
            # Follow the path
            cmd = self._follow_path()
            
            if cmd is not None:
                self.nav_cmd_pub.publish(cmd)
            
            # Check if reached goal or obstacle cleared
            if not self.obstacle_detected:
                self.get_logger().info('Obstacle cleared - returning to direct follow')
                self.nav_state = NavState.DIRECT_FOLLOW
                self.is_navigating = False
            
            # Check for stuck condition
            if self._is_stuck():
                if self.recovery_enabled and self.recovery_attempts < self.max_recovery_attempts:
                    self.nav_state = NavState.RECOVERY
                    self.recovery_attempts += 1
                    self.stuck_start_time = current_time
                else:
                    self.get_logger().warn('Stuck with no recovery options')
                    self.nav_state = NavState.WAITING
        
        elif self.nav_state == NavState.RECOVERY:
            cmd = self._execute_recovery()
            if cmd is not None:
                self.nav_cmd_pub.publish(cmd)
            
            # Recovery complete after timeout
            if current_time - self.stuck_start_time > 2.0:
                self.nav_state = NavState.PLANNING
                self.trigger_replan()
        
        elif self.nav_state == NavState.WAITING:
            # Wait for obstacle to clear
            self.is_navigating = False
            if not self.obstacle_detected:
                self.nav_state = NavState.DIRECT_FOLLOW
                self.recovery_attempts = 0
    
    def _follow_path(self) -> Optional[Twist]:
        """
        Follow the current path using pure pursuit controller.
        Returns velocity command or None if path following complete.
        """
        if self.current_path is None or self.current_pose is None:
            return None
        
        if self.path_index >= len(self.current_path.poses):
            self.get_logger().info('Path following complete')
            self.nav_state = NavState.DIRECT_FOLLOW
            self.is_navigating = False
            return None
        
        # Get current position
        robot_x = self.current_pose.position.x
        robot_y = self.current_pose.position.y
        robot_yaw = self._quaternion_to_yaw(self.current_pose.orientation)
        
        # Find lookahead point on path
        lookahead_point = self._find_lookahead_point(robot_x, robot_y)
        
        if lookahead_point is None:
            return None
        
        # Calculate control commands using pure pursuit
        dx = lookahead_point[0] - robot_x
        dy = lookahead_point[1] - robot_y
        
        # Distance to lookahead point
        dist = math.sqrt(dx**2 + dy**2)
        
        # Angle to lookahead point
        angle_to_point = math.atan2(dy, dx)
        angle_error = self._normalize_angle(angle_to_point - robot_yaw)
        
        # Pure pursuit curvature
        curvature = 2.0 * math.sin(angle_error) / max(dist, 0.1)
        
        # Calculate velocities
        cmd = Twist()
        
        # Linear velocity - reduce when turning
        cmd.linear.x = self.nav_linear_velocity * (1.0 - min(abs(angle_error) / math.pi, 0.5))
        cmd.linear.x = max(cmd.linear.x, 0.05)  # Minimum velocity
        
        # Angular velocity from curvature
        cmd.angular.z = curvature * cmd.linear.x
        cmd.angular.z = max(-self.nav_angular_velocity, 
                           min(self.nav_angular_velocity, cmd.angular.z))
        
        return cmd
    
    def _find_lookahead_point(self, robot_x: float, robot_y: float) -> Optional[Tuple[float, float]]:
        """Find lookahead point on path."""
        if self.current_path is None:
            return None
        
        # Update path index to closest point
        min_dist = float('inf')
        closest_idx = self.path_index
        
        for i in range(self.path_index, len(self.current_path.poses)):
            pose = self.current_path.poses[i]
            dx = pose.pose.position.x - robot_x
            dy = pose.pose.position.y - robot_y
            dist = math.sqrt(dx**2 + dy**2)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        self.path_index = closest_idx
        
        # Find lookahead point
        accumulated_dist = 0.0
        lookahead_idx = closest_idx
        
        for i in range(closest_idx, len(self.current_path.poses) - 1):
            p1 = self.current_path.poses[i].pose.position
            p2 = self.current_path.poses[i + 1].pose.position
            segment_dist = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            
            if accumulated_dist + segment_dist >= self.lookahead_distance:
                # Interpolate
                t = (self.lookahead_distance - accumulated_dist) / max(segment_dist, 0.01)
                x = p1.x + t * (p2.x - p1.x)
                y = p1.y + t * (p2.y - p1.y)
                return (x, y)
            
            accumulated_dist += segment_dist
            lookahead_idx = i + 1
        
        # Use last point if path is shorter than lookahead
        if lookahead_idx < len(self.current_path.poses):
            p = self.current_path.poses[-1].pose.position
            return (p.x, p.y)
        
        return None
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _is_stuck(self) -> bool:
        """Check if robot is stuck."""
        if self.current_path is None or len(self.current_path.poses) == 0:
            return False
        
        # Check if we've been navigating too long without progress
        current_time = time.time()
        
        if current_time - self.navigation_start_time > self.stuck_timeout:
            if self.path_index < len(self.current_path.poses) * 0.3:
                return True
        
        return False
    
    def _execute_recovery(self) -> Twist:
        """Execute recovery behavior (backup and spin)."""
        cmd = Twist()
        
        elapsed = time.time() - self.stuck_start_time
        
        if elapsed < 1.0:
            # Backup
            cmd.linear.x = -0.1
            cmd.angular.z = 0.0
        else:
            # Spin in place
            direction = 1.0 if self.obstacle_info.get('left_distance', 0) > \
                              self.obstacle_info.get('right_distance', 0) else -1.0
            cmd.linear.x = 0.0
            cmd.angular.z = direction * 0.3
        
        return cmd
    
    def publish_status(self):
        """Publish navigation status."""
        status_msg = String()
        status_data = {
            'state': self.nav_state.value,
            'enabled': self.smart_nav_enabled,
            'navigating': self.is_navigating,
            'path_points': len(self.current_path.poses) if self.current_path else 0,
            'path_index': self.path_index,
            'recovery_attempts': self.recovery_attempts
        }
        status_msg.data = json.dumps(status_data)
        self.nav_status_pub.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = SmartNavigatorNode()
    
    # Use multi-threaded executor for action clients
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
