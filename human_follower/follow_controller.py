import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from geometry_msgs.msg import Twist

IDLE = 0
FOLLOW = 1


class FollowController(Node):
    def __init__(self):
        super().__init__("follow_controller")

        # ===== Parameters (GLOBAL TUNING) =====
        self.declare_parameter("min_distance", 1.0)             # Minimum distance to maintain
        self.declare_parameter("linear_kp", 0.6)                # kP of PID Controller for smoothing
        self.declare_parameter("max_linear_speed", 0.2)         # Maximum speed

        self.declare_parameter("offset_left_threshold", -0.1)   # Person's offset to the left (Always < 0)
        self.declare_parameter("offset_right_threshold", 0.1)   # Person's offset to the rigth (Always > 0)
        self.declare_parameter("angular_speed", 0.25)           # Angular speed for turning

        self.min_distance = self.get_parameter("min_distance").value
        self.linear_kp = self.get_parameter("linear_kp").value
        self.max_linear_speed = self.get_parameter("max_linear_speed").value

        self.offset_left_th = self.get_parameter("offset_left_threshold").value
        self.offset_right_th = self.get_parameter("offset_right_threshold").value
        self.angular_speed = self.get_parameter("angular_speed").value

        # ===== State =====
        self.state = IDLE
        self.person_detected = False
        self.distance = 999
        self.offset = 0.0
        self.gesture = "NONE"

        # ===== Subscribers =====
        self.create_subscription(Bool, "/person_detected", self.detect_cb, 10)
        self.create_subscription(Float32, "/person_distance", self.distance_cb, 10)
        self.create_subscription(Float32, "/person_offset_x", self.offset_cb, 10)
        self.create_subscription(String, "/person_gesture", self.gesture_cb, 10)

        # ===== Publisher =====
        # self.cmd_pub = self.create_publisher(Twist, "/mobile_base/commands/velocity", 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.timer = self.create_timer(0.1, self.update)   # 10Hz

        self.get_logger().info("FOLLOW controller started")

    # ===== Callbacks =====
    def detect_cb(self, msg):
        self.person_detected = msg.data

    def distance_cb(self, msg):
        self.distance = msg.data

    def offset_cb(self, msg):
        self.offset = msg.data

    def gesture_cb(self, msg):
        self.gesture = msg.data

    # ===== Robot STOP =====
    def stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)

    # ===== Main FSM Logic =====
    def update(self):
        twist = Twist()

        # ---------------- IDLE STATE ----------------
        if self.state == IDLE:
            self.stop_robot()

            if self.gesture == "LEFT_HAND":
                self.state = FOLLOW
                self.get_logger().info("STATE -> FOLLOW")
                return

        # ---------------- FOLLOW STATE ----------------
        elif self.state == FOLLOW:

            # Stop following gesture
            if self.gesture == "RIGHT_HAND":
                self.state = IDLE
                self.stop_robot()
                self.get_logger().info("STATE -> IDLE")
                return

            # If human lost → Stop but stay in FOLLOW until gesture stop
            if not self.person_detected:
                self.stop_robot()
                self.get_logger().info("Person lost")
                return

            # ===== Distance Control =====
            if self.distance > self.min_distance:
                error = self.distance - self.min_distance
                speed = min(error * self.linear_kp, self.max_linear_speed)
                twist.linear.x = speed

            elif self.distance < self.min_distance:
                twist.linear.x = 0.0

            # ===== Alignment Control =====
            if self.offset < self.offset_left_th:
                twist.angular.z = self.angular_speed   # Turn Left

            elif self.offset > self.offset_right_th:
                twist.angular.z = -self.angular_speed  # Turn Right

            self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = FollowController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
