import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32, Bool
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

class PersonNode(Node):
    def __init__(self):
        super().__init__('person_detector')

        self.bridge = CvBridge()
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

        # self.rgb_sub = self.create_subscription(Image, "/camera/rgb/image_raw", self.rgb_callback, 10)
        self.rgb_sub = self.create_subscription(Image, "/image_raw", self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, "/camera/depth/image_raw", self.depth_callback, 10)

        self.detect_pub = self.create_publisher(Bool, "/person_detected", 10)
        self.pub_gesture = self.create_publisher(String, "/person_gesture", 10)
        self.pub_offsetX = self.create_publisher(Float32, "/person_offsetX", 10)
        self.pub_dist = self.create_publisher(Float32, "/person_distance", 10)

        self.depth_image = None

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def rgb_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        gesture = "NONE"
        offset = 0.0
        distance = -1.0  # unknown unless detected

        detected = result.pose_landmarks is not None

        if detected:
            lm = result.pose_landmarks.landmark

            cx = (lm[mp_pose.PoseLandmark.LEFT_HIP].x +
                  lm[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2
            offset = (cx - 0.5)

            left_up = lm[mp_pose.PoseLandmark.LEFT_WRIST].y < lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_up = lm[mp_pose.PoseLandmark.RIGHT_WRIST].y < lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y

            if left_up and not right_up:
                gesture = "LEFT_HAND"
            elif right_up and not left_up:
                gesture = "RIGHT_HAND"

            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_style.get_default_pose_landmarks_style()
            )

            # ---------- Distance from depth image ----------
            if self.depth_image is not None:
                px = int(cx * w)
                py = int(lm[mp_pose.PoseLandmark.LEFT_HIP].y * h)

                if 0 <= px < self.depth_image.shape[1] and 0 <= py < self.depth_image.shape[0]:
                    d = self.depth_image[py, px]

                    if d > 0:
                        if self.depth_image.dtype == np.uint16:
                            distance = float(d) / 1000.0
                        else:
                            distance = float(d)

        text = f"offset={offset:.2f}  gesture={gesture}  dist={distance:.2f}m"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Person Detection", frame)
        cv2.waitKey(1)

        self.detect_pub.publish(Bool(data=detected))
        self.pub_gesture.publish(String(data=gesture))
        self.pub_offsetX.publish(Float32(data=float(offset)))
        self.pub_dist.publish(Float32(data=float(distance)))

def main():
    rclpy.init()
    node = PersonNode()
    rclpy.spin(node)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
