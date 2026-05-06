#!/usr/bin/python3

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from carla_msgs.msg import CarlaWeatherParameters
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np

DEFAULT_WEIGHTS = "/home/peeradon/lka-carla-yolo/models/best_vision.pt"


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__("lane_detection_node")

        self.declare_parameter("weights", DEFAULT_WEIGHTS)
        self.declare_parameter("conf_threshold", 0.25) # >= 0.25 can find if lower don't care
        self.declare_parameter("weather_mode", "clear")

        weights = self.get_parameter("weights").get_parameter_value().string_value
        self.conf = self.get_parameter("conf_threshold").get_parameter_value().double_value
        self.weather_mode = self.get_parameter("weather_mode").get_parameter_value().string_value

        self.model = YOLO(weights)
        self.bridge = CvBridge()

        self.create_subscription(
            Image,
            "/carla/ego_vehicle/CAM_FRONT/image",
            self.image_callback,
            10,
        )
        self.create_subscription(
            CarlaWeatherParameters,
            "/carla/weather_control",
            self._weather_callback,
            10,
        )
        self.pub_enhanced = self.create_publisher(Image, "/lka/enhanced_image", 10)
        self.pub_center = self.create_publisher(Float32, "/lka/lane_center", 10)
        self.pub_conf = self.create_publisher(Float32, "/lka/detection_confidence", 10)

        self.get_logger().info(
            f"Lane detection node ready | weather: {self.weather_mode} | weights: {weights}"
        )

    # ------------------------------------------------------------------
    # Weather callback
    # ------------------------------------------------------------------

    def _weather_callback(self, msg: CarlaWeatherParameters):
        if msg.fog_density > 40:
            mode = 'fog'
        elif msg.precipitation > 30:
            mode = 'rain'
        elif msg.sun_altitude_angle < 0:
            mode = 'night'
        else:
            mode = 'clear'
        if mode != self.weather_mode:
            self.weather_mode = mode
            self.get_logger().info(
                f'Weather → {self.weather_mode} '
                f'(fog={msg.fog_density:.0f} rain={msg.precipitation:.0f} '
                f'sun={msg.sun_altitude_angle:.1f}°)'
            )

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _extract_lane_info(self, results):
        """Return (lane_center_x_px, mean_confidence). lane_center_x is None on no detection."""
        if results.masks is None or len(results.boxes) == 0:
            return None, 0.0

        masks = results.masks.xy
        confs = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        left_xs, right_xs = [], []
        for poly, cls in zip(masks, class_ids):
            if len(poly) == 0:
                continue
            cx = float(poly[:, 0].mean())
            if cls == 0:   # left_marking
                left_xs.append(cx)
            else:          # right_edge
                right_xs.append(cx)

        LANE_WIDTH_PX = 760  # fallback half-lane offset in pixels (same as pure_vision_node)
        if left_xs and right_xs:
            lane_center = (np.mean(left_xs) + np.mean(right_xs)) / 2.0
        elif left_xs:
            lx = float(np.mean(left_xs))
            lane_center = (lx + lx + LANE_WIDTH_PX) / 2.0
        elif right_xs:
            rx = float(np.mean(right_xs))
            lane_center = (rx - LANE_WIDTH_PX + rx) / 2.0
        else:
            return None, 0.0

        return lane_center, float(confs.mean())

    @staticmethod
    def _draw_center_overlay(img: np.ndarray, lane_center_x, conf: float, w: int, h: int):
        img_cx = w // 2
        bottom_y = h - 10
        top_y = h // 2

        if lane_center_x is None:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, 95), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            cv2.putText(img, "!! NO DETECTION !!", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(img, f"conf: {conf:.3f}", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return

        cx = int(lane_center_x)
        lateral_error = cx - img_cx
        color = (0, 255, 0) if abs(lateral_error) < 50 else (0, 0, 255)

        cv2.line(img, (cx, top_y), (cx, bottom_y), color, 2)
        cv2.circle(img, (cx, (top_y + bottom_y) // 2), 6, color, -1)
        cv2.arrowedLine(img, (img_cx, top_y + 20), (cx, top_y + 20), (0, 255, 255), 2, tipLength=0.2)

        error_norm = lateral_error / w
        cv2.putText(img, f"center: {lane_center_x / w:.3f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"error:  {error_norm:+.3f}", (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, f"conf:   {conf:.3f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------

    def image_callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w = img.shape[:2]

        results = self.model.predict(img, conf=self.conf, device=0, verbose=False)[0]
        lane_center_x, mean_conf = self._extract_lane_info(results)

        annotated = results.plot()
        self._draw_center_overlay(annotated, lane_center_x, mean_conf, w, h)
        self.pub_enhanced.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))

        center_norm = float(lane_center_x / w) if lane_center_x is not None else -1.0
        self.pub_center.publish(Float32(data=center_norm))
        self.pub_conf.publish(Float32(data=float(mean_conf)))

        self.get_logger().info(
            f"[{self.weather_mode}] conf={mean_conf:.3f}  center={center_norm:.3f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
