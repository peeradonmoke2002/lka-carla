#!/usr/bin/env python3
"""
Lane Detection Node (Phase 3-4)

Subscribes to /carla/ego_vehicle/CAM_FRONT/image, applies weather-adaptive
preprocessing, runs YOLOv8-seg, and publishes lane center + confidence.

Parameters:
  weights         (string) – path to .pt weights file
  conf_threshold  (float)  – YOLO confidence threshold (default 0.25)
  weather_mode    (string) – clear | fog | rain | night

Published topics:
  /lka/enhanced_image              sensor_msgs/Image  – annotated enhanced frame
  /lka/degraded_image              sensor_msgs/Image  – synthetically degraded frame
  /lka/degraded_detection          sensor_msgs/Image  – annotated degraded frame (baseline)
  /lka/lane_center                 std_msgs/Float32   – normalized x in [0,1]; -1 = no detection
  /lka/detection_confidence        std_msgs/Float32   – mean YOLO confidence (enhanced)
  /lka/detection_confidence_degraded std_msgs/Float32 – mean YOLO confidence (degraded baseline)
"""

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

DEFAULT_WEIGHTS = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..', '..', '..', '..', '..', '..', 'models', 'best_vision.pt'
)


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__("lane_detection_node")

        self.declare_parameter("weights", DEFAULT_WEIGHTS)
        self.declare_parameter("conf_threshold", 0.25)
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
        self.pub_degraded = self.create_publisher(Image, "/lka/degraded_image", 10)
        self.pub_degraded_det = self.create_publisher(Image, "/lka/degraded_detection", 10)
        self.pub_center = self.create_publisher(Float32, "/lka/lane_center", 10)
        self.pub_conf = self.create_publisher(Float32, "/lka/detection_confidence", 10)
        self.pub_conf_degraded = self.create_publisher(Float32, "/lka/detection_confidence_degraded", 10)

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
    # Preprocessing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gamma_lut(gamma: float) -> np.ndarray:
        return np.array(
            [min(255, int((i / 255.0) ** (1.0 / gamma) * 255)) for i in range(256)],
            dtype=np.uint8,
        )

    def degrade(self, img: np.ndarray) -> np.ndarray:
        """Synthetically degrade image to simulate weather conditions."""
        if self.weather_mode == "fog":
            white = np.full_like(img, 220)
            hazy = cv2.addWeighted(img, 0.45, white, 0.55, 0)
            return cv2.GaussianBlur(hazy, (21, 21), 0)

        if self.weather_mode == "rain":
            dark = (img * 0.65).astype(np.uint8)
            noise = np.random.normal(0, 18, img.shape).astype(np.int16)
            return np.clip(dark.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if self.weather_mode == "night":
            return (img * 0.18).astype(np.uint8)

        return img  # clear

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Enhance degraded image to recover lane visibility."""
        if self.weather_mode == "fog":
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab = cv2.merge([clahe.apply(l), a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if self.weather_mode == "rain":
            denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
            return self._gamma_lut(1.3)[denoised]

        if self.weather_mode == "night":
            return self._gamma_lut(1.8)[img]

        return img  # clear

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

        # Step 1: synthetic degradation (simulates real-world weather)
        degraded = self.degrade(img)

        # Step 2: YOLO on degraded image — baseline (no preprocessing)
        results_deg = self.model.predict(degraded, conf=self.conf, device=0, verbose=False)[0]
        lane_center_deg, conf_deg = self._extract_lane_info(results_deg)

        annotated_deg = results_deg.plot()
        self._draw_center_overlay(annotated_deg, lane_center_deg, conf_deg, w, h)
        self.pub_degraded.publish(self.bridge.cv2_to_imgmsg(degraded, encoding="bgr8"))
        self.pub_degraded_det.publish(self.bridge.cv2_to_imgmsg(annotated_deg, encoding="bgr8"))
        self.pub_conf_degraded.publish(Float32(data=float(conf_deg)))

        # Step 3: weather-adaptive preprocessing
        enhanced = self.preprocess(degraded)

        # Step 4: YOLO on enhanced image
        results = self.model.predict(enhanced, conf=self.conf, device=0, verbose=False)[0]
        lane_center_x, mean_conf = self._extract_lane_info(results)

        annotated = results.plot()
        self._draw_center_overlay(annotated, lane_center_x, mean_conf, w, h)
        self.pub_enhanced.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))

        center_norm = float(lane_center_x / w) if lane_center_x is not None else -1.0
        self.pub_center.publish(Float32(data=center_norm))
        self.pub_conf.publish(Float32(data=float(mean_conf)))

        self.get_logger().info(
            f"[{self.weather_mode}] "
            f"conf_degraded={conf_deg:.3f}  conf_enhanced={mean_conf:.3f}  "
            f"delta={mean_conf - conf_deg:+.3f}  center={center_norm:.3f}"
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
