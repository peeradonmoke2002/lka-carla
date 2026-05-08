#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from carla_msgs.msg import CarlaWeatherParameters
from lka_msgs.msg import LaneCenter
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__("lane_detection_node")

        # ── Declare parameters ────────────────────────────────────────
        self.declare_parameter("weights", "/home/peeradon/lka-carla-yolo/models/best_vision.pt")
        self.declare_parameter("weather_mode", "rain")
        self.declare_parameter("min_detection_conf", 0.60)
        self.declare_parameter("poly_degree", 2)
        self.declare_parameter("min_pixels", 50)
        self.declare_parameter("y_top_ratio", 0.50)
        self.declare_parameter("y_ref_ratio", 0.85)

        # ── Load parameters ───────────────────────────────────────────
        weights           = self.get_parameter("weights").value
        self.weather_mode = self.get_parameter("weather_mode").value
        self.min_det_conf = self.get_parameter("min_detection_conf").value
        self.poly_degree  = self.get_parameter("poly_degree").value
        self.min_pixels   = self.get_parameter("min_pixels").value
        self.y_top_ratio  = self.get_parameter("y_top_ratio").value
        self.y_ref_ratio  = self.get_parameter("y_ref_ratio").value

        self.model  = YOLO(weights)
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
            self.weather_callback,
            10,
        )
        self.pub_enhanced = self.create_publisher(Image,      "/lka/enhanced_image", 10)
        self.pub_center   = self.create_publisher(LaneCenter, "/lka/lane_center", 10)

        self.get_logger().info(
            f"Lane detection node ready | weather: {self.weather_mode} | weights: {weights}"
        )

    # ── Weather ────────────────────────────────────────────────────────

    def weather_callback(self, msg: CarlaWeatherParameters):
        if msg.fog_density > 40.0:
            mode = 'fog'
        elif msg.precipitation > 30.0:
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

    # ── Lane fitting ───────────────────────────────────────────────────

    def fit_lane(self, mask: np.ndarray, roi_top: int, roi_bot: int):
        """Fit polynomial x=f(y) through all mask pixels in [roi_top, roi_bot].
        Returns (coeffs, pixel_y_top, pixel_y_bot) or None when too few pixels."""
        ys, xs = np.where(mask[roi_top:roi_bot] > 0)
        if len(xs) < self.min_pixels:
            return None
        ys_abs = ys + roi_top
        coeffs = np.polyfit(ys_abs, xs, self.poly_degree)
        return coeffs, int(ys_abs.min()), int(ys_abs.max())

    def build_lane_masks(self, results, h: int, w: int):
        """Rasterize YOLO segmentation polygons into left/right binary masks.
        Skips any detection below min_det_conf."""
        left_mask  = np.zeros((h, w), dtype=np.uint8)
        right_mask = np.zeros((h, w), dtype=np.uint8)

        for poly, conf, cls in zip(results.masks.xy,
                                   results.boxes.conf.cpu().numpy(),
                                   results.boxes.cls.cpu().numpy().astype(int)):
            if len(poly) < 3 or conf < self.min_det_conf:
                continue
            pts = poly.astype(np.int32)
            if cls == 0:
                cv2.fillPoly(left_mask, [pts], 255)
            else:
                cv2.fillPoly(right_mask, [pts], 255)

        return left_mask, right_mask

    def detect_lanes(self, results, h: int, w: int):
        """Detect ego-lane center from YOLO results.

        Returns (center_px, mean_conf, left_fit, right_fit).
        center_px is None when both sides are not found.
        left_fit / right_fit = (coeffs, y_top, y_bot) or None.
        """
        if results.masks is None or len(results.boxes) == 0:
            return None, 0.0, None, None

        left_mask, right_mask = self.build_lane_masks(results, h, w)

        roi_top = int(h * self.y_top_ratio)
        y_ref   = int(h * self.y_ref_ratio)

        left_fit  = self.fit_lane(left_mask,  roi_top, h - 1)
        right_fit = self.fit_lane(right_mask, roi_top, h - 1)

        if left_fit is None or right_fit is None:
            return None, 0.0, left_fit, right_fit

        lx = np.polyval(left_fit[0],  y_ref)
        rx = np.polyval(right_fit[0], y_ref)
        center_px = (lx + rx) / 2.0
        mean_conf = float(results.boxes.conf.cpu().numpy().mean())

        return center_px, mean_conf, left_fit, right_fit

    # ── Visualization ──────────────────────────────────────────────────

    @staticmethod
    def draw_lane_line(img, fit, color, label: str, label_offset_x: int):
        """Draw a fitted line from its detected pixel top down to the image bottom."""
        h, w = img.shape[:2]
        coeffs, y_detected_top, _ = fit
        ys = np.linspace(y_detected_top, h - 1, 60).astype(int)
        xs = np.polyval(coeffs, ys).astype(int)
        for i in range(len(ys) - 1):
            if 0 <= xs[i] < w and 0 <= xs[i + 1] < w:
                cv2.line(img, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), color, 3)
        label_x = max(0, int(xs[-1]) + label_offset_x)
        cv2.putText(img, label, (label_x, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def draw_debug(self, img: np.ndarray, center_px, conf: float,
                   left_fit, right_fit, y_ref: int):
        """Draw fitted lines, center dot, and HUD text onto the debug image."""
        h, w   = img.shape[:2]
        img_cx = w // 2

        if left_fit is not None:
            self.draw_lane_line(img, left_fit,  (0, 255, 0),   'left_marking',  10)
        if right_fit is not None:
            self.draw_lane_line(img, right_fit, (0, 255, 255), 'right_edge', -160)

        if center_px is None:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, 95), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            cv2.putText(img, "!! NO DETECTION !!", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(img, f"conf: {conf:.3f}", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return

        cx            = int(center_px)
        lateral_error = cx - img_cx
        color         = (0, 255, 0) if abs(lateral_error) < 50 else (0, 0, 255)

        cv2.circle(img, (cx, y_ref), 10, (0, 0, 255), -1)
        cv2.line(img, (img_cx, y_ref - 25), (img_cx, y_ref + 25), (255, 255, 0), 2)
        cv2.arrowedLine(img, (img_cx, y_ref), (cx, y_ref), (0, 255, 255), 2, tipLength=0.2)

        cv2.putText(img, f"center: {center_px / w:.3f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"error:  {lateral_error / w:+.3f}", (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, f"conf:   {conf:.3f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ── Main image callback ────────────────────────────────────────────

    def image_callback(self, msg: Image):
        img  = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w = img.shape[:2]

        results                               = self.model.predict(img, conf=self.min_det_conf, device=0, verbose=False)[0]
        center_px, mean_conf, left_fit, right_fit = self.detect_lanes(results, h, w)

        y_ref = int(h * self.y_ref_ratio)

        annotated = results.plot()
        self.draw_debug(annotated, center_px, mean_conf, left_fit, right_fit, y_ref)
        self.pub_enhanced.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))

        detected    = center_px is not None
        center_norm = float(center_px / w) if detected else -1.0

        lane_msg              = LaneCenter()
        lane_msg.header.stamp = self.get_clock().now().to_msg()
        lane_msg.center       = center_norm
        lane_msg.confidence   = float(mean_conf)
        lane_msg.detected     = detected
        self.pub_center.publish(lane_msg)

        self.get_logger().info(
            f"[{self.weather_mode}] conf={mean_conf:.3f}  center={center_norm:.3f}",
            throttle_duration_sec=1.0,
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
