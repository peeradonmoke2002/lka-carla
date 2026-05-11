#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from carla_msgs.msg import CarlaWeatherParameters
from lka_msgs.msg import LaneCenter
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np


class YoloNode(Node):
    def __init__(self):
        super().__init__("yolo_node")

        # ── Declare parameters ────────────────────────────────────────
        self.declare_parameter("weights", "/home/peeradon/lka-carla-yolo/models/best_vision.pt")
        self.declare_parameter("weather_mode", "rain")
        self.declare_parameter("min_detection_conf", 0.60)
        self.declare_parameter("poly_degree", 2)
        self.declare_parameter("min_pixels", 50)
        self.declare_parameter("y_top_ratio", 0.50)
        self.declare_parameter("y_ref_ratio", 0.85)
        self.declare_parameter("lane_width_m", 4.0)  # Town01 driving lane width
        self.declare_parameter("lane_width_px", 760)  # fallback when one side missing
        self.declare_parameter("enable_hysteresis", False)
        self.declare_parameter("confirm_frames", 3)
        self.declare_parameter("lost_frames", 5)
        self.declare_parameter("jump_thresh", 0.12)

        # ── Load parameters ───────────────────────────────────────────
        weights           = self.get_parameter("weights").value
        self.weather_mode = self.get_parameter("weather_mode").value
        self.min_det_conf = self.get_parameter("min_detection_conf").value
        self.poly_degree  = self.get_parameter("poly_degree").value
        self.min_pixels   = self.get_parameter("min_pixels").value
        self.y_top_ratio  = self.get_parameter("y_top_ratio").value
        self.y_ref_ratio  = self.get_parameter("y_ref_ratio").value
        self.lane_width_m  = self.get_parameter("lane_width_m").value
        self.lane_width_px = self.get_parameter("lane_width_px").value
        self.enable_hysteresis = self.get_parameter("enable_hysteresis").value
        self.confirm_frames    = self.get_parameter("confirm_frames").value
        self.lost_frames       = self.get_parameter("lost_frames").value
        self.jump_thresh       = self.get_parameter("jump_thresh").value

        self.model  = YOLO(weights)
        self.bridge = CvBridge()
        self._cte   = None   # latest GT cross-track error (m)

        # Hysteresis state (used when enable_hysteresis=True)
        self.tracking    = False
        self.good_streak = 0
        self.bad_streak  = 0
        self.prev_center = None

        self.create_subscription(Image, "/carla/ego_vehicle/CAM_FRONT/image", self.image_callback, 10)
        self.create_subscription(CarlaWeatherParameters, "/carla/weather_control", self.weather_callback, 10)
        self.create_subscription(Float64, "/lka/gt/cross_track_m", self._cte_cb, 10)

        self.pub_enhanced     = self.create_publisher(Image,      "/lka/enhanced_image",    10)
        # self.pub_center       = self.create_publisher(LaneCenter, "/lka/lane_center",       10)
        self.pub_center_debug = self.create_publisher(LaneCenter, "/lka/yolo/lane_center",  10)

        self.get_logger().info(
            f"YOLO node ready | weather: {self.weather_mode} | weights: {weights}"
        )

    # ── GT callback ───────────────────────────────────────────────────

    def _cte_cb(self, msg: Float64):
        self._cte = msg.data

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

        if left_fit is not None and right_fit is not None:
            common_y_top = max(left_fit[1], right_fit[1])
            self.draw_lane_line(img, (left_fit[0],  common_y_top, left_fit[2]),  (0, 255, 0),   'left_marking',  10)
            self.draw_lane_line(img, (right_fit[0], common_y_top, right_fit[2]), (0, 255, 255), 'right_edge', -160)
        else:
            if left_fit  is not None: self.draw_lane_line(img, left_fit,  (0, 255, 0),   'left_marking',  10)
            if right_fit is not None: self.draw_lane_line(img, right_fit, (0, 255, 255), 'right_edge', -160)

        if center_px is None:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, 95), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            cv2.putText(img, "!! NO DETECTION !!", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(img, f"conf: {conf:.3f}", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return

        cx           = int(center_px)
        center_norm  = center_px / w

        # Real GT error (when GT available), else fallback to image-centre error
        if self._cte is not None:
            true_center = 0.5 - (self._cte / self.lane_width_m)
            real_error  = center_norm - true_center
            error_label = f"GT err: {real_error:+.3f}"
            color = (0, 255, 0) if abs(real_error) < 0.05 else (0, 0, 255)
        else:
            real_error  = center_norm - 0.5
            error_label = f"err:    {real_error:+.3f}  (no GT)"
            color = (0, 255, 0) if abs(cx - img_cx) < 50 else (0, 0, 255)

        cv2.circle(img, (cx, y_ref), 10, (0, 0, 255), -1)
        cv2.line(img, (img_cx, y_ref - 25), (img_cx, y_ref + 25), (255, 255, 0), 2)
        cv2.arrowedLine(img, (img_cx, y_ref), (cx, y_ref), (0, 255, 255), 2, tipLength=0.2)

        cv2.putText(img, f"YOLO   center: {center_norm:.3f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, error_label, (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, f"conf:   {conf:.3f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ── Hysteresis ─────────────────────────────────────────────────────

    def is_good_detection(self, center_norm, both_sides):
        if center_norm is None or not both_sides:
            return False
        if self.prev_center is not None and abs(center_norm - self.prev_center) > self.jump_thresh:
            return False
        return True

    def stability_filter(self, center_norm, both_sides):
        good = self.is_good_detection(center_norm, both_sides)
        if not self.tracking:
            if good:
                self.good_streak += 1
                self.prev_center  = center_norm
                if self.good_streak >= self.confirm_frames:
                    self.tracking   = True
                    self.bad_streak = 0
                    self.get_logger().info('Hysteresis → TRACKING')
            else:
                self.good_streak = 0
                self.prev_center = None
            return None
        else:
            if good:
                self.bad_streak  = 0
                self.prev_center = center_norm
                return center_norm
            else:
                self.bad_streak += 1
                if self.bad_streak >= self.lost_frames:
                    self.tracking    = False
                    self.good_streak = 0
                    self.prev_center = None
                    self.get_logger().info('Hysteresis → SEARCHING')
                return self.prev_center

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

        lx_raw = float(np.polyval(left_fit[0],  y_ref)) if left_fit  is not None else None
        rx_raw = float(np.polyval(right_fit[0], y_ref)) if right_fit is not None else None
        both_sides = left_fit is not None and right_fit is not None

        if self.enable_hysteresis:
            # Synthesize missing side, then apply hysteresis filter
            if center_px is not None:
                center_norm_raw = float(center_px / w)
            elif lx_raw is not None:
                center_norm_raw = (lx_raw + lx_raw + self.lane_width_px) / 2.0 / w
            elif rx_raw is not None:
                center_norm_raw = (rx_raw - self.lane_width_px + rx_raw) / 2.0 / w
            else:
                center_norm_raw = None
            stable = self.stability_filter(center_norm_raw, both_sides)
            detected    = stable is not None
            center_norm = float(stable) if detected else -1.0
        else:
            # Raw: only publish when both sides found (no synthesis)
            detected    = center_px is not None
            center_norm = float(center_px / w) if detected else -1.0

        lx_val = lx_raw if lx_raw is not None else -1.0
        rx_val = rx_raw if rx_raw is not None else -1.0

        lane_msg              = LaneCenter()
        lane_msg.header.stamp = self.get_clock().now().to_msg()
        lane_msg.center       = center_norm
        lane_msg.confidence   = float(mean_conf)
        lane_msg.detected     = detected
        lane_msg.lx           = lx_val
        lane_msg.rx           = rx_val
        self.pub_center_debug.publish(lane_msg)

        self.get_logger().info(
            f"[{self.weather_mode}] conf={mean_conf:.3f}  center={center_norm:.3f}",
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
