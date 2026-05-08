#!/usr/bin/python3
"""
Pure Vision Lane Detection Node

Detects lanes via HSV yellow (left) + grayscale Canny (right) + Hough lines,
then publishes the ego-lane center. All parameters are loaded from a YAML file.
"""

import os
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from carla_msgs.msg import CarlaWeatherParameters
from lka_msgs.msg import LaneCenter
from cv_bridge import CvBridge
import cv2
import numpy as np

DEFAULT_ROI_YAML = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..', '..', '..', 'lka_dataset_collection', 'config', 'roi.yaml'
)


class PureVisionNode(Node):
    def __init__(self):
        super().__init__('pure_vision_node')

        # ── Declare parameters ────────────────────────────────────────
        self.declare_parameter('roi_yaml', DEFAULT_ROI_YAML)
        self.declare_parameter('roi_margin_ratio', 0.08)
        self.declare_parameter('y_top_ratio', 0.55)
        self.declare_parameter('y_ref_ratio', 0.85)
        self.declare_parameter('lane_width_px', 760)
        self.declare_parameter('hough_threshold', 15)
        self.declare_parameter('hough_min_line_length', 20)
        self.declare_parameter('hough_max_line_gap', 80)
        self.declare_parameter('left_slope_min', -2.5)
        self.declare_parameter('left_slope_max', -0.3)
        self.declare_parameter('right_slope_min', 0.3)
        self.declare_parameter('right_slope_max', 2.5)
        self.declare_parameter('confirm_frames', 3)
        self.declare_parameter('lost_frames', 5)
        self.declare_parameter('jump_thresh', 0.12)
        self.declare_parameter('weather_fog_thresh', 40.0)
        self.declare_parameter('weather_rain_thresh', 30.0)
        self.declare_parameter('hsv_lo_clear', [10,  30, 250])
        self.declare_parameter('hsv_hi_clear', [40, 120, 255])
        self.declare_parameter('hsv_lo_fog',   [10,   5, 180])
        self.declare_parameter('hsv_hi_fog',   [40, 120, 255])
        self.declare_parameter('hsv_lo_night', [10, 150,  30])
        self.declare_parameter('hsv_hi_night', [35, 255, 255])
        self.declare_parameter('hsv_lo_rain',  [15,  25, 150])
        self.declare_parameter('hsv_hi_rain',  [35, 255, 255])
        self.declare_parameter('canny_clear', [30, 90])
        self.declare_parameter('canny_fog',   [20, 60])
        self.declare_parameter('canny_night', [20, 60])
        self.declare_parameter('canny_rain',  [20, 60])

        # ── Load parameters ───────────────────────────────────────────
        roi_path = self.get_parameter('roi_yaml').value

        self.roi_margin_ratio  = self.get_parameter('roi_margin_ratio').value
        self.y_top_ratio       = self.get_parameter('y_top_ratio').value
        self.y_ref_ratio       = self.get_parameter('y_ref_ratio').value
        self.lane_width_px     = self.get_parameter('lane_width_px').value
        self.hough_threshold   = self.get_parameter('hough_threshold').value
        self.hough_min_line_len = self.get_parameter('hough_min_line_length').value
        self.hough_max_line_gap = self.get_parameter('hough_max_line_gap').value
        self.left_slope_min    = self.get_parameter('left_slope_min').value
        self.left_slope_max    = self.get_parameter('left_slope_max').value
        self.right_slope_min   = self.get_parameter('right_slope_min').value
        self.right_slope_max   = self.get_parameter('right_slope_max').value
        self.confirm_frames    = self.get_parameter('confirm_frames').value
        self.lost_frames       = self.get_parameter('lost_frames').value
        self.jump_thresh       = self.get_parameter('jump_thresh').value
        self.fog_thresh        = self.get_parameter('weather_fog_thresh').value
        self.rain_thresh       = self.get_parameter('weather_rain_thresh').value

        def load_hsv(name):
            return np.array(self.get_parameter(name).value, dtype=np.uint8)

        def load_canny(name):
            v = self.get_parameter(name).value
            return (int(v[0]), int(v[1]))

        self.hsv_lo = {
            'clear': load_hsv('hsv_lo_clear'),
            'fog':   load_hsv('hsv_lo_fog'),
            'night': load_hsv('hsv_lo_night'),
            'rain':  load_hsv('hsv_lo_rain'),
        }
        self.hsv_hi = {
            'clear': load_hsv('hsv_hi_clear'),
            'fog':   load_hsv('hsv_hi_fog'),
            'night': load_hsv('hsv_hi_night'),
            'rain':  load_hsv('hsv_hi_rain'),
        }
        self.gray_canny = {
            'clear': load_canny('canny_clear'),
            'fog':   load_canny('canny_fog'),
            'night': load_canny('canny_night'),
            'rain':  load_canny('canny_rain'),
        }

        self.roi_polygon  = self.load_roi(roi_path)
        self.bridge       = CvBridge()
        self.weather_mode = 'rain'

        # Hysteresis state machine
        self.tracking    = False
        self.good_streak = 0
        self.bad_streak  = 0
        self.prev_center = None

        self.create_subscription(
            Image,
            '/carla/ego_vehicle/CAM_FRONT/image',
            self.image_callback,
            10,
        )
        self.create_subscription(
            CarlaWeatherParameters,
            '/carla/weather_control',
            self.weather_callback,
            10,
        )
        self.pub_center = self.create_publisher(LaneCenter, '/lka/lane_center', 10)
        self.pub_image  = self.create_publisher(Image,      '/lka/pure_vision_image', 10)

        self.get_logger().info(f'Pure vision node ready | roi: {roi_path}')

    # ── ROI helpers ────────────────────────────────────────────────────

    @staticmethod
    def load_roi(path):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = yaml.full_load(f)
        raw = data.get('roi', {}).get('polygon', None)
        if raw is None:
            return None
        return np.array([list(p) for p in raw], dtype=np.int32)

    def make_roi_mask(self, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        if self.roi_polygon is not None:
            cv2.fillPoly(mask, [self.roi_polygon], 255)
        else:
            mask[:] = 255
        return mask

    def roi_center_x(self, w):
        if self.roi_polygon is not None:
            top2 = self.roi_polygon[self.roi_polygon[:, 1].argsort()][:2]
            return int(top2[:, 0].mean())
        return w // 2

    # ── Edge detection ─────────────────────────────────────────────────

    def build_edge_images(self, img, weather, roi_mask, roi_cx, margin):
        """Build left (HSV yellow) and right (Canny gray) edge images."""
        # Left: HSV yellow → dilate → mask to left half of ROI
        hsv         = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, self.hsv_lo[weather], self.hsv_hi[weather])
        kernel      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        left_edges  = cv2.dilate(yellow_mask, kernel, iterations=2)
        left_edges  = cv2.bitwise_and(left_edges, roi_mask)
        left_edges[:, roi_cx + margin:] = 0

        # Right: grayscale Canny → mask to right half of ROI
        gray        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur   = cv2.GaussianBlur(gray, (5, 5), 0)
        canny_lo, canny_hi = self.gray_canny[weather]
        right_edges = cv2.Canny(gray_blur, canny_lo, canny_hi)
        right_edges = cv2.bitwise_and(right_edges, roi_mask)
        right_edges[:, :roi_cx - margin] = 0

        return left_edges, right_edges

    # ── Line fitting ───────────────────────────────────────────────────

    @staticmethod
    def fit_line(points, y_bottom, y_top):
        """Fit a line through (x,y) points. Returns (x_bottom, x_top, coeffs) or None."""
        if len(points) < 2:
            return None
        pts    = np.array(points)
        coeffs = np.polyfit(pts[:, 1], pts[:, 0], 1)
        return int(np.polyval(coeffs, y_bottom)), int(np.polyval(coeffs, y_top)), coeffs

    def hough_fit(self, edge_img, y_bottom, y_top, slope_min, slope_max):
        """Run Hough on edge_img, keep lines within slope range, fit one line."""
        lines = cv2.HoughLinesP(
            edge_img, rho=1, theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_len,
            maxLineGap=self.hough_max_line_gap,
        )
        pts = []
        if lines is not None:
            for seg in lines:
                x1, y1, x2, y2 = seg[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if slope_min <= slope <= slope_max:
                    pts.extend([(x1, y1), (x2, y2)])
        return self.fit_line(pts, y_bottom, y_top)

    # ── Lane detection ─────────────────────────────────────────────────

    def detect(self, img, weather):
        """Detect left and right lane lines and compute the normalized ego-lane center.

        Returns (left_fit, right_fit, center_norm, lx, rx, y_bottom, y_top, y_ref, both_sides).
        center_norm is None when detection fails. both_sides is True only when both lines found.
        """
        h, w     = img.shape[:2]
        roi_mask = self.make_roi_mask(h, w)
        margin   = int(w * self.roi_margin_ratio)
        roi_cx   = self.roi_center_x(w)

        left_edges, right_edges = self.build_edge_images(img, weather, roi_mask, roi_cx, margin)

        y_bottom = h - 1
        y_top    = int(h * self.y_top_ratio)
        y_ref    = int(h * self.y_ref_ratio)

        left_fit  = self.hough_fit(left_edges,  y_bottom, y_top, self.left_slope_min,  self.left_slope_max)
        right_fit = self.hough_fit(right_edges, y_bottom, y_top, self.right_slope_min, self.right_slope_max)

        # Sanity checks — discard lines that crossed to the wrong side
        if left_fit and left_fit[0] > roi_cx:
            left_fit = None
        if right_fit and right_fit[0] < roi_cx:
            right_fit = None
        if left_fit and left_fit[1] > roi_cx * 1.1:
            left_fit = None
        if right_fit and right_fit[1] < roi_cx * 0.9:
            right_fit = None
        if left_fit and right_fit and left_fit[1] >= right_fit[1]:
            left_fit = None

        center_norm = lx = rx = None
        both_sides = False

        if left_fit and right_fit:
            lx = int(np.polyval(left_fit[2],  y_ref))
            rx = int(np.polyval(right_fit[2], y_ref))
            center_norm = ((lx + rx) / 2.0) / w
            both_sides  = True
        elif left_fit:
            lx = int(np.polyval(left_fit[2], y_ref))
            rx = lx + self.lane_width_px
            center_norm = ((lx + rx) / 2.0) / w
        elif right_fit:
            rx = int(np.polyval(right_fit[2], y_ref))
            lx = rx - self.lane_width_px
            center_norm = ((lx + rx) / 2.0) / w

        return left_fit, right_fit, center_norm, lx, rx, y_bottom, y_top, y_ref, both_sides

    # ── Visualization ──────────────────────────────────────────────────

    @staticmethod
    def draw_debug(img, weather, left_fit, right_fit, center_norm, lx, rx, y_bottom, y_top, y_ref):
        h, w = img.shape[:2]
        vis  = img.copy()

        if left_fit and right_fit:
            lane_poly = np.array([
                [left_fit[1],  y_top],   [right_fit[1], y_top],
                [right_fit[0], y_bottom], [left_fit[0],  y_bottom],
            ], dtype=np.int32)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [lane_poly], (0, 180, 0))
            vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)

        if left_fit:
            cv2.line(vis, (left_fit[0], y_bottom), (left_fit[1], y_top), (0, 255, 0), 3)
            cv2.putText(vis, 'left_marking', (left_fit[0] + 10, y_bottom - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if right_fit:
            cv2.line(vis, (right_fit[0], y_bottom), (right_fit[1], y_top), (0, 255, 255), 3)
            cv2.putText(vis, 'right_edge', (right_fit[0] - 220, y_bottom - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if center_norm is not None and lx is not None and rx is not None:
            cx  = int(center_norm * w)
            err = (w / 2) - cx
            cv2.line(vis, (lx, y_ref), (rx, y_ref), (180, 180, 0), 1)
            cv2.circle(vis, (cx, y_ref), 14, (0, 0, 255), -1)
            cv2.line(vis, (w // 2, y_ref - 25), (w // 2, y_ref + 25), (255, 255, 0), 2)
            cv2.putText(vis, f'center={center_norm:.3f}  err={err:+.0f}px',
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            cv2.putText(vis, 'NO DETECTION', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.putText(vis, f'weather: {weather}', (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        return vis

    # ── Hysteresis ─────────────────────────────────────────────────────

    def is_good_detection(self, center_norm, both_sides):
        """A frame is 'good' when both sides are detected and the center didn't jump."""
        if center_norm is None or not both_sides:
            return False
        if self.prev_center is not None and abs(center_norm - self.prev_center) > self.jump_thresh:
            return False
        return True

    def stability_filter(self, center_norm, both_sides):
        """Hysteresis state machine: require several good frames before publishing,
        and tolerate several bad frames before dropping back to searching."""
        good = self.is_good_detection(center_norm, both_sides)

        if not self.tracking:
            if good:
                self.good_streak += 1
                self.prev_center  = center_norm
                if self.good_streak >= self.confirm_frames:
                    self.tracking    = True
                    self.bad_streak  = 0
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
                reason = 'MISS' if center_norm is None else \
                         'single-side' if not both_sides else 'jump'
                self.get_logger().warn(
                    f'Bad frame ({reason}) bad_streak={self.bad_streak}/{self.lost_frames}',
                    throttle_duration_sec=0.5,
                )
                if self.bad_streak >= self.lost_frames:
                    self.tracking    = False
                    self.good_streak = 0
                    self.prev_center = None
                    self.get_logger().info('Hysteresis → SEARCHING')
                return self.prev_center

    # ── Callbacks ──────────────────────────────────────────────────────

    def weather_callback(self, msg: CarlaWeatherParameters):
        if msg.fog_density > self.fog_thresh:
            new_mode = 'fog'
        elif msg.precipitation > self.rain_thresh:
            new_mode = 'rain'
        elif msg.sun_altitude_angle < 0:
            new_mode = 'night'
        else:
            new_mode = 'clear'
        if new_mode != self.weather_mode:
            self.weather_mode = new_mode
            self.get_logger().info(
                f'Weather → {self.weather_mode} '
                f'(fog={msg.fog_density:.0f} rain={msg.precipitation:.0f} '
                f'sun={msg.sun_altitude_angle:.1f}°)'
            )

    def image_callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        left_fit, right_fit, center_norm, lx, rx, y_bottom, y_top, y_ref, both_sides = \
            self.detect(img, self.weather_mode)

        stable_center = self.stability_filter(center_norm, both_sides)
        detected      = stable_center is not None
        center_val    = float(stable_center) if detected else -1.0

        lane_msg              = LaneCenter()
        lane_msg.header.stamp = self.get_clock().now().to_msg()
        lane_msg.center       = center_val
        lane_msg.confidence   = 0.0
        lane_msg.detected     = detected
        self.pub_center.publish(lane_msg)

        vis = self.draw_debug(img, self.weather_mode, left_fit, right_fit,
                              stable_center, lx, rx, y_bottom, y_top, y_ref)
        self.pub_image.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))

        raw_str = f'{center_norm:.3f}' if center_norm is not None else 'MISS'
        sides   = 'both' if both_sides else ('one' if center_norm is not None else 'none')
        state   = 'TRACKING' if self.tracking else f'SEARCHING({self.good_streak}/{self.confirm_frames})'
        self.get_logger().info(
            f'[{self.weather_mode}] {state} raw={raw_str}({sides}) out={center_val:.3f}',
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = PureVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
