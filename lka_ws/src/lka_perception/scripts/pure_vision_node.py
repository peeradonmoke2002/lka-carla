#!/usr/bin/env python3
"""
Pure Vision Lane Detection Node

Subscribes to /carla/ego_vehicle/CAM_FRONT/image and /carla/weather_control,
detects lanes via HSV yellow (left) + grayscale Canny (right) + Hough, and
publishes ego-lane center on the same topic as the YOLO node.

Weather classification (from CarlaWeatherParameters):
  fog_density > 40   → fog
  precipitation > 30 → rain
  sun_altitude < 0   → night
  else               → clear

Parameters:
  roi_yaml  (string) – path to roi.yaml

Published topics:
  /lka/lane_center         std_msgs/Float32  – normalized x [0,1]; -1 = no detection
  /lka/pure_vision_image   sensor_msgs/Image – annotated debug frame
"""

import os
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from carla_msgs.msg import CarlaWeatherParameters
from cv_bridge import CvBridge
import cv2
import numpy as np

DEFAULT_ROI_YAML = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..', '..', '..', 'lka_dataset_collection', 'config', 'roi.yaml'
)

# HSV yellow thresholds per weather (left marking)
_HSV_LO = {
    'clear': np.array([10,  30, 250]),
    'fog':   np.array([10,   5, 180]),
    'night': np.array([10, 150,  30]),
    'rain':  np.array([15,  25, 150]),
}
_HSV_HI = {
    'clear': np.array([40, 120, 255]),
    'fog':   np.array([40, 120, 255]),
    'night': np.array([35, 255, 255]),
    'rain':  np.array([35, 255, 255]),
}
_GRAY_CANNY = {
    'clear': (30, 90), 'fog': (20, 60), 'night': (20, 60), 'rain': (20, 60),
}


def _classify_weather(msg: CarlaWeatherParameters) -> str:
    if msg.fog_density > 40:
        return 'fog'
    if msg.precipitation > 30:
        return 'rain'
    if msg.sun_altitude_angle < 0:
        return 'night'
    return 'clear'


class PureVisionNode(Node):
    def __init__(self):
        super().__init__('pure_vision_node')

        self.declare_parameter('roi_yaml', DEFAULT_ROI_YAML)
        roi_path = self.get_parameter('roi_yaml').get_parameter_value().string_value

        self.roi_polygon  = self._load_roi(roi_path)
        self.bridge       = CvBridge()
        self.weather_mode = 'rain'

        # Hysteresis state machine
        #   SEARCHING → TRACKING : need CONFIRM_FRAMES consecutive "good" frames
        #   TRACKING  → SEARCHING: need LOST_FRAMES   consecutive "bad"  frames
        # "good" = both sides detected, center jump < JUMP_THRESH
        # "bad"  = MISS, single-side, or large jump
        self._tracking      = False  # current state
        self._good_streak   = 0      # consecutive good frames (used while SEARCHING)
        self._bad_streak    = 0      # consecutive bad  frames (used while TRACKING)
        self._prev_center   = None   # last published center (for jump check)
        self._CONFIRM  = 3           # good frames needed to enter TRACKING
        self._LOST     = 5           # bad  frames needed to exit  TRACKING
        self._JUMP     = 0.12        # max center shift per frame to count as "good"

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
        self.pub_center = self.create_publisher(Float32, '/lka/lane_center', 10)
        self.pub_image  = self.create_publisher(Image,   '/lka/pure_vision_image', 10)

        self.get_logger().info(f'Pure vision node ready | roi: {roi_path}')

    # ── ROI ────────────────────────────────────────────────────────────

    @staticmethod
    def _load_roi(path):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = yaml.full_load(f)
        raw = data.get('roi', {}).get('polygon', None)
        if raw is None:
            return None
        return np.array([list(p) for p in raw], dtype=np.int32)

    def _make_roi_mask(self, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        if self.roi_polygon is not None:
            cv2.fillPoly(mask, [self.roi_polygon], 255)
        else:
            mask[:] = 255
        return mask

    # ── Detection ──────────────────────────────────────────────────────

    @staticmethod
    def _fit_line(points, y_bottom, y_top):
        if len(points) < 2:
            return None
        pts = np.array(points)
        coeffs = np.polyfit(pts[:, 1], pts[:, 0], 1)
        return int(np.polyval(coeffs, y_bottom)), int(np.polyval(coeffs, y_top)), coeffs

    @staticmethod
    def _hough_fit(edge_img, y_bottom, y_top, slope_min, slope_max):
        lines = cv2.HoughLinesP(edge_img, rho=1, theta=np.pi / 180,
                                 threshold=15, minLineLength=20, maxLineGap=80)
        pts = []
        if lines is not None:
            for seg in lines:
                x1, y1, x2, y2 = seg[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if slope_min <= slope <= slope_max:
                    pts.extend([(x1, y1), (x2, y2)])
        return PureVisionNode._fit_line(pts, y_bottom, y_top)

    def _roi_center_x(self, w):
        """Midpoint of the ROI top edge — used as left/right split."""
        if self.roi_polygon is not None:
            top2 = self.roi_polygon[self.roi_polygon[:, 1].argsort()][:2]
            return int(top2[:, 0].mean())
        return w // 2

    def detect(self, img, weather):
        h, w = img.shape[:2]
        roi_mask = self._make_roi_mask(h, w)
        MARGIN   = int(w * 0.08)
        roi_cx   = self._roi_center_x(w)

        # Left: HSV yellow → dilate → left half of ROI
        hsv         = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, _HSV_LO[weather], _HSV_HI[weather])
        kernel      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        left_edges  = cv2.dilate(yellow_mask, kernel, iterations=2)
        left_edges  = cv2.bitwise_and(left_edges, roi_mask)
        left_edges[:, roi_cx + MARGIN:] = 0

        # Right: grayscale Canny → right half of ROI
        gray        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur   = cv2.GaussianBlur(gray, (5, 5), 0)
        glo, ghi    = _GRAY_CANNY[weather]
        right_edges = cv2.Canny(gray_blur, glo, ghi)
        right_edges = cv2.bitwise_and(right_edges, roi_mask)
        right_edges[:, :roi_cx - MARGIN] = 0

        y_bottom = h - 1
        y_top    = int(h * 0.55)
        y_ref    = int(h * 0.85)

        left_fit  = self._hough_fit(left_edges,  y_bottom, y_top, slope_min=-2.5, slope_max=-0.3)
        right_fit = self._hough_fit(right_edges, y_bottom, y_top, slope_min= 0.3, slope_max= 2.5)

        # Sanity checks (use roi_cx as the left/right boundary)
        if left_fit  and left_fit[0]  > roi_cx:         left_fit  = None
        if right_fit and right_fit[0] < roi_cx:         right_fit = None
        if left_fit  and left_fit[1]  > roi_cx * 1.1:   left_fit  = None
        if right_fit and right_fit[1] < roi_cx * 0.9:   right_fit = None
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
            rx = lx + 760
            center_norm = ((lx + rx) / 2.0) / w
        elif right_fit:
            rx = int(np.polyval(right_fit[2], y_ref))
            lx = rx - 760
            center_norm = ((lx + rx) / 2.0) / w

        return left_fit, right_fit, center_norm, lx, rx, y_bottom, y_top, y_ref, both_sides

    # ── Visualization ──────────────────────────────────────────────────

    @staticmethod
    def _draw(img, weather, left_fit, right_fit, center_norm, lx, rx, y_bottom, y_top, y_ref):
        h, w = img.shape[:2]
        vis  = img.copy()

        if left_fit and right_fit:
            poly = np.array([[left_fit[1], y_top],  [right_fit[1], y_top],
                              [right_fit[0], y_bottom], [left_fit[0], y_bottom]], dtype=np.int32)
            ov = vis.copy()
            cv2.fillPoly(ov, [poly], (0, 180, 0))
            vis = cv2.addWeighted(ov, 0.25, vis, 0.75, 0)

        if left_fit:
            cv2.line(vis, (left_fit[0], y_bottom), (left_fit[1], y_top), (0, 255, 0), 3)
            cv2.putText(vis, 'left_marking', (left_fit[0] + 10, y_bottom - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if right_fit:
            cv2.line(vis, (right_fit[0], y_bottom), (right_fit[1], y_top), (0, 255, 255), 3)
            cv2.putText(vis, 'right_edge', (right_fit[0] - 220, y_bottom - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if center_norm is not None and lx is not None and rx is not None:
            cx = int(center_norm * w)
            cv2.line(vis, (lx, y_ref), (rx, y_ref), (180, 180, 0), 1)
            cv2.circle(vis, (cx, y_ref), 14, (0, 0, 255), -1)
            cv2.line(vis, (w // 2, y_ref - 25), (w // 2, y_ref + 25), (255, 255, 0), 2)
            err = (w / 2) - cx
            cv2.putText(vis, f'center={center_norm:.3f}  err={err:+.0f}px',
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            cv2.putText(vis, 'NO DETECTION', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.putText(vis, f'weather: {weather}', (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        return vis

    # ── Callbacks ──────────────────────────────────────────────────────

    def weather_callback(self, msg: CarlaWeatherParameters):
        new_mode = _classify_weather(msg)
        if new_mode != self.weather_mode:
            self.weather_mode = new_mode
            self.get_logger().info(
                f'Weather → {self.weather_mode} '
                f'(fog={msg.fog_density:.0f} rain={msg.precipitation:.0f} '
                f'sun={msg.sun_altitude_angle:.1f}°)'
            )

    def _is_good(self, center_norm, both_sides):
        """True if this frame counts as a good detection."""
        if center_norm is None or not both_sides:
            return False
        if self._prev_center is not None and abs(center_norm - self._prev_center) > self._JUMP:
            return False
        return True

    def _stability_filter(self, center_norm, both_sides):
        """Hysteresis state machine. Returns accepted center or None."""
        good = self._is_good(center_norm, both_sides)

        if not self._tracking:
            # ── SEARCHING ──────────────────────────────────────────────
            if good:
                self._good_streak += 1
                self._prev_center  = center_norm
                if self._good_streak >= self._CONFIRM:
                    self._tracking    = True
                    self._bad_streak  = 0
                    self.get_logger().info('Hysteresis → TRACKING')
            else:
                self._good_streak = 0
                self._prev_center = None
            return None  # never publish while still searching

        else:
            # ── TRACKING ───────────────────────────────────────────────
            if good:
                self._bad_streak  = 0
                self._prev_center = center_norm
                return center_norm
            else:
                self._bad_streak += 1
                reason = 'MISS' if center_norm is None else \
                         'single-side' if not both_sides else 'jump'
                self.get_logger().warn(
                    f'Bad frame ({reason}) bad_streak={self._bad_streak}/{self._LOST}'
                )
                if self._bad_streak >= self._LOST:
                    self._tracking    = False
                    self._good_streak = 0
                    self._prev_center = None
                    self.get_logger().info('Hysteresis → SEARCHING')
                # Hold last known center during grace period
                return self._prev_center

    def image_callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        left_fit, right_fit, center_norm, lx, rx, y_bottom, y_top, y_ref, both_sides = self.detect(img, self.weather_mode)

        stable_center = self._stability_filter(center_norm, both_sides)
        center_val    = float(stable_center) if stable_center is not None else -1.0
        self.pub_center.publish(Float32(data=center_val))

        # Pass stable_center to draw so NO DETECTION shows when rejected
        vis = self._draw(img, self.weather_mode, left_fit, right_fit,
                         stable_center, lx, rx, y_bottom, y_top, y_ref)
        self.pub_image.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))

        raw_str = f'{center_norm:.3f}' if center_norm is not None else 'MISS'
        sides   = 'both' if both_sides else ('one' if center_norm is not None else 'none')
        state   = 'TRACKING' if self._tracking else f'SEARCHING({self._good_streak}/{self._CONFIRM})'
        self.get_logger().info(
            f'[{self.weather_mode}] {state} raw={raw_str}({sides}) out={center_val:.3f}'
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
