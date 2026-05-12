#!/usr/bin/python3
import os
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from carla_msgs.msg import CarlaWeatherParameters
from lka_msgs.msg import LaneCenter
from std_msgs.msg import Float64
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
        self.declare_parameter('lane_width_m', 4.0)  # Town01 driving lane width
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
        self.declare_parameter('enable_hysteresis', False)
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
        self.lane_width_m      = self.get_parameter('lane_width_m').value
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
        self.enable_hysteresis = self.get_parameter('enable_hysteresis').value
        
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
        self._cte         = None   # latest GT cross-track error (m)

        # Hysteresis state machine
        self.tracking    = False
        self.good_streak = 0
        self.bad_streak  = 0
        self.prev_center = None

        self.create_subscription(Image, '/carla/ego_vehicle/CAM_FRONT/image', self.image_callback, 10)
        self.create_subscription(CarlaWeatherParameters, '/carla/weather_control', self.weather_callback, 10)
        self.create_subscription(Float64, '/lka/gt/cross_track_m', self._cte_cb, 10)

        # self.pub_center       = self.create_publisher(LaneCenter, '/lka/lane_center',             10)
        self.pub_center_debug = self.create_publisher(LaneCenter, '/lka/pure_vision/lane_center', 10)
        self.pub_image        = self.create_publisher(Image,      '/lka/pure_vision_image',       10)

        self.get_logger().info(f'Pure vision node ready | roi: {roi_path}')

    # ── GT callback ───────────────────────────────────────────────────

    def _cte_cb(self, msg: Float64):
        self._cte = msg.data

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
    def fit_lane(points, y_bottom, y_top):
        """Fit a line through (x,y) points. Returns (coeffs, y_top, y_bottom) or None."""
        if len(points) < 2:
            return None
        pts    = np.array(points)
        coeffs = np.polyfit(pts[:, 1], pts[:, 0], 1)
        return coeffs, y_top, y_bottom

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
        return self.fit_lane(pts, y_bottom, y_top)

    # ── Lane detection ─────────────────────────────────────────────────

    def detect_lanes(self, img, weather):
        """Detect left and right lane lines and compute the normalized ego-lane center.

        Returns (left_fit, right_fit, center_norm, lx, rx, y_bottom, y_top, y_ref, both_sides).
        center_norm is None when detection fails. both_sides is True only when both lines found.
        left_fit / right_fit = (coeffs, y_top, y_bot) or None.
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
        if left_fit and int(np.polyval(left_fit[0], y_bottom)) > roi_cx:
            left_fit = None
        if right_fit and int(np.polyval(right_fit[0], y_bottom)) < roi_cx:
            right_fit = None
        if left_fit and int(np.polyval(left_fit[0], y_top)) > roi_cx * 1.1:
            left_fit = None
        if right_fit and int(np.polyval(right_fit[0], y_top)) < roi_cx * 0.9:
            right_fit = None
        if left_fit and right_fit and \
                int(np.polyval(left_fit[0], y_top)) >= int(np.polyval(right_fit[0], y_top)):
            left_fit = None

        center_norm = lx = rx = None
        both_sides = False

        if left_fit and right_fit:
            lx = int(np.polyval(left_fit[0],  y_ref))
            rx = int(np.polyval(right_fit[0], y_ref))
            center_norm = ((lx + rx) / 2.0) / w
            both_sides  = True
        elif left_fit:
            lx = int(np.polyval(left_fit[0], y_ref))
            rx = lx + self.lane_width_px
            center_norm = ((lx + rx) / 2.0) / w
        elif right_fit:
            rx = int(np.polyval(right_fit[0], y_ref))
            lx = rx - self.lane_width_px
            center_norm = ((lx + rx) / 2.0) / w

        return left_fit, right_fit, center_norm, lx, rx, y_bottom, y_top, y_ref, both_sides

    # ── Visualization ──────────────────────────────────────────────────

    @staticmethod
    def draw_lane_line(img, fit, color, label, label_offset_x):
        h, w = img.shape[:2]
        coeffs, y_top, _ = fit
        ys = np.linspace(y_top, h - 1, 60).astype(int)
        xs = np.polyval(coeffs, ys).astype(int)
        for i in range(len(ys) - 1):
            if 0 <= xs[i] < w and 0 <= xs[i + 1] < w:
                cv2.line(img, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), color, 3)
        label_x = max(0, int(xs[-1]) + label_offset_x)
        cv2.putText(img, label, (label_x, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def draw_debug(self, img, weather, left_fit, right_fit, center_norm, lx, rx, y_bottom, y_top, y_ref):
        h, w   = img.shape[:2]
        vis    = img.copy()
        img_cx = w // 2

        if left_fit is not None and right_fit is not None:
            common_y_top = max(left_fit[1], right_fit[1])
            PureVisionNode.draw_lane_line(vis, (left_fit[0],  common_y_top, left_fit[2]),  (0, 255, 0),   'left_marking',  10)
            PureVisionNode.draw_lane_line(vis, (right_fit[0], common_y_top, right_fit[2]), (0, 255, 255), 'right_edge', -160)
        else:
            if left_fit  is not None: PureVisionNode.draw_lane_line(vis, left_fit,  (0, 255, 0),   'left_marking',  10)
            if right_fit is not None: PureVisionNode.draw_lane_line(vis, right_fit, (0, 255, 255), 'right_edge', -160)

        if center_norm is None:
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, 0), (w, 95), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)
            cv2.putText(vis, '!! NO DETECTION !!', (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(vis, f'weather: {weather}', (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return vis

        cx = int(center_norm * w)

        # Real GT error (when GT available), else fallback to image-centre error
        if self._cte is not None:
            true_center = 0.5 - (self._cte / self.lane_width_m)
            real_error  = center_norm - true_center
            error_label = f'GT err: {real_error:+.3f}'
            err_color   = (0, 255, 0) if abs(real_error) < 0.05 else (0, 0, 255)
        else:
            real_error  = center_norm - 0.5
            error_label = f'err:    {real_error:+.3f}  (no GT)'
            err_color   = (0, 255, 0) if abs(cx - img_cx) < 50 else (0, 0, 255)

        cv2.circle(vis, (cx, y_ref), 10, (0, 0, 255), -1)
        cv2.line(vis, (img_cx, y_ref - 25), (img_cx, y_ref + 25), (255, 255, 0), 2)
        cv2.arrowedLine(vis, (img_cx, y_ref), (cx, y_ref), (0, 255, 255), 2, tipLength=0.2)

        cv2.putText(vis, f'PV     center: {center_norm:.3f}', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, error_label,                         (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, err_color, 2)
        cv2.putText(vis, f'weather: {weather}',               (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

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
        if msg.fog_density > 40.0:
            new_mode = 'fog'
        elif msg.precipitation > 30.0:
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
            self.detect_lanes(img, self.weather_mode)

        if self.enable_hysteresis:
            stable_center = self.stability_filter(center_norm, both_sides)
            detected      = stable_center is not None
            center_val    = float(stable_center) if detected else -1.0
            pub_lx        = float(lx) if lx is not None else -1.0
            pub_rx        = float(rx) if rx is not None else -1.0
        else:
            # Raw: only publish when both sides are found; discard single-side synthesis
            if both_sides:
                stable_center = center_norm
                detected      = True
                center_val    = float(center_norm)
                pub_lx        = float(lx)
                pub_rx        = float(rx)
            else:
                stable_center = None
                detected      = False
                center_val    = -1.0
                pub_lx        = -1.0
                pub_rx        = -1.0

        lane_msg              = LaneCenter()
        lane_msg.header.stamp = self.get_clock().now().to_msg()
        lane_msg.center       = center_val
        lane_msg.confidence   = 0.0
        lane_msg.detected     = detected
        lane_msg.lx           = pub_lx
        lane_msg.rx           = pub_rx
        self.pub_center_debug.publish(lane_msg)

        vis = self.draw_debug(img, self.weather_mode, left_fit, right_fit,
                              stable_center, lx, rx, y_bottom, y_top, y_ref)
        self.pub_image.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))

        raw_str = f'{center_norm:.3f}' if center_norm is not None else 'MISS'
        sides   = 'both' if both_sides else ('one' if center_norm is not None else 'none')
        if self.enable_hysteresis:
            state = 'TRACKING' if self.tracking else f'SEARCHING({self.good_streak}/{self.confirm_frames})'
        else:
            state = 'RAW'
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
