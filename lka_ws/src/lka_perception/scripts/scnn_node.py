#!/usr/bin/python3
"""
SCNN Lane Detection Node (pytorch-auto-drive backend)
https://github.com/voldemortX/pytorch-auto-drive

Prerequisites
─────────────
1. Clone the repo and install deps:
     git clone https://github.com/voldemortX/pytorch-auto-drive.git ~/pytorch-auto-drive
     cd ~/pytorch-auto-drive && pip install -r requirements.txt

2. Convert dataset (run once):
     python3 /home/peeradon/lka-carla-yolo/training/yolo2scnn.py

3. Copy custom config and dataset class into pytorch-auto-drive:
     cp /home/peeradon/lka-carla-yolo/training/scnn_lka_config.py \
        ~/pytorch-auto-drive/configs/lane_detection/scnn/resnet18_lka.py
     # Register LkaAsSegmentation in utils/datasets/__init__.py

4. Train:
     cd ~/pytorch-auto-drive
     python main_landet.py --config configs/lane_detection/scnn/resnet18_lka.py

5. Set 'weights' param below to the trained checkpoint.

Publishes
─────────
  /lka/scnn/lane_center   lka_msgs/LaneCenter
  /lka/scnn_image         sensor_msgs/Image  (debug overlay)
"""
import sys
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from carla_msgs.msg import CarlaWeatherParameters
from lka_msgs.msg import LaneCenter
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np

PAD_ROOT = os.path.expanduser('~/lka-carla-yolo/pytorch-auto-drive')
if PAD_ROOT not in sys.path:
    sys.path.insert(0, PAD_ROOT)

# pytorch-auto-drive colour → class index mapping used at training time
# (CULane palette, re-mapped to our 3-class case)
# class 0 = background, 1 = left_marking, 2 = right_edge
NUM_CLASSES = 3


class SCNNNode(Node):
    def __init__(self):
        super().__init__('scnn_node')

        self.declare_parameter('weights',     '/home/peeradon/pytorch-auto-drive/checkpoints/resnet18_scnn_lka/model.pt')
        self.declare_parameter('input_w',     800)
        self.declare_parameter('input_h',     288)
        self.declare_parameter('y_ref_ratio', 0.85)
        self.declare_parameter('prob_thresh',  0.3)   # min softmax prob to count a pixel
        self.declare_parameter('lane_width_px', 760)  # fallback when one side missing
        self.declare_parameter('lane_width_m', 4.0)   # Town01 driving lane width
        self.declare_parameter('enable_hysteresis', False)
        self.declare_parameter('confirm_frames', 3)
        self.declare_parameter('lost_frames', 5)
        self.declare_parameter('jump_thresh', 0.12)

        weights        = self.get_parameter('weights').value
        self.in_w      = self.get_parameter('input_w').value
        self.in_h      = self.get_parameter('input_h').value
        self.y_ref_r   = self.get_parameter('y_ref_ratio').value
        self.thresh    = self.get_parameter('prob_thresh').value
        self.lane_w    = self.get_parameter('lane_width_px').value
        self.lane_width_m      = self.get_parameter('lane_width_m').value
        self.enable_hysteresis = self.get_parameter('enable_hysteresis').value
        self.confirm_frames    = self.get_parameter('confirm_frames').value
        self.lost_frames       = self.get_parameter('lost_frames').value
        self.jump_thresh       = self.get_parameter('jump_thresh').value

        self.model, self.device = self._load_model(weights)
        self.bridge = CvBridge()
        self._cte   = None   # latest GT cross-track error (m)

        # Hysteresis state (used when enable_hysteresis=True)
        self.tracking    = False
        self.good_streak = 0
        self.bad_streak  = 0
        self.prev_center = None

        self.create_subscription(Image, '/carla/ego_vehicle/CAM_FRONT/image', self.image_callback, 10)
        self.create_subscription(CarlaWeatherParameters, '/carla/weather_control', self.weather_callback, 10)
        self.create_subscription(Float64, '/lka/gt/cross_track_m', self._cte_cb, 10)

        self.pub_center = self.create_publisher(LaneCenter, '/lka/scnn/lane_center', 10)
        self.pub_image  = self.create_publisher(Image,      '/lka/scnn_image',       10)

        self.get_logger().info(f'SCNN node ready | weights: {weights} | device: {self.device}')

    # ── Model ──────────────────────────────────────────────────────────

    def _load_model(self, weights: str):
        import torch
        from utils.models.builder import MODELS

        model_cfg = dict(
            name='standard_segmentation_model',
            backbone_cfg=dict(
                name='predefined_resnet_backbone',
                backbone_name='resnet18',
                return_layer='layer4',
                pretrained=False,
                replace_stride_with_dilation=[False, True, True],
            ),
            reducer_cfg=dict(name='RESAReducer', in_channels=512, reduce=128),
            spatial_conv_cfg=dict(name='SpatialConv', num_channels=128),
            classifier_cfg=dict(
                name='DeepLabV1Head',
                in_channels=128,
                num_classes=NUM_CLASSES,
                dilation=1,
            ),
            lane_classifier_cfg=dict(
                name='SimpleLaneExist',
                num_output=NUM_CLASSES - 1,
                # num_classes * (in_h/8/2) * (in_w/8/2) — must match training config
                flattened_size=(NUM_CLASSES * (self.in_h // 16) * (self.in_w // 16)),
            ),
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = MODELS.from_dict(model_cfg).to(device)
        ckpt = torch.load(weights, map_location=device)
        net.load_state_dict(ckpt.get('model', ckpt))
        net.eval()
        self.get_logger().info(f'SCNN loaded on {device}')
        return net, device

    # ── Inference ──────────────────────────────────────────────────────

    def _infer(self, img_bgr: np.ndarray):
        """Returns (left_prob, right_prob) at original image resolution."""
        import torch

        h_orig, w_orig = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.in_w, self.in_h)).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = torch.from_numpy(((resized - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self.model(x)        # OrderedDict with 'out' and 'lane'
            seg_pred = result['out']      # (1, NUM_CLASSES, H_feat, W_feat)
            prob = torch.softmax(seg_pred, dim=1)[0].cpu().numpy()  # (C, H_feat, W_feat)

        left_prob  = cv2.resize(prob[1], (w_orig, h_orig))  # class 1 = left_marking
        right_prob = cv2.resize(prob[2], (w_orig, h_orig))  # class 2 = right_edge
        return left_prob, right_prob

    # ── Lane fitting ───────────────────────────────────────────────────

    def _fit_lane(self, prob_map: np.ndarray, x_min: int = 0, x_max: int = None):
        """Fit line x=f(y) through high-prob pixels within x bounds.
        Returns (coeffs, y_top, y_bot) or None — same tuple as YOLO/PV."""
        h, w = prob_map.shape
        if x_max is None:
            x_max = w
        ys, xs = np.where(prob_map > self.thresh)
        mask = (xs >= x_min) & (xs < x_max)
        ys, xs = ys[mask], xs[mask]
        if len(xs) < 10:
            return None
        weights = prob_map[ys, xs]
        try:
            coeffs = np.polyfit(ys, xs, 1, w=weights)
        except np.linalg.LinAlgError:
            return None
        return coeffs, int(ys.min()), int(ys.max())

    @staticmethod
    def _draw_lane_line(img, fit, color, label, label_offset_x):
        """Draw fitted line within its detected y range — same as YOLO/PV."""
        h, w = img.shape[:2]
        coeffs, y_top, _ = fit
        ys = np.linspace(y_top, h - 1, 60).astype(int)
        xs = np.polyval(coeffs, ys).astype(int)
        for i in range(len(ys) - 1):
            if 0 <= xs[i] < w and 0 <= xs[i + 1] < w:
                cv2.line(img, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), color, 3)
        label_x = max(0, int(xs[-1]) + label_offset_x)
        cv2.putText(img, label, (label_x, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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

    # ── Callbacks ──────────────────────────────────────────────────────

    def _cte_cb(self, msg: Float64):
        self._cte = msg.data

    def weather_callback(self, _msg):
        pass   # SCNN is weather-agnostic

    def image_callback(self, msg: Image):
        if self.model is None:
            return

        img  = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = img.shape[:2]
        y_ref = int(h * self.y_ref_r)
        img_cx = w // 2

        try:
            left_prob, right_prob = self._infer(img)
        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}', throttle_duration_sec=5.0)
            return

        left_fit  = self._fit_lane(left_prob,  x_min=0,      x_max=w * 3 // 4)
        right_fit = self._fit_lane(right_prob, x_min=w // 4, x_max=w)

        # Sanity checks — discard lines that crossed to the wrong side
        if left_fit  is not None and np.polyval(left_fit[0],  y_ref) > img_cx:
            left_fit = None
        if right_fit is not None and np.polyval(right_fit[0], y_ref) < img_cx:
            right_fit = None
        if left_fit and right_fit and \
                np.polyval(left_fit[0], y_ref) >= np.polyval(right_fit[0], y_ref):
            left_fit = None

        lx = float(np.polyval(left_fit[0],  y_ref)) if left_fit  is not None else None
        rx = float(np.polyval(right_fit[0], y_ref)) if right_fit is not None else None

        both_sides = lx is not None and rx is not None

        if self.enable_hysteresis:
            # Synthesize missing side, then apply hysteresis filter
            if both_sides:
                center_norm_raw = (lx + rx) / 2.0 / w
            elif lx is not None:
                center_norm_raw = (lx + lx + self.lane_w) / 2.0 / w
            elif rx is not None:
                center_norm_raw = (rx - self.lane_w + rx) / 2.0 / w
            else:
                center_norm_raw = None
            stable   = self.stability_filter(center_norm_raw, both_sides)
            detected = stable is not None
            center_norm = float(stable) if detected else -1.0
        else:
            # Raw: only publish when both sides found; discard single-side synthesis
            if both_sides:
                center_norm = (lx + rx) / 2.0 / w
                detected    = True
            else:
                center_norm = -1.0
                detected    = False

        lane_msg              = LaneCenter()
        lane_msg.header.stamp = self.get_clock().now().to_msg()
        lane_msg.center       = center_norm
        lane_msg.confidence   = 0.0
        lane_msg.detected     = detected
        lane_msg.lx           = float(lx) if lx is not None else -1.0
        lane_msg.rx           = float(rx) if rx is not None else -1.0
        self.pub_center.publish(lane_msg)

        vis = img.copy()
        overlay = np.zeros_like(img)
        overlay[:, :, 1] = (left_prob  * 255).clip(0, 255).astype(np.uint8)
        overlay[:, :, 0] = (right_prob * 255).clip(0, 255).astype(np.uint8)
        cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

        if not detected:
            bg = vis.copy()
            cv2.rectangle(bg, (0, 0), (w, 95), (0, 0, 180), -1)
            cv2.addWeighted(bg, 0.45, vis, 0.55, 0, vis)
            cv2.putText(vis, '!! NO DETECTION !!', (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(vis, 'SCNN', (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            # Use same y_top for both lines so they have equal length
            if left_fit is not None and right_fit is not None:
                common_y_top = max(left_fit[1], right_fit[1])
                l_fit_draw = (left_fit[0],  common_y_top, left_fit[2])
                r_fit_draw = (right_fit[0], common_y_top, right_fit[2])
            else:
                l_fit_draw = left_fit
                r_fit_draw = right_fit
            if l_fit_draw is not None: self._draw_lane_line(vis, l_fit_draw, (0, 255, 0),   'left_marking',  10)
            if r_fit_draw is not None: self._draw_lane_line(vis, r_fit_draw, (0, 255, 255), 'right_edge',  -160)
            if lx is not None:
                cv2.circle(vis, (int(np.clip(lx, 0, w-1)), y_ref), 8, (0, 255, 0), -1)
            if rx is not None:
                cv2.circle(vis, (int(np.clip(rx, 0, w-1)), y_ref), 8, (255, 0, 0), -1)
            cx = int(np.clip(center_norm * w, 0, w-1))
            cv2.circle(vis, (cx, y_ref), 10, (0, 0, 255), -1)
            cv2.line(vis, (img_cx, y_ref - 25), (img_cx, y_ref + 25), (255, 255, 0), 2)
            cv2.arrowedLine(vis, (img_cx, y_ref), (cx, y_ref), (0, 255, 255), 2, tipLength=0.2)

            cv2.putText(vis, f'SCNN   center: {center_norm:.3f}', (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if self._cte is not None:
                true_center = 0.5 - (self._cte / self.lane_width_m)
                real_error  = center_norm - true_center
                err_label   = f'GT err: {real_error:+.3f}'
                err_color   = (0, 255, 0) if abs(real_error) < 0.05 else (0, 0, 255)
            else:
                real_error = center_norm - 0.5
                err_label  = f'err:    {real_error:+.3f}  (no GT)'
                err_color  = (255, 255, 0)
            cv2.putText(vis, err_label, (10, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, err_color, 2)

        self.pub_image.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))

        self.get_logger().info(
            f'center={center_norm:.3f}  lx={lx}  rx={rx}',
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = SCNNNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
