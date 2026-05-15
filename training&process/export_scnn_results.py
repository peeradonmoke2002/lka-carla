#!/usr/bin/python3
"""
Standalone SCNN inference script.
Loads model, runs on 4 weather reference images, exports scnn_result_*.png.

Usage:
    python3 training&process/export_scnn_results.py
    python3 training&process/export_scnn_results.py --weights models/scnn.pt --input Images/ --output Images/
"""
import argparse
import os
import sys

import cv2
import numpy as np

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAD_ROOT   = os.path.join(REPO_ROOT, 'pytorch-auto-drive')
sys.path.insert(0, PAD_ROOT)

NUM_CLASSES  = 3
INPUT_W      = 800
INPUT_H      = 288
Y_REF_RATIO  = 0.85
PROB_THRESH  = 0.3
LANE_WIDTH_PX = 760   # fallback when one side missing

WEATHERS = ['clear', 'fog', 'night', 'rain']


# ── Model ──────────────────────────────────────────────────────────────────────

def load_model(weights: str):
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
            flattened_size=(NUM_CLASSES * (INPUT_H // 16) * (INPUT_W // 16)),
        ),
    )

    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    import torch
    net = MODELS.from_dict(model_cfg).to(device)
    ckpt = torch.load(weights, map_location=device)
    net.load_state_dict(ckpt.get('model', ckpt))
    net.eval()
    print(f'[SCNN] model loaded from {weights} on {device}')
    return net, device


# ── Inference ──────────────────────────────────────────────────────────────────

def infer(model, device, img_bgr: np.ndarray):
    """Returns (left_prob, right_prob) resized to original image resolution."""
    import torch

    h, w = img_bgr.shape[:2]
    rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_W, INPUT_H)).astype(np.float32) / 255.0
    mean    = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std     = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = torch.from_numpy(((resized - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        result   = model(x)
        seg_pred = result['out']
        prob     = torch.softmax(seg_pred, dim=1)[0].cpu().numpy()

    left_prob  = cv2.resize(prob[1], (w, h))
    right_prob = cv2.resize(prob[2], (w, h))
    return left_prob, right_prob


# ── Lane fitting ───────────────────────────────────────────────────────────────

def fit_lane(prob_map: np.ndarray, x_min: int = 0, x_max: int = None):
    h, w = prob_map.shape
    if x_max is None:
        x_max = w
    ys, xs = np.where(prob_map > PROB_THRESH)
    mask   = (xs >= x_min) & (xs < x_max)
    ys, xs = ys[mask], xs[mask]
    if len(xs) < 10:
        return None
    weights = prob_map[ys, xs]
    try:
        coeffs = np.polyfit(ys, xs, 1, w=weights)
    except np.linalg.LinAlgError:
        return None
    return coeffs, int(ys.min()), int(ys.max())


def draw_lane_line(img, fit, color, label, label_offset_x):
    h, w         = img.shape[:2]
    coeffs, y_top, _ = fit
    ys = np.linspace(y_top, h - 1, 60).astype(int)
    xs = np.polyval(coeffs, ys).astype(int)
    for i in range(len(ys) - 1):
        if 0 <= xs[i] < w and 0 <= xs[i + 1] < w:
            cv2.line(img, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), color, 3)
    label_x = max(0, int(xs[-1]) + label_offset_x)
    cv2.putText(img, label, (label_x, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


# ── Process one image ──────────────────────────────────────────────────────────

def process(model, device, img_bgr: np.ndarray, weather: str) -> np.ndarray:
    h, w   = img_bgr.shape[:2]
    y_ref  = int(h * Y_REF_RATIO)
    img_cx = w // 2

    left_prob, right_prob = infer(model, device, img_bgr)

    left_fit  = fit_lane(left_prob,  x_min=0,      x_max=w * 3 // 4)
    right_fit = fit_lane(right_prob, x_min=w // 4, x_max=w)

    # Sanity: discard lines that crossed to the wrong side
    if left_fit  is not None and np.polyval(left_fit[0],  y_ref) > img_cx:
        left_fit = None
    if right_fit is not None and np.polyval(right_fit[0], y_ref) < img_cx:
        right_fit = None
    if left_fit and right_fit and \
            np.polyval(left_fit[0], y_ref) >= np.polyval(right_fit[0], y_ref):
        left_fit = None

    lx = float(np.polyval(left_fit[0],  y_ref)) if left_fit  is not None else None
    rx = float(np.polyval(right_fit[0], y_ref)) if right_fit is not None else None

    # Probability overlay
    vis     = img_bgr.copy()
    overlay = np.zeros_like(img_bgr)
    overlay[:, :, 1] = (left_prob  * 255).clip(0, 255).astype(np.uint8)
    overlay[:, :, 0] = (right_prob * 255).clip(0, 255).astype(np.uint8)
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

    if lx is None and rx is None:
        # No detection overlay
        bg = vis.copy()
        cv2.rectangle(bg, (0, 0), (w, 95), (0, 0, 180), -1)
        cv2.addWeighted(bg, 0.45, vis, 0.55, 0, vis)
        cv2.putText(vis, '!! NO DETECTION !!', (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(vis, f'SCNN   weather: {weather}', (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        return vis

    # Compute center
    if lx is not None and rx is not None:
        center_norm = (lx + rx) / 2.0 / w
    elif lx is not None:
        center_norm = (lx + lx + LANE_WIDTH_PX) / 2.0 / w
    else:
        center_norm = (rx - LANE_WIDTH_PX + rx) / 2.0 / w

    # Draw fitted lines
    if left_fit is not None and right_fit is not None:
        common_y_top = max(left_fit[1], right_fit[1])
        l_draw = (left_fit[0],  common_y_top, left_fit[2])
        r_draw = (right_fit[0], common_y_top, right_fit[2])
    else:
        l_draw, r_draw = left_fit, right_fit

    if l_draw is not None: draw_lane_line(vis, l_draw, (0, 255, 0),   'left_marking',   10)
    if r_draw is not None: draw_lane_line(vis, r_draw, (0, 255, 255), 'right_edge',    -160)

    # Keypoints
    if lx is not None:
        cv2.circle(vis, (int(np.clip(lx, 0, w - 1)), y_ref), 8, (0, 255, 0), -1)
    if rx is not None:
        cv2.circle(vis, (int(np.clip(rx, 0, w - 1)), y_ref), 8, (255, 0, 0), -1)
    cx = int(np.clip(center_norm * w, 0, w - 1))
    cv2.circle(vis, (cx, y_ref), 10, (0, 0, 255), -1)
    cv2.line(vis, (img_cx, y_ref - 25), (img_cx, y_ref + 25), (255, 255, 0), 2)
    cv2.arrowedLine(vis, (img_cx, y_ref), (cx, y_ref), (0, 255, 255), 2, tipLength=0.2)

    lx_txt = f'{lx:.0f}' if lx is not None else 'N/A'
    rx_txt = f'{rx:.0f}' if rx is not None else 'N/A'
    cv2.putText(vis, f'SCNN  lx={lx_txt}  rx={rx_txt}  center={center_norm:.3f}',
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f'weather: {weather}', (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    print(f'  [{weather}] lx={lx_txt}  rx={rx_txt}  center={center_norm:.3f}')
    return vis


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Export SCNN result images for 4 weather conditions')
    parser.add_argument('--weights', default=os.path.join(REPO_ROOT, 'models', 'scnn.pt'),
                        help='Path to SCNN model weights')
    parser.add_argument('--input',   default=os.path.join(REPO_ROOT, 'Images'),
                        help='Directory containing clear.png / fog.png / night.png / rain.png')
    parser.add_argument('--output',  default=os.path.join(REPO_ROOT, 'Images'),
                        help='Directory to save scnn_result_*.png')
    args = parser.parse_args()

    model, device = load_model(args.weights)
    os.makedirs(args.output, exist_ok=True)

    for weather in WEATHERS:
        in_path  = os.path.join(args.input,  f'{weather}.png')
        out_path = os.path.join(args.output, f'scnn_result_{weather}.png')

        if not os.path.exists(in_path):
            print(f'[SKIP] {in_path} not found')
            continue

        print(f'Processing {weather} ...')
        img = cv2.imread(in_path)
        if img is None:
            print(f'[ERROR] could not read {in_path}')
            continue

        result = process(model, device, img, weather)
        cv2.imwrite(out_path, result)
        print(f'  → saved {out_path}')

    print('Done.')


if __name__ == '__main__':
    main()
