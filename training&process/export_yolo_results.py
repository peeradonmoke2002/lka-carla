#!/usr/bin/python3
"""
Standalone YOLO inference script.
Loads model, runs on 4 weather reference images, exports yolo_result_*.png.

Usage:
    python3 training&process/export_yolo_results.py
    python3 training&process/export_yolo_results.py --weights models/best_vision.pt --input Images/ --output Images/
"""
import argparse
import os

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MIN_CONF      = 0.60
POLY_DEGREE   = 2
MIN_PIXELS    = 50
Y_TOP_RATIO   = 0.50
Y_REF_RATIO   = 0.85
LANE_WIDTH_PX = 760

WEATHERS = ['clear', 'fog', 'night', 'rain']


# ── Lane mask & fitting ────────────────────────────────────────────────────────

def build_lane_masks(results, h, w):
    left_mask  = np.zeros((h, w), dtype=np.uint8)
    right_mask = np.zeros((h, w), dtype=np.uint8)
    if results.masks is None or len(results.boxes) == 0:
        return left_mask, right_mask
    for poly, conf, cls in zip(results.masks.xy,
                                results.boxes.conf.cpu().numpy(),
                                results.boxes.cls.cpu().numpy().astype(int)):
        if len(poly) < 3 or conf < MIN_CONF:
            continue
        pts = poly.astype(np.int32)
        if cls == 0:
            cv2.fillPoly(left_mask,  [pts], 255)
        else:
            cv2.fillPoly(right_mask, [pts], 255)
    return left_mask, right_mask


def fit_lane(mask, roi_top, h):
    ys, xs = np.where(mask[roi_top:h] > 0)
    if len(xs) < MIN_PIXELS:
        return None
    ys_abs = ys + roi_top
    coeffs = np.polyfit(ys_abs, xs, POLY_DEGREE)
    return coeffs, int(ys_abs.min()), int(ys_abs.max())


# ── Visualization ──────────────────────────────────────────────────────────────

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


# ── Process one image ──────────────────────────────────────────────────────────

def process(model, img_bgr: np.ndarray, weather: str) -> np.ndarray:
    h, w   = img_bgr.shape[:2]
    y_ref  = int(h * Y_REF_RATIO)
    y_top  = int(h * Y_TOP_RATIO)
    img_cx = w // 2

    results   = model.predict(img_bgr, conf=MIN_CONF, verbose=False)[0]
    mean_conf = float(results.boxes.conf.cpu().numpy().mean()) if len(results.boxes) else 0.0

    left_mask, right_mask = build_lane_masks(results, h, w)
    left_fit  = fit_lane(left_mask,  y_top, h - 1)
    right_fit = fit_lane(right_mask, y_top, h - 1)

    annotated = results.plot()

    if left_fit is None and right_fit is None:
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 95), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0, annotated)
        cv2.putText(annotated, '!! NO DETECTION !!', (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(annotated, f'conf: {mean_conf:.3f}  weather: {weather}', (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        return annotated

    lx = float(np.polyval(left_fit[0],  y_ref)) if left_fit  is not None else None
    rx = float(np.polyval(right_fit[0], y_ref)) if right_fit is not None else None

    if lx is not None and rx is not None:
        center_px = (lx + rx) / 2.0
    elif lx is not None:
        center_px = lx + LANE_WIDTH_PX / 2.0
    else:
        center_px = rx - LANE_WIDTH_PX / 2.0

    center_norm = center_px / w

    if left_fit is not None and right_fit is not None:
        common_y_top = max(left_fit[1], right_fit[1])
        draw_lane_line(annotated, (left_fit[0],  common_y_top, left_fit[2]),  (0, 255, 0),   'left_marking',   10)
        draw_lane_line(annotated, (right_fit[0], common_y_top, right_fit[2]), (0, 255, 255), 'right_edge',    -160)
    else:
        if left_fit  is not None: draw_lane_line(annotated, left_fit,  (0, 255, 0),   'left_marking',   10)
        if right_fit is not None: draw_lane_line(annotated, right_fit, (0, 255, 255), 'right_edge',    -160)

    cx = int(center_px)
    cv2.circle(annotated, (cx, y_ref), 10, (0, 0, 255), -1)
    cv2.line(annotated, (img_cx, y_ref - 25), (img_cx, y_ref + 25), (255, 255, 0), 2)
    cv2.arrowedLine(annotated, (img_cx, y_ref), (cx, y_ref), (0, 255, 255), 2, tipLength=0.2)

    lx_txt = f'{lx:.0f}' if lx is not None else 'N/A'
    rx_txt = f'{rx:.0f}' if rx is not None else 'N/A'
    cv2.putText(annotated, f'YOLO  lx={lx_txt}  rx={rx_txt}  center={center_norm:.3f}',
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f'conf: {mean_conf:.3f}  weather: {weather}',
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    print(f'  [{weather}] lx={lx_txt}  rx={rx_txt}  center={center_norm:.3f}  conf={mean_conf:.3f}')
    return annotated


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Export YOLO result images for 4 weather conditions')
    parser.add_argument('--weights', default=os.path.join(REPO_ROOT, 'models', 'best_vision.pt'))
    parser.add_argument('--input',   default=os.path.join(REPO_ROOT, 'Images'))
    parser.add_argument('--output',  default=os.path.join(REPO_ROOT, 'Images'))
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    print(f'[YOLO] model loaded from {args.weights}')
    os.makedirs(args.output, exist_ok=True)

    for weather in WEATHERS:
        in_path  = os.path.join(args.input,  f'{weather}.png')
        out_path = os.path.join(args.output, f'yolo_result_{weather}.png')
        if not os.path.exists(in_path):
            print(f'[SKIP] {in_path} not found')
            continue
        print(f'Processing {weather} ...')
        img = cv2.imread(in_path)
        if img is None:
            print(f'[ERROR] could not read {in_path}')
            continue
        result = process(model, img, weather)
        cv2.imwrite(out_path, result)
        print(f'  → saved {out_path}')

    print('Done.')


if __name__ == '__main__':
    main()
