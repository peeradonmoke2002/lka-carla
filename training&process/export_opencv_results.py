#!/usr/bin/python3
"""
Standalone Pure Vision (OpenCV) inference script.
Runs HSV + Canny + Hough pipeline on 4 weather reference images, exports opencv_result_*.png.

Usage:
    python3 training&process/export_opencv_results.py
    python3 training&process/export_opencv_results.py --input Images/ --output Images/
"""
import argparse
import os

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROI_YAML  = os.path.join(REPO_ROOT, 'lka_ws', 'src',
                          'lka_dataset_collection', 'config', 'roi.yaml')

Y_TOP_RATIO       = 0.55
Y_REF_RATIO       = 0.85
ROI_MARGIN_RATIO  = 0.08
LANE_WIDTH_PX     = 760
HOUGH_THRESH      = 15
HOUGH_MIN_LEN     = 20
HOUGH_MAX_GAP     = 80
LEFT_SLOPE_MIN    = -2.5
LEFT_SLOPE_MAX    = -0.3
RIGHT_SLOPE_MIN   =  0.3
RIGHT_SLOPE_MAX   =  2.5

HSV_LO = {
    'clear': np.array([10,  30, 250], dtype=np.uint8),
    'fog':   np.array([10,   5, 180], dtype=np.uint8),
    'night': np.array([10, 150,  30], dtype=np.uint8),
    'rain':  np.array([15,  25, 150], dtype=np.uint8),
}
HSV_HI = {
    'clear': np.array([40, 120, 255], dtype=np.uint8),
    'fog':   np.array([40, 120, 255], dtype=np.uint8),
    'night': np.array([35, 255, 255], dtype=np.uint8),
    'rain':  np.array([35, 255, 255], dtype=np.uint8),
}
CANNY = {
    'clear': (30, 90),
    'fog':   (20, 60),
    'night': (20, 60),
    'rain':  (20, 60),
}

WEATHERS = ['clear', 'fog', 'night', 'rain']


# ── ROI helpers ────────────────────────────────────────────────────────────────

def load_roi(path):
    if not os.path.exists(path):
        print(f'[WARN] roi.yaml not found at {path}, using full image')
        return None
    import yaml
    with open(path) as f:
        data = yaml.full_load(f)
    raw = data.get('roi', {}).get('polygon', None)
    if raw is None:
        return None
    return np.array([list(p) for p in raw], dtype=np.int32)


def make_roi_mask(h, w, polygon):
    mask = np.zeros((h, w), dtype=np.uint8)
    if polygon is not None:
        cv2.fillPoly(mask, [polygon], 255)
    else:
        mask[:] = 255
    return mask


def roi_center_x(w, polygon):
    if polygon is not None:
        top2 = polygon[polygon[:, 1].argsort()][:2]
        return int(top2[:, 0].mean())
    return w // 2


# ── Edge detection ─────────────────────────────────────────────────────────────

def build_edge_images(img, weather, roi_mask, roi_cx, margin):
    hsv         = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, HSV_LO[weather], HSV_HI[weather])
    kernel      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    left_edges  = cv2.dilate(yellow_mask, kernel, iterations=2)
    left_edges  = cv2.bitwise_and(left_edges, roi_mask)
    left_edges[:, roi_cx + margin:] = 0

    gray         = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur    = cv2.GaussianBlur(gray, (5, 5), 0)
    clo, chi     = CANNY[weather]
    right_edges  = cv2.Canny(gray_blur, clo, chi)
    right_edges  = cv2.bitwise_and(right_edges, roi_mask)
    right_edges[:, :roi_cx - margin] = 0

    return left_edges, right_edges


# ── Line fitting ───────────────────────────────────────────────────────────────

def fit_lane_from_points(points, y_bottom, y_top):
    if len(points) < 2:
        return None
    pts    = np.array(points)
    coeffs = np.polyfit(pts[:, 1], pts[:, 0], 1)
    return coeffs, y_top, y_bottom


def hough_fit(edge_img, y_bottom, y_top, slope_min, slope_max):
    lines = cv2.HoughLinesP(
        edge_img, rho=1, theta=np.pi / 180,
        threshold=HOUGH_THRESH,
        minLineLength=HOUGH_MIN_LEN,
        maxLineGap=HOUGH_MAX_GAP,
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
    return fit_lane_from_points(pts, y_bottom, y_top)


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

def process(img_bgr: np.ndarray, weather: str, polygon) -> np.ndarray:
    h, w   = img_bgr.shape[:2]
    y_ref  = int(h * Y_REF_RATIO)
    y_top  = int(h * Y_TOP_RATIO)
    img_cx = w // 2

    roi_mask = make_roi_mask(h, w, polygon)
    margin   = int(w * ROI_MARGIN_RATIO)
    roi_cx   = roi_center_x(w, polygon)

    left_edges, right_edges = build_edge_images(img_bgr, weather, roi_mask, roi_cx, margin)

    left_fit  = hough_fit(left_edges,  h - 1, y_top, LEFT_SLOPE_MIN,  LEFT_SLOPE_MAX)
    right_fit = hough_fit(right_edges, h - 1, y_top, RIGHT_SLOPE_MIN, RIGHT_SLOPE_MAX)

    # Sanity checks
    if left_fit  and int(np.polyval(left_fit[0],  h - 1)) > roi_cx:   left_fit  = None
    if right_fit and int(np.polyval(right_fit[0], h - 1)) < roi_cx:   right_fit = None
    if left_fit  and int(np.polyval(left_fit[0],  y_top)) > roi_cx * 1.1: left_fit = None
    if right_fit and int(np.polyval(right_fit[0], y_top)) < roi_cx * 0.9: right_fit = None
    if left_fit and right_fit and \
            int(np.polyval(left_fit[0], y_top)) >= int(np.polyval(right_fit[0], y_top)):
        left_fit = None

    lx = rx = center_norm = None
    if left_fit and right_fit:
        lx = int(np.polyval(left_fit[0],  y_ref))
        rx = int(np.polyval(right_fit[0], y_ref))
        center_norm = (lx + rx) / 2.0 / w
    elif left_fit:
        lx = int(np.polyval(left_fit[0], y_ref))
        rx = lx + LANE_WIDTH_PX
        center_norm = (lx + rx) / 2.0 / w
    elif right_fit:
        rx = int(np.polyval(right_fit[0], y_ref))
        lx = rx - LANE_WIDTH_PX
        center_norm = (lx + rx) / 2.0 / w

    vis = img_bgr.copy()

    if left_fit is not None and right_fit is not None:
        common_y_top = max(left_fit[1], right_fit[1])
        draw_lane_line(vis, (left_fit[0],  common_y_top, left_fit[2]),  (0, 255, 0),   'left_marking',   10)
        draw_lane_line(vis, (right_fit[0], common_y_top, right_fit[2]), (0, 255, 255), 'right_edge',    -160)
    else:
        if left_fit  is not None: draw_lane_line(vis, left_fit,  (0, 255, 0),   'left_marking',   10)
        if right_fit is not None: draw_lane_line(vis, right_fit, (0, 255, 255), 'right_edge',    -160)

    if center_norm is None:
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 95), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)
        cv2.putText(vis, '!! NO DETECTION !!', (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(vis, f'weather: {weather}', (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        print(f'  [{weather}] NO DETECTION')
        return vis

    cx = int(center_norm * w)
    cv2.circle(vis, (cx, y_ref), 10, (0, 0, 255), -1)
    cv2.line(vis, (img_cx, y_ref - 25), (img_cx, y_ref + 25), (255, 255, 0), 2)
    cv2.arrowedLine(vis, (img_cx, y_ref), (cx, y_ref), (0, 255, 255), 2, tipLength=0.2)

    lx_txt = f'{lx}' if lx is not None else 'N/A'
    rx_txt = f'{rx}' if rx is not None else 'N/A'
    err    = center_norm - 0.5
    cv2.putText(vis, f'PV   lx={lx_txt}  rx={rx_txt}  center={center_norm:.3f}',
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f'err: {err:+.3f}  weather: {weather}',
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    print(f'  [{weather}] lx={lx_txt}  rx={rx_txt}  center={center_norm:.3f}  err={err:+.3f}')
    return vis


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Export Pure Vision result images for 4 weather conditions')
    parser.add_argument('--input',   default=os.path.join(REPO_ROOT, 'Images'))
    parser.add_argument('--output',  default=os.path.join(REPO_ROOT, 'Images'))
    parser.add_argument('--roi',     default=ROI_YAML)
    args = parser.parse_args()

    polygon = load_roi(args.roi)
    print(f'[PV] ROI polygon: {polygon.tolist() if polygon is not None else "full image"}')
    os.makedirs(args.output, exist_ok=True)

    for weather in WEATHERS:
        in_path  = os.path.join(args.input,  f'{weather}.png')
        out_path = os.path.join(args.output, f'opencv_result_{weather}.png')
        if not os.path.exists(in_path):
            print(f'[SKIP] {in_path} not found')
            continue
        print(f'Processing {weather} ...')
        img = cv2.imread(in_path)
        if img is None:
            print(f'[ERROR] could not read {in_path}')
            continue
        result = process(img, weather, polygon)
        cv2.imwrite(out_path, result)
        print(f'  → saved {out_path}')

    print('Done.')


if __name__ == '__main__':
    main()
