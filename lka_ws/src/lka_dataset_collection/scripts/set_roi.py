#!/usr/bin/python3
"""
Interactive ROI selector — click points to define a polygon ROI.

Controls:
  Left click  — add point
  ENTER       — confirm polygon
  r           — reset all points
  ESC         — quit

Output: roi.yaml with bounding box of the polygon (x1,y1,x2,y2)
Preview saved to /home/peeradon/lka_dataset/sample_roi_preview.jpg
"""

import cv2
import numpy as np
import yaml
import os

IMAGE_PATH = '/home/peeradon/lka-carla-yolo/Images/sample_rgb.jpg'
SEM_PATH   = '/home/peeradon/lka-carla-yolo/Images/sample_sem.jpg'
OUT_YAML   = os.path.join(os.path.dirname(__file__), 'roi.yaml')

SW_LOWER = np.array([227,  30, 238], dtype=np.uint8)
SW_UPPER = np.array([237,  40, 249], dtype=np.uint8)
RL_LOWER = np.array([ 45, 229, 152], dtype=np.uint8)
RL_UPPER = np.array([ 55, 239, 162], dtype=np.uint8)


def draw_preview(rgb, sem, pts_orig):
    out  = rgb.copy()
    h, w = rgb.shape[:2]

    # Draw polygon ROI with semi-transparent fill
    poly    = np.array(pts_orig, dtype=np.int32)
    overlay = out.copy()
    cv2.fillPoly(overlay, [poly], (0, 255, 255))
    cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
    cv2.polylines(out, [poly], isClosed=True, color=(0,255,255), thickness=3)
    for i, p in enumerate(pts_orig):
        cv2.circle(out, p, 8, (0,255,255), -1)
        cv2.putText(out, str(i+1), (p[0]+10, p[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Bounding box from polygon
    x1 = min(p[0] for p in pts_orig)
    y1 = min(p[1] for p in pts_orig)
    x2 = max(p[0] for p in pts_orig)
    y2 = max(p[1] for p in pts_orig)

    # RoadLine
    rl_mask = cv2.inRange(sem, RL_LOWER, RL_UPPER)
    cnts, _ = cv2.findContours(rl_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) >= 30:
            cv2.polylines(out, [c], True, (0,255,0), 3)

    # Sidewalk inside bounding box only
    sw_mask  = cv2.inRange(sem, SW_LOWER, SW_UPPER)
    roi_mask = np.zeros_like(sw_mask)
    roi_mask[y1:y2, x1:x2] = sw_mask[y1:y2, x1:x2]

    px = np.where(roi_mask > 0)
    if len(px[1]) > 0:
        lx = int(px[1].min())
        cv2.line(out, (lx, y1), (lx, y2), (0,0,255), 4)
        cv2.putText(out, f'right_edge x={lx} ({lx/w*100:.0f}%)',
                    (lx+8, y1+40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    else:
        cv2.putText(out, 'No sidewalk in ROI',
                    (x1+8, y1+40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    return out, (x1, y1, x2, y2)


def main():
    rgb = cv2.imread(IMAGE_PATH)
    sem = cv2.imread(SEM_PATH)
    if rgb is None or sem is None:
        print('ERROR: Run save_sem_sample.py first to generate sample images.')
        return

    h, w = rgb.shape[:2]
    scale  = min(1400/w, 800/h)
    disp_w = int(w * scale)
    disp_h = int(h * scale)

    WIN    = 'Set ROI'
    points = []   # display-scale points
    mouse  = {'pos': (0, 0)}

    def on_mouse(event, x, y, flags, param):
        mouse['pos'] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    # imshow first — window must exist before setMouseCallback
    cv2.imshow(WIN, cv2.resize(rgb, (disp_w, disp_h)))
    cv2.waitKey(1)
    cv2.setMouseCallback(WIN, on_mouse)

    print(f'Image: {w}x{h}')
    print('Click points around the right-side road boundary.')
    print('ENTER=confirm | r=reset | ESC=quit')

    while True:
        disp = cv2.resize(rgb, (disp_w, disp_h))

        # Draw completed segments
        for i, p in enumerate(points):
            cv2.circle(disp, p, 6, (0,255,255), -1)
            cv2.putText(disp, str(i+1), (p[0]+8, p[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            if i > 0:
                cv2.line(disp, points[i-1], p, (0,255,255), 2)

        # Preview closing line to cursor
        if points:
            cv2.line(disp, points[-1], mouse['pos'], (0,255,255), 1)
            if len(points) >= 3:
                cv2.line(disp, points[0], mouse['pos'], (0,200,200), 1)

        hint = f'Points: {len(points)} | ENTER=confirm(need>=3) | r=reset | ESC=quit'
        cv2.putText(disp, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow(WIN, disp)

        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):          # ENTER / SPACE
            if len(points) >= 3:
                break
            else:
                print('Need at least 3 points.')
        elif key == ord('r'):
            points.clear()
            print('Reset.')
        elif key == 27:              # ESC
            print('Cancelled.')
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow(WIN)

    # Convert display coords → original image coords
    pts_orig = [(int(x/scale), int(y/scale)) for x, y in points]

    preview, (x1,y1,x2,y2) = draw_preview(rgb, sem, pts_orig)
    out_path = '/home/peeradon/lka-carla-yolo/Images/sample_roi_preview.jpg'
    cv2.imwrite(out_path, preview)

    # Resize preview to fit screen
    prev_disp = cv2.resize(preview, (disp_w, disp_h))
    cv2.putText(prev_disp, 'Preview — press any key to close',
                (10, disp_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imshow('Preview', prev_disp)
    print(f'\nPolygon points (original): {pts_orig}')
    print(f'Bounding box: x1={x1} y1={y1} x2={x2} y2={y2}')
    print(f'Preview saved: {out_path}')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    roi_data = {
        'roi': {
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'x1_norm': round(x1/w, 4),
            'y1_norm': round(y1/h, 4),
            'x2_norm': round(x2/w, 4),
            'y2_norm': round(y2/h, 4),
            'image_w': w,
            'image_h': h,
            'polygon': pts_orig,
        }
    }
    with open(OUT_YAML, 'w') as f:
        yaml.dump(roi_data, f, default_flow_style=False)
    print(f'ROI saved: {OUT_YAML}')


if __name__ == '__main__':
    main()
