#!/usr/bin/env python3
"""
Convert YOLO segmentation dataset → pytorch-auto-drive CULane-compatible format.

Input (lka.yolo26):
  train/images/*.jpg    valid/images/*.jpg
  train/labels/*.txt    valid/labels/*.txt
  YOLO seg: <class> x1 y1 x2 y2 ... (normalised 0-1)
    class 0 = left_marking
    class 1 = right_edge

Output (lka.scnn  — mirrors CULane structure for pytorch-auto-drive):
  images/train/*.jpg          images/val/*.jpg
  laneseg_label_w16/train/*.png   laneseg_label_w16/val/*.png
      8-bit indexed PNG: 0=background  1=left_marking  2=right_edge
  list/train_gt.txt           list/val_gt.txt
      <img_rel> <mask_rel> <exist_left> <exist_right>

Usage:
  python3 training/yolo2scnn.py
  python3 training/yolo2scnn.py --src lka.yolo26 --dst lka.scnn --w 1600 --h 900
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

LANE_MAP = {0: 1, 1: 2}   # YOLO class → mask pixel value


def convert_split(src_img_dir: Path, src_lbl_dir: Path,
                  dst_img_dir: Path, dst_mask_dir: Path,
                  list_file: Path, W: int, H: int) -> None:
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)
    list_file.parent.mkdir(parents=True, exist_ok=True)

    images  = sorted(src_img_dir.glob('*.jpg'))
    lines   = []
    skipped = 0

    for img_path in images:
        stem = img_path.stem
        lbl_path = src_lbl_dir / f'{stem}.txt'

        dst_img  = dst_img_dir  / img_path.name
        dst_mask = dst_mask_dir / f'{stem}.png'

        shutil.copy2(img_path, dst_img)

        mask   = np.zeros((H, W), dtype=np.uint8)
        exist  = [0, 0]   # [left_marking, right_edge]

        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7:
                        continue
                    cls = int(parts[0])
                    if cls not in LANE_MAP:
                        continue
                    coords = list(map(float, parts[1:]))
                    if len(coords) % 2 != 0:
                        coords = coords[:-1]
                    xs = [round(coords[i]     * W) for i in range(0, len(coords), 2)]
                    ys = [round(coords[i + 1] * H) for i in range(0, len(coords), 2)]
                    pts = np.array(list(zip(xs, ys)), dtype=np.int32)
                    cv2.fillPoly(mask, [pts], LANE_MAP[cls])
                    exist[cls] = 1
        else:
            skipped += 1

        cv2.imwrite(str(dst_mask), mask)

        # Paths relative to lka.scnn root (pytorch-auto-drive convention)
        img_rel  = str(dst_img.relative_to(dst_img.parent.parent.parent))
        mask_rel = str(dst_mask.relative_to(dst_mask.parent.parent.parent))
        lines.append(f'{img_rel} {mask_rel} {exist[0]} {exist[1]}')

    with open(list_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'  {len(images)} images  |  list → {list_file}  |  skipped {skipped} missing labels')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/home/peeradon/lka-carla-yolo/lka.yolo26')
    ap.add_argument('--dst', default='/home/peeradon/lka-carla-yolo/lka.scnn')
    ap.add_argument('--w',   type=int, default=1600)
    ap.add_argument('--h',   type=int, default=900)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    W, H = args.w, args.h

    print(f'Source : {src}  →  Dest: {dst}  ({W}×{H})')

    for split, list_name in [('train', 'train_gt.txt'), ('valid', 'val_gt.txt')]:
        print(f'\n[{split}]')
        convert_split(
            src_img_dir  = src / split / 'images',
            src_lbl_dir  = src / split / 'labels',
            dst_img_dir  = dst / 'images'              / split,
            dst_mask_dir = dst / 'laneseg_label_w16'   / split,
            list_file    = dst / 'list' / list_name,
            W=W, H=H,
        )

    print(f'\nDone. SCNN dataset at: {dst}')
    print('Next steps:')
    print('  1. Set LKA_ROOT in pytorch-auto-drive utils/lane_as_segmentation.py')
    print('  2. python main_landet.py --config configs/lka_scnn.py')


if __name__ == '__main__':
    main()
