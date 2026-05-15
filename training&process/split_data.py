# split_dataset.py  — run once from the lka.yolo26/ directory
import os, shutil, random

base = "/home/peeradon/lka-carla-yolo/lka.yolo26"
img_dir = f"{base}/train/images"
lbl_dir = f"{base}/train/labels"

val_img = f"{base}/valid/images"
val_lbl = f"{base}/valid/labels"
os.makedirs(val_img, exist_ok=True)
os.makedirs(val_lbl, exist_ok=True)

images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
random.seed(42)
random.shuffle(images)
val_set = images[:400]  # 20% of 2000

for img in val_set:
    stem = os.path.splitext(img)[0]
    shutil.move(f"{img_dir}/{img}", f"{val_img}/{img}")
    shutil.move(f"{lbl_dir}/{stem}.txt", f"{val_lbl}/{stem}.txt")

print(f"Moved {len(val_set)} images to valid/")
