from roboflow import Roboflow
import os, glob

rf      = Roboflow(api_key="5KSltVzJaS7kDpK4fgC3")
project = rf.workspace("peeradon-eueih").project("lka-ckden")

img_dir = os.path.expanduser("~/lka_dataset/images/val")
lbl_dir = os.path.expanduser("~/lka_dataset/labels/val")

for img_path in glob.glob(f"{img_dir}/*.jpg"):
    stem     = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = f"{lbl_dir}/{stem}.txt"
    if os.path.exists(lbl_path):
        project.upload(
            image_path       = img_path,
            annotation_path  = lbl_path,
            annotation_labelmap = {0: "left_marking", 1: "right_edge"},
            split            = "train"
        )
        print(f"uploaded: {stem}")
