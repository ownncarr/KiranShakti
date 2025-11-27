# preprocess.py
"""
Preprocess YOLO segmentation .txt -> binary mask PNG files (640x640).
Edit CONFIG only.
"""

import os
from glob import glob
from PIL import Image
import numpy as np
import cv2

# ---------------- CONFIG ----------------
DATA_ROOT = r"C:/Users/onkar/Desktop/Onkar personal/Programming/projects/KiranShakti/data/yolo/"
IMG_SIZE = 640
SPLITS = ["train", "val", "test"]
# ----------------------------------------

DATA_ROOT = os.path.normpath(DATA_ROOT)

def list_images(split_folder):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(split_folder, "images", e)))
    return sorted(files)

def yolo_to_mask(img_path, label_path, out_size=IMG_SIZE):
    img = np.array(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if not os.path.exists(label_path):
        return cv2.resize(mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    with open(label_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    for ln in lines:
        parts = ln.split()
        if len(parts) < 3:
            continue
        coords = list(map(float, parts[1:]))
        if len(coords) < 6:
            continue
        pts = []
        for i in range(0, len(coords), 2):
            xn = coords[i]
            yn = coords[i+1]
            x = int(round(xn * w))
            y = int(round(yn * h))
            pts.append([x, y])
        if len(pts) >= 3:
            pts_np = np.array([pts], dtype=np.int32)
            cv2.fillPoly(mask, pts_np, 255)
    mask_resized = cv2.resize(mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return mask_resized

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def run_preprocess():
    for split in SPLITS:
        split_folder = os.path.join(DATA_ROOT, split)
        images = list_images(split_folder)
        if len(images) == 0:
            print(f"[skipping] no images found for split: {split} at {split_folder}")
            continue
        mask_dir = os.path.join(split_folder, "labels_mask")
        ensure_dir(mask_dir)
        lbl_dir = os.path.join(split_folder, "labels")
        print(f"[{split}] images: {len(images)} -> masks will be written to {mask_dir}")
        for img_path in images:
            name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, f"{name}.txt")
            mask = yolo_to_mask(img_path, lbl_path, out_size=IMG_SIZE)
            out_path = os.path.join(mask_dir, f"{name}.png")
            # write PNG with 0/255
            Image.fromarray(mask).save(out_path)
    print("Preprocessing finished. Masks written to <split>/labels_mask/*.png")

if __name__ == "__main__":
    run_preprocess()
