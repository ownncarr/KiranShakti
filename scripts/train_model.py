# train_model.py
"""
UNet++ (EfficientNet-B3) training with gradient accumulation + AMP.
Dataset layout expected (set DATA_ROOT below):
DATA_ROOT/
  train/
    images/*.jpg
    labels/*.txt
  val/
    images/*.jpg
    labels/*.txt
  test/   (optional)
    images/*.jpg
    labels/*.txt

Set BATCH_SIZE to a value that fits your GPU (e.g., 1 or 2).
Set ACCUM_STEPS so that effective_batch = BATCH_SIZE * ACCUM_STEPS.
Example: BATCH_SIZE=2, ACCUM_STEPS=4 -> effective batch 8.

This script:
- Converts YOLO polygon segmentation in .txt to binary masks (640×640).
- Trains with AMP and gradient accumulation.
- Validates using IoU and saves best model to 'solar_detector.pt'.
"""

import os
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
from math import ceil

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

# ---------------- USER CONFIG ----------------
DATA_ROOT = "c:/Users/onkar/Desktop/Onkar personal/Programming/projects/KiranShakti/data/yolo"   # <<--- Set this to the folder containing train/, val/, test/
IMG_SIZE = 640
BATCH_SIZE = 2         # pick small number that fits your GPU (1 or 2 recommended)
ACCUM_STEPS = 4        # gradient accumulation steps; effective_batch = BATCH_SIZE * ACCUM_STEPS
LR = 1e-4
EPOCHS = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_OUT = "solar_detector.pt"
ENCODER = "efficientnet-b3"
ENCODER_WEIGHTS = "imagenet"
NUM_WORKERS = 4
# ---------------------------------------------

# ---------------- helpers ----------------
def list_images(folder):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(folder, "images", e)))
    return sorted(files)

def yolo_to_mask(img_path, label_path, size=IMG_SIZE):
    """
    Convert YOLO segmentation txt to binary mask.
    Each label line: cls x1 y1 x2 y2 ... (normalized coords in [0,1])
    Returns mask resized to (size,size) dtype uint8 with {0,255}.
    """
    img = np.array(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not os.path.exists(label_path):
        return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

    with open(label_path, "r") as f:
        lines = [l.strip() for l in f.read().splitlines() if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
        except Exception:
            continue
        pts = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i + 1] * h)
            pts.append([x, y])
        if len(pts) >= 3:
            pts = np.array([pts], dtype=np.int32)
            cv2.fillPoly(mask, pts, 255)

    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return mask

# ---------------- Dataset ----------------
class SolarDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.img_dir = os.path.join(folder, "images")
        self.lbl_dir = os.path.join(folder, "labels")
        self.images = list_images(folder)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.lbl_dir, f"{name}.txt")

        img = np.array(Image.open(img_path).convert("RGB"))
        # Resize image to IMG_SIZE (bilinear)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        mask = yolo_to_mask(img_path, lbl_path, size=IMG_SIZE)
        # Albumentations expects masks as HxW single channel ints
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"].unsqueeze(0).float() / 255.0  # normalize mask to 0/1
        else:
            # fallback transforms
            img = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return img, mask

# ---------------- Augmentations ----------------
def get_train_aug():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.25),
        A.Rotate(limit=12, p=0.4, border_mode=0),
        A.ColorJitter(brightness=0.20, saturation=0.25, contrast=0.15, p=0.4),
        A.GaussNoise(p=0.15),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

def get_val_aug():
    return A.Compose([
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

# ---------------- Model & Loss ----------------
def get_model():
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=1
    )
    return model.to(DEVICE)

# Combined loss: BCEWithLogits + Dice (binary)
def combined_loss_fn(logits, targets):
    bce = nn.BCEWithLogitsLoss()(logits, targets)
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2,3))
    den = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + 1e-7
    dice_loss = 1 - (num / den).mean()
    return bce + dice_loss

# ---------------- Metrics ----------------
def batch_iou(preds, targets, threshold=0.5, eps=1e-7):
    """
    preds: torch.Tensor (N,1,H,W) in probability space [0,1]
    targets: torch.Tensor (N,1,H,W) {0,1}
    returns average IoU across batch
    """
    preds_bin = (preds > threshold).float()
    preds_flat = preds_bin.view(preds_bin.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersection = (preds_flat * targets_flat).sum(1)
    union = preds_flat.sum(1) + targets_flat.sum(1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

# ---------------- Training ----------------
def train():
    # sanity checks
    for s in ("train","val"):
        folder = os.path.join(DATA_ROOT, s)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing folder: {folder}. Please set DATA_ROOT correctly.")
        cnt = len(list_images(folder))
        print(f"{s}: {cnt} images found under {folder}")

    train_ds = SolarDataset(os.path.join(DATA_ROOT, "train"), transform=get_train_aug())
    val_ds = SolarDataset(os.path.join(DATA_ROOT, "val"), transform=get_val_aug())

    if len(train_ds) == 0:
        raise RuntimeError("No training images found. Check DATA_ROOT and train/images.")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

    best_iou = 0.0
    steps_per_epoch = len(train_loader)

    print(f"Training on device={DEVICE} | batch_size={BATCH_SIZE} | accum_steps={ACCUM_STEPS} | effective_batch={BATCH_SIZE * ACCUM_STEPS}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{EPOCHS} [train]")
        for step, (imgs, masks) in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                logits = model(imgs)
                loss = combined_loss_fn(logits, masks)
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUM_STEPS  # un-normalize for logging
            pbar.set_postfix({"loss": f"{running_loss / (step+1):.4f}"})

        avg_train_loss = running_loss / steps_per_epoch if steps_per_epoch > 0 else running_loss

        # Validation
        model.eval()
        iou_accum = 0.0
        val_batches = 0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc="Validation"):
                imgs = imgs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                    probs = torch.sigmoid(model(imgs))
                batch_iou_score = batch_iou(probs, masks, threshold=0.5)
                iou_accum += batch_iou_score
                val_batches += 1

        mean_iou = iou_accum / val_batches if val_batches > 0 else 0.0
        print(f"Epoch {epoch}: TrainLoss={avg_train_loss:.4f}, Val IoU={mean_iou:.4f}")

        # Save best
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"✔ Saved best model -> {MODEL_OUT} (val_iou={best_iou:.4f})")

    print("Training complete.")

if __name__ == "__main__":
    train()
