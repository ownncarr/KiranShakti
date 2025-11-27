# predict_pv.py
"""
predict_pv.py

Usage:
    from PIL import Image
    from predict_pv import predict_pv, load_model, MODEL

    img = Image.open("some_640_image.jpg").convert("RGB")
    meters_per_pixel = 0.15  # compute from your tile/zoom
    out = predict_pv(img, meters_per_pixel)
    print(out)

Notes:
- Assumes your model file is 'solar_detector.pt' in the same folder, or set MODEL_PATH below.
- Assumes model architecture is UNet++ with EfficientNet-B3 encoder (match your training).
- Normalizes with ImageNet mean/std (common when encoder_weights="imagenet").
"""

import os
from typing import Optional, Dict, Any, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import cv2
import segmentation_models_pytorch as smp

# -----------------------------
# USER-CONFIG / PATHS
# -----------------------------
MODEL_PATH = os.environ.get("SOLAR_MODEL_PATH", "solar_detector.pt")
INPUT_SIZE = 640   # must match training input
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Minimum connected component area (sqm) to consider a real PV panel (tunable)
MIN_COMP_AREA_SQM = 0.5  # 0.5 sqm (adjust if you want larger threshold)

# Probability threshold to binarize masks
PROB_THRESH = 0.5

# -----------------------------
# MODEL LOADING (cached)
# -----------------------------
_MODEL = None

def load_model(model_path: str = MODEL_PATH, device: str = DEVICE):
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    # Build model architecture - must match training
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b3",
        encoder_weights=None,  # weights already baked in state_dict
        in_channels=3,
        classes=1
    )
    state = torch.load(model_path, map_location=device)
    # if saved state is state_dict vs full checkpoint, handle both
    if isinstance(state, dict) and ("state_dict" in state) and not any(k.startswith("module.") for k in state.keys()):
        # saved as {"state_dict": ...}
        sd = state["state_dict"]
        model.load_state_dict(sd)
    else:
        # assume direct state_dict
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    _MODEL = model
    return _MODEL

# load once at import (but silently continue if model missing)
try:
    MODEL = load_model()
except Exception as e:
    MODEL = None
    # do not raise here; will raise on predict if model missing


# -----------------------------
# Preprocess helpers
# -----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def preprocess_pil_to_tensor(pil_img: Image.Image, size: int = INPUT_SIZE) -> torch.Tensor:
    """
    Convert PIL image -> torch tensor (1,3,H,W) normalized with ImageNet stats.
    """
    img = pil_img.convert("RGB")
    # Resize with aspect-preserve fit, pad black to square, then to size x size
    w, h = img.size
    # compute scale
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), resample=Image.BILINEAR)
    # paste onto black canvas
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))
    arr = np.array(canvas).astype(np.float32) / 255.0
    # normalize
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    # to tensor C,H,W
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr).unsqueeze(0).float()  # 1,3,H,W
    return tensor.to(DEVICE)

def postprocess_prob_map(prob_map: np.ndarray, prob_thresh: float = PROB_THRESH):
    """
    prob_map: 2D float array (H,W) values in [0,1]
    returns binary_mask (uint8 0/1)
    """
    bin_mask = (prob_map >= prob_thresh).astype(np.uint8)
    return bin_mask

# -----------------------------
# Core predict function
# -----------------------------
def predict_pv(
    pil_img: Image.Image,
    meters_per_pixel: float,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    min_comp_area_sqm: float = MIN_COMP_AREA_SQM,
    prob_thresh: float = PROB_THRESH,
    save_annotated_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run PV segmentation on a PIL image.

    Args:
        pil_img: PIL.Image input (any size). Function will resize/pad to INPUT_SIZE.
        meters_per_pixel: meters represented by one pixel in the original image input.
                          IMPORTANT: if input image was padded to square during preprocess,
                          meters_per_pixel should correspond to the scaled image pixels (after resizing).
        model_path: optional path to model .pt
        device: "cuda" or "cpu"
        min_comp_area_sqm: filter small components below this area
        prob_thresh: threshold to binarize probability map
        save_annotated_path: if provided, saves an overlay PNG with contours and areas

    Returns:
        dict with keys: has_solar, pv_area_sqm, confidence, prob_map, binary_mask, components
    """
    # device + model
    use_device = device if device is not None else DEVICE
    model = None
    if model_path:
        model = load_model(model_path, use_device)
    else:
        model = load_model()
    if model is None:
        raise RuntimeError("Model not loaded. Check MODEL_PATH or model file.")

    # Preprocess
    input_tensor = preprocess_pil_to_tensor(pil_img, size=INPUT_SIZE)  # 1,3,H,W
    with torch.no_grad():
        logits = model(input_tensor)  # (1,1,H,W)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # H,W float32 in [0,1]

    # postprocess
    prob_map = probs.astype(np.float32)
    bin_mask = postprocess_prob_map(prob_map, prob_thresh)

    # Connected components on binary mask
    # use OpenCV connectedComponentsWithStats (expects 8-bit single-channel)
    mask_u8 = (bin_mask * 255).astype(np.uint8)
    n_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    components: List[Dict[str, Any]] = []
    total_area_sqm = 0.0
    total_prob_vals = []
    # meters_per_pixel: meters represented by one pixel in prob_map.
    # area per pixel in sqm:
    sqm_per_pixel = (meters_per_pixel ** 2)

    h, w = prob_map.shape  # should be INPUT_SIZE, INPUT_SIZE

    for label in range(1, n_labels):  # skip background label 0
        x, y, width, height, area_px = stats[label]
        if area_px <= 0:
            continue
        mean_prob = float(np.mean(prob_map[labels_im == label]))
        area_sqm = float(area_px * sqm_per_pixel)

        # filter small components
        if area_sqm < min_comp_area_sqm:
            continue

        total_area_sqm += area_sqm
        total_prob_vals.extend(prob_map[labels_im == label].tolist())

        x1, y1, x2, y2 = int(x), int(y), int(x + width), int(y + height)
        cx, cy = float(centroids[label][0]), float(centroids[label][1])

        components.append({
            "label": int(label),
            "area_px": int(area_px),
            "area_sqm": float(area_sqm),
            "mean_prob": float(mean_prob),
            "bbox": [x1, y1, x2, y2],
            "centroid": [cx, cy]
        })

    has_solar = len(components) > 0
    pv_area_sqm = float(total_area_sqm)
    confidence = float(np.mean(total_prob_vals)) if len(total_prob_vals) > 0 else 0.0

    result = {
        "has_solar": bool(has_solar),
        "pv_area_sqm": pv_area_sqm,
        "confidence": confidence,
        "prob_map": prob_map,          # 2D numpy array (H,W) float32
        "binary_mask": bin_mask,       # 2D numpy array (H,W) uint8 0/1
        "components": components
    }

    # Optionally save annotated image: overlay contours and area labels on original (resized/padded) image
    if save_annotated_path:
        try:
            # Build annotation on the resized/padded image produced in preprocess step
            # Recreate the resized canvas used in preprocess so that contour coordinates align
            w0, h0 = pil_img.size
            scale = INPUT_SIZE / max(w0, h0)
            new_w, new_h = int(w0 * scale), int(h0 * scale)
            paste_x = (INPUT_SIZE - new_w) // 2
            paste_y = (INPUT_SIZE - new_h) // 2

            canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (0, 0, 0))
            img_resized = pil_img.resize((new_w, new_h), resample=Image.BILINEAR)
            canvas.paste(img_resized, (paste_x, paste_y))
            draw = ImageDraw.Draw(canvas)

            # draw contours
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i, cnt in enumerate(contours):
                # contour points are in mask coordinates (already matching canvas)
                cnt = cnt.squeeze().astype(int)
                if cnt.ndim == 1:
                    continue
                pts = [tuple(pt) for pt in cnt]
                # only draw if component area passes threshold
                # find which label this contour belongs to
                # compute centroid to find the component entry
                M = cv2.moments(cnt.astype(np.int32))
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = pts[0]
                # find component in list with centroid near (cx,cy)
                comp_found = None
                for comp in components:
                    comp_cx, comp_cy = comp["centroid"]
                    if abs(comp_cx - cx) < 5 and abs(comp_cy - cy) < 5:
                        comp_found = comp
                        break
                if comp_found is None:
                    continue
                # draw polygon
                draw.polygon(pts, outline=(255, 0, 0))
                # label with area and mean_prob
                label_text = f"{comp_found['area_sqm']:.1f}m² {comp_found['mean_prob']:.2f}"
                # choose a simple font (PIL default)
                draw.text((cx + 3, cy + 3), label_text, fill=(255, 255, 0))

            # save
            canvas.save(save_annotated_path)
            result["annotated_image_path"] = save_annotated_path
        except Exception as e:
            # don't fail the main pipeline if annotation fails
            result["annotated_image_error"] = str(e)

    return result

# If called as script, demo a minimal run (requires valid model and sample image)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Path to input image (PIL readable)")
    parser.add_argument("--meters_per_pixel", required=True, type=float, help="meters per pixel")
    parser.add_argument("--model", default=None, help="path to model .pt")
    parser.add_argument("--out", default=None, help="optional annotated output path (png)")
    args = parser.parse_args()

    pil = Image.open(args.img).convert("RGB")
    out = predict_pv(pil, args.meters_per_pixel, model_path=args.model, save_annotated_path=args.out)
    # print summary
    print({
        "has_solar": out["has_solar"],
        "pv_area_sqm": out["pv_area_sqm"],
        "confidence": out["confidence"],
        "n_components": len(out["components"])
    })
