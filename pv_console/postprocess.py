# pv_console/postprocess.py
print("[postprocess.py] imported")
import base64
import io
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple
try:
    from shapely.geometry import Polygon
    SHAPELY_OK = True
except Exception:
    SHAPELY_OK = False

try:
    from skimage.draw import disk
    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False

from math import cos, pi
from pv_console.config import MODEL_INPUT_SIZE

def encode_mask_as_polygon(mask_img: Image.Image) -> Optional[str]:
    print("[postprocess.encode_mask_as_polygon] called")
    try:
        arr = np.array(mask_img)
        ys, xs = np.where(arr == 255)
        if len(xs) == 0:
            print("[postprocess] mask empty -> returning None")
            return None
        coords = list(zip(xs.tolist(), ys.tolist()))
        if SHAPELY_OK:
            poly = Polygon(coords).convex_hull
            encoded = json.dumps(list(poly.exterior.coords))
            return base64.b64encode(encoded.encode()).decode()
        else:
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        import traceback; traceback.print_exc()
        print("[postprocess] encode_mask_as_polygon error:", e)
        return None

def qc_status(confidence: float, pv_area_m2: float) -> str:
    if pv_area_m2 > 1.0 and confidence > 0.8:
        return "VERIFIABLE"
    elif confidence > 0.4:
        return "REVIEW"
    return "LOW_CONFIDENCE"

def compute_gsd_m_per_px(lat: float, zoom: int = 20, scale: int = 2) -> float:
    """
    Return meters-per-pixel for an image returned at the given zoom and scale.
    base_gsd is meters/pixel at scale=1; actual meters/pixel = base_gsd / scale.
    """
    try:
        base_gsd = 156543.03392 * cos(lat * pi / 180) / (2 ** zoom)
        return base_gsd / float(scale)
    except Exception:
        return 0.3


def compute_buffer_radius_px(area_sqft: float, gsd_m: float) -> int:
    try:
        area_m2 = area_sqft * 0.092903
        pixel_area_m2 = gsd_m ** 2 if gsd_m > 0 else 0.09
        radius_px = int(np.sqrt(area_m2 / (pixel_area_m2 + 1e-9) / pi))
        return max(1, radius_px)
    except Exception:
        return int(area_sqft ** 0.5)

def create_circular_buffer(img_size: Tuple[int,int], radius_px: int) -> np.ndarray:
    H, W = img_size
    mask = np.zeros((H, W), dtype=np.uint8)
    cy, cx = H // 2, W // 2
    if SKIMAGE_OK:
        rr, cc = disk((cy, cx), radius_px, shape=mask.shape)
        mask[rr, cc] = 1
    else:
        Y, X = np.ogrid[:H, :W]
        mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= (radius_px ** 2)
        mask = mask.astype(np.uint8)
    return mask

def compute_overlap_area(pv_mask_arr: np.ndarray, buffer_mask: np.ndarray, gsd_m: float) -> float:
    try:
        pv_binary = (pv_mask_arr == 255).astype(np.uint8)
        inter = (pv_binary & buffer_mask).sum()
        return float(inter) * (gsd_m ** 2)
    except Exception:
        return 0.0

def _text_size(draw: ImageDraw.ImageDraw, text: str, font: Optional[ImageFont.ImageFont] = None) -> tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return (w, h)
    except Exception:
        pass
    try:
        return draw.textsize(text, font=font)
    except Exception:
        pass
    try:
        if font is not None:
            return font.getsize(text)
    except Exception:
        pass
    approx_w = max(10, int(len(text) * 7))
    approx_h = 12
    return (approx_w, approx_h)

def make_overlay(image: Image.Image, mask_img: Image.Image, r1200_px: int, r2400_px: int, confidence: float) -> Image.Image:
    print("[postprocess.make_overlay] called")
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    W, H = base.size
    center = (W // 2, H // 2)
    mask_has_data = False
    try:
        mask_arr = np.array(mask_img.convert("L"))
        if mask_arr.ndim == 2 and mask_arr.sum() > 0:
            mask_has_data = True
            red = Image.new("RGBA", base.size, (255, 0, 0, 0))
            mask_alpha = Image.fromarray(((mask_arr == 255).astype(np.uint8) * 140).astype(np.uint8))
            red.putalpha(mask_alpha)
            overlay = Image.alpha_composite(overlay, red)
    except Exception as e:
        print("[postprocess.make_overlay] mask overlay failed:", e)

    try:
        box1200 = [center[0] - r1200_px, center[1] - r1200_px, center[0] + r1200_px, center[1] + r1200_px]
        box2400 = [center[0] - r2400_px, center[1] - r2400_px, center[0] + r2400_px, center[1] + r2400_px]
        draw.ellipse(box2400, outline=(255, 200, 40, 220), width=3)
        draw.ellipse(box1200, outline=(120, 220, 130, 240), width=4)
    except Exception as e:
        print("[postprocess.make_overlay] drawing circles failed:", e)

    if not mask_has_data:
        try:
            cx, cy = center
            draw.line((cx - 12, cy, cx + 12, cy), fill=(255,255,255,200), width=2)
            draw.line((cx, cy - 12, cx, cy + 12), fill=(255,255,255,200), width=2)
            font = ImageFont.load_default()
            draw.text((8, 8), "no mask", fill=(255,255,255,220), font=font)
        except Exception as e:
            print("[postprocess.make_overlay] placeholder text failed:", e)

    try:
        txt = f"conf: {confidence:.3f}"
        font = ImageFont.load_default()
        text_pos = (8, 8)
        tw, th = _text_size(draw, txt, font)
        draw.rectangle([text_pos, (text_pos[0] + tw + 6, text_pos[1] + th + 4)], fill=(0,0,0,160))
        draw.text((text_pos[0] + 3, text_pos[1] + 2), txt, fill=(255,255,255,240), font=font)
    except Exception as e:
        print("[postprocess.make_overlay] conf text failed:", e)

    result = Image.alpha_composite(base, overlay).convert("RGB")
    print("[postprocess.make_overlay] done")
    return result
