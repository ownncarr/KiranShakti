# pipeline/backend.py
import io
import traceback
from typing import Tuple, Optional

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

# import pipeline modules (assumes pipeline/ is importable)
from pipeline.fetch_image import fetch_satellite_image
from pipeline.model_inference import load_model, predict, get_default_device
from pipeline.postprocess import encode_mask_as_polygon, format_output
from pipeline.utils.buffer_utils import (
    compute_gsd_m_per_px,
    compute_buffer_radius_px,
    create_circular_buffer,
    compute_overlap_area,
)
from pipeline.utils.io_utils import ensure_dir

# Cache model and device to avoid reloading on every sample
_MODEL = None
_DEVICE = None


def _load_model_once():
    """
    Returns (model, device). Loads model on first call and caches it.
    """
    global _MODEL, _DEVICE
    if _MODEL is None:
        try:
            _DEVICE = get_default_device()
            _MODEL = load_model(device=_DEVICE)
        except Exception:
            _MODEL = None
            _DEVICE = get_default_device(prefer_cuda=False)
    return _MODEL, _DEVICE


def _make_overlay(image: Image.Image, mask_img: Image.Image, r1200_px: int, r2400_px: int, confidence: float) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    W, H = base.size
    center = (W // 2, H // 2)

    # mask overlay (semi transparent red)
    try:
        mask_arr = np.array(mask_img.convert("L"))
        if mask_arr.ndim == 2:
            red = Image.new("RGBA", base.size, (255, 0, 0, 0))
            mask_alpha = Image.fromarray(((mask_arr == 255).astype(np.uint8) * 120).astype(np.uint8))
            red.putalpha(mask_alpha)
            overlay = Image.alpha_composite(overlay, red)
    except Exception:
        pass

    # buffer rings
    try:
        box1200 = [center[0] - r1200_px, center[1] - r1200_px, center[0] + r1200_px, center[1] + r1200_px]
        box2400 = [center[0] - r2400_px, center[1] - r2400_px, center[0] + r2400_px, center[1] + r2400_px]
        draw.ellipse(box2400, outline=(255, 200, 40, 220), width=3)
        draw.ellipse(box1200, outline=(120, 220, 130, 240), width=4)
    except Exception:
        pass

    # confidence text
    try:
        txt = f"conf: {confidence:.3f}"
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        text_pos = (8, 8)
        tw, th = draw.textsize(txt, font=font)
        draw.rectangle([text_pos, (text_pos[0] + tw + 6, text_pos[1] + th + 4)], fill=(0, 0, 0, 160))
        draw.text((text_pos[0] + 3, text_pos[1] + 2), txt, fill=(255, 255, 255, 240), font=font)
    except Exception:
        pass

    result = Image.alpha_composite(base, overlay).convert("RGB")
    return result


def _placeholder_overlay(size=(640, 640), message: Optional[str] = None) -> Image.Image:
    img = Image.new("RGB", size, (28, 28, 30))
    d = ImageDraw.Draw(img)
    txt = message or "No overlay"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    w, h = d.textsize(txt, font=font)
    d.text(((size[0] - w) // 2, (size[1] - h) // 2), txt, fill=(255, 255, 255), font=font)
    return img


def process_sample(row, mode: str = "Smart Review", api_key: str = None) -> Tuple[dict, Image.Image, bool]:
    """
    Main entry used by the Streamlit UI.
    Accepts `row` (pandas Series or dict-like) with `sample_id`, `lat`, `lon`.
    mode: "Automatic" | "Smart Review" | "Review All"
    api_key: Google Static Maps API key (must be provided by UI)

    Returns:
      rec (dict), overlay (PIL Image), needs_review (bool)
    """
    sid = row.get("sample_id", None)
    try:
        lat = float(row.get("lat", 0.0))
        lon = float(row.get("lon", 0.0))
    except Exception:
        lat = 0.0
        lon = 0.0

    # default conservative record
    rec = {
        "sample_id": sid,
        "lat": lat,
        "lon": lon,
        "has_solar": False,
        "confidence": 0.0,
        "pv_area_sqm_est": 0.0,
        "buffer_radius_sqft": None,
        "qc_status": "NOT_VERIFIABLE",
        "bbox_or_mask": None,
        "image_metadata": {"source": None, "capture_date": None},
    }
    overlay = _placeholder_overlay(message="processing failed")

    try:
        if not api_key:
            raise RuntimeError("No API key provided to backend.process_sample().")

        # fetch image
        img = fetch_satellite_image(lat, lon, api_key=api_key)
        W, H = img.size

        # compute gsd & buffers
        gsd = compute_gsd_m_per_px(lat)
        r1200 = compute_buffer_radius_px(1200, gsd)
        r2400 = compute_buffer_radius_px(2400, gsd)

        # model + device
        model, device = _load_model_once()
        if model is None:
            rec.update({
                "image_metadata": {"source": "GOOGLE_STATIC_MAPS", "capture_date": "UNKNOWN"},
                "qc_status": "NOT_VERIFIABLE"
            })
            overlay = _placeholder_overlay(size=img.size, message="model missing")
            needs_review = True
            return rec, overlay, needs_review

        # ensure image is PIL
        mask_img, confidence = predict(model, img, device=device)

        # compute areas
        pv_arr = np.array(mask_img)
        buf1200 = create_circular_buffer((H, W), r1200)
        buf2400 = create_circular_buffer((H, W), r2400)

        area_1200 = compute_overlap_area(pv_arr, buf1200, gsd)
        area_2400 = compute_overlap_area(pv_arr, buf2400, gsd)

        # decide presence and buffer used
        if area_1200 > 1.0:
            has_solar = True
            pv_area_m2 = area_1200
            buffer_used_sqft = 1200
        else:
            has_solar = area_2400 > 1.0
            pv_area_m2 = area_2400
            buffer_used_sqft = 2400

        polygon_b64 = encode_mask_as_polygon(mask_img)

        rec = format_output(
            sample_id=sid,
            lat=lat,
            lon=lon,
            has_solar=has_solar,
            confidence=confidence,
            pv_area=pv_area_m2,
            buffer_radius_sqft=buffer_used_sqft,
            bbox_or_mask=polygon_b64,
            img_source="GOOGLE_STATIC_MAPS",
            capture_date="UNKNOWN",
        )

        overlay = _make_overlay(img, mask_img, r1200, r2400, confidence)

        qc = rec.get("qc_status", "REVIEW")
        if mode == "Review All":
            needs_review = True
        elif mode == "Automatic":
            needs_review = (qc != "VERIFIABLE")
        else:  # Smart Review or default
            needs_review = (qc != "VERIFIABLE")

        return rec, overlay, needs_review

    except Exception:
        traceback.print_exc()
        rec["qc_status"] = "NOT_VERIFIABLE"
        overlay = _placeholder_overlay(message="error")
        needs_review = True
        return rec, overlay, needs_review
