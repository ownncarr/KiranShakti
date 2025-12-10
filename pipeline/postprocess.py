# pipeline/postprocess.py
import numpy as np
import json
import base64
from shapely.geometry import Polygon


def encode_mask_as_polygon(mask_img):
    """
    Very simple convex hull polygon encoding.
    """
    mask = np.array(mask_img)
    ys, xs = np.where(mask == 255)
    if len(xs) == 0:
        return None

    coords = list(zip(xs, ys))
    poly = Polygon(coords).convex_hull

    encoded = json.dumps(list(poly.exterior.coords))
    return base64.b64encode(encoded.encode()).decode()


def qc_status(confidence, pv_area_m2):
    if pv_area_m2 > 1.0 and confidence > 0.8:
        return "VERIFIABLE"
    elif confidence > 0.4:
        return "REVIEW"
    return "LOW_CONFIDENCE"


def format_output(
    sample_id,
    lat,
    lon,
    has_solar,
    confidence,
    pv_area,
    buffer_radius_sqft,
    bbox_or_mask,
    img_source="GOOGLE_STATIC_MAPS",
    capture_date="UNKNOWN"
):
    return {
        "sample_id": sample_id,
        "lat": lat,
        "lon": lon,
        "has_solar": bool(has_solar),
        "confidence": round(confidence, 4),
        "pv_area_sqm_est": round(pv_area, 2),
        "buffer_radius_sqft": round(buffer_radius_sqft, 2),
        "qc_status": qc_status(confidence, pv_area),
        "bbox_or_mask": bbox_or_mask,
        "image_metadata": {
            "source": img_source,
            "capture_date": capture_date
        }
    }
