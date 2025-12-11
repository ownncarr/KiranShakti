# pv_console/pipeline.py  (corrected)
print("[pipeline.py] imported")
import io
import traceback
from typing import Tuple, Optional

import numpy as np
from PIL import Image            # <- fixed: Image import for Image.new(...)
from pv_console.fetch_image import fetch_satellite_image
from pv_console.model_inference import load_model, predict
from pv_console.postprocess import (
    create_circular_buffer, compute_overlap_area,
    encode_mask_as_polygon, make_overlay, compute_gsd_m_per_px, compute_buffer_radius_px
)
from pv_console.config import MODEL_INPUT_SIZE

def format_output(sample_id, lat, lon, has_solar, confidence, pv_area, buffer_radius_sqft, bbox_or_mask, img_source="GOOGLE_STATIC_MAPS", capture_date="UNKNOWN"):
    return {
        "sample_id": sample_id,
        "lat": lat,
        "lon": lon,
        "has_solar": bool(has_solar),
        "confidence": round(float(confidence), 4),
        "pv_area_sqm_est": round(float(pv_area or 0.0), 2),
        "buffer_radius_sqft": round(float(buffer_radius_sqft or 0.0), 2),
        "qc_status": "REVIEW" if confidence < 0.9 else "VERIFIABLE",
        "bbox_or_mask": bbox_or_mask,
        "image_metadata": {"source": img_source, "capture_date": capture_date},
    }

def process_sample(row, mode: str = "Smart Review", api_key: Optional[str] = None) -> Tuple[dict, object, bool]:
    print("[pipeline.process_sample] called for sample:", row.get("sample_id"))
    sid = row.get("sample_id", None)
    try:
        lat = float(row.get("lat", 0.0))
        lon = float(row.get("lon", 0.0))
    except Exception:
        lat = 0.0; lon = 0.0

    rec = {
        "sample_id": sid,
        "lat": lat, "lon": lon,
        "has_solar": False, "confidence": 0.0, "pv_area_sqm_est": 0.0,
        "buffer_radius_sqft": None, "qc_status": "NOT_VERIFIABLE",
        "bbox_or_mask": None, "image_metadata": {"source": None, "capture_date": None},
    }
    try:
        if not api_key:
            print("[pipeline] no api_key -> early return with placeholder")
            overlay = make_overlay(Image.new("RGB", (640,640), (28,28,30)), Image.new("L", (640,640), 0), 0, 0, 0.0)
            return rec, overlay, True

        img = fetch_satellite_image(lat, lon, api_key=api_key)
        W, H = img.size
        gsd = compute_gsd_m_per_px(lat)
        r1200 = compute_buffer_radius_px(1200, gsd)
        r2400 = compute_buffer_radius_px(2400, gsd)

        print("[pipeline] loading model")
        model = load_model()
        if model is None:
            print("[pipeline] model missing")
            overlay = make_overlay(img, Image.new("L", img.size, 0), r1200, r2400, 0.0)
            rec.update({"image_metadata": {"source": "GOOGLE_STATIC_MAPS", "capture_date": "UNKNOWN"}, "qc_status": "NOT_VERIFIABLE"})
            return rec, overlay, True

        print("[pipeline] running predict")
        mask_img, confidence = predict(model, img)

        # normalize mask to numpy uint8
        if hasattr(mask_img, "convert"):
            pv_arr_np = (np.array(mask_img)).astype(np.uint8)
        else:
            pv_arr_np = np.array(mask_img).astype(np.uint8)

        buf1200 = create_circular_buffer((H, W), r1200)
        buf2400 = create_circular_buffer((H, W), r2400)

        area_1200 = compute_overlap_area(pv_arr_np, buf1200, gsd)
        area_2400 = compute_overlap_area(pv_arr_np, buf2400, gsd)

        if area_1200 > 1.0:
            has_solar = True; pv_area_m2 = area_1200; buffer_used_sqft = 1200
        else:
            has_solar = area_2400 > 1.0; pv_area_m2 = area_2400; buffer_used_sqft = 2400

        polygon_b64 = encode_mask_as_polygon(mask_img)
        rec = format_output(sid, lat, lon, has_solar, confidence, pv_area_m2, buffer_used_sqft, polygon_b64, "GOOGLE_STATIC_MAPS", "UNKNOWN")
        overlay = make_overlay(img, mask_img, r1200, r2400, confidence)

        needs_review = rec.get("qc_status", "REVIEW") != "VERIFIABLE"
        print(f"[pipeline.process_sample] done sid={sid} needs_review={needs_review}")
        return rec, overlay, needs_review

    except Exception as e:
        print("[pipeline.process_sample] exception:", e)
        traceback.print_exc()
        overlay = make_overlay(Image.new("RGB", (640,640), (28,28,30)), Image.new("L", (640,640), 0), 0, 0, 0.0)
        rec["qc_status"] = "NOT_VERIFIABLE"
        return rec, overlay, True

def run_batch_process(df, mode="Smart Review", api_key=None, session_state=None):
    print("[pipeline.run_batch_process] called, total rows:", len(df))
    results = []
    review_queue = []
    overlays = {}
    for idx, row in df.iterrows():
        sid = str(row["sample_id"])
        print(f"[pipeline] processing idx={idx} sid={sid}")
        rec, overlay, needs_review = process_sample(row, mode, api_key=api_key)
        # serialize overlay bytes
        buf = io.BytesIO()
        try:
            overlay.convert("RGB").save(buf, format="PNG")
            overlay_bytes = buf.getvalue()
        except Exception as e:
            print("[pipeline.run_batch_process] overlay save failed:", e)
            overlay_bytes = None

        if needs_review:
            review_queue.append({"record": rec, "overlay": overlay_bytes, "sid": sid})
        else:
            results.append(rec)
            overlays[sid] = overlay_bytes

    print("[pipeline.run_batch_process] done")
    return results, review_queue, overlays
