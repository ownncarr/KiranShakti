# backend_stub.py
# compatibility shim so the UI import `import backend_stub as backend` continues to work
try:
    # prefer real pipeline backend if available
    from pipeline.backend import process_sample  # type: ignore
except Exception:
    # graceful fallback with matching signature (accepts api_key)
    def process_sample(row, mode="Smart Review", api_key=None):
        rec = {
            "sample_id": row.get("sample_id"),
            "lat": row.get("lat"),
            "lon": row.get("lon"),
            "has_solar": False,
            "confidence": 0.0,
            "pv_area_sqm_est": 0.0,
            "buffer_radius_sqft": None,
            "qc_status": "NOT_VERIFIABLE",
            "bbox_or_mask": None,
            "image_metadata": {"source": None, "capture_date": None},
        }
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (640, 640), (28, 28, 30))
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        msg = "backend missing"
        d.text((10, 10), msg, fill=(255, 255, 255), font=font)
        return rec, img, True
