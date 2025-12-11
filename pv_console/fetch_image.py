# fetch_image.py — replace fetch_satellite_image with this
import io
from typing import Optional
from PIL import Image
import requests

def _placeholder_overlay(size=(640, 640), message: Optional[str] = None) -> Image.Image:
    from PIL import ImageDraw, ImageFont
    img = Image.new("RGB", size, (28, 28, 30))
    d = ImageDraw.Draw(img)
    txt = message or "No overlay"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    w = len(txt) * 7
    h = 12
    d.text(((size[0] - w) // 2, (size[1] - h) // 2), txt, fill=(255,255,255), font=font)
    return img

def fetch_satellite_image(lat: float, lon: float, api_key: str, zoom: int = 20, size_px: int = 640, scale:int = 2) -> Image.Image:
    """
    Fetch satellite image centered at (lat, lon).
    By default requests scale=1 so returned image pixels match the GSD formula.
    Returns PIL.Image (RGB), guaranteed to be (size_px * scale, size_px * scale).
    """
    print(f"[fetch_image] fetch lat={lat}, lon={lon}, zoom={zoom}, size_px={size_px}, scale={scale}")
    if not api_key:
        print("[fetch_image] no api key -> placeholder")
        return _placeholder_overlay((size_px, size_px), "no api key")
    try:
        url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            "center": f"{lat},{lon}",
            "zoom": int(zoom),
            "size": f"{size_px}x{size_px}",
            "maptype": "satellite",
            "key": api_key,
            "scale": str(scale),
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        # Ensure returned size is exactly (size_px*scale, size_px*scale)
        expected = (size_px * scale, size_px * scale)
        if img.size != expected:
            print(f"[fetch_image] returned size {img.size} != expected {expected}, resizing (LANCZOS)")
            img = img.resize(expected, resample=Image.LANCZOS)
        else:
            print(f"[fetch_image] returned size OK: {img.size}")
        return img
    except Exception as e:
        print("[fetch_image] fetch failed:", e)
        return _placeholder_overlay((size_px, size_px), "fetch failed")
