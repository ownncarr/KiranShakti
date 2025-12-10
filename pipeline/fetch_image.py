# pipeline/fetch_image.py
import requests
from io import BytesIO
from PIL import Image


def fetch_satellite_image(lat: float, lon: float, api_key: str, zoom: int = 20, size_px: int = 640) -> Image.Image:
    """
    Fetch a top-down satellite image centered at (lat, lon)
    using Google Maps Static API.

    The API key must be passed explicitly (no .env reading).
    """
    if not api_key:
        raise RuntimeError("Google Static Maps API key was not provided to fetch_satellite_image().")

    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": f"{size_px}x{size_px}",
        "maptype": "satellite",
        "key": api_key,
    }

    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Google Static Maps API error {resp.status_code}:\n{resp.text}")

    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return img
