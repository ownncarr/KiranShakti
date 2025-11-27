# pipeline/fetch_image.py

import os
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

def fetch_satellite_image(lat: float, lon: float, zoom: int = 20, size_px: int = 640) -> Image.Image:
    """
    Fetch a top-down satellite image centered at (lat, lon)
    using Google Maps Static API.

    We do NOT resize here. The image we get back is the one
    we treat as 'ground truth' for geometric scale.
    """
    if not API_KEY:
        raise RuntimeError(
            f"GOOGLE_MAPS_API_KEY not found. Expected in {env_path}"
        )

    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": f"{size_px}x{size_px}",  # actual image size
        "maptype": "satellite",
        "key": API_KEY,
    }

    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Google Static Maps API error {resp.status_code}:\n{resp.text}"
        )

    img = Image.open(BytesIO(resp.content)).convert("RGB")
    # At this point img.size == (size_px, size_px)
    return img
