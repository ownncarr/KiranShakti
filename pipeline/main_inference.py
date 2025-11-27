# pipeline/main_inference.py

import os
import math
import pandas as pd
from PIL import Image, ImageDraw

from fetch_image import fetch_satellite_image

DATA_DIR = "../data"
PREPROCESS_DIR = "./preprocessing"

os.makedirs(PREPROCESS_DIR, exist_ok=True)

SQFT_TO_SQM = 0.092903
INNER_AREA_SQFT = 1200
OUTER_AREA_SQFT = 2400

def area_sqft_to_radius_m(area_sqft: float) -> float:
    """Convert area (sq.ft) → radius (m) assuming circular buffer."""
    area_m2 = area_sqft * SQFT_TO_SQM
    return math.sqrt(area_m2 / math.pi)

def meters_per_pixel_web_mercator(lat_deg: float, zoom: int) -> float:
    """
    Approximate ground resolution in meters per pixel
    for Web Mercator at a given latitude & zoom.
    """
    lat_rad = math.radians(lat_deg)
    earth_circumference = 40075016.686  # meters
    return math.cos(lat_rad) * earth_circumference / (256 * (2 ** zoom))

def downscale_image(img: Image.Image, target_size: int = 1200) -> Image.Image:
    """Downscale an image to target_size x target_size (for model input)."""
    return img.resize((target_size, target_size), Image.BILINEAR)

def draw_buffer_circles(img: Image.Image, lat: float, zoom: int) -> Image.Image:
    """
    Draw 1200 sq.ft (inner) and 2400 sq.ft (outer) circles on the image,
    using the image center as the (lat, lon) location.
    """
    width, height = img.size
    cx, cy = width // 2, height // 2  # this is (lat, lon) in pixel space

    # Real-world radii in meters
    inner_r_m = area_sqft_to_radius_m(INNER_AREA_SQFT)
    outer_r_m = area_sqft_to_radius_m(OUTER_AREA_SQFT)

    # Ground resolution at this lat & zoom
    m_per_px = meters_per_pixel_web_mercator(lat, zoom)

    # Meters → pixels
    inner_r_px = inner_r_m / m_per_px
    outer_r_px = outer_r_m / m_per_px

    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)

    # Outer circle (2400 sq.ft)
    bbox_outer = [
        cx - outer_r_px,
        cy - outer_r_px,
        cx + outer_r_px,
        cy + outer_r_px,
    ]

    # Inner circle (1200 sq.ft)
    bbox_inner = [
        cx - inner_r_px,
        cy - inner_r_px,
        cx + inner_r_px,
        cy + inner_r_px,
    ]

    # Draw outlines only
    draw.ellipse(bbox_outer, outline="red", width=3)
    draw.ellipse(bbox_inner, outline="yellow", width=3)

    return overlay

def run_preprocessing(input_excel: str):
    # 1) Read Excel
    excel_path = os.path.join(DATA_DIR, input_excel)
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Input Excel not found: {excel_path}")

    df = pd.read_excel(excel_path)

    required_cols = {"sample_id", "lat", "lon"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in Excel: {missing}")

    # 2) First record
    first_row = df.iloc[0]
    sample_id = first_row["sample_id"]
    lat = float(first_row["lat"])
    lon = float(first_row["lon"])

    print(f"Using first record → sample_id={sample_id}, lat={lat}, lon={lon}")

    # 3) Fetch base image (no fake 2400x2400)
    zoom = 20
    base_img = fetch_satellite_image(lat, lon, zoom=zoom, size_px=640)

    # 4) Draw accurate-scale circles centered at (lat, lon)
    base_with_circles = draw_buffer_circles(base_img, lat=lat, zoom=zoom)

    # 5) Downscale copy for model input (1200x1200)
    downscaled_img = downscale_image(base_img, target_size=1200)

    # 6) Save
    base_path = os.path.join(PREPROCESS_DIR, f"{sample_id}_base_raw.png")
    overlay_path = os.path.join(PREPROCESS_DIR, f"{sample_id}_base_with_circles.png")
    small_path = os.path.join(PREPROCESS_DIR, f"{sample_id}_downscaled_1200.png")

    base_img.save(base_path)
    base_with_circles.save(overlay_path)
    downscaled_img.save(small_path)

    print(f"✅ Saved base raw image → {base_path}")
    print(f"✅ Saved base image with circles → {overlay_path}")
    print(f"✅ Saved downscaled image → {small_path}")
    print("🎯 Preprocessing with accurate buffer circles complete.")

if __name__ == "__main__":
    run_preprocessing("coordinates.xlsx")
