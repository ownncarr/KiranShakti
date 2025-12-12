# app.py

import os
import sys
import io
import json
import zipfile
import traceback
from typing import Any, Tuple, Optional

import base64
from math import cos, pi

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import streamlit as st
from streamlit_option_menu import option_menu

# -------------------------
# Optional dependencies
try:
    import requests
    REQ_OK = True
except Exception:
    REQ_OK = False

try:
    import torch
    import torchvision.transforms as T
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# segmentation-models-pytorch (only needed if we must reconstruct architecture)
try:
    import segmentation_models_pytorch as smp
    SMP_OK = True
except Exception:
    SMP_OK = False

# shapely for polygon hulls (postprocess)
try:
    from shapely.geometry import Polygon
    SHAPELY_OK = True
except Exception:
    SHAPELY_OK = False

# skimage disk function for buffer mask
try:
    from skimage.draw import disk
    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False

# cryptography for encrypted key storage
try:
    from cryptography.fernet import Fernet, InvalidToken
    CRYPTO_OK = True
except Exception:
    CRYPTO_OK = False

# -------------------------
# Project root safety
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------
# App config + encrypted storage helpers
APP_NAME = "pv_console"
CONFIG_DIR = os.path.join(os.path.expanduser("~"), f".{APP_NAME}")
KEY_FILE = os.path.join(CONFIG_DIR, "key.bin")
API_FILE = os.path.join(CONFIG_DIR, "api.enc")
os.makedirs(CONFIG_DIR, exist_ok=True)

def _make_restricted(path: str):
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass

def generate_and_store_key() -> bytes:
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    _make_restricted(KEY_FILE)
    return key

def load_key() -> Optional[bytes]:
    if not CRYPTO_OK:
        return None
    if not os.path.exists(KEY_FILE):
        return None
    try:
        with open(KEY_FILE, "rb") as f:
            return f.read()
    except Exception:
        return None

def store_api_key_encrypted(raw_api: str) -> bool:
    if not CRYPTO_OK:
        return False
    try:
        key = load_key() or generate_and_store_key()
        f = Fernet(key)
        token = f.encrypt(raw_api.encode("utf-8"))
        with open(API_FILE, "wb") as fa:
            fa.write(token)
        _make_restricted(API_FILE)
        return True
    except Exception:
        return False

def load_api_key_encrypted() -> Optional[str]:
    if not CRYPTO_OK:
        return None
    try:
        if not os.path.exists(API_FILE):
            return None
        key = load_key()
        if key is None:
            return None
        f = Fernet(key)
        with open(API_FILE, "rb") as fa:
            token = fa.read()
        return f.decrypt(token).decode("utf-8")
    except Exception:
        return None

def store_api_key_session(api: str):
    st.session_state.api_key = api

def load_api_key_session():
    return st.session_state.get("api_key")

# -------------------------
# Streamlit rerun compatibility helper
def safe_rerun():
    """
    Compatibility wrapper for streamlit rerun across versions.
    Tries several mechanisms; falls back to st.stop().
    """
    try:
        return st.experimental_rerun()
    except Exception:
        pass
    try:
        return st.rerun()
    except Exception:
        pass
    try:
        # older internal exception
        from streamlit.runtime.scriptrunner.script_runner import RerunException  # type: ignore
        raise RerunException()
    except Exception:
        try:
            return st.stop()
        except Exception:
            return None

# -------------------------
# UI theme (copied from your app)
ACCENT_1 = "#acc3a6"
ACCENT_2 = "#d1cdb0"
ACCENT_3 = "#F5D6BA"
ACCENT_4 = "#F49D6E"
BG = "#0f1115"
SURFACE = "#161a1e"
TEXT = "#e6edf3"
MUTED = "#9BA3AE"

st.set_page_config(page_title="PV Detection Console", layout="wide")

st.markdown(f"""
<style>
:root {{
  --accent-1: {ACCENT_1};
  --accent-2: {ACCENT_2};
  --accent-3: {ACCENT_3};
  --accent-4: {ACCENT_4};
  --bg: {BG};
  --surface: {SURFACE};
  --text: {TEXT};
  --muted: {MUTED};
}}
body, .stApp {{
  background: radial-gradient(1200px 600px at 10% 10%, var(--accent-1)14%, transparent 28%),
              radial-gradient(1100px 550px at 90% 90%, var(--accent-3)10%, transparent 26%),
              linear-gradient(180deg, #0f1115 0%, #141518 100%) !important;
  color: var(--text) !important;
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}}
[data-testid="stSidebar"] {{
  background: var(--surface) !important;
  border-right: 1px solid rgba(255,255,255,0.07);
}}
.option-menu .nav-link {{
  color: var(--muted) !important;
  background: transparent !important;
  border-radius: 6px;
}}
.option-menu .nav-link-selected {{
  background: linear-gradient(90deg, var(--accent-2), var(--accent-4)) !important;
  color: #0f1115 !important;
  font-weight: 700 !important;
}}
.section-header {{
  font-size: 1.06rem;
  font-weight: 650;
  margin-bottom: 10px;
  color: var(--text);
}}
.card {{
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
  border: 1px solid rgba(255,255,255,0.05);
  padding: 14px;
  border-radius: 10px;
}}
.small-muted {{
  color: var(--muted);
  font-size: 0.93rem;
}}
.btn-primary > button {{
  background: linear-gradient(90deg, var(--accent-1), var(--accent-3)) !important;
  color: #0b0b0b !important;
  border-radius: 10px !important;
  padding: 10px 16px !important;
  font-weight: 700 !important;
  border: none !important;
}}
.btn-ghost > button {{
  background: transparent !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
  padding: 8px 14px !important;
}}
.workflow-pills {{
  display: flex;
  gap: 10px;
  align-items: center;
  margin-bottom: 14px;
}}
.workflow-pill {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 8px 14px;
  border-radius: 999px;
  font-weight: 600;
  font-size: 0.95rem;
  color: var(--muted);
  background: transparent;
  border: 1px solid rgba(255,255,255,0.04);
}}
.workflow-pill--active {{
  background: linear-gradient(90deg, var(--accent-2), var(--accent-4));
  color: #0b0b0b !important;
  border: none;
}}
.workflow-pill--small {{
  padding: 6px 10px;
  font-size: 0.88rem;
  font-weight: 700;
}}
.workflow-label {{
  margin-right: 12px;
  font-weight: 700;
  color: var(--text);
  font-size: 0.98rem;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "Main",
        ["Home", "Settings", "How to use"],
        icons=["house", "gear", "question-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
    )

# -------------------------
# Session state defaults
if "df" not in st.session_state:
    st.session_state.df = None
if "in_workflow" not in st.session_state:
    st.session_state.in_workflow = False
if "workflow_step" not in st.session_state:
    st.session_state.workflow_step = "upload"
if "results" not in st.session_state:
    st.session_state.results = []
if "review_queue" not in st.session_state:
    st.session_state.review_queue = []
if "overlays" not in st.session_state:
    st.session_state.overlays = {}
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# Load encrypted API on boot
if CRYPTO_OK:
    try:
        s = load_api_key_encrypted()
        if s:
            st.session_state.api_key = s
    except Exception:
        pass

# -------------------------
# Inline pipeline functions (merged from your modules)
# fetch_satellite_image (pipeline/fetch_image.py)
def fetch_satellite_image(lat: float, lon: float, api_key: str, zoom: int = 20, size_px: int = 640) -> Image.Image:
    """
    Fetch satellite image from Google Static Maps API.
    Always returns RGB image (canonical channel order) for preprocessing consistency.
    """
    if not api_key:
        return _placeholder_overlay((size_px, size_px), "no api key")
    if not REQ_OK:
        return _placeholder_overlay((size_px, size_px), "requests missing")
    try:
        url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            "center": f"{lat},{lon}",
            "zoom": int(zoom),
            "size": f"{size_px}x{size_px}",
            "maptype": "satellite",
            "key": api_key,
            "scale": "2",
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        return img.convert("RGB")  # Explicit RGB conversion for reproducibility
    except Exception:
        return _placeholder_overlay((size_px, size_px), "fetch failed")

# model_inference (merged)
# ---------------------------------------------------------
# MODEL LOADING — use a clean relative path (one deterministic path)
# ---------------------------------------------------------

# Absolute directory where app.py sits
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path relative to this file (one deterministic location)
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "pv_detector.pt")
)

# Create models dir (no harm)
os.makedirs(os.path.join(APP_DIR, "..", "models"), exist_ok=True)

# -------------------------
# PREPROCESSING CONSTANTS & METADATA (must match training exactly)
# -------------------------
MODEL_INPUT_SIZE = (512, 512)  # (width, height) — exact model input dimensions
TEMPERATURE = 1.5  # calibration scale for sigmoid
INTERPOLATION = Image.LANCZOS  # Lanczos for high-quality resize (matches training)
PAD_VALUE = (28, 28, 30)  # RGB padding color

# ImageNet normalization (standard; match training exactly)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Preprocessing metadata for reproducibility & validation
PREPROCESS_METADATA = {
    "version": "1.0",
    "img_size": MODEL_INPUT_SIZE,
    "pad_value": PAD_VALUE,
    "mean": IMAGENET_MEAN,
    "std": IMAGENET_STD,
    "interpolation": "LANCZOS",
    "channel_order": "RGB",
    "tensor_layout": "CHW",
    "description": "Letterbox resize (aspect-ratio preserving) + pad + normalize(ImageNet)"
}

if TORCH_OK:
    preprocess = T.Compose([
        T.ToTensor(),  # uint8 [0,255] → float32 [0,1], reorder to CHW
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
else:
    preprocess = None

def get_default_device(prefer_cuda: bool = True):
    if TORCH_OK and prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if TORCH_OK:
        return torch.device("cpu")
    return None

_MODEL = None
_MODEL_DEVICE = None

def load_model():
    """
    Robust loader with allowlist for segmentation_models_pytorch.UnetPlusPlus
    (Option A). Attempts safe object unpickling using torch.serialization.add_safe_globals
    when available and falls back gracefully to older torch behavior.
    """
    import traceback

    global _MODEL, _MODEL_DEVICE
    if _MODEL is not None:
        return _MODEL

    if not TORCH_OK:
        print("Torch not available → running in dummy mode.")
        _MODEL = None
        return None

    if not os.path.exists(MODEL_PATH):
        print("MODEL FILE NOT FOUND AT:", MODEL_PATH)
        _MODEL = None
        return None

    device = get_default_device() or torch.device("cpu")
    _MODEL_DEVICE = device

    try:
        # Build an allowlist for safe unpickling if smp is present
        allowlist = []
        if SMP_OK:
            try:
                import segmentation_models_pytorch as smp  # may raise if SMP not actually importable
                # The exact class referenced in the error message:
                UnetPP = smp.decoders.unetplusplus.model.UnetPlusPlus
                allowlist.append(UnetPP)
            except Exception:
                # if we can't import or access the class, continue without allowlist
                allowlist = []

        # Try to use torch.serialization.add_safe_globals when available.
        # Also try to use weights_only=False (PyTorch >=2.6); handle older torch versions.
        model = None
        ser = getattr(torch, "serialization", None)
        add_safe = getattr(ser, "add_safe_globals", None) if ser is not None else None

        # Primary attempt: with add_safe_globals (if available) and weights_only=False
        try:
            if add_safe is not None and allowlist:
                with add_safe(allowlist):
                    # weights_only=False allows object unpickling for allowlisted globals.
                    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            else:
                # If we don't have an allowlist, still attempt to load with weights_only=False;
                # this can raise on newer PyTorch if globals aren't allowed.
                model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        except TypeError:
            # torch.load doesn't accept weights_only parameter (older torch) — retry without it
            try:
                # Skip add_safe if unavailable; just load normally
                model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            except TypeError:
                # Older torch: weights_only not supported, remove it
                model = torch.load(MODEL_PATH, map_location=device)
            except Exception:
                # Propagate to outer handler
                raise
        except Exception:
            # If the above failed (e.g. due to allowlist missing/unsafe), attempt a safer path below
            # and let outer except handle it if it ultimately fails.
            raise

        # If loaded object is an actual nn.Module — use it directly
        if isinstance(model, torch.nn.Module):
            model.to(device)
            model.eval()
            _MODEL = model
            print(f"Loaded full model object from {MODEL_PATH}")
            return _MODEL

        # If we got a dict-like checkpoint (weights/state_dict) — reconstruct UNet++
        if isinstance(model, dict):
            print("Model file contains state_dict-style checkpoint; reconstructing UNet++...")
            if not SMP_OK:
                print("segmentation_models_pytorch not available; cannot rebuild UNet++.")
                _MODEL = None
                return None
            import segmentation_models_pytorch as smp
            m = smp.UnetPlusPlus(
                encoder_name="efficientnet-b3",
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None
            )
            # Support both {'model_state': state_dict} and raw state_dict
            state = model.get("model_state", model)
            try:
                m.load_state_dict(state)
            except Exception as e:
                print("Failed to load state_dict into reconstructed UnetPlusPlus:", e)
                traceback.print_exc()
                _MODEL = None
                return None
            m.to(device)
            m.eval()
            _MODEL = m
            print("Reconstructed model from state_dict successfully.")
            return _MODEL

        # Unknown object type
        print(f"Unknown object type loaded from {MODEL_PATH}: {type(model)}")
        _MODEL = None
        return None

    except Exception as e:
        print(f"Failed to load model from {MODEL_PATH}: {e}")
        traceback.print_exc()
        _MODEL = None
        return None



# -------------------------
# CANONICAL PREPROCESSING PIPELINE
# Order: decode → RGB → EXIF rotate → letterbox → pad → normalize
# -------------------------
def _handle_exif_rotation(pil_img: Image.Image) -> Image.Image:
    """Apply EXIF rotation for reproducibility across camera sources."""
    try:
        from PIL import ExifTags
        exif = pil_img._getexif() if hasattr(pil_img, '_getexif') else None
        if exif is None:
            return pil_img
        exif_dict = {ExifTags.TAGS[k]: v for k, v in exif.items() if k in ExifTags.TAGS}
        orientation = exif_dict.get('Orientation', 1)
        rotations = {3: 180, 6: 270, 8: 90}
        if orientation in rotations:
            return pil_img.rotate(rotations[orientation], expand=True)
    except Exception:
        pass
    return pil_img

def resize_and_pad(pil_img: Image.Image, target_size=MODEL_INPUT_SIZE, fill_color=None):
    """
    Canonical preprocessing: aspect-ratio preserving resize + centerpad.
    
    Pipeline:
      1. RGB conversion (drop alpha, convert grayscale)
      2. EXIF rotation (camera orientation)
      3. Scale: longer side → target size (maintain aspect)
      4. Pad to exact target_size
    
    Returns:
        (padded_img, metadata) — metadata for prediction denormalization
    """
    if fill_color is None:
        fill_color = PAD_VALUE
    
    # Step 1: Ensure RGB (handles RGBA, grayscale, etc.)
    if pil_img.mode == "RGBA":
        bg = Image.new("RGB", pil_img.size, fill_color)
        bg.paste(pil_img, mask=pil_img.split()[3])
        pil_img = bg
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    
    # Step 2: EXIF rotation
    pil_img = _handle_exif_rotation(pil_img)
    
    # Step 3: Aspect-ratio preserving scale + pad
    target_w, target_h = target_size
    src_w, src_h = pil_img.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = pil_img.resize((new_w, new_h), resample=INTERPOLATION)
    
    # Step 4: Pad centered to exact target size
    padded = Image.new("RGB", (target_w, target_h), fill_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))
    
    meta = {
        "orig_size": (src_w, src_h),
        "resized_size": (new_w, new_h),
        "paste": (paste_x, paste_y),
        "scale": scale,
        "pad_value": fill_color,
        "preprocess_version": PREPROCESS_METADATA["version"]
    }
    return padded, meta

# -------------------------
def predict(model, pil_image: Image.Image, device: Optional[Any] = None):
    """
    Inference with canonical preprocessing (training-inference parity).
    
    Pipeline (matching training exactly):
      1. Preprocess: RGB convert → EXIF rotate → letterbox → pad
      2. Normalize: uint8 [0,255] → float32 [0,1] → ImageNet normalized
      3. Infer: logits → sigmoid(logits / TEMPERATURE)
      4. Denormalize: mask → original image coordinates
    
    Returns:
        (mask_resized, confidence): mask (PIL L uint8 [0,255]), confidence (float [0,1])
    """
    if model is None or not TORCH_OK:
        w, h = pil_image.size
        return Image.new("L", (w, h), 0), 0.0
    try:
        device = device or _MODEL_DEVICE or get_default_device()
        
        # PREPROCESS: RGB + EXIF + letterbox + pad (uses PREPROCESS_METADATA)
        pil_preprocessed, meta = resize_and_pad(pil_image, MODEL_INPUT_SIZE)
        
        # NORMALIZE: uint8 [0,255] → float32 [0,1] (ToTensor) → ImageNet normalized
        # Order: ToTensor (uint8→float [0,1], HWC→CHW) then Normalize (subtract mean, divide std)
        img_tensor = preprocess(pil_preprocessed).unsqueeze(0).to(device)
        
        # INFERENCE: logits → probabilities via sigmoid
        with torch.no_grad():
            logits = model(img_tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            prob_map = torch.sigmoid(logits / TEMPERATURE)[0, 0].detach().cpu().numpy()
        
        # BINARIZE at 0.5 threshold to create binary mask
        mask_bool = (prob_map > 0.5)
        mask = (mask_bool.astype(np.uint8) * 255)
        mask_img_512 = Image.fromarray(mask.astype(np.uint8))

        # DENORMALIZE: resize mask from preprocessed coords back to original image size
        orig_w, orig_h = meta["orig_size"]
        mask_resized = mask_img_512.resize((orig_w, orig_h), resample=Image.NEAREST)

        # CONFIDENCE: prefer statistics inside predicted mask when available
        try:
            if mask_bool.sum() > 0:
                mask_probs = prob_map[mask_bool]
                mean_in_mask = float(mask_probs.mean())
                p95_in_mask = float(np.percentile(mask_probs, 95))
                # Combine mean and high-percentile to be robust and reflect strong peaks
                confidence = float(0.6 * mean_in_mask + 0.4 * p95_in_mask)
            else:
                # No mask: fall back to 95th percentile of global prob map
                confidence = float(np.percentile(prob_map, 95))
        except Exception:
            # Last fallback: global mean
            confidence = float(np.mean(prob_map))
        return mask_resized, confidence
    except Exception:
        traceback.print_exc()
        w, h = pil_image.size
        return Image.new("L", (w, h), 0), 0.0

# postprocess (merged)
def encode_mask_as_polygon(mask_img: Image.Image) -> Optional[str]:
    """
    Try convex hull via shapely. Fallback: return base64-encoded PNG bytes.
    """
    try:
        arr = np.array(mask_img)
        ys, xs = np.where(arr == 255)
        if len(xs) == 0:
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
    except Exception:
        traceback.print_exc()
        return None

def qc_status(confidence: float, pv_area_m2: float) -> str:
    if pv_area_m2 > 1.0 and confidence > 0.8:
        return "VERIFIABLE"
    elif confidence > 0.4:
        return "REVIEW"
    return "LOW_CONFIDENCE"

def format_output(sample_id, lat, lon, has_solar, confidence, pv_area, buffer_radius_sqft, bbox_or_mask, img_source="GOOGLE_STATIC_MAPS", capture_date="UNKNOWN"):
    return {
        "sample_id": sample_id,
        "lat": lat,
        "lon": lon,
        "has_solar": bool(has_solar),
        "confidence": round(float(confidence), 4),
        "pv_area_sqm_est": round(float(pv_area or 0.0), 2),
        "buffer_radius_sqft": round(float(buffer_radius_sqft or 0.0), 2),
        "qc_status": qc_status(confidence, pv_area),
        "bbox_or_mask": bbox_or_mask,
        "image_metadata": {"source": img_source, "capture_date": capture_date},
    }

# buffer utils (merged)
def compute_gsd_m_per_px(lat: float, zoom: int = 20) -> float:
    try:
        return 156543.03392 * cos(lat * pi / 180) / (2 ** zoom)
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

# overlay helpers (adapted from backend)
def _make_overlay(image: Image.Image, mask_img: Image.Image, r1200_px: int, r2400_px: int, confidence: float, show_1200: bool = True, show_2400: bool = True) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
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
            if mask_alpha.mode != "L":
                mask_alpha = mask_alpha.convert("L")
            red.putalpha(mask_alpha)
            overlay = Image.alpha_composite(overlay, red)
            # Recreate drawing context on the composited overlay so subsequent
            # shapes are drawn onto the actual overlay image (Image.alpha_composite
            # returns a new Image object).
            draw = ImageDraw.Draw(overlay)

            # Draw bounding boxes around detected regions
            try:
                from scipy import ndimage
                labeled, num_features = ndimage.label(mask_arr == 255)
                for i in range(1, num_features + 1):
                    region = (labeled == i)
                    y_coords, x_coords = np.where(region)
                    if len(y_coords) > 0 and len(x_coords) > 0:
                        x_min, x_max = int(x_coords.min()), int(x_coords.max())
                        y_min, y_max = int(y_coords.min()), int(y_coords.max())
                        # Draw only a cyan outline box (no fill)
                        outline_color = (0, 255, 255, 255)
                        try:
                            draw.rectangle([x_min, y_min, x_max, y_max], outline=outline_color, width=2)
                        except Exception:
                            # Pillow fallback - try without width
                            draw.rectangle([x_min, y_min, x_max, y_max], outline=outline_color)
            except Exception:
                pass
    except Exception:
        mask_has_data = False

    try:
        box1200 = [center[0] - r1200_px, center[1] - r1200_px, center[0] + r1200_px, center[1] + r1200_px]
        box2400 = [center[0] - r2400_px, center[1] - r2400_px, center[0] + r2400_px, center[1] + r2400_px]
        # Draw only the selected circle(s)
        if show_2400:
            draw.ellipse(box2400, outline=(255, 200, 40, 220), width=3)
        if show_1200:
            draw.ellipse(box1200, outline=(120, 220, 130, 240), width=4)
    except Exception:
        pass

    if not mask_has_data:
        try:
            cx, cy = center
            draw.line((cx - 12, cy, cx + 12, cy), fill=(255, 255, 255, 200), width=2)
            draw.line((cx, cy - 12, cx, cy + 12), fill=(255, 255, 255, 200), width=2)
            font = ImageFont.load_default()
            draw.text((8, 8), "no mask", fill=(255, 255, 255, 220), font=font)
        except Exception:
            pass

    try:
        txt = f"conf: {confidence:.3f}"
        font = ImageFont.load_default()
        text_pos = (8, 8)
        tw, th = _text_size(draw, txt, font)
        draw.rectangle([text_pos, (text_pos[0] + tw + 6, text_pos[1] + th + 4)], fill=(0, 0, 0, 160))
        draw.text((text_pos[0] + 3, text_pos[1] + 2), txt, fill=(255, 255, 255, 240), font=font)
    except Exception:
        pass

    result = Image.alpha_composite(base, overlay).convert("RGB")
    return result

# Robust text-size helper (Pillow compatibility)
def _text_size(draw: ImageDraw.ImageDraw, text: str, font: Optional[ImageFont.ImageFont] = None) -> tuple[int, int]:
    """
    Return (width, height) for given text and font, using
    textbbox -> textsize -> font.getsize -> fallback estimate.
    """
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

def _placeholder_overlay(size=(640, 640), message: Optional[str] = None) -> Image.Image:
    img = Image.new("RGB", size, (28, 28, 30))
    d = ImageDraw.Draw(img)
    txt = message or "No overlay"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    w, h = _text_size(d, txt, font)
    d.text(((size[0] - w) // 2, (size[1] - h) // 2), txt, fill=(255, 255, 255), font=font)
    return img

# -------------------------
# process_sample (single-entrypoint used by UI)
def process_sample(row, mode: str = "Smart Review", api_key: Optional[str] = None) -> Tuple[dict, Image.Image, bool]:
    sid = row.get("sample_id", None)
    try:
        lat = float(row.get("lat", 0.0))
        lon = float(row.get("lon", 0.0))
    except Exception:
        lat = 0.0
        lon = 0.0

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
            # offline placeholder (no network)
            overlay = _placeholder_overlay(message="no api key")
            return rec, overlay, True

        img = fetch_satellite_image(lat, lon, api_key=api_key)
        W, H = img.size

        # compute gsd & buffer radii
        gsd = compute_gsd_m_per_px(lat)
        r1200 = compute_buffer_radius_px(1200, gsd)
        r2400 = compute_buffer_radius_px(2400, gsd)

        model = load_model()
        if model is None:
            rec.update({"image_metadata": {"source": "GOOGLE_STATIC_MAPS", "capture_date": "UNKNOWN"}, "qc_status": "NOT_VERIFIABLE"})
            overlay = _placeholder_overlay(size=img.size, message="model missing")
            needs_review = True
            return rec, overlay, needs_review

        mask_img, confidence = predict(model, img)

        pv_arr = np.array(mask_img)
        buf1200 = create_circular_buffer((H, W), r1200)
        buf2400 = create_circular_buffer((H, W), r2400)

        area_1200 = compute_overlap_area(pv_arr, buf1200, gsd)
        area_2400 = compute_overlap_area(pv_arr, buf2400, gsd)

        # Decide which buffer to use for reporting and visualization.
        if area_1200 > 1.0:
            # PV detected within 1200 sqft circle — focus prediction there
            used_buf = buf1200
            show_1200, show_2400 = True, False
            has_solar = True
            pv_area_m2 = area_1200
            buffer_used_sqft = 1200
        elif area_2400 > 1.0:
            # No PV in 1200, but present in 2400 — focus prediction in 2400 circle
            used_buf = buf2400
            show_1200, show_2400 = False, True
            has_solar = True
            pv_area_m2 = area_2400
            buffer_used_sqft = 2400
        else:
            # No PV found in either; default to 1200 circle for visualization
            used_buf = buf1200
            show_1200, show_2400 = True, False
            has_solar = False
            pv_area_m2 = area_1200
            buffer_used_sqft = 1200

        # Mask the predicted mask so overlay and polygon reflect only the chosen buffer
        try:
            masked_arr = ((pv_arr == 255) & (used_buf == 1)).astype(np.uint8) * 255
            masked_mask_img = Image.fromarray(masked_arr.astype(np.uint8))
        except Exception:
            masked_mask_img = mask_img

        polygon_b64 = encode_mask_as_polygon(masked_mask_img)

        # If the model predicts PV outside the 2400 sqft buffer, include an annulus just outside
        # the circle and merge any predicted pixels within that annulus into a single combined mask.
        try:
            outside_px = ((pv_arr == 255) & (buf2400 == 0)).sum()
            if outside_px > 0:
                # Expand search area by a margin (half the 2400 radius or at least 60px),
                # and limit to image bounds.
                margin_px = max(int(r2400 * 0.5), 60)
                extended_radius = r2400 + margin_px
                # create ring buffer = extended radius minus original 2400 radius
                extended_buf = create_circular_buffer((H, W), extended_radius)
                ring_buf = ((extended_buf == 1) & (buf2400 == 0)).astype(np.uint8)
                # Check whether predicted PV exists inside that ring
                ring_px = ((pv_arr == 255) & (ring_buf == 1)).sum()
                if ring_px > 0:
                    # Merge predicted pixels inside 2400 and ring into final mask
                    combined_arr = ((pv_arr == 255) & ((buf2400 == 1) | (ring_buf == 1))).astype(np.uint8) * 255
                    masked_mask_img = Image.fromarray(combined_arr.astype(np.uint8))
                    # Recompute area: area_in_2400 + area_in_ring
                    area_in_2400 = compute_overlap_area(pv_arr, buf2400, gsd)
                    area_in_ring = compute_overlap_area(pv_arr, ring_buf, gsd)
                    pv_area_m2 = area_in_2400 + area_in_ring
                    # Re-encode polygon / mask for export
                    polygon_b64 = encode_mask_as_polygon(masked_mask_img)
                    # Ensure buffer used remains 2400 for reporting
                    buffer_used_sqft = 2400
        except Exception:
            # Non-fatal; fall back to previous masked result if anything fails
            pass

        # If extremely low confidence, mark directly as solar absent (auto-accepted)
        if confidence < 0.05:
            # Enforce absent, zero area, and no mask in outputs
            has_solar = False
            pv_area_m2 = 0.0
            buffer_used_sqft = buffer_used_sqft or 1200
            masked_mask_img = Image.new("L", img.size, 0)
            polygon_b64 = None
            rec = format_output(
                sample_id=sid,
                lat=lat,
                lon=lon,
                has_solar=False,
                confidence=confidence,
                pv_area=pv_area_m2,
                buffer_radius_sqft=buffer_used_sqft,
                bbox_or_mask=polygon_b64,
                img_source="GOOGLE_STATIC_MAPS",
                capture_date="UNKNOWN",
            )

            overlay = _make_overlay(img, masked_mask_img, r1200, r2400, confidence, show_1200=show_1200, show_2400=show_2400)
            needs_review = False
            return rec, overlay, needs_review

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

        overlay = _make_overlay(img, masked_mask_img, r1200, r2400, confidence, show_1200=show_1200, show_2400=show_2400)

        qc = rec.get("qc_status", "REVIEW")
        if mode == "Review All":
            # Flag all samples for human review
            needs_review = True
        elif mode == "Automatic":
            # Skip review entirely - go straight to download/results (trust automation)
            needs_review = False
        else:  # "Smart Review"
            # Only keep images with confidence < 0.65 for manual review; skip images with higher confidence scores
            needs_review = confidence < 0.65

        return rec, overlay, needs_review

    except Exception:
        traceback.print_exc()
        rec["qc_status"] = "NOT_VERIFIABLE"
        overlay = _placeholder_overlay(message="error")
        needs_review = True
        return rec, overlay, needs_review

# -------------------------
# UI helpers and batch runner (copied/adapted)
def render_step_pills(current):
    steps = [("upload", "Upload"), ("process", "Process"), ("verify", "Verify"), ("download", "Download")]
    html_parts = []
    html_parts.append("<div class='workflow-label'>Workflow</div>")
    html_parts.append("<div class='workflow-pills'>")
    for key, title in steps:
        cls = "workflow-pill workflow-pill--active" if key == current else "workflow-pill"
        if key in ("verify", "download"):
            cls += " workflow-pill--small"
        html_parts.append(f"<div class='{cls}' role='button'>{title}</div>")
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)

def run_batch_process(df, mode="Smart Review"):
    st.session_state.results = []
    st.session_state.review_queue = []
    st.session_state.overlays = {}

    total = len(df)
    prog = st.progress(0)
    status = st.empty()

    api_key = st.session_state.get("api_key")
    if not api_key and CRYPTO_OK:
        api_key = load_api_key_encrypted()
        if api_key:
            st.session_state.api_key = api_key

    if not api_key:
        st.error("No API key available. Go to Settings and save your Google Maps API key (encrypted on device).")
        return

    debug_dir = os.path.join(CONFIG_DIR, "debug_overlays")
    os.makedirs(debug_dir, exist_ok=True)

    for idx, row in df.iterrows():
        sid = str(row["sample_id"])
        status.text(f"Processing {sid} ({idx+1}/{total})")
        try:
            rec, overlay, needs_review = process_sample(row, mode, api_key=api_key)

            try:
                buf = io.BytesIO()
                overlay.convert("RGB").save(buf, format="PNG")
                overlay_bytes = buf.getvalue()
                try:
                    with open(os.path.join(debug_dir, f"{sid}.png"), "wb") as f:
                        f.write(overlay_bytes)
                except Exception:
                    pass
            except Exception:
                overlay_bytes = None

            if needs_review:
                st.session_state.review_queue.append({"record": rec, "overlay": overlay_bytes, "sid": sid})
            else:
                st.session_state.results.append(rec)
                st.session_state.overlays[sid] = overlay_bytes

        except Exception as e:
            st.session_state.review_queue.append({
                "record": {
                    "sample_id": sid,
                    "lat": row.get("lat"),
                    "lon": row.get("lon"),
                    "has_solar": False,
                    "confidence": 0.0,
                    "pv_area_sqm_est": 0.0,
                    "buffer_radius_sqft": None,
                    "qc_status": "NOT_VERIFIABLE",
                    "bbox_or_mask": None,
                    "image_metadata": {"source": None, "capture_date": None},
                },
                "overlay": None,
                "sid": sid
            })

        prog.progress((idx+1) / total)

    st.session_state.workflow_step = "verify" if len(st.session_state.review_queue) else "download"
    status.text("Processing complete.")

# -------------------------
# Main UI pages (Home / Settings / How to use)
if selected == "Home":
    st.markdown("<div class='section-header'>Rooftop PV Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>A clean, audit-ready workflow for rooftop solar detection.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([3,1])

    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Professional PV Detection Workflow")
        st.markdown("<div class='small-muted'>Upload, process, verify, and export — all in one seamless dark-mode console.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
        if st.button("Start new analysis", key="start_x"):
            st.session_state.in_workflow = True
            st.session_state.workflow_step = "upload"
            safe_rerun()
    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Session snapshot")
        st.markdown(f"<div class='small-muted'>Approved: {len(st.session_state.results)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>Pending review: {len(st.session_state.review_queue)}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.in_workflow:
        st.markdown("<div class='section-header'></div>", unsafe_allow_html=True)
        render_step_pills(st.session_state.workflow_step)
        st.write("")

        # UPLOAD
        if st.session_state.workflow_step == "upload":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("**Step 1 — Upload dataset**")
            up = st.file_uploader("Upload .xlsx with sample_id, lat, lon", type=["xlsx"])
            if up:
                try:
                    import pandas as pd
                    df = pd.read_excel(up)
                    if {"sample_id", "lat", "lon"}.issubset(df.columns):
                        def _normalize_sid(v):
                            import pandas as pd
                            if pd.isna(v):
                                return ""
                            if isinstance(v, float) and v.is_integer():
                                return str(int(v))
                            if isinstance(v, (int,)):
                                return str(v)
                            return str(v).strip()
                        df["sample_id"] = df["sample_id"].apply(_normalize_sid)
                        st.success(f"{len(df)} samples loaded.")
                        st.session_state.df = df
                    else:
                        st.error("File must contain columns: sample_id, lat, lon")
                except Exception as e:
                    st.error(str(e))

            cN, cC = st.columns(2)
            with cN:
                st.markdown("<div class='btn-primary'>", unsafe_allow_html=True)
                if st.button("Next →", key="up_next"):
                    if st.session_state.df is None:
                        st.error("Upload dataset first.")
                    else:
                        st.session_state.workflow_step = "process"
                        safe_rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            with cC:
                st.markdown("<div class='btn-ghost'>", unsafe_allow_html=True)
                if st.button("Cancel"):
                    st.session_state.in_workflow = False
                    safe_rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # PROCESS
        elif st.session_state.workflow_step == "process":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("**Step 2 — Process (inference & QC)**")
            mode = st.selectbox("Processing mode", ["Automatic", "Smart Review", "Review All"], index=1)

            cRun, cBack = st.columns(2)
            with cRun:
                st.markdown("<div class='btn-primary'>", unsafe_allow_html=True)
                if st.button("Run processing", key="runproc"):
                    # Prevent running without an uploaded dataset
                    if st.session_state.get("df") is None:
                        st.error("Upload a data first (Home → Start new analysis → Upload).")
                    else:
                        api_key = st.session_state.get("api_key")
                        if not api_key and CRYPTO_OK:
                            api_key = load_api_key_encrypted()
                            if api_key:
                                st.session_state.api_key = api_key

                        if not api_key:
                            st.error("No API key found. Save your API key in Settings (encrypted on device) or paste it into the field.")
                        else:
                            run_batch_process(st.session_state.df, mode)
                    safe_rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            with cBack:
                st.markdown("<div class='btn-ghost'>", unsafe_allow_html=True)
                if st.button("← Back"):
                    st.session_state.workflow_step = "upload"
                    safe_rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # VERIFY
        elif st.session_state.workflow_step == "verify":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("**Step 3 — Verify flagged samples**")

            queue = st.session_state.review_queue
            if len(queue) == 0:
                st.info("No flagged items.")
                if st.button("Next →"):
                    st.session_state.workflow_step = "download"
                    safe_rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                idx = st.number_input("Queue index", min_value=0, max_value=len(queue)-1, step=1)
                row = queue[int(idx)]
                rec = row["record"]
                overlay_bytes = row["overlay"]
                sid = row["sid"]

                st.subheader(f"Sample {sid}")
                if overlay_bytes:
                    try:
                        buf = io.BytesIO(overlay_bytes)
                        overlay_pil = Image.open(buf).convert("RGB")
                        st.image(overlay_pil, width=512)
                    except Exception:
                        st.image(_placeholder_overlay(), width=512)
                else:
                    st.image(_placeholder_overlay(), width=512)

                st.write(f"Confidence: {rec.get('confidence', 0.0)}")
                st.write(f"Suggested QC: {rec.get('qc_status', 'UNKNOWN')}")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("<div class='btn-primary'>", unsafe_allow_html=True)
                    if st.button("Solar present"):
                        rec["has_solar"] = True; rec["qc_status"] = "VERIFIABLE"
                        st.session_state.results.append(rec)
                        st.session_state.overlays[sid] = overlay_bytes or None
                        queue.pop(int(idx))
                        safe_rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

                with c2:
                    st.markdown("<div class='btn-ghost'>", unsafe_allow_html=True)
                    if st.button("Solar absent"):
                        rec["has_solar"] = False; rec["qc_status"] = "VERIFIABLE"
                        st.session_state.results.append(rec)
                        st.session_state.overlays[sid] = overlay_bytes or None
                        queue.pop(int(idx))
                        safe_rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

                with c3:
                    st.markdown("<div class='btn-ghost'>", unsafe_allow_html=True)
                    if st.button("Mark unusable"):
                        rec["has_solar"] = False; rec["qc_status"] = "NOT_VERIFIABLE"
                        st.session_state.results.append(rec)
                        st.session_state.overlays[sid] = overlay_bytes or None
                        queue.pop(int(idx))
                        safe_rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

                cn, cb = st.columns(2)
                with cn:
                    st.markdown("<div class='btn-primary'>", unsafe_allow_html=True)
                    if st.button("Next →", key="v_next"):
                        if len(queue) == 0:
                            st.session_state.workflow_step = "download"
                        safe_rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                with cb:
                    st.markdown("<div class='btn-ghost'>", unsafe_allow_html=True)
                    if st.button("← Back", key="v_back"):
                        st.session_state.workflow_step = "process"
                        safe_rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # DOWNLOAD
        elif st.session_state.workflow_step == "download":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("**Step 4 — Download results**")
            total = len(st.session_state.results)
            st.write(f"Approved records: {total}")

            if total > 0:
                mem = io.BytesIO()
                with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
                    z.writestr("predictions.json", json.dumps(st.session_state.results, indent=2))
                    for sid, img_bytes in st.session_state.overlays.items():
                        if img_bytes:
                            z.writestr(f"overlays/{sid}.png", img_bytes)
                mem.seek(0)
                st.download_button("Download ZIP", data=mem, file_name="pv_results.zip")

            cCLOSE, cNEW = st.columns(2)
            with cCLOSE:
                st.markdown("<div class='btn-ghost'>", unsafe_allow_html=True)
                if st.button("Close workflow"):
                    st.session_state.in_workflow = False
                    st.session_state.workflow_step = "upload"
                    safe_rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            with cNEW:
                st.markdown("<div class='btn-primary'>", unsafe_allow_html=True)
                if st.button("Restart workflow"):
                    st.session_state.df = None
                    st.session_state.results = []
                    st.session_state.review_queue = []
                    st.session_state.overlays = {}
                    st.session_state.workflow_step = "upload"
                    safe_rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

# SETTINGS page
elif selected == "Settings":
    st.markdown("<div class='section-header'>Settings</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**API Key**")
    st.markdown("<div class='small-muted'>Stored securely on device using encryption (if cryptography installed).</div>", unsafe_allow_html=True)

    current = st.session_state.api_key or ""
    api = st.text_input("API key", value=current, type="password")

    cSAVE, cDEL = st.columns(2)
    with cSAVE:
        st.markdown("<div class='btn-primary'>", unsafe_allow_html=True)
        if st.button("Save API key"):
            if api.strip() == "":
                st.error("API key cannot be empty.")
            else:
                if CRYPTO_OK:
                    ok = store_api_key_encrypted(api.strip())
                    if ok:
                        st.session_state.api_key = api.strip()
                        st.success("API key saved (encrypted).")
                    else:
                        st.error("Could not save encrypted key. Using session memory.")
                        store_api_key_session(api.strip())
                else:
                    store_api_key_session(api.strip())
                    st.warning("cryptography not installed — stored only in session.")
        st.markdown("</div>", unsafe_allow_html=True)
    with cDEL:
        st.markdown("<div class='btn-ghost'>", unsafe_allow_html=True)
        if st.button("Clear stored key"):
            try:
                if os.path.exists(API_FILE): os.remove(API_FILE)
                if os.path.exists(KEY_FILE): os.remove(KEY_FILE)
                st.session_state.api_key = None
                st.success("Stored key removed.")
            except Exception as e:
                st.error(str(e))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:8px'/>", unsafe_allow_html=True)
    if CRYPTO_OK:
        if os.path.exists(API_FILE):
            st.success("Encrypted API key found on device.")
        else:
            st.info("No encrypted API key stored on device. Use 'Save API key' to store it securely.")
    else:
        st.info("No encrypted storage available. Key is stored in browser session only.")

    if not TORCH_OK:
        st.warning("PyTorch not available — app will run in 'dummy' mode (no real model).")
    else:
        st.info("PyTorch available. If a model exists in ../models it will be attempted to load.")

    st.markdown("</div>", unsafe_allow_html=True)

# HOW TO USE
elif selected == "How to use":
    st.markdown("<div class='section-header'>How to use</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
1. Go to Settings and paste your Google Static Maps API key; click Save API key to store it encrypted on this device.  
2. Go to Home → Start new analysis  
3. Upload an Excel: `sample_id`, `lat`, `lon`  
4. Run processing (inference & QC) — the UI will use the encrypted key automatically.  
5. Verify flagged samples  
6. Download ZIP with predictions.json + overlays  
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
