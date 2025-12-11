# pv_console/model.py
print("[model.py] imported")
import os
import traceback
from typing import Optional, Any
import numpy as np
from PIL import Image
from pv_console.config import MODEL_INPUT_SIZE, TEMPERATURE
from pv_console.config import MODEL_RELATIVE_PATH
try:
    import torch
    import torchvision.transforms as T
    TORCH_OK = True
    print("[model.py] PyTorch available")
except Exception:
    TORCH_OK = False
    print("[model.py] PyTorch not available")

try:
    import segmentation_models_pytorch as smp
    SMP_OK = True
except Exception:
    SMP_OK = False

# Preprocess defined only if torch available
if TORCH_OK:
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
else:
    preprocess = None

_MODEL = None
_MODEL_DEVICE = None

def get_default_device(prefer_cuda: bool = True):
    if TORCH_OK and prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if TORCH_OK:
        return torch.device("cpu")
    return None

def load_model():
    global _MODEL, _MODEL_DEVICE
    print("[model.load_model] called")
    if _MODEL is not None:
        print("[model.load_model] model already loaded; returning cached")
        return _MODEL
    if not TORCH_OK:
        print("[model.load_model] TORCH not available -> dummy mode")
        _MODEL = None
        return None
    if not os.path.exists(MODEL_RELATIVE_PATH):
        print(f"[model.load_model] MODEL FILE NOT FOUND AT: {MODEL_RELATIVE_PATH}")
        _MODEL = None
        return None
    device = get_default_device() or torch.device("cpu")
    _MODEL_DEVICE = device
    try:
        model_obj = torch.load(MODEL_RELATIVE_PATH, map_location=device)
        if isinstance(model_obj, torch.nn.Module):
            model_obj.to(device); model_obj.eval()
            _MODEL = model_obj
            print("[model.load_model] loaded full model object")
            return _MODEL
        elif isinstance(model_obj, dict):
            print("[model.load_model] state-dict loaded, try reconstructing Unet++")
            if not SMP_OK:
                print("[model.load_model] segmentation_models_pytorch not available; cannot reconstruct")
                _MODEL = None
                return None
            m = smp.UnetPlusPlus(encoder_name="efficientnet-b3", encoder_weights=None, in_channels=3, classes=1, activation=None)
            state = model_obj.get("model_state", model_obj)
            m.load_state_dict(state)
            m.to(device); m.eval()
            _MODEL = m
            print("[model.load_model] reconstructed model from state_dict")
            return _MODEL
        else:
            print("[model.load_model] unknown object type in checkpoint:", type(model_obj))
            _MODEL = None
            return None
    except Exception as e:
        print("[model.load_model] exception while loading model:", e)
        traceback.print_exc()
        _MODEL = None
        return None

def resize_and_pad(pil_img: Image.Image, target_size=MODEL_INPUT_SIZE, fill_color=(28,28,30)):
    print("[model.resize_and_pad] called")
    target_w, target_h = target_size
    src_w, src_h = pil_img.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = pil_img.resize((new_w, new_h), resample=Image.LANCZOS)
    new_img = Image.new("RGB", (target_w, target_h), fill_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    new_img.paste(resized, (paste_x, paste_y))
    meta = {"orig_size": (src_w, src_h), "resized_size": (new_w, new_h), "paste": (paste_x, paste_y), "scale": scale}
    print("[model.resize_and_pad] done", meta)
    return new_img, meta

def predict(model, pil_image: Image.Image, device: Optional[Any] = None):
    print("[model.predict] called")
    if model is None or not TORCH_OK:
        print("[model.predict] no model or torch missing -> returning blank mask")
        w, h = pil_image.size
        from PIL import Image as PILImage
        return PILImage.new("L", (w, h), 0), 0.0
    try:
        device = device or _MODEL_DEVICE or get_default_device()
        pil_in = pil_image.convert("RGB")
        pil_letterboxed, meta = resize_and_pad(pil_in)
        img_tensor = preprocess(pil_letterboxed).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            prob_map = torch.sigmoid(logits / TEMPERATURE)[0, 0].detach().cpu().numpy()
        mask = (prob_map > 0.5).astype(np.uint8) * 255
        from PIL import Image as PILImage
        mask_img_512 = PILImage.fromarray(mask.astype(np.uint8))
        orig_w, orig_h = meta["orig_size"]
        mask_resized = mask_img_512.resize((orig_w, orig_h), resample=PILImage.NEAREST)
        confidence = float(prob_map.mean())
        print(f"[model.predict] done confidence={confidence:.4f}")
        return mask_resized, confidence
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("[model.predict] exception:", e)
        w, h = pil_image.size
        from PIL import Image as PILImage
        return PILImage.new("L", (w, h), 0), 0.0
