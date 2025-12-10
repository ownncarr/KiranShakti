import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path

MODEL_PATH = Path("models/solar_detector.pt")

# Preprocessing
preprocess = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

TEMPERATURE = 1.5  # for calibrated confidence


def get_default_device(prefer_cuda: bool = True) -> torch.device:
 
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device: torch.device | None = None):
  
    device = device or get_default_device()
    # use map_location=device so weights are loaded in-place on the right device
    model = torch.load(MODEL_PATH, map_location=device)
    # ensure model is on device (some saved objects may need explicit .to)
    try:
        model.to(device)
    except Exception:
        # if model is a state dict or incompatible, we still return whatever torch.load returned
        pass
    model.eval()
    return model


def predict(model, pil_image: Image.Image, device: torch.device | None = None):
    device = device or get_default_device()
    img_tensor = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        # logits might be on GPU; move result to CPU for numpy
        prob_map = torch.sigmoid(logits / TEMPERATURE)[0, 0].detach().cpu().numpy()

    mask = (prob_map > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask)

    confidence = float(prob_map.mean())

    return mask_img, confidence
