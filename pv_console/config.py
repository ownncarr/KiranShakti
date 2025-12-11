# pipeline/config.py
print("[config.py] imported")

import os

APP_NAME = "pv_console"
CONFIG_DIR = os.path.join(os.path.expanduser("~"), f".{APP_NAME}")
KEY_FILE = os.path.join(CONFIG_DIR, "key.bin")
API_FILE = os.path.join(CONFIG_DIR, "api.enc")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_RELATIVE_PATH = os.path.abspath(os.path.join(APP_DIR, "..", "models", "best_model.pth"))
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(os.path.join(APP_DIR, "..", "models"), exist_ok=True)

MODEL_INPUT_SIZE = (512, 512)
TEMPERATURE = 1.5
