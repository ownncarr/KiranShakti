# pipeline/utils/io_utils.py
import json
import os


def ensure_dir(path: str):
    """
    Create directory if not present.
    """
    os.makedirs(path, exist_ok=True)


def save_json(data, path: str):
    """
    Save Python list or dict to JSON.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
