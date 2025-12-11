# pv_console/crypto.py
print("[crypto.py] imported")
from typing import Optional
import os
from pv_console.config import KEY_FILE, API_FILE, CONFIG_DIR

try:
    from cryptography.fernet import Fernet
    CRYPTO_OK = True
except Exception:
    CRYPTO_OK = False

def _make_restricted(path: str):
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass

def generate_and_store_key() -> bytes:
    print("[crypto] generate_and_store_key")
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    _make_restricted(KEY_FILE)
    return key

def load_key() -> Optional[bytes]:
    print("[crypto] load_key")
    if not CRYPTO_OK:
        print("[crypto] cryptography not available")
        return None
    if not os.path.exists(KEY_FILE):
        print("[crypto] no key file")
        return None
    try:
        with open(KEY_FILE, "rb") as f:
            return f.read()
    except Exception as e:
        print("[crypto] load_key exception:", e)
        return None

def store_api_key_encrypted(raw_api: str) -> bool:
    print("[crypto] store_api_key_encrypted")
    if not CRYPTO_OK:
        print("[crypto] cryptography not available -> cannot encrypt")
        return False
    try:
        key = load_key() or generate_and_store_key()
        f = Fernet(key)
        token = f.encrypt(raw_api.encode("utf-8"))
        with open(API_FILE, "wb") as fa:
            fa.write(token)
        _make_restricted(API_FILE)
        return True
    except Exception as e:
        print("[crypto] store_api_key_encrypted failed:", e)
        return False

def load_api_key_encrypted() -> Optional[str]:
    print("[crypto] load_api_key_encrypted")
    if not CRYPTO_OK:
        print("[crypto] cryptography not available -> cannot decrypt")
        return None
    try:
        if not os.path.exists(API_FILE):
            print("[crypto] api file missing")
            return None
        key = load_key()
        if key is None:
            print("[crypto] key missing while trying to decrypt")
            return None
        f = Fernet(key)
        with open(API_FILE, "rb") as fa:
            token = fa.read()
        return f.decrypt(token).decode("utf-8")
    except Exception as e:
        print("[crypto] load_api_key_encrypted error:", e)
        return None
