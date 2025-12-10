# streamlit_app.py
import os
import io
import json
import zipfile
from typing import Any
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu

# Try importing the real backend; fall back to shim that matches signature
try:
    import pipeline.backend as backend  # type: ignore
except Exception:
    import backend_stub as backend  # project-root shim (must accept api_key param)

# Encryption for API key storage
try:
    from cryptography.fernet import Fernet, InvalidToken
    CRYPTO_OK = True
except Exception:
    CRYPTO_OK = False


# --------------------------------------------------
# LOCAL ENCRYPTED STORAGE CONFIG
# --------------------------------------------------
APP_NAME = "pv_console"
CONFIG_DIR = os.path.join(os.path.expanduser("~"), f".{APP_NAME}")
KEY_FILE = os.path.join(CONFIG_DIR, "key.bin")
API_FILE = os.path.join(CONFIG_DIR, "api.enc")

os.makedirs(CONFIG_DIR, exist_ok=True)


def _make_restricted(path: str):
    try:
        os.chmod(path, 0o600)
    except:
        pass


def generate_and_store_key() -> bytes:
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    _make_restricted(KEY_FILE)
    return key


def load_key() -> bytes | None:
    if not CRYPTO_OK:
        return None
    if not os.path.exists(KEY_FILE):
        return None
    try:
        with open(KEY_FILE, "rb") as f:
            return f.read()
    except:
        return None


def store_api_key_encrypted(raw_api: str) -> bool:
    """
    Store API key encrypted on device.
    Returns True on success.
    """
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


def load_api_key_encrypted() -> str | None:
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


# --------------------------------------------------
# DARK MODE COLOR PALETTE
# --------------------------------------------------
ACCENT_1 = "#acc3a6"
ACCENT_2 = "#d1cdb0"
ACCENT_3 = "#F5D6BA"
ACCENT_4 = "#F49D6E"
BG = "#0f1115"
SURFACE = "#161a1e"
TEXT = "#e6edf3"
MUTED = "#9BA3AE"

st.set_page_config(page_title="PV Detection Console", layout="wide")


# --------------------------------------------------
# DARK MODE CSS (FULL)
# --------------------------------------------------
st.markdown(f"""
<style>
:root {{
  --accent-1: {ACCENT_1};
  --accent-2: {ACCENT_2};
  --accent-3: {ACCENT_3};
  --accent-4: {ACCENT_4};
  --bg: #0f1115;
  --surface: #161a1e;
  --text: #e6edf3;
  --muted: #9BA3AE;
}}

body, .stApp {{
  background: radial-gradient(1200px 600px at 10% 10%, var(--accent-1)14%, transparent 28%),
              radial-gradient(1100px 550px at 90% 90%, var(--accent-3)10%, transparent 26%),
              linear-gradient(180deg, #0f1115 0%, #141518 100%) !important;
  color: var(--text) !important;
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}}

/* ---------------- SIDEBAR ---------------- */
[data-testid="stSidebar"] {{
  background: var(--surface) !important;
  border-right: 1px solid rgba(255,255,255,0.07);
}}

[data-testid="stSidebar"] * {{
  color: var(--text) !important;
}}

.option-menu .nav-link {{
  color: var(--muted) !important;
  background: transparent !important;
  border-radius: 6px;
}}
.option-menu .nav-link:hover {{
  background: rgba(255,255,255,0.06) !important;
}}
.option-menu .nav-link-selected {{
  background: linear-gradient(90deg, var(--accent-2), var(--accent-4)) !important;
  color: #0f1115 !important;
  font-weight: 700 !important;
}}

/* ---------------- TITLES ---------------- */
.section-header {{
  font-size: 1.06rem;
  font-weight: 650;
  margin-bottom: 10px;
  color: var(--text);
}}
/* NO underline, per request */

/* ---------------- CARDS ---------------- */
.card {{
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
  border: 1px solid rgba(255,255,255,0.05);
  padding: 14px;
  border-radius: 10px;
  transition: transform .14s ease, box-shadow .14s ease;
}}
.card:hover {{
  transform: translateY(-5px);
  box-shadow: 0 10px 28px rgba(0,0,0,0.45);
}}

.small-muted {{
  color: var(--muted);
  font-size: 0.93rem;
}}

/* ---------------- BUTTONS ---------------- */
.btn-primary > button {{
  background: linear-gradient(90deg, var(--accent-1), var(--accent-3)) !important;
  color: #0b0b0b !important;
  border-radius: 10px !important;
  padding: 10px 16px !important;
  font-weight: 700 !important;
  border: none !important;
}}
.btn-primary > button:hover {{
  transform: translateY(-2px);
  transition: 0.12s;
}}

.btn-ghost > button {{
  background: transparent !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
  padding: 8px 14px !important;
}}
.btn-ghost > button:hover {{
  background: rgba(255,255,255,0.06) !important;
}}

.btn-small > button {{
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
  color: var(--text) !important;
  padding: 6px 10px !important;
  border-radius: 8px !important;
}}

# --- Replace the existing CSS rules for step pills with this (paste inside your existing <style> block) ---
/* ---------- WORKFLOW PILL STYLES (improved) ---------- */
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
  box-shadow: 0 1px 0 rgba(255,255,255,0.02) inset;
  transition: transform 0.14s ease, box-shadow 0.14s ease, color 0.14s ease, background 0.14s ease;
  cursor: default;
}}
.workflow-pill:hover {{
  transform: translateY(-3px);
  box-shadow: 0 8px 20px rgba(0,0,0,0.45);
  color: var(--text);
  background: rgba(255,255,255,0.03);
}}
.workflow-pill--active {{
  background: linear-gradient(90deg, var(--accent-2), var(--accent-4));
  color: #0b0b0b !important;
  border: none;
  box-shadow: 0 8px 30px rgba(0,0,0,0.55);
  transform: translateY(-2px);
}}
.workflow-pill--small {{
  padding: 6px 10px;
  font-size: 0.88rem;
  font-weight: 700;
}}

/* optional: visually separate label from pills */
.workflow-label {{
  margin-right: 12px;
  font-weight: 700;
  color: var(--text);
  font-size: 0.98rem;
  letter-spacing: 0.2px;
}}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR MENU
# --------------------------------------------------
with st.sidebar:
    selected = option_menu(
        "Main",
        ["Home", "Settings", "How to use"],
        icons=["house", "gear", "question-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
    )


# --------------------------------------------------
# STATE
# --------------------------------------------------
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


# Load encrypted API on boot if present (preferred)
if CRYPTO_OK:
    try:
        stored = load_api_key_encrypted()
        if stored:
            st.session_state.api_key = stored
    except Exception:
        # don't crash if decryption fails; keep session key as-is
        pass


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
# --- Replace the old render_step_pills function with this improved version ---
def render_step_pills(current):
    """
    Renders a compact, modern set of pills for the workflow steps.
    Paste this function in place of your previous render_step_pills.
    """
    steps = [("upload", "Upload"), ("process", "Process"), ("verify", "Verify"), ("download", "Download")]
    html_parts = []
    # optional label at left
    html_parts.append("<div class='workflow-label'>Workflow</div>")
    html_parts.append("<div class='workflow-pills'>")
    for key, title in steps:
        cls = "workflow-pill workflow-pill--active" if key == current else "workflow-pill"
        # add smaller style for later steps (example: make verify/download slightly smaller)
        if key in ("verify", "download"):
            cls += " workflow-pill--small"
        html_parts.append(f"<div class='{cls}' role='button'>{title}</div>")
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)



def _create_error_overlay(overlay_size=(640, 640)):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", overlay_size, (28, 28, 30))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    d.text((8, 8), "error", fill=(255, 255, 255), font=font)
    return img


def run_batch_process(df, mode="Smart Review"):
    """
    This function:
    - obtains the API key from session state or encrypted file
    - refuses to run if there is no API key (unless offline mode is implemented)
    - calls backend.process_sample(row, mode, api_key=api_key)
    """
    st.session_state.results = []
    st.session_state.review_queue = []
    st.session_state.overlays = {}

    total = len(df)
    prog = st.progress(0)
    status = st.empty()

    # Ensure we have a key: first check session, then encrypted file
    api_key = st.session_state.get("api_key")
    if not api_key and CRYPTO_OK:
        api_key = load_api_key_encrypted()
        if api_key:
            st.session_state.api_key = api_key

    if not api_key:
        st.error("No API key available. Go to Settings and save your Google Maps API key (encrypted on device).")
        return

    for idx, row in df.iterrows():
        sid = str(row["sample_id"])
        status.text(f"Processing {sid} ({idx+1}/{total})")
        try:
            rec, overlay, needs_review = backend.process_sample(row, mode, api_key=api_key)

            if needs_review:
                st.session_state.review_queue.append({"record": rec, "overlay": overlay, "sid": sid})
            else:
                st.session_state.results.append(rec)
                st.session_state.overlays[sid] = overlay

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
                "overlay": _create_error_overlay(overlay_size=(640, 640)),
                "sid": sid
            })

        prog.progress((idx+1) / total)

    st.session_state.workflow_step = "verify" if len(st.session_state.review_queue) else "download"
    status.text("Processing complete.")


# --------------------------------------------------
# PAGE: HOME
# --------------------------------------------------
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
            st.rerun()
    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### Session snapshot")
        st.markdown(f"<div class='small-muted'>Approved: {len(st.session_state.results)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>Pending review: {len(st.session_state.review_queue)}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Workflow
    if st.session_state.in_workflow:
        st.markdown("<div class='section-header'>Workflow</div>", unsafe_allow_html=True)
        render_step_pills(st.session_state.workflow_step)
        st.write("")

        # ---------------- UPLOAD ----------------
        if st.session_state.workflow_step == "upload":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("**Step 1 — Upload dataset**")
            up = st.file_uploader("Upload .xlsx with sample_id, lat, lon", type=["xlsx"])
            if up:
                try:
                    df = pd.read_excel(up)
                    if {"sample_id", "lat", "lon"}.issubset(df.columns):
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
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            with cC:
                st.markdown("<div class='btn-ghost'>", unsafe_allow_html=True)
                if st.button("Cancel"):
                    st.session_state.in_workflow = False
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- PROCESS ----------------
        elif st.session_state.workflow_step == "process":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("**Step 2 — Process (inference & QC)**")
            mode = st.selectbox("Processing mode", ["Automatic", "Smart Review", "Review All"], index=1)

            cRun, cBack = st.columns(2)
            with cRun:
                st.markdown("<div class='btn-primary'>", unsafe_allow_html=True)
                if st.button("Run processing", key="runproc"):
                    # make sure API key exists (either in session or encrypted file)
                    api_key = st.session_state.get("api_key")
                    if not api_key and CRYPTO_OK:
                        api_key = load_api_key_encrypted()
                        if api_key:
                            st.session_state.api_key = api_key

                    if not api_key:
                        st.error("No API key found. Save your API key in Settings (encrypted on device) or paste it into the field.")
                    else:
                        run_batch_process(st.session_state.df, mode)
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            with cBack:
                st.markdown("<div class='btn-ghost'>", unsafe_allow_html=True)
                if st.button("← Back"):
                    st.session_state.workflow_step = "upload"
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- VERIFY ----------------
        elif st.session_state.workflow_step == "verify":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("**Step 3 — Verify flagged samples**")

            queue = st.session_state.review_queue
            if len(queue) == 0:
                st.info("No flagged items.")
                if st.button("Next →"):
                    st.session_state.workflow_step = "download"
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                idx = st.number_input("Queue index", min_value=0, max_value=len(queue)-1, step=1)
                row = queue[idx]
                rec = row["record"]; overlay = row["overlay"]; sid = row["sid"]

                st.subheader(f"Sample {sid}")
                st.image(overlay, width="stretch")  
                st.write(f"Confidence: {rec['confidence']}")
                st.write(f"Suggested QC: {rec['qc_status']}")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("<div class='btn-primary'>", unsafe_allow_html=True)
                    if st.button("Solar present"):
                        rec["has_solar"] = True; rec["qc_status"] = "VERIFIABLE"
                        st.session_state.results.append(rec)
                        st.session_state.overlays[sid] = overlay
                        queue.pop(idx)
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

                with c2:
                    st.markdown("<div class='btn-ghost'>", unsafe_allow_html=True)
                    if st.button("Solar absent"):
                        rec["has_solar"] = False; rec["qc_status"] = "VERIFIABLE"
                        st.session_state.results.append(rec)
                        st.session_state.overlays[sid] = overlay
                        queue.pop(idx)
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

                with c3:
                    st.markdown("<div class='btn-ghost'>", unsafe_allow_html=True)
                    if st.button("Mark unusable"):
                        rec["has_solar"] = False; rec["qc_status"] = "NOT_VERIFIABLE"
                        st.session_state.results.append(rec)
                        st.session_state.overlays[sid] = overlay
                        queue.pop(idx)
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

                # navigation
                cn, cb = st.columns(2)
                with cn:
                    st.markdown("<div class='btn-primary'>", unsafe_allow_html=True)
                    if st.button("Next →", key="v_next"):
                        if len(queue) == 0:
                            st.session_state.workflow_step = "download"
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                with cb:
                    st.markdown("<div class='btn-ghost'>", unsafe_allow_html=True)
                    if st.button("← Back", key="v_back"):
                        st.session_state.workflow_step = "process"
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- DOWNLOAD ----------------
        elif st.session_state.workflow_step == "download":
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("**Step 4 — Download results**")
            total = len(st.session_state.results)
            st.write(f"Approved records: {total}")

            if total > 0:
                mem = io.BytesIO()
                with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
                    z.writestr("predictions.json", json.dumps(st.session_state.results, indent=2))
                    for sid, img in st.session_state.overlays.items():
                        buf = io.BytesIO(); img.save(buf, "PNG")
                        z.writestr(f"overlays/{sid}.png", buf.getvalue())
                mem.seek(0)
                st.download_button("Download ZIP", data=mem, file_name="pv_results.zip")

            cCLOSE, cNEW = st.columns(2)
            with cCLOSE:
                st.markdown("<div class='btn-ghost'>", unsafe_allow_html=True)
                if st.button("Close workflow"):
                    st.session_state.in_workflow = False
                    st.session_state.workflow_step = "upload"
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            with cNEW:
                st.markdown("<div class='btn-primary'>", unsafe_allow_html=True)
                if st.button("Restart workflow"):
                    st.session_state.df = None
                    st.session_state.results = []
                    st.session_state.review_queue = []
                    st.session_state.overlays = {}
                    st.session_state.workflow_step = "upload"
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------
# SETTINGS: ONLY API KEY
# --------------------------------------------------
elif selected == "Settings":
    st.markdown("<div class='section-header'>Settings</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**API Key**")
    st.markdown("<div class='small-muted'>Stored securely on device using encryption.</div>", unsafe_allow_html=True)

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

    # Show current storage status
    st.markdown("<div style='margin-top:8px'/>", unsafe_allow_html=True)
    if CRYPTO_OK:
        if os.path.exists(API_FILE):
            st.success("Encrypted API key found on device.")
        else:
            st.info("No encrypted API key stored on device. Use 'Save API key' to store it securely.")
    else:
        st.info("No encrypted storage available. Key is stored in browser session only.")

    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------
# HOW TO USE
# --------------------------------------------------
elif selected == "How to use":
    st.markdown("<div class='section-header'>How to use</div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
1. Go to **Settings** and paste your Google Static Maps API key; click **Save API key** to store it encrypted on this device.  
2. Go to **Home → Start new analysis**  
3. Upload an Excel: `sample_id`, `lat`, `lon`  
4. Run processing (inference & QC) — the UI will use the encrypted key automatically.  
5. Verify flagged samples  
6. Download ZIP with predictions.json + overlays  
""")
    st.markdown("</div>", unsafe_allow_html=True)
