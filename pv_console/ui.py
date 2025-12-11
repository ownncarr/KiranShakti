# pv_console/ui.py
print("[ui.py] imported")

import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import json, io, zipfile
from PIL import Image

# internal imports
from pv_console.crypto import load_api_key_encrypted, store_api_key_encrypted
from pv_console.pipeline import run_batch_process, process_sample
from pv_console.fetch_image import _placeholder_overlay

# CSS ---------------------------------------------------
CSS = """<style>
:root {
  --accent-1: #acc3a6;
  --accent-2: #d1cdb0;
  --accent-3: #F5D6BA;
  --accent-4: #F49D6E;
  --bg: #0f1115;
  --surface: #161a1e;
  --text: #e6edf3;
  --muted: #9BA3AE;
}
body, .stApp {
  background: radial-gradient(1200px 600px at 10% 10%, var(--accent-1)14%, transparent 28%),
              radial-gradient(1100px 550px at 90% 90%, var(--accent-3)10%, transparent 26%),
              linear-gradient(180deg, #0f1115 0%, #141518 100%) !important;
  color: var(--text) !important;
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid rgba(255,255,255,0.07);
}
.option-menu .nav-link {
  color: var(--muted) !important;
  background: transparent !important;
  border-radius: 6px;
}
.option-menu .nav-link-selected {
  background: linear-gradient(90deg, var(--accent-2), var(--accent-4)) !important;
  color: #0f1115 !important;
  font-weight: 700 !important;
}
.section-header {
  font-size: 1.06rem;
  font-weight: 650;
  margin-bottom: 10px;
  color: var(--text);
}
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
  border: 1px solid rgba(255,255,255,0.05);
  padding: 14px;
  border-radius: 10px;
}
.small-muted {
  color: var(--muted);
  font-size: 0.93rem;
}
.btn-primary > button {
  background: linear-gradient(90deg, var(--accent-1), var(--accent-3)) !important;
  color: #0b0b0b !important;
  border-radius: 10px !important;
  padding: 10px 16px !important;
  font-weight: 700 !important;
  border: none !important;
}
.btn-ghost > button {
  background: transparent !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
  padding: 8px 14px !important;
}
.workflow-pills {
  display: flex;
  gap: 10px;
  align-items: center;
  margin-bottom: 14px;
}
.workflow-pill {
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
}
.workflow-pill--active {
  background: linear-gradient(90deg, var(--accent-2), var(--accent-4));
  color: #0b0b0b !important;
  border: none;
}
.workflow-pill--small {
  padding: 6px 10px;
  font-size: 0.88rem;
  font-weight: 700;
}
.workflow-label {
  margin-right: 12px;
  font-weight: 700;
  color: var(--text);
  font-size: 0.98rem;
}
</style>"""


# Step pills renderer ----------------------------------
def render_step_pills(current):
    steps = [
        ("upload", "Upload"),
        ("process", "Process"),
        ("verify", "Verify"),
        ("download", "Download"),
    ]
    html = "<div class='workflow-label'>Workflow</div><div class='workflow-pills'>"
    for key, label in steps:
        cls = "workflow-pill"
        if key == current:
            cls += " workflow-pill--active"
        if key in ("verify", "download"):
            cls += " workflow-pill--small"
        html += f"<div class='{cls}'>{label}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# MAIN UI ------------------------------------------------
def main():
    print("[ui.main] start")
    st.markdown(CSS, unsafe_allow_html=True)

    # Sidebar menu
    with st.sidebar:
        selected = option_menu(
            "Main",
            ["Home", "Settings", "How to use"],
            icons=["house", "gear", "question-circle"],
            menu_icon="cast",
            default_index=0,
        )

    # Session state defaults
    ss = st.session_state
    ss.setdefault("df", None)
    ss.setdefault("in_workflow", False)
    ss.setdefault("workflow_step", "upload")
    ss.setdefault("results", [])
    ss.setdefault("review_queue", [])
    ss.setdefault("overlays", {})
    ss.setdefault("api_key", None)

    # Load encrypted API key
    try:
      saved = load_api_key_encrypted()
      if saved:
        ss.api_key = saved
        # show a clear success message in Settings/Home snapshot
        st.sidebar.success("Encrypted API key found on device.")
        # also show a non-modal status on the Settings page later (we add extra signage below)
    except Exception as e:
        print("[ui] load_api_key_encrypted failed:", e)

    # ============ HOME PAGE =============
    if selected == "Home":
        st.markdown("<div class='section-header'>Rooftop PV Detection</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-muted'>A clean, audit-ready workflow.</div>", unsafe_allow_html=True)

        # Start workflow
        if st.button("Start new analysis"):
            ss.in_workflow = True
            ss.workflow_step = "upload"
            st.rerun()

        if ss.in_workflow:
            render_step_pills(ss.workflow_step)

            # ---- UPLOAD ----
            if ss.workflow_step == "upload":
                st.subheader("Step 1 — Upload dataset")
                up = st.file_uploader("Upload .xlsx with sample_id, lat, lon", type=["xlsx"])
                if up:
                    try:
                        df = pd.read_excel(up)
                        if {"sample_id", "lat", "lon"}.issubset(df.columns):
                            df["sample_id"] = df["sample_id"].astype(str)
                            ss.df = df
                            st.success(f"{len(df)} rows loaded.")
                        else:
                            st.error("File must contain columns: sample_id, lat, lon")
                    except Exception as e:
                        st.error(str(e))

                if st.button("Next →"):
                    if ss.df is None:
                        st.error("Upload dataset first.")
                    else:
                        ss.workflow_step = "process"
                        st.rerun()

            # ---- PROCESS ----
            elif ss.workflow_step == "process":
                st.subheader("Step 2 — Process (inference & QC)")
                mode = st.selectbox("Processing mode", ["Automatic", "Smart Review", "Review All"], index=1)

                if st.button("Run processing"):
                    if not ss.api_key:
                        st.error("No API key found. Add one in Settings.")
                    else:
                        results, reviews, overlays = run_batch_process(ss.df, mode, ss.api_key, ss)
                        ss.results = results
                        ss.review_queue = reviews
                        ss.overlays = overlays
                        ss.workflow_step = "verify" if reviews else "download"
                        st.rerun()

                if st.button("← Back"):
                    ss.workflow_step = "upload"
                    st.rerun()

            # ---- VERIFY ----
            elif ss.workflow_step == "verify":
                st.subheader("Step 3 — Verify flagged samples")

                if len(ss.review_queue) == 0:
                    st.info("No flagged items.")
                    if st.button("Next →"):
                        ss.workflow_step = "download"
                        st.rerun()
                else:
                    idx = st.number_input("Queue index", 0, len(ss.review_queue) - 1)
                    item = ss.review_queue[int(idx)]
                    rec = item["record"]
                    overlay_bytes = item["overlay"]

                    st.write(f"Sample: {rec['sample_id']}")
                    if overlay_bytes:
    # display at native 640x640 so the image is not stretched by Streamlit
                           st.image(Image.open(io.BytesIO(overlay_bytes)), width=640)
                    else:
                           st.image(_placeholder_overlay(size=(640,640)), width=640)


                    c1, c2, c3 = st.columns(3)
                    if c1.button("Solar present"):
                        rec["has_solar"] = True
                        ss.results.append(rec)
                        ss.overlays[rec["sample_id"]] = overlay_bytes
                        ss.review_queue.pop(int(idx))
                        st.rerun()

                    if c2.button("Solar absent"):
                        rec["has_solar"] = False
                        ss.results.append(rec)
                        ss.overlays[rec["sample_id"]] = overlay_bytes
                        ss.review_queue.pop(int(idx))
                        st.rerun()

                    if c3.button("Mark unusable"):
                        rec["has_solar"] = False
                        rec["qc_status"] = "NOT_VERIFIABLE"
                        ss.results.append(rec)
                        ss.overlays[rec["sample_id"]] = overlay_bytes
                        ss.review_queue.pop(int(idx))
                        st.rerun()

            # ---- DOWNLOAD ----
            elif ss.workflow_step == "download":
                st.subheader("Step 4 — Download results")

                if len(ss.results) > 0:
                    mem = io.BytesIO()
                    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
                        z.writestr("predictions.json", json.dumps(ss.results, indent=2))
                        for sid, img_bytes in ss.overlays.items():
                            if img_bytes:
                                z.writestr(f"overlays/{sid}.png", img_bytes)
                    mem.seek(0)

                    st.download_button(
                        "Download ZIP",
                        mem,
                        "pv_results.zip",
                    )

                if st.button("Close workflow"):
                    ss.in_workflow = False
                    ss.workflow_step = "upload"
                    st.rerun()

    # ============ SETTINGS PAGE =============
    elif selected == "Settings":
        st.subheader("Settings")
        st.write("API Key (stored encrypted locally)")

        current = ss.api_key or ""
        api = st.text_input("API key", value=current, type="password")

        if st.button("Save API key"):
            if api.strip() == "":
                st.error("API key cannot be empty.")
            else:
                if store_api_key_encrypted(api.strip()):
                    ss.api_key = api.strip()
                    st.success("Saved securely.")
                else:
                    st.error("Failed to save encrypted. Stored in session only.")

        if st.button("Clear stored key"):
            ss.api_key = None
            st.success("Removed stored key.")

    # ============ HOW TO USE PAGE =============
    elif selected == "How to use":
        st.subheader("How to use")
        st.markdown(
            """
1. Save your Google Maps Static API key in Settings  
2. Start a new analysis from Home  
3. Upload `.xlsx` with columns: sample_id, lat, lon  
4. Click **Run processing**  
5. Review flagged samples  
6. Download results ZIP  
"""
        )

    print("[ui.main] end")
