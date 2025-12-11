# pv_console/app.py
"""
Entry point for Streamlit. We force the project root (one level above this file)
onto sys.path so package-style imports like `from pv_console.ui import main`
work even when Streamlit runs this file as a script.
"""
import os
import sys

# --- Add project root to sys.path very early ---
THIS_FILE = os.path.abspath(__file__)
THIS_DIR = os.path.dirname(THIS_FILE)
PROJECT_ROOT = os.path.dirname(THIS_DIR)  # one level above pv_console
if PROJECT_ROOT not in sys.path:
    # insert at front so it takes precedence over the script dir
    sys.path.insert(0, PROJECT_ROOT)

print("[app.py] start")
print(f"[app.py] __file__ = {THIS_FILE}")
print(f"[app.py] THIS_DIR = {THIS_DIR}")
print(f"[app.py] PROJECT_ROOT = {PROJECT_ROOT}")
print(f"[app.py] sys.path[0] = {sys.path[0]}")

# Now safe to import streamlit and the package modules
import streamlit as st

# page config
st.set_page_config(page_title="PV Detection Console", layout="wide")

print("[app.py] importing ui.main()")
from pv_console.ui import main as ui_main

if __name__ == "__main__":
    print("[app.py] launching UI")
    ui_main()
