# Dockerfile (universal: cpu or gpu depending on build args)
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS base

LABEL maintainer="you@example.com"
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    STREAMLIT_SERVER_PORT=8501 \
    TZ=UTC

WORKDIR /app

# Install system dependencies required for Pillow / shapely / scikit-image / ffmpeg etc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       wget \
       ca-certificates \
       bash \
       libgl1 \
       libglib2.0-0 \
       libsm6 \
       libxrender1 \
       libxext6 \
       libjpeg-dev \
       zlib1g-dev \
       pkg-config \
       libgeos-dev \
       ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy project files early (allows caching)
COPY . /app

# Use bash for the following multi-step install
SHELL ["/bin/bash", "-c"]

# Upgrade pip and wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Install torch only if not present in the base image.
# Try CPU wheels from the official PyTorch CPU index, fall back to PyPI.
RUN python - <<'PY'
import importlib, subprocess, sys
try:
    import torch  # type: ignore
    print("torch detected in base image; skipping torch install")
except Exception:
    print("torch not found; installing CPU torch + torchvision (best-effort)")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
            '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch', 'torchvision'
        ])
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'torch', 'torchvision'])
PY

# Install the rest of requirements but skip torch and torchvision lines (avoids double install)
RUN if [ -f requirements.txt ]; then \
      grep -vE "^\s*(torch|torchvision)\b" requirements.txt > /tmp/reqs_no_torch.txt || true; \
      python -m pip install --no-cache-dir -r /tmp/reqs_no_torch.txt; \
    fi

# Create a non-root user and fix permissions
RUN useradd --create-home --shell /bin/bash appuser || true \
 && mkdir -p /app && chown -R appuser:appuser /app

USER appuser
ENV HOME=/home/appuser

EXPOSE ${STREAMLIT_SERVER_PORT}

ENV STREAMLIT_SERVER_HEADLESS=true

ENTRYPOINT ["streamlit", "run", "pv_console/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
