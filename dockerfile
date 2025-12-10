ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} as base

# metadata
LABEL maintainer="you@example.com"
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    STREAMLIT_SERVER_PORT=8501 \
    TZ=UTC

WORKDIR /app

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

COPY . /app

SHELL ["/bin/bash", "-c"]

RUN python -m pip install --upgrade pip setuptools wheel \
 && python -c "import importlib, sys\ntry:\n import torch\n print('torch already present; skipping torch install')\nexcept Exception:\n print('torch not found')\n sys.exit(0 if ('torch' in __import__('sys').argv) else 0)" || true

RUN python - <<'PY'\nimport importlib, subprocess, sys\ntry:\n    import torch\n    print('torch detected; skipping torch pip install')\nexcept Exception:\n    print('Installing CPU torch + torchvision (best-effort).')\n    try:\n        subprocess.check_call([\n            sys.executable, '-m', 'pip', 'install', '--no-cache-dir',\n            '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch', 'torchvision'\n        ])\n    except Exception:\n        # fallback to pip (may install cpu wheel from PyPI)\n        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'torch', 'torchvision'])\nPY

RUN if [ -f requirements.txt ]; then \\\n  grep -vE \"^\\s*(torch|torchvision)\\b\" requirements.txt > /tmp/reqs_no_torch.txt || true; \\\n  python -m pip install --no-cache-dir -r /tmp/reqs_no_torch.txt; \\\nfi

RUN useradd --create-home --shell /bin/bash appuser || true \
 && mkdir -p /app && chown -R appuser:appuser /app

USER appuser
ENV HOME=/home/appuser

EXPOSE ${STREAMLIT_SERVER_PORT}

ENV STREAMLIT_SERVER_HEADLESS=true

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
