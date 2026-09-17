FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# -- system libs for opencv-python runtime (headless, no curl/wget needed) --
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -- CPU-only torch, pinned to uv.lock parity (2.13.0 / 0.28.0, +cpu build) --
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.13.0" \
    "torchvision==0.28.0"

# -- remaining runtime deps, pinned to uv.lock versions (PyPI) --
RUN pip install --no-cache-dir \
    "fastapi==0.141.1" \
    "uvicorn==0.52.4" \
    "python-multipart==0.0.32" \
    "pillow==12.3.0" \
    "numpy==2.5.2" \
    "opencv-python==5.0.0.93" \
    "scipy==1.18.0" \
    "ultralytics==8.4.118" \
    "pyyaml==6.0.3" \
    "bcrypt==5.0.0" \
    "pyjwt==2.14.0"

# -- app code + deployment-owned inference scripts --
# Models and config.yaml are NOT baked in: they are mounted at runtime
# (see docker-compose.yml), so swapping a model or editing config.yaml only
# needs a container restart, never a rebuild.
COPY backend/ ./backend/
COPY scripts/ ./scripts/

# -- Vite-built frontend (from stage 1; absent locally until `npm run build`) --
COPY --from=frontend-build /build/frontend/dist ./frontend/dist

# -- writable mountpoint for sessions (host ./data/sessions overlays this) --
RUN mkdir -p /app/data/sessions

EXPOSE 8000

# stdlib only: / returns the frontend when the app is up
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
