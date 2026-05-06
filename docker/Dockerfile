# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.10-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# ── App stage ────────────────────────────────────────────────────────────────
FROM base AS app

COPY . .

# Default port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default: run API server
CMD ["python", "-m", "src.api", "--weights", "weights/best.onnx", "--host", "0.0.0.0", "--port", "8000", "--imgsz", "640"]

# ── Training stage ───────────────────────────────────────────────────────────
FROM base AS train

COPY . .

# Default: run training
CMD ["python", "-m", "src.train", "--cfg", "configs/train_cfg.yaml"]
