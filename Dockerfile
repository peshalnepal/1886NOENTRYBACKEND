FROM python:3.12-slim

ARG APP_MODULE=main:app

# ----------
# Runtime env
# ----------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_MODULE=${APP_MODULE} \
    PORT=8080

WORKDIR /app

# ----------
# OS deps
# - curl: healthcheck / debugging
# - libgl1, libglib2.0-0: common runtime deps for OpenCV wheels even in headless contexts
# - libxcb1 + X libs: prevents "libxcb.so.1" crash if opencv-python sneaks in
# - gcc/build-essential: keep only if you truly need builds (safe to keep for now)
# ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential curl \
    libgl1 libglib2.0-0 \
    libxcb1 libx11-6 libxext6 libxrender1 libsm6 \
    && rm -rf /var/lib/apt/lists/*

# 1) Install Python deps first (better layer caching)
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip uninstall -y opencv-python || true \
    && pip install --no-cache-dir --no-deps --force-reinstall opencv-python-headless==4.10.0.84

# 2) Copy app code
COPY . .

ENV PYTHONPATH=/app

EXPOSE 8080

# Healthcheck: hit "/" on the internal port
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/' % os.getenv('PORT','8080')).read()" || exit 1

# App entrypoint
CMD ["sh", "-lc", "uvicorn ${APP_MODULE} --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips='*'"]
