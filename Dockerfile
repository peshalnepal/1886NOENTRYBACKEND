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
# - gcc/build-essential: for wheels without a prebuilt binary
# ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 1) Install Python deps first (better layer caching)
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 2) Copy app code
COPY . .

ENV PYTHONPATH=/app

EXPOSE 8080

# Healthcheck: hit "/" on the internal port
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/' % os.getenv('PORT','8080')).read()" || exit 1

# App entrypoint
CMD ["sh", "-lc", "uvicorn ${APP_MODULE} --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips='*'"]
