FROM python:3.12-slim

# ----------
# Runtime env
# ----------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Minimal OS deps (keep slim)
# - gcc/build-essential are only needed for packages that don't ship wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 1) Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 2) Copy app code
COPY . .

ENV PYTHONPATH=/app

EXPOSE 8080

# Your main.py already has a "/" health endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/' % os.getenv('PORT','8080')).read()" || exit 1

# ✅ Your app entrypoint:
# uvicorn main:app --host 0.0.0.0 --port 8080
CMD ["sh", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips='*'"]
