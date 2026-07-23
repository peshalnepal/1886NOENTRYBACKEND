#!/usr/bin/env bash
# =============================================================================
# setup_orin.sh — one-shot setup for the tensort camera service on a
#                 Jetson Orin Nano (JetPack 7.x / L4T R39 / Ubuntu 24.04 /
#                 Python 3.12 / TensorRT 10).
#
# Mirrors installation_doc.md. Run it ON the Jetson, from the tensort/ folder:
#
#     cd ~/tensort
#     chmod +x setup_orin.sh
#     ./setup_orin.sh
#
# Idempotent: safe to re-run. It will NOT pip-install tensorrt / opencv / numpy
# (those come from JetPack and must stay the system copies).
#
# Env overrides:
#   MODEL=yolo26n        base model name in models/ (default yolo26n)
#   VENV=.venv_trt       venv directory name
#   SKIP_APT=1           skip the apt/nvidia-jetpack install (already done)
#   SKIP_ENGINE=1        skip the trtexec engine build
#   WORKSPACE_MB=2048    trtexec build scratch memory
# =============================================================================
set -euo pipefail

MODEL="${MODEL:-yolo26n}"
VENV="${VENV:-.venv_trt}"
WORKSPACE_MB="${WORKSPACE_MB:-2048}"
ONNX="models/${MODEL}.onnx"
ENGINE="models/${MODEL}.engine"

log()  { printf '\033[1;36m[orin-setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[orin-setup] WARN:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[orin-setup] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

cd "$(dirname "$0")"
[ -f main.py ] || die "run this from the tensort/ folder (main.py not found here)"

# --- 1. Confirm board / OS -----------------------------------------------------
log "Board:   $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
log "L4T:     $(head -n1 /etc/nv_tegra_release 2>/dev/null || echo unknown)"
PYVER="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
log "Python:  ${PYVER}"
case "$PYVER" in
  3.1[0-9]) : ;;  # 3.10+ ok
  *) warn "expected Python 3.12 on JetPack 7; found ${PYVER}. For Python 3.6 / old Nano use setup_nano_py36.sh." ;;
esac

# --- 2. NVIDIA stack + build tools --------------------------------------------
if [ "${SKIP_APT:-0}" != "1" ]; then
  log "Installing NVIDIA stack (nvidia-jetpack) + build tools — this is the big download…"
  sudo apt-get update
  sudo apt-get install -y nvidia-jetpack
  sudo apt-get install -y \
      python3-venv python3-dev python3-pip build-essential \
      gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
else
  log "SKIP_APT=1 — skipping nvidia-jetpack / apt step."
fi

# --- Verify the stack ----------------------------------------------------------
log "Verifying CUDA / TensorRT / OpenCV on the SYSTEM python…"
/usr/local/cuda/bin/nvcc --version >/dev/null 2>&1 || die "nvcc missing — nvidia-jetpack did not install. Re-run without SKIP_APT."
python3 -c "import tensorrt as t; print('  TensorRT', t.__version__)" || die "system TensorRT import failed."
python3 -c "import cv2; print('  cv2', cv2.__version__)" || die "system OpenCV import failed."
python3 -c "import cv2,re,sys; sys.exit(0 if re.search(r'GStreamer:\s+YES', cv2.getBuildInformation()) else 1)" \
  && log "  OpenCV GStreamer: YES" \
  || die "system OpenCV has GStreamer:NO — a pip opencv wheel is shadowing it. Remove it."
[ -x /usr/src/tensorrt/bin/trtexec ] || die "trtexec missing at /usr/src/tensorrt/bin/trtexec."

# --- 3. Max performance --------------------------------------------------------
log "Setting max power mode + clocks (resets on reboot; systemd unit re-applies)…"
sudo nvpmodel -m 0 || warn "nvpmodel failed (non-fatal)"
sudo jetson_clocks  || warn "jetson_clocks failed (non-fatal)"

# --- 5. Python venv + pip deps -------------------------------------------------
if [ ! -d "$VENV" ]; then
  log "Creating venv ${VENV} with --system-site-packages (sees JetPack TRT/OpenCV)…"
  python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
log "venv python: $(python3 --version)"
python3 -m pip install --upgrade pip setuptools wheel

# CUDA on PATH so pycuda compiles
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_ROOT="/usr/local/cuda"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

log "Installing pinned pip deps (requirements-jetson.txt; pycuda compiles — may take minutes)…"
pip install -r requirements-jetson.txt

log "Verifying end-to-end imports…"
python3 -c "import flask, sqlalchemy, aiosqlite, pycuda.driver, tensorrt, cv2; print('  All core imports OK')"

# --- 6. Build the TensorRT engine ON THIS device ------------------------------
if [ "${SKIP_ENGINE:-0}" != "1" ]; then
  [ -f "$ONNX" ] || die "missing ${ONNX} — cannot build engine. Set MODEL=… or restore the onnx."
  log "Building TensorRT engine for THIS Jetson (fp16): ${ENGINE}"
  /usr/src/tensorrt/bin/trtexec \
      --onnx="$ONNX" \
      --saveEngine="$ENGINE" \
      --fp16 \
      --memPoolSize="workspace:${WORKSPACE_MB}"
  log "Engine built: ${ENGINE}"
else
  log "SKIP_ENGINE=1 — not rebuilding the engine."
fi

# --- 7. .env -------------------------------------------------------------------
if [ ! -f .env ] && [ -f .env.example ]; then
  log "Creating .env from .env.example — set DET_ENGINE=./${ENGINE}"
  cp .env.example .env
fi

cat <<EOF

\033[1;32m[orin-setup] Done.\033[0m
Next:
  1. Edit .env and set:  DET_ENGINE=./${ENGINE}
  2. Run:                source ${VENV}/bin/activate && python3 main.py
  3. Check:              curl http://localhost:8080/health
Cameras accept any source_url scheme: rtsp/rtsps/webrtc/whep/wheps/http/https/rtmp/rtmps/srt.
EOF
