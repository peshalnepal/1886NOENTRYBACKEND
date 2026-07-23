#!/usr/bin/env bash
# =============================================================================
# setup_nano_py36.sh — setup for the tensort camera service on the ORIGINAL
#                      Jetson Nano (JetPack 4.x / L4T R32 / Ubuntu 18.04 /
#                      Python 3.6 / CUDA 10.2 / TensorRT 7.x–8.2).
#
# This is the LEGACY board. Use setup_orin.sh for the Orin Nano (JetPack 7).
#
# Run it ON the Nano, from the tensort/ folder:
#     cd ~/tensort
#     chmod +x setup_nano_py36.sh
#     ./setup_nano_py36.sh
#
# ┌───────────────────────────────────────────────────────────────────────────┐
# │ IMPORTANT COMPATIBILITY NOTE                                               │
# │ trt_infer.py is written for the TensorRT 10 name-based tensor API          │
# │ (io_tensor / execute_async_v3). JetPack 4.x ships TensorRT 7/8, whose      │
# │ engines use the older binding API (num_bindings / execute_async_v2).       │
# │ The VIDEO INGEST layer (multi-scheme GStreamer in channels/) runs fine on  │
# │ Python 3.6, but trt_infer.py will need the TRT-8 binding API restored      │
# │ before inference works on this board. This script sets up everything       │
# │ EXCEPT that code change and prints the warning again at the end.           │
# └───────────────────────────────────────────────────────────────────────────┘
#
# Env overrides:
#   MODEL=yolov8n        base model in models/ (default yolov8n)
#   VENV=.venv_trt       venv directory name
#   PYCUDA_VER=2020.1    pycuda version pinned for Python 3.6
#   SKIP_APT=1           skip apt installs
#   SKIP_ENGINE=1        skip the trtexec engine build
#   WORKSPACE_MB=1024    trtexec build scratch memory (MB) — Nano has little RAM
# =============================================================================
set -euo pipefail

MODEL="${MODEL:-yolov8n}"
VENV="${VENV:-.venv_trt}"
PYCUDA_VER="${PYCUDA_VER:-2020.1}"
WORKSPACE_MB="${WORKSPACE_MB:-1024}"
ONNX="models/${MODEL}.onnx"
ENGINE="models/${MODEL}.engine"

log()  { printf '\033[1;36m[nano-setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[nano-setup] WARN:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[nano-setup] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

cd "$(dirname "$0")"
[ -f main.py ] || die "run this from the tensort/ folder (main.py not found here)"

# --- 1. Confirm board / OS -----------------------------------------------------
log "Board:   $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
log "L4T:     $(head -n1 /etc/nv_tegra_release 2>/dev/null || echo unknown)"
PYVER="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
log "Python:  ${PYVER}"
case "$PYVER" in
  3.6|3.7) : ;;
  *) warn "this script targets Python 3.6 (old Nano). Found ${PYVER}. For Orin Nano use setup_orin.sh." ;;
esac

# --- 2. NVIDIA stack + build tools --------------------------------------------
if [ "${SKIP_APT:-0}" != "1" ]; then
  log "Installing NVIDIA stack (nvidia-jetpack) + build tools…"
  sudo apt-get update
  # nvidia-jetpack pulls CUDA 10.2, cuDNN, TensorRT and the GStreamer-enabled
  # system OpenCV for JetPack 4.x.
  sudo apt-get install -y nvidia-jetpack || warn "nvidia-jetpack not found via apt; on some 4.x flashes it is preinstalled."
  sudo apt-get install -y \
      python3-venv python3-dev python3-pip build-essential \
      libboost-all-dev \
      gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
else
  log "SKIP_APT=1 — skipping apt step."
fi

# --- Verify the stack ----------------------------------------------------------
log "Verifying CUDA / TensorRT / OpenCV on the SYSTEM python…"
/usr/local/cuda/bin/nvcc --version >/dev/null 2>&1 || warn "nvcc not found on PATH — CUDA may be at /usr/local/cuda-10.2."
python3 -c "import tensorrt as t; print('  TensorRT', t.__version__)" || warn "system TensorRT import failed — check JetPack flash."
python3 -c "import cv2; print('  cv2', cv2.__version__)" || warn "system OpenCV import failed."
python3 -c "import cv2,re,sys; sys.exit(0 if re.search(r'GStreamer:\s+YES', cv2.getBuildInformation()) else 1)" \
  && log "  OpenCV GStreamer: YES" \
  || warn "system OpenCV reports GStreamer:NO — hardware/scheme decode will be limited."
[ -x /usr/src/tensorrt/bin/trtexec ] || warn "trtexec not at /usr/src/tensorrt/bin/trtexec."

# --- 3. Max performance --------------------------------------------------------
log "Setting max power mode + clocks…"
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
# Python 3.6 needs an old pip; never bootstrap a newer one.
python3 -m pip install --upgrade "pip<22" "setuptools<60" "wheel<0.38"

# CUDA on PATH so pycuda compiles (JetPack 4.x -> CUDA 10.2)
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_ROOT="/usr/local/cuda"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

log "Installing pinned pip deps (requirements.txt — Python 3.6 set)…"
pip install -r requirements.txt

log "Installing pycuda==${PYCUDA_VER} (compiles against CUDA 10.2 — may take minutes)…"
pip install "pycuda==${PYCUDA_VER}" || warn "pycuda build failed — confirm CUDA 10.2 paths and nvcc, then re-run."

log "Verifying imports…"
python3 -c "import flask, sqlalchemy, aiosqlite, cv2; print('  core imports OK')" || warn "core import check failed."
python3 -c "import pycuda.driver, tensorrt; print('  cuda/trt imports OK')" || warn "pycuda/tensorrt import failed."

# --- 6. Build the TensorRT engine ON THIS device ------------------------------
# NOTE: TRT 7/8 trtexec uses --workspace=<MB>, NOT --memPoolSize (that is TRT 10).
if [ "${SKIP_ENGINE:-0}" != "1" ]; then
  if [ -f "$ONNX" ] && [ -x /usr/src/tensorrt/bin/trtexec ]; then
    log "Building TensorRT engine for THIS Nano (fp16, --workspace=${WORKSPACE_MB}): ${ENGINE}"
    /usr/src/tensorrt/bin/trtexec \
        --onnx="$ONNX" \
        --saveEngine="$ENGINE" \
        --fp16 \
        --workspace="${WORKSPACE_MB}"
    log "Engine built: ${ENGINE}"
  else
    warn "skipping engine build — missing ${ONNX} or trtexec."
  fi
else
  log "SKIP_ENGINE=1 — not rebuilding the engine."
fi

# --- 7. .env -------------------------------------------------------------------
if [ ! -f .env ] && [ -f .env.example ]; then
  log "Creating .env from .env.example — set DET_ENGINE=./${ENGINE}"
  cp .env.example .env
fi

cat <<EOF

\033[1;33m[nano-setup] Environment ready — but READ THIS:\033[0m
JetPack 4.x ships TensorRT 7/8. trt_infer.py targets the TensorRT 10 tensor API,
so inference will NOT run until trt_infer.py is adapted to the TRT-8 binding API
(num_bindings / execute_async_v2). The camera ingest layer works as-is.

Next:
  1. Adapt trt_infer.py to the TensorRT 8 binding API (see the note above).
  2. Edit .env and set:  DET_ENGINE=./${ENGINE}
  3. Run:                source ${VENV}/bin/activate && python3 main.py
  4. Check:              curl http://localhost:8080/health
Cameras accept any source_url scheme: rtsp/rtsps/webrtc/whep/wheps/http/https/rtmp/rtmps/srt.
EOF
