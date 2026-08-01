#!/usr/bin/env bash
# =============================================================================
# setup_nano.sh — full deployment of the tensort camera detection service on the
#                 ORIGINAL Jetson Nano (JetPack 4.6.x / L4T R32 / Ubuntu 18.04 /
#                 Python 3.6 / CUDA 10.2 / TensorRT 8.2 / 4 GB shared RAM).
#
#   THIS IS THE SCRIPT FOR THE ORIGINAL NANO, and it must be run against the
#   'main' git branch, whose trt_infer.py uses the TensorRT 8 binding API.
#   The 'orin-nano' branch requires TensorRT 10 and will NOT run here.
#
#   For a Jetson ORIN Nano use setup_orin.sh (on the orin-nano branch).
#
# Run it ON the Nano:
#
#     cd ~/tensort/deployment
#     chmod +x setup_nano.sh
#     ./setup_nano.sh
#
# Idempotent: safe to re-run after an interruption.
#
# Env overrides:
#   MODEL=yolov8n         base model name in models/
#   VENV=.venv_trt        venv directory name
#   PYCUDA_VER=2020.1     pycuda version pinned for Python 3.6 / CUDA 10.2
#   IMG_SZ=640            engine input resolution (must match .env IMG_SZ)
#   PORT=8080             service port
#   SERVICE_NAME=jetson-cameras
#   WORKSPACE_MB=512      trtexec scratch memory (small: the Nano has 4 GB total)
#   SKIP_APT=1            skip the apt install
#   SKIP_ENGINE=1         skip the TensorRT engine build
#   SKIP_SERVICE=1        skip systemd install + health check
# =============================================================================
set -euo pipefail

LOG_PREFIX="nano-setup"

HERE="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "${HERE}/.." && pwd)"      # the tensort/ folder
# shellcheck source=common.sh
. "${HERE}/common.sh"

cd "$APP_DIR"
[ -f main.py ] || die "main.py not found in ${APP_DIR} — is the deployment/ folder inside tensort/?"

MODEL="${MODEL:-yolov8n}"
VENV="${VENV:-.venv_trt}"
PYCUDA_VER="${PYCUDA_VER:-2020.1}"
IMG_SZ="${IMG_SZ:-640}"
PORT="${PORT:-8080}"
SERVICE_NAME="${SERVICE_NAME:-jetson-cameras}"
# Keep the workspace small: trtexec competes with everything else for the same
# 4 GB, and a large workspace is the usual cause of an OOM-killed engine build.
WORKSPACE_MB="${WORKSPACE_MB:-512}"
ONNX="models/${MODEL}.onnx"
ENGINE="models/${MODEL}.engine"
RUN_USER="${SUDO_USER:-$(id -un)}"

# --- 1. Preflight --------------------------------------------------------------
log "=== Step 1/8: Preflight checks ==="
# JetPack 4.x is usually preinstalled on the SD image, so the disk requirement
# is lower than the Orin's, but the engine build still needs room.
preflight 4000 3500 "$APP_DIR"

BOARD="$(detect_board)"
case "$BOARD" in
  *Orin*)  die "This is an ORIN board ('${BOARD}'). Use setup_orin.sh on the \
orin-nano branch instead — this script targets the original Nano." ;;
  *Nano*)  ok "Original Jetson Nano detected." ;;
  *)       warn_track "Board reports '${BOARD}' — expected a Jetson Nano." ;;
esac

JP="$(detect_jetpack_major)"
[ "$JP" = "4" ] && ok "JetPack 4.x (L4T R32) detected." \
                || warn_track "Expected JetPack 4.x (L4T R32); detected major '${JP}'."

PYVER="$(py_version)"
case "$PYVER" in
  3.6|3.7) ok "Python ${PYVER}" ;;
  *) warn_track "This script targets Python 3.6; found ${PYVER}. \
On an Orin Nano use setup_orin.sh." ;;
esac

# The 4 GB Nano cannot build an engine without swap. Make this loud.
MEM="$(total_mem_mb)"
if [ "$MEM" -lt 5000 ]; then
  log "4 GB-class board: the service is memory-bound. Plan camera count accordingly \
(see deployment/README.md for the memory budget)."
fi

# --- 2. NVIDIA stack + build tools ---------------------------------------------
log "=== Step 2/8: NVIDIA stack and build tools ==="
if [ "${SKIP_APT:-0}" != "1" ]; then
  log "Installing build tools and GStreamer plugins…"
  sudo apt-get update
  # On most JetPack 4.x SD images CUDA/TensorRT/OpenCV are ALREADY installed;
  # the nvidia-jetpack meta-package may not be available from apt. Non-fatal.
  sudo apt-get install -y nvidia-jetpack \
    || warn_track "nvidia-jetpack not available via apt — normal on JetPack 4.x SD images \
where it is preinstalled. Continuing."
  sudo apt-get install -y \
      python3-venv python3-dev python3-pip build-essential \
      libboost-all-dev \
      curl \
      gstreamer1.0-tools \
      gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
else
  log "SKIP_APT=1 — skipping apt step."
fi

# --- 3. Verify the system stack ------------------------------------------------
log "=== Step 3/8: Verifying CUDA / TensorRT / OpenCV ==="
if ! /usr/local/cuda/bin/nvcc --version >/dev/null 2>&1; then
  warn_track "nvcc not on the default path — CUDA 10.2 may be at /usr/local/cuda-10.2."
else
  ok "CUDA toolkit present"
fi

python3 -c "import tensorrt as t; print('  TensorRT', t.__version__)" \
  || die "system TensorRT import failed — check the JetPack 4.x installation."

TRT_MAJOR="$(python3 -c 'import tensorrt as t;print(t.__version__.split(".")[0])' 2>/dev/null || echo 0)"
if [ "$TRT_MAJOR" -ge 10 ]; then
  die "TensorRT ${TRT_MAJOR}.x found — that is an Orin-class stack. Use setup_orin.sh \
on the orin-nano branch."
fi
ok "TensorRT ${TRT_MAJOR}.x (matches the 'main' branch binding API)"

# Guard against running the wrong branch: TRT10-only calls in trt_infer.py mean
# the orin-nano branch is checked out, which cannot work on this board.
if grep -q "execute_async_v3\|num_io_tensors" trt_infer.py 2>/dev/null; then
  die "trt_infer.py uses the TensorRT 10 API (execute_async_v3 / num_io_tensors) but \
this board has TensorRT ${TRT_MAJOR}.x. You are on the wrong branch. Run: \
git checkout main"
fi
ok "trt_infer.py uses the TensorRT 8 binding API (correct branch)"

GST_OUT="$(check_opencv_gstreamer || true)"
case "$GST_OUT" in
  YES*) ok "OpenCV with GStreamer: ${GST_OUT}" ;;
  NO*)  warn_track "System OpenCV reports GStreamer:NO (${GST_OUT}). Hardware decode \
will not work and camera ingest will be slow or fail. A pip 'opencv-python' wheel \
is probably shadowing the JetPack build — remove it with: \
pip uninstall opencv-python opencv-python-headless" ;;
  *)    warn_track "OpenCV import failed: ${GST_OUT}" ;;
esac

[ -x /usr/src/tensorrt/bin/trtexec ] || warn_track "trtexec not at /usr/src/tensorrt/bin/trtexec."

# --- 4. Max performance --------------------------------------------------------
log "=== Step 4/8: Enabling maximum performance mode ==="
# -m 0 is the 10 W mode on the Nano. Requires the barrel-jack supply; on
# micro-USB power the board can brown out under load.
sudo nvpmodel -m 0 || warn_track "nvpmodel failed (non-fatal)"
sudo jetson_clocks  || warn_track "jetson_clocks failed (non-fatal)"
log "Note: max performance (10 W) needs the barrel-jack PSU, not micro-USB."

# --- 5. Python venv + dependencies ---------------------------------------------
log "=== Step 5/8: Python environment ==="
if [ ! -d "$VENV" ]; then
  log "Creating venv ${VENV} with --system-site-packages (so it sees JetPack TRT/OpenCV)…"
  python3 -m venv --system-site-packages "$VENV" \
    || die "venv creation failed. On Ubuntu 18.04 run: sudo apt-get install -y python3-venv"
fi
# shellcheck disable=SC1090,SC1091
source "${VENV}/bin/activate"
log "venv python: $(python3 --version)"

# Python 3.6 needs OLD packaging tools; a modern pip drops py36 support and will
# refuse to resolve the pinned requirements.
log "Pinning pip/setuptools/wheel to Python 3.6-compatible versions…"
python3 -m pip install --upgrade "pip<22" "setuptools<60" "wheel<0.38"

export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_ROOT="/usr/local/cuda"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

log "Installing pinned dependencies (requirements.txt — Python 3.6 set)…"
pip install -r requirements.txt

log "Installing pycuda==${PYCUDA_VER} (compiles against CUDA 10.2 — can take 10+ minutes)…"
pip install "pycuda==${PYCUDA_VER}" \
  || warn_track "pycuda build failed. Confirm nvcc works and CUDA_ROOT=/usr/local/cuda, then re-run."

log "Verifying imports…"
python3 -c "import flask, sqlalchemy, aiosqlite, cv2; print('  core imports OK')" \
  || warn_track "core import check failed."
python3 -c "import pycuda.driver, tensorrt; print('  cuda/trt imports OK')" \
  || warn_track "pycuda/tensorrt import failed — inference will not start."

# requirements.txt pins opencv-python, which has NO GStreamer and shadows the
# JetPack system cv2. Detect that here rather than during a silent camera failure.
GST_OUT="$(check_opencv_gstreamer || true)"
case "$GST_OUT" in
  YES*) ok "venv OpenCV with GStreamer: ${GST_OUT}" ;;
  *)    warn_track "Inside the venv, OpenCV has no GStreamer (${GST_OUT}). A pip \
opencv-python wheel is shadowing the JetPack build. Fix with: \
pip uninstall -y opencv-python && python3 -c 'import cv2;print(cv2.__file__)'" ;;
esac

# --- 6. Build the TensorRT engine ON THIS DEVICE -------------------------------
# A TensorRT engine is device- and version-specific; it cannot be copied in.
log "=== Step 6/8: Building the TensorRT engine ==="
if [ "${SKIP_ENGINE:-0}" != "1" ]; then
  if [ -f "$ONNX" ] && [ -x /usr/src/tensorrt/bin/trtexec ]; then
    # TensorRT 7/8 uses --workspace=<MB>. (--memPoolSize is TensorRT 10 only.)
    log "Building FP16 engine (--workspace=${WORKSPACE_MB} MB): ${ENGINE}"
    log "This is memory-intensive and may take 10-20 minutes on this board."
    /usr/src/tensorrt/bin/trtexec \
        --onnx="$ONNX" \
        --saveEngine="$ENGINE" \
        --fp16 \
        --workspace="${WORKSPACE_MB}" \
      || die "Engine build failed. If the process was killed, the board ran out of \
memory — add swap (see the warning in step 1) or lower WORKSPACE_MB, then re-run."
    ok "Engine built: ${ENGINE}"
  else
    warn_track "Skipping engine build — missing ${ONNX} or trtexec."
  fi
else
  log "SKIP_ENGINE=1 — not rebuilding the engine."
  [ -f "$ENGINE" ] || warn_track "SKIP_ENGINE=1 but ${ENGINE} does not exist; the service will fail to start."
fi

# --- 7. Configuration ----------------------------------------------------------
log "=== Step 7/8: Configuration (.env) ==="
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    cp .env.example .env
    log "Created .env from .env.example"
  else
    warn_track ".env.example missing — creating a minimal .env"
    : > .env
  fi
fi

set_env_var() {
  local key="$1" val="$2"
  if grep -qE "^[#[:space:]]*${key}=" .env; then
    sed -i -E "s|^[#[:space:]]*${key}=.*|${key}=${val}|" .env
  else
    printf '%s=%s\n' "$key" "$val" >> .env
  fi
}

# Conservative defaults for a 4 GB board. The TRT8 path has no dynamic batching,
# so INFER_MAX_BATCH must be 1 — anything higher just splits into serial calls.
# This board sustains only ~25-40 frames/sec TOTAL across all cameras:
# one 128-core GPU, no batching in the TRT8 path, and CPU preprocessing that
# does not overlap GPU execution. Budget = fps x cameras, not fps per camera.
#
# 6 fps suits the recommended 2-4 cameras (12-24 img/s). If you run more
# cameras, LOWER this: 10 cameras needs <= 3. Watch infer_dropped in /health.
set_env_var DET_ENGINE "./${ENGINE}"
set_env_var IMG_SZ "${IMG_SZ}"
# TRT8 on this branch is fixed batch=1; a higher value only splits into extra
# serial GPU calls. INFER_BATCH_LINGER_MS is likewise unused here.
set_env_var INFER_MAX_BATCH 1
# Explicit 1, not 0: two CUDA contexts + two engines on a 4 GB board costs
# ~600 MB and makes the GPU time-slice, which LOWERS throughput.
set_env_var INFER_NUM_WORKERS 1
set_env_var INFER_NUM_WORKERS_MAX 1
set_env_var DEFAULT_SAMPLE_FPS 6
set_env_var MAX_SAMPLE_FPS 8
set_env_var DEFAULT_RESIZE_W 640
set_env_var DEFAULT_RESIZE_H 480
# Keep true even on this bandwidth-limited board: the cloud tracker ages tracks
# by frame arrival, so withholding empty frames makes a departing object look
# like a stalled stream and its box lingers/blinks. At 6 fps x 2-4 cameras the
# extra SSE traffic is small.
set_env_var EMIT_EMPTY_DETECTIONS true
# Must stay <= the cloud tracker's low_th so its low-confidence rescue band is
# not empty (see Backend/application/services/tracker.py).
set_env_var CONF 0.20
set_env_var PORT "${PORT}"
ok ".env configured for a 4 GB board (batch=1, 1 worker, 6 fps/cam @ 2-4 cameras)"

# Final guard: .env must point at an engine that exists, or the service starts
# and then fails on every frame with FileNotFoundError.
if [ ! -f "$ENGINE" ]; then
  warn_track "DET_ENGINE points at ${ENGINE}, which does not exist. \
Available: $(ls models/*.engine 2>/dev/null | tr '\n' ' ' || echo 'none'). \
Re-run without SKIP_ENGINE=1, or set MODEL= to a model whose .onnx is present."
fi

# --- 8. systemd service + health check -----------------------------------------
log "=== Step 8/8: Service installation and health check ==="
if [ "${SKIP_SERVICE:-0}" != "1" ]; then
  install_systemd_unit "$SERVICE_NAME" "$APP_DIR" "$VENV" "$RUN_USER"

  log "Starting ${SERVICE_NAME}…"
  sudo systemctl restart "$SERVICE_NAME"

  # The Nano is slow to load an engine; allow longer than the Orin.
  if wait_for_health "$PORT" 180; then
    ok "Deployment verified."
  else
    warn_track "Health check did not pass. Inspect: sudo journalctl -u ${SERVICE_NAME} -n 80 --no-pager"
  fi
else
  log "SKIP_SERVICE=1 — systemd install and health check skipped."
fi

# --- Done ----------------------------------------------------------------------
print_warn_summary

cat <<EOF

$(printf '%s' "${_C_OK}")Jetson Nano setup complete.${_C_OFF}

  Service:   ${SERVICE_NAME}   (starts automatically on boot)
  API:       http://$(hostname -I 2>/dev/null | awk '{print $1}'):${PORT}
  Engine:    ${ENGINE}
  Config:    ${APP_DIR}/.env

This board is 4 GB and single-GPU-core class. Start with 2-3 cameras at
2 fps each and watch 'infer_dropped' in /health before adding more.

Useful commands:
  sudo systemctl status ${SERVICE_NAME}
  sudo journalctl -u ${SERVICE_NAME} -f
  curl http://localhost:${PORT}/health
  tegrastats                      # live CPU/GPU/RAM usage

EOF
