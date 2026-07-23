#!/usr/bin/env bash
# =============================================================================
# setup_orin.sh — full deployment of the tensort camera detection service on a
#                 Jetson ORIN Nano (JetPack 6.x/7.x, Ubuntu 22.04/24.04,
#                 Python 3.10+, TensorRT 10).
#
#   THIS IS THE SCRIPT FOR THE ORIN NANO. For the original Jetson Nano
#   (JetPack 4.x, Python 3.6) use setup_nano.sh instead.
#
# Run it ON the Jetson:
#
#     cd ~/tensort/deployment
#     chmod +x setup_orin.sh
#     ./setup_orin.sh
#
# Idempotent: safe to re-run after an interruption.
#
# It will NOT pip-install tensorrt / opencv / numpy — those come from JetPack
# and must remain the system copies (a pip opencv wheel has no GStreamer and
# breaks hardware video decode).
#
# Env overrides:
#   MODEL=yolo26n         base model name in models/
#   VENV=.venv_trt        venv directory name
#   MAX_BATCH=10          dynamic-engine max batch = plan for this many cameras
#   OPT_BATCH=6           batch size the engine is optimised for
#   IMG_SZ=640            engine input resolution (must match .env IMG_SZ)
#   PORT=8080             service port
#   SERVICE_NAME=jetson-cameras
#   WORKSPACE_MB=2048     trtexec build scratch memory
#   SKIP_APT=1            skip the nvidia-jetpack/apt install
#   SKIP_ENGINE=1         skip the TensorRT engine build
#   SKIP_SERVICE=1        skip systemd install + health check
# =============================================================================
set -euo pipefail

LOG_PREFIX="orin-setup"

HERE="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$(cd "${HERE}/.." && pwd)"      # the tensort/ folder
# shellcheck source=common.sh
. "${HERE}/common.sh"

cd "$APP_DIR"
[ -f main.py ] || die "main.py not found in ${APP_DIR} — is the deployment/ folder inside tensort/?"

MODEL="${MODEL:-yolo26n}"
VENV="${VENV:-.venv_trt}"
MAX_BATCH="${MAX_BATCH:-10}"
OPT_BATCH="${OPT_BATCH:-6}"
IMG_SZ="${IMG_SZ:-640}"
PORT="${PORT:-8080}"
SERVICE_NAME="${SERVICE_NAME:-jetson-cameras}"
WORKSPACE_MB="${WORKSPACE_MB:-2048}"
ONNX="models/${MODEL}.onnx"
ENGINE="models/${MODEL}.engine"
RUN_USER="${SUDO_USER:-$(id -un)}"

# --- 1. Preflight --------------------------------------------------------------
log "=== Step 1/8: Preflight checks ==="
# nvidia-jetpack is a multi-GB download; require headroom for it plus the engine.
preflight 12000 6000 "$APP_DIR"

BOARD="$(detect_board)"
case "$BOARD" in
  *Orin*) ok "Orin board detected." ;;
  *)      warn_track "Board reports '${BOARD}', not an Orin. If this is the ORIGINAL \
Jetson Nano, stop and run setup_nano.sh instead." ;;
esac

PYVER="$(py_version)"
case "$PYVER" in
  3.1[0-9]) ok "Python ${PYVER}" ;;
  *) warn_track "Expected Python 3.10+ on JetPack 6/7; found ${PYVER}. \
For Python 3.6 / original Nano use setup_nano.sh." ;;
esac

# --- 2. NVIDIA stack + build tools ---------------------------------------------
log "=== Step 2/8: NVIDIA stack and build tools ==="
if [ "${SKIP_APT:-0}" != "1" ]; then
  log "Installing nvidia-jetpack + build tools (large download, may take 30+ min)…"
  sudo apt-get update
  sudo apt-get install -y nvidia-jetpack
  sudo apt-get install -y \
      python3-venv python3-dev python3-pip build-essential \
      curl \
      gstreamer1.0-tools \
      gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
else
  log "SKIP_APT=1 — skipping apt step."
fi

# --- 3. Verify the system stack ------------------------------------------------
log "=== Step 3/8: Verifying CUDA / TensorRT / OpenCV ==="
/usr/local/cuda/bin/nvcc --version >/dev/null 2>&1 \
  || die "nvcc missing — nvidia-jetpack did not install. Re-run without SKIP_APT=1."
ok "CUDA toolkit present"

python3 -c "import tensorrt as t; print('  TensorRT', t.__version__)" \
  || die "system TensorRT import failed — check the JetPack installation."

TRT_MAJOR="$(python3 -c 'import tensorrt as t;print(t.__version__.split(".")[0])' 2>/dev/null || echo 0)"
if [ "$TRT_MAJOR" -lt 10 ]; then
  die "TensorRT ${TRT_MAJOR}.x found, but this branch's trt_infer.py requires TensorRT 10. \
On a JetPack 4.x board use the 'main' branch + setup_nano.sh."
fi
ok "TensorRT ${TRT_MAJOR}.x (matches this code path)"

GST_OUT="$(check_opencv_gstreamer || true)"
case "$GST_OUT" in
  YES*) ok "OpenCV with GStreamer: ${GST_OUT}" ;;
  NO*)  die "System OpenCV reports GStreamer:NO — a pip 'opencv-python' wheel is \
shadowing the JetPack build. Remove it (pip uninstall opencv-python \
opencv-python-headless) and re-run. Hardware decode cannot work otherwise." ;;
  *)    die "OpenCV import failed: ${GST_OUT}" ;;
esac

[ -x /usr/src/tensorrt/bin/trtexec ] || die "trtexec missing at /usr/src/tensorrt/bin/trtexec."
ok "trtexec present"

# --- 4. Max performance --------------------------------------------------------
log "=== Step 4/8: Enabling maximum performance mode ==="
# Resets on reboot; the systemd unit re-applies both on every start.
sudo nvpmodel -m 0 || warn_track "nvpmodel failed (non-fatal)"
sudo jetson_clocks  || warn_track "jetson_clocks failed (non-fatal)"

# --- 5. Python venv + dependencies ---------------------------------------------
log "=== Step 5/8: Python environment ==="
if [ ! -d "$VENV" ]; then
  log "Creating venv ${VENV} with --system-site-packages (so it sees JetPack TRT/OpenCV)…"
  python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1090,SC1091
source "${VENV}/bin/activate"
log "venv python: $(python3 --version)"
python3 -m pip install --upgrade pip setuptools wheel

# CUDA on PATH so pycuda can compile against it.
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_ROOT="/usr/local/cuda"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

log "Installing pinned dependencies (pycuda compiles from source — may take minutes)…"
pip install -r requirements-jetson.txt

log "Verifying end-to-end imports inside the venv…"
python3 -c "import flask, sqlalchemy, aiosqlite, pycuda.driver, tensorrt, cv2; \
print('  All core imports OK')" || die "dependency verification failed."

# The venv must still see the GStreamer-enabled system OpenCV.
GST_OUT="$(check_opencv_gstreamer || true)"
case "$GST_OUT" in
  YES*) ok "venv OpenCV with GStreamer: ${GST_OUT}" ;;
  *)    die "Inside the venv, OpenCV lost GStreamer (${GST_OUT}). A pip opencv wheel \
was pulled in. Run: pip uninstall opencv-python opencv-python-headless" ;;
esac

# --- 6. Build the TensorRT engine ON THIS DEVICE -------------------------------
# A TensorRT engine is device- and version-specific. It CANNOT be copied from
# another machine — it must be built here.
log "=== Step 6/8: Building the TensorRT engine ==="
if [ "${SKIP_ENGINE:-0}" != "1" ]; then
  [ -f "$ONNX" ] || die "missing ${ONNX} — cannot build the engine. Set MODEL=… or restore the .onnx file."

  # Detect the ONNX input tensor name and whether it has a dynamic batch axis.
  # Only a dynamic ONNX can produce a batched engine; a static one silently
  # yields max_batch=1 and cross-camera batching does nothing.
  read -r IN_NAME IS_DYNAMIC <<EOF
$(python3 - "$ONNX" <<'PY'
import sys
try:
    import onnx
    m = onnx.load(sys.argv[1])
    i = m.graph.input[0]
    name = i.name
    d0 = i.type.tensor_type.shape.dim[0]
    dyn = (d0.dim_param != "") or (d0.dim_value <= 0)
    print("%s %s" % (name, "1" if dyn else "0"))
except Exception:
    # onnx module not installed — assume the ultralytics default name.
    print("images ?")
PY
)
EOF
  IN_NAME="${IN_NAME:-images}"
  log "ONNX input tensor: ${IN_NAME} (dynamic batch: ${IS_DYNAMIC})"

  if [ "$IS_DYNAMIC" = "0" ]; then
    warn_track "${ONNX} has a FIXED batch axis. The engine will be batch=1 and \
cross-camera batching will not work (INFER_MAX_BATCH is then ignored). \
To enable batching, re-export with: \
yolo export model=${MODEL}.pt format=onnx dynamic=True imgsz=${IMG_SZ} simplify=True"
    log "Building fixed-batch FP16 engine: ${ENGINE}"
    /usr/src/tensorrt/bin/trtexec \
        --onnx="$ONNX" \
        --saveEngine="$ENGINE" \
        --fp16 \
        --memPoolSize="workspace:${WORKSPACE_MB}"
  else
    # Dynamic-batch build: min=1, opt=OPT_BATCH, max=MAX_BATCH. INFER_MAX_BATCH
    # in .env must be <= MAX_BATCH or run_batch silently splits into extra calls.
    log "Building DYNAMIC-batch FP16 engine (1..${MAX_BATCH}, opt=${OPT_BATCH}) at ${IMG_SZ}px: ${ENGINE}"
    /usr/src/tensorrt/bin/trtexec \
        --onnx="$ONNX" \
        --saveEngine="$ENGINE" \
        --fp16 \
        --memPoolSize="workspace:${WORKSPACE_MB}" \
        --minShapes="${IN_NAME}:1x3x${IMG_SZ}x${IMG_SZ}" \
        --optShapes="${IN_NAME}:${OPT_BATCH}x3x${IMG_SZ}x${IMG_SZ}" \
        --maxShapes="${IN_NAME}:${MAX_BATCH}x3x${IMG_SZ}x${IMG_SZ}"
  fi
  ok "Engine built: ${ENGINE}"
else
  log "SKIP_ENGINE=1 — not rebuilding the engine."
  [ -f "$ENGINE" ] || warn_track "SKIP_ENGINE=1 but ${ENGINE} does not exist; the service will fail to start."
fi

# --- 7. Configuration ----------------------------------------------------------
log "=== Step 7/8: Configuration (.env) ==="
if [ ! -f .env ]; then
  [ -f .env.example ] || die ".env.example missing — cannot generate .env"
  cp .env.example .env
  log "Created .env from .env.example"
fi

# Keep .env consistent with what was actually built, so the operator never has
# to hand-edit these three coupled values.
set_env_var() {
  local key="$1" val="$2"
  if grep -qE "^[#[:space:]]*${key}=" .env; then
    sed -i -E "s|^[#[:space:]]*${key}=.*|${key}=${val}|" .env
  else
    printf '%s=%s\n' "$key" "$val" >> .env
  fi
}
set_env_var DET_ENGINE "./${ENGINE}"
set_env_var IMG_SZ "${IMG_SZ}"
if [ "${IS_DYNAMIC:-1}" = "0" ]; then
  set_env_var INFER_MAX_BATCH 1
else
  set_env_var INFER_MAX_BATCH "${MAX_BATCH}"
fi
set_env_var PORT "${PORT}"
ok ".env set: DET_ENGINE=./${ENGINE}, IMG_SZ=${IMG_SZ}, INFER_MAX_BATCH=$(grep -E '^INFER_MAX_BATCH=' .env | cut -d= -f2)"

# --- 8. systemd service + health check -----------------------------------------
log "=== Step 8/8: Service installation and health check ==="
if [ "${SKIP_SERVICE:-0}" != "1" ]; then
  install_systemd_unit "$SERVICE_NAME" "$APP_DIR" "$VENV" "$RUN_USER"

  log "Starting ${SERVICE_NAME}…"
  sudo systemctl restart "$SERVICE_NAME"

  if wait_for_health "$PORT" 120; then
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

$(printf '%s' "${_C_OK}")Orin Nano setup complete.${_C_OFF}

  Service:   ${SERVICE_NAME}   (starts automatically on boot)
  API:       http://$(hostname -I 2>/dev/null | awk '{print $1}'):${PORT}
  Engine:    ${ENGINE}
  Config:    ${APP_DIR}/.env

Useful commands:
  sudo systemctl status ${SERVICE_NAME}
  sudo journalctl -u ${SERVICE_NAME} -f
  curl http://localhost:${PORT}/health

Add a camera:
  curl -X POST http://localhost:${PORT}/cameras \\
    -H 'Content-Type: application/json' \\
    -d '{"source_url":"rtsp://user:pass@192.168.1.50:554/stream1","camera_uuid":"front-door"}'

EOF
