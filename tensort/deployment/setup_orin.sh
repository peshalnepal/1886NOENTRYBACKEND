#!/usr/bin/env bash
# setup_orin.sh — one-command install of the tensort detection service on a
# Jetson Orin Nano (JetPack 6/7, TensorRT 10). Run it ON the Jetson, as your
# normal user (it asks for sudo once):
#
#     ./deployment/setup_orin.sh
#     MAX_BATCH=8 OPT_BATCH=4 ./deployment/setup_orin.sh
#
# Safe to re-run: finished steps are detected and skipped (JetPack already
# installed, engine already built with the same settings, venv present).
#
# Env overrides:
#   MODEL=yolo26m  MAX_BATCH=8  OPT_BATCH=MAX_BATCH  IMG_SZ=640  WORKSPACE_MB=3072
#   PORT=8080  SERVICE_NAME=jetson-cameras  VENV=.venv_trt
#   DISCOVERY=false   camera discovery on the Jetson (the MiniPC owns it in a tower)
#   FORCE_ENGINE=1    rebuild the engine even if an identical one exists
#   SKIP_APT=1 / SKIP_ENGINE=1 / SKIP_SERVICE=1   skip a step outright
#
# Never pip-installs tensorrt / opencv / numpy: those are JetPack's and a pip
# opencv wheel has no GStreamer, which breaks hardware video decode.
set -euo pipefail

LOG_PREFIX="orin-setup"
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
APP_DIR="$(cd "${HERE}/.." && pwd)"
. "${HERE}/common.sh"
cd "$APP_DIR"
[ -f main.py ] || die "main.py not found in ${APP_DIR}; deployment/ must sit inside tensort/"
[ "$(id -u)" -ne 0 ] || die "run as your normal user, not root/sudo (a root-owned venv breaks the service)"

MODEL="${MODEL:-yolo26m}"
MAX_BATCH="${MAX_BATCH:-8}"
OPT_BATCH="${OPT_BATCH:-${MAX_BATCH}}"
IMG_SZ="${IMG_SZ:-640}"
WORKSPACE_MB="${WORKSPACE_MB:-3072}"
PORT="${PORT:-8080}"
SERVICE_NAME="${SERVICE_NAME:-jetson-cameras}"
VENV="${VENV:-.venv_trt}"
DISCOVERY="${DISCOVERY:-false}"
export TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ONNX="models/${MODEL}.onnx"
ENGINE="models/${MODEL}.engine"
[[ "$MAX_BATCH" =~ ^[1-8]$ ]] || die "MAX_BATCH must be 1..8"
[[ "$OPT_BATCH" =~ ^[1-8]$ ]] && (( OPT_BATCH <= MAX_BATCH )) || die "OPT_BATCH must be 1..MAX_BATCH"
[[ "$DISCOVERY" =~ ^(true|false)$ ]] || die "DISCOVERY must be true or false"

# --- 1. Preflight ---------------------------------------------------------------
log "=== 1/7 Preflight ==="
preflight 12000 6000 "$APP_DIR"
case "$(detect_board)" in
  *Orin*) ok "Orin board" ;;
  *) warn_track "Board is '$(detect_board)', not an Orin. For the original Jetson Nano use setup_nano.sh." ;;
esac
[ "${SKIP_ENGINE:-0}" = "1" ] || [ -f "$ONNX" ] || die "${ONNX} missing — copy the model to models/ first"
log "Asking for sudo once now so the long steps don't stall on a password prompt"
sudo -v || die "sudo is required"

# --- 2. System packages ---------------------------------------------------------
log "=== 2/7 System packages ==="
have_jetpack() { [ -x "$NVCC" ] && [ -x "$TRTEXEC" ] && python3 -c 'import tensorrt' 2>/dev/null; }
if [ "${SKIP_APT:-0}" = "1" ]; then
  log "SKIP_APT=1"
elif have_jetpack; then
  ok "JetPack (CUDA + TensorRT) already installed; skipping the download"
  sudo apt-get install -y -qq python3-venv python3-dev python3-pip build-essential curl \
    gstreamer1.0-tools python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
else
  log "Installing nvidia-jetpack (multi-GB download, 30+ min) and build/GStreamer tools"
  sudo apt-get update
  sudo apt-get install -y nvidia-jetpack python3-venv python3-dev python3-pip build-essential curl \
    gstreamer1.0-tools python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav
fi

# --- 3. Verify the system stack -------------------------------------------------
log "=== 3/7 Verify CUDA / TensorRT / OpenCV ==="
have_jetpack || die "CUDA/TensorRT/trtexec missing — JetPack did not install. Re-run without SKIP_APT=1."
TRT_VERSION="$(python3 -c 'import tensorrt;print(tensorrt.__version__)')"
[ "${TRT_VERSION%%.*}" -ge 10 ] || die "TensorRT ${TRT_VERSION} found; this code needs TensorRT 10 (JetPack 6/7)."
ok "CUDA + TensorRT ${TRT_VERSION}"
GST_OUT="$(check_opencv_gstreamer || true)"
case "$GST_OUT" in
  YES*) ok "OpenCV with GStreamer: ${GST_OUT}" ;;
  NO*)  die "OpenCV has no GStreamer: a pip opencv wheel shadows JetPack's. Fix: python3 -m pip uninstall -y opencv-python opencv-python-headless; sudo apt-get install --reinstall python3-opencv" ;;
  *)    die "OpenCV import failed: ${GST_OUT}" ;;
esac
sudo nvpmodel -m 0 >/dev/null 2>&1 || warn_track "nvpmodel -m 0 failed (non-fatal)"
sudo jetson_clocks   >/dev/null 2>&1 || warn_track "jetson_clocks failed (non-fatal)"

# --- 4. Python venv -------------------------------------------------------------
log "=== 4/7 Python environment (${VENV}) ==="
[ -d "$VENV" ] || python3 -m venv --system-site-packages "$VENV"
# shellcheck disable=SC1091
. "${VENV}/bin/activate"
export PATH="/usr/local/cuda/bin:$PATH" CUDA_ROOT="/usr/local/cuda"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
python3 -m pip install -q --upgrade pip setuptools wheel
log "pip install -r requirements-jetson.txt (pycuda compiles from source on first run)"
pip install -q -r requirements-jetson.txt
python3 -c "import flask, sqlalchemy, aiosqlite, pycuda.driver, tensorrt, cv2, dotenv" || die "venv import check failed"
GST_OUT="$(check_opencv_gstreamer || true)"
case "$GST_OUT" in
  YES*) ok "venv imports OK, OpenCV still has GStreamer" ;;
  *) die "inside the venv OpenCV lost GStreamer (${GST_OUT}). Fix: pip uninstall -y opencv-python opencv-python-headless numpy" ;;
esac

# --- 5. TensorRT engine ---------------------------------------------------------
log "=== 5/7 TensorRT engine ==="
WANT_STAMP="MODEL=${MODEL} MAX_BATCH=${MAX_BATCH} OPT_BATCH=${OPT_BATCH} IMG_SZ=${IMG_SZ} TRT=${TRT_VERSION}"
if [ "${SKIP_ENGINE:-0}" = "1" ]; then
  log "SKIP_ENGINE=1"
  [ -f "$ENGINE" ] || warn_track "${ENGINE} does not exist; the service will fail to start"
elif [ "${FORCE_ENGINE:-0}" != "1" ] && [ -f "$ENGINE" ] && [ "$(cat "${ENGINE}.stamp" 2>/dev/null)" = "$WANT_STAMP" ]; then
  ok "${ENGINE} already built with these settings (${WANT_STAMP}); set FORCE_ENGINE=1 to rebuild"
else
  log "Building ${ENGINE} on this device (10-20 min): ${WANT_STAMP}"
  MODEL="$MODEL" MAX_BATCH="$MAX_BATCH" OPT_BATCH="$OPT_BATCH" IMG_SZ="$IMG_SZ" WORKSPACE_MB="$WORKSPACE_MB" \
    SKIP_CLOCKS=1 bash "${HERE}/build_engine.sh" || die "engine build failed"
fi

# --- 6. .env --------------------------------------------------------------------
log "=== 6/7 Configuration (.env) ==="
if [ ! -f .env ]; then
  [ -f .env.example ] || die ".env.example missing"
  cp .env.example .env && log "Created .env from .env.example"
fi
set_env_var .env DET_ENGINE "./${ENGINE}"
set_env_var .env IMG_SZ "$IMG_SZ"
set_env_var .env PORT "$PORT"
set_env_var .env DISCOVERY_ENABLED "$DISCOVERY"
# The service never batches above the engine's max; keep .env in step with the build.
[ "${SKIP_ENGINE:-0}" = "1" ] || set_env_var .env INFER_MAX_BATCH "$MAX_BATCH"
ok ".env: DET_ENGINE=./${ENGINE} IMG_SZ=${IMG_SZ} INFER_MAX_BATCH=$(grep -E '^INFER_MAX_BATCH=' .env | cut -d= -f2) PORT=${PORT} DISCOVERY_ENABLED=${DISCOVERY}"

# --- 7. systemd service + health check ------------------------------------------
log "=== 7/7 Service ==="
if [ "${SKIP_SERVICE:-0}" = "1" ]; then
  log "SKIP_SERVICE=1"
else
  install_systemd_unit "$SERVICE_NAME" "$APP_DIR" "$VENV" "$(id -un)"
  sudo systemctl restart "$SERVICE_NAME"
  wait_for_health "$PORT" 180 || warn_track "Inspect with: sudo journalctl -u ${SERVICE_NAME} -n 80 --no-pager"
fi

print_warn_summary
cat <<EOF

Orin Nano setup complete.
  Service:  ${SERVICE_NAME} (starts on boot)      sudo systemctl status ${SERVICE_NAME}
  Logs:     sudo journalctl -u ${SERVICE_NAME} -f
  API:      http://$(hostname -I 2>/dev/null | awk '{print $1}'):${PORT}/health
  Engine:   ${ENGINE}  (batch 1..${MAX_BATCH}, opt ${OPT_BATCH}, ${IMG_SZ}px)
  Config:   ${APP_DIR}/.env  (DISCOVERY_ENABLED=${DISCOVERY})

Only the MiniPC should reach port ${PORT} (the API has no login):
  sudo ufw allow OpenSSH && sudo ufw allow from <MINIPC_IP> to any port ${PORT} proto tcp && sudo ufw enable
EOF
