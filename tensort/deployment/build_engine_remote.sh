#!/usr/bin/env bash
# Build the TensorRT engine ON the Jetson, driven from your dev machine.
#
# WHY THIS EXISTS: a TensorRT engine cannot be built locally and copied to the
# Jetson. The serialized plan is tied to the CPU architecture (x86_64 vs
# aarch64), the GPU compute capability (8.6 on a desktop RTX vs 8.7 on Orin) and
# the exact TensorRT version. A foreign plan fails to deserialize at startup.
# The ONNX, however, IS portable — so this script ships the ONNX over and runs
# only the build remotely.
#
#   ./deployment/build_engine_remote.sh peshal@jetson.local
#   MODEL=yolo26n ./deployment/build_engine_remote.sh peshal@192.168.1.50
#
# Env:
#   REMOTE_DIR   path to tensort/ on the Jetson (default ~/1886NOENTRY/Backend/tensort)
#   MODEL        model basename (default yolo26s)
#   MAX_BATCH    engine max batch (default 10)
#   FETCH=1      copy the finished .engine back here for archiving
set -euo pipefail

TARGET="${1:-}"
[ -n "$TARGET" ] || { echo "usage: $0 user@jetson-host"; exit 1; }

SELF="$0"
while [ -L "$SELF" ]; do
  LINK="$(readlink "$SELF")"
  case "$LINK" in
    /*) SELF="$LINK" ;;
    *)  SELF="$(dirname "$SELF")/$LINK" ;;
  esac
done
HERE="$(cd "$(dirname "$SELF")" && pwd)"
APP_DIR="$(cd "${HERE}/.." && pwd)"   # the tensort/ folder
cd "$APP_DIR"

MODEL="${MODEL:-yolo26s}"
MAX_BATCH="${MAX_BATCH:-10}"
IMG_SZ="${IMG_SZ:-640}"
REMOTE_DIR="${REMOTE_DIR:-~/1886NOENTRY/Backend/tensort}"
ONNX="models/${MODEL}.onnx"

[ -f "$ONNX" ] || { echo "ERROR: ${ONNX} not found locally. Export it first:"; \
  echo "  yolo export model=models/${MODEL}.pt format=onnx dynamic=True imgsz=${IMG_SZ} simplify=True nms=False"; exit 1; }

echo "==> Copying ${ONNX} to ${TARGET}:${REMOTE_DIR}/models/"
ssh "$TARGET" "mkdir -p ${REMOTE_DIR}/models"
scp "$ONNX" "${TARGET}:${REMOTE_DIR}/models/"
scp "${HERE}/build_engine.sh" "${TARGET}:${REMOTE_DIR}/deployment/"

echo "==> Building on the Jetson (this takes ~10-20 min on an Orin Nano)"
ssh -t "$TARGET" "cd ${REMOTE_DIR} && chmod +x deployment/build_engine.sh && \
  MODEL=${MODEL} MAX_BATCH=${MAX_BATCH} IMG_SZ=${IMG_SZ} ./deployment/build_engine.sh"

if [ "${FETCH:-0}" = "1" ]; then
  echo "==> Fetching the built engine back (for archiving only — it runs on the Jetson)"
  scp "${TARGET}:${REMOTE_DIR}/models/${MODEL}.engine" "models/${MODEL}.engine.${TARGET##*@}"
fi

echo
echo "Done. On the Jetson, point the service at it:"
echo "  sed -i 's|^DET_ENGINE=.*|DET_ENGINE=./models/${MODEL}.engine|' ${REMOTE_DIR}/.env"
echo "  python3 tests/diag_batch.py <a_frame.jpg> ${MAX_BATCH}"
echo "  sudo systemctl restart jetson-cameras"
