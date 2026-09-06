#!/usr/bin/env bash
# Build a dynamic-batch TensorRT engine from an already-exported ONNX.
#
# MUST run ON THE JETSON. A TensorRT engine is compiled for one GPU
# architecture and one TensorRT version — an engine built on a dev box (or on a
# different Jetson model) will fail to deserialize here.
#
#   ./deployment/build_engine.sh                 # yolo26s, batch 1..10, 640px
#   MODEL=yolo26n ./deployment/build_engine.sh   # roll back to the small model
#   MAX_BATCH=6 ./deployment/build_engine.sh     # fewer cameras
#
# Use deployment/setup_orin.sh instead if you also want the venv, systemd unit
# and health check — this script only does the engine.
set -euo pipefail

# Resolve through symlinks so the script works no matter where it is invoked
# from, or whether it is reached via a symlink on PATH.
SELF="$0"
while [ -L "$SELF" ]; do
  LINK="$(readlink "$SELF")"
  case "$LINK" in
    /*) SELF="$LINK" ;;
    *)  SELF="$(dirname "$SELF")/$LINK" ;;
  esac
done
HERE="$(cd "$(dirname "$SELF")" && pwd)"
APP_DIR="$(cd "${HERE}/.." && pwd)"   # the tensort/ folder — all paths below are relative to it
cd "$APP_DIR"

MODEL="${MODEL:-yolo26s}"
MAX_BATCH="${MAX_BATCH:-10}"
OPT_BATCH="${OPT_BATCH:-${MAX_BATCH}}"
IMG_SZ="${IMG_SZ:-640}"
WORKSPACE_MB="${WORKSPACE_MB:-3072}"
ONNX="models/${MODEL}.onnx"
ENGINE="models/${MODEL}.engine"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

[ -f "$ONNX" ] || { echo "ERROR: ${ONNX} not found. Export it first:"; \
  echo "  yolo export model=models/${MODEL}.pt format=onnx dynamic=True imgsz=${IMG_SZ} simplify=True nms=False"; exit 1; }
[ -x "$TRTEXEC" ] || { echo "ERROR: trtexec not found at ${TRTEXEC} — is this a Jetson with JetPack installed?"; exit 1; }

# A static-batch ONNX silently yields max_batch=1 and cross-camera batching does
# nothing. Catch that here rather than after a 10-minute build.
python3 - "$ONNX" <<'PY'
import sys
try:
    import onnx
except ImportError:
    print("WARN: onnx not installed, skipping dynamic-batch check"); raise SystemExit(0)
m = onnx.load(sys.argv[1]); i = m.graph.input[0]
d = i.type.tensor_type.shape.dim[0]
if not (d.dim_param or d.dim_value <= 0):
    raise SystemExit("ERROR: %s has a FIXED batch axis. Re-export with dynamic=True." % sys.argv[1])
print("ONNX dynamic batch: OK (input '%s')" % i.name)
PY

echo "Setting max clocks (build tactics are timed, so clocks must be stable)…"
sudo nvpmodel -m 0 || true
sudo jetson_clocks || true

echo "Building ${ENGINE}: batch 1..${MAX_BATCH} (opt=${OPT_BATCH}) at ${IMG_SZ}px, FP16"
"$TRTEXEC" \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  --fp16 \
  --memPoolSize="workspace:${WORKSPACE_MB}" \
  --minShapes="images:1x3x${IMG_SZ}x${IMG_SZ}" \
  --optShapes="images:${OPT_BATCH}x3x${IMG_SZ}x${IMG_SZ}" \
  --maxShapes="images:${MAX_BATCH}x3x${IMG_SZ}x${IMG_SZ}"

echo
echo "=== Throughput gate ==="
echo "images/sec = ${MAX_BATCH}000 / (GPU Compute Mean ms); per-camera FPS = that / camera_count"
"$TRTEXEC" --loadEngine="$ENGINE" --shapes="images:${MAX_BATCH}x3x${IMG_SZ}x${IMG_SZ}" 2>&1 \
  | grep -E "GPU Compute Time|Throughput" || true

echo
echo "Engine built: ${ENGINE}"
echo "Next:"
echo "  sed -i 's|^DET_ENGINE=.*|DET_ENGINE=./${ENGINE}|' .env"
echo "  python3 tests/diag_batch.py <a_frame.jpg> ${MAX_BATCH}   # all rows must match"
echo "  sudo systemctl restart jetson-cameras"
