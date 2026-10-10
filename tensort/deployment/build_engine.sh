#!/usr/bin/env bash
# Build the TensorRT engine from models/<MODEL>.onnx. MUST run ON THE JETSON:
# an engine only loads on the GPU + TensorRT version it was built with.
#
#   ./deployment/build_engine.sh                        # yolo26m, batch 1..8 (opt 8), 640px
#   MAX_BATCH=8 OPT_BATCH=4 ./deployment/build_engine.sh
#
# Env: MODEL=yolo26m  MAX_BATCH=8  OPT_BATCH=MAX_BATCH  IMG_SZ=640  WORKSPACE_MB=3072
#      INPUT_NAME=images  TRTEXEC=/usr/src/tensorrt/bin/trtexec  SKIP_CLOCKS=1
#
# Writes models/<MODEL>.engine and models/<MODEL>.engine.stamp (what was built,
# so setup_orin.sh can skip an identical rebuild). Does not touch .env or systemd.
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "${HERE}/.."

MODEL="${MODEL:-yolo26m}"
MAX_BATCH="${MAX_BATCH:-8}"
OPT_BATCH="${OPT_BATCH:-${MAX_BATCH}}"
IMG_SZ="${IMG_SZ:-640}"
WORKSPACE_MB="${WORKSPACE_MB:-3072}"
INPUT_NAME="${INPUT_NAME:-images}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
ONNX="models/${MODEL}.onnx"
ENGINE="models/${MODEL}.engine"

[[ "$MAX_BATCH" =~ ^[1-8]$ ]] || { echo "ERROR: MAX_BATCH must be 1..8" >&2; exit 1; }
[[ "$OPT_BATCH" =~ ^[1-8]$ ]] && (( OPT_BATCH <= MAX_BATCH )) || { echo "ERROR: OPT_BATCH must be 1..MAX_BATCH" >&2; exit 1; }
[ -f "$ONNX" ] || { echo "ERROR: ${ONNX} not found. Export it on a dev machine and copy it here:" >&2
  echo "  yolo export model=${MODEL}.pt format=onnx dynamic=True imgsz=${IMG_SZ} simplify=True nms=False" >&2; exit 1; }
[ -x "$TRTEXEC" ] || { echo "ERROR: ${TRTEXEC} not found — is JetPack installed?" >&2; exit 1; }
TRT_VERSION="$(python3 -c 'import tensorrt;print(tensorrt.__version__)' 2>/dev/null)" \
  || { echo "ERROR: 'import tensorrt' failed — is JetPack installed?" >&2; exit 1; }

if [ "${SKIP_CLOCKS:-0}" != "1" ]; then
  echo "Pinning max clocks (build tactics are timed)"
  sudo nvpmodel -m 0 || true
  sudo jetson_clocks || true
fi

# The ONNX has dynamic batch/height/width; the shape profile pins all of them, so
# the engine is fixed at IMG_SZ (which the service asserts at startup).
echo "Building ${ENGINE}: batch 1..${MAX_BATCH} (opt ${OPT_BATCH}), ${IMG_SZ}px, FP16, TensorRT ${TRT_VERSION}"
rm -f "${ENGINE}.stamp"   # a half-finished or failed build must never pass as "already built"
"$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" --fp16 --memPoolSize="workspace:${WORKSPACE_MB}" \
  --minShapes="${INPUT_NAME}:1x3x${IMG_SZ}x${IMG_SZ}" \
  --optShapes="${INPUT_NAME}:${OPT_BATCH}x3x${IMG_SZ}x${IMG_SZ}" \
  --maxShapes="${INPUT_NAME}:${MAX_BATCH}x3x${IMG_SZ}x${IMG_SZ}" \
  || { echo "ERROR: trtexec failed. If it reported that '${INPUT_NAME}' is not an input, set INPUT_NAME=<name>." >&2; exit 1; }

# Verify the engine really has the requested batch profile. A static-batch ONNX
# builds without error but yields a batch-1 engine, which disables cross-camera
# batching — catch that here instead of in production.
verify_engine() {
  python3 - "$@" <<'PY'
import sys
import tensorrt as trt
engine_path, name, max_b, opt_b, img = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
rt = trt.Runtime(trt.Logger(trt.Logger.ERROR))
with open(engine_path, "rb") as f:
    eng = rt.deserialize_cuda_engine(f.read())
if eng is None:
    sys.exit("ERROR: built engine does not deserialize")
try:
    mn, op, mx = (tuple(s) for s in eng.get_tensor_profile_shape(name, 0))
except Exception as e:
    sys.exit("ERROR: no shape profile for '%s' (%s): the ONNX has a FIXED batch axis. "
             "Re-export with dynamic=True." % (name, e))
print("Engine profile min/opt/max:", mn, op, mx)
want = [(1, 3, img, img), (opt_b, 3, img, img), (max_b, 3, img, img)]
if [mn, op, mx] != want:
    sys.exit("ERROR: profile %s != requested %s. Re-export the ONNX with dynamic=True." % ([mn, op, mx], want))
PY
}
verify_engine "$ENGINE" "$INPUT_NAME" "$MAX_BATCH" "$OPT_BATCH" "$IMG_SZ" || { rm -f "$ENGINE"; exit 1; }

printf 'MODEL=%s MAX_BATCH=%s OPT_BATCH=%s IMG_SZ=%s TRT=%s\n' \
  "$MODEL" "$MAX_BATCH" "$OPT_BATCH" "$IMG_SZ" "$TRT_VERSION" > "${ENGINE}.stamp"

echo
echo "=== Throughput at batch ${MAX_BATCH} (images/sec = ${MAX_BATCH}000 / GPU Compute mean ms; per-camera FPS = that / cameras) ==="
"$TRTEXEC" --loadEngine="$ENGINE" --shapes="${INPUT_NAME}:${MAX_BATCH}x3x${IMG_SZ}x${IMG_SZ}" 2>&1 \
  | grep -E "GPU Compute Time|Throughput" || true
echo
echo "Engine built: ${ENGINE}"
