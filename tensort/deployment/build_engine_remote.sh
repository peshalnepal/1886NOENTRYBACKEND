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
#   MODEL=yolo26m ./deployment/build_engine_remote.sh peshal@192.168.1.50
#
# Env:
#   REMOTE_DIR   path to tensort/ on the Jetson, relative to the remote home
#                directory unless it starts with "/"
#                (default 1886NOENTRY/Backend/tensort)
#   MODEL        model basename (default yolo26m)
#   MAX_BATCH    engine max batch (default 8; min batch is always 1)
#   OPT_BATCH    batch the engine is tuned for (default MAX_BATCH)
#   IMG_SZ       input image size used for the build (default 640)
#   WORKSPACE_MB trtexec build scratch memory (default 3072)
#   FETCH=1      copy the finished .engine back here for archiving
set -euo pipefail

# --- Arguments ---------------------------------------------------------------
# First argument is the SSH target, e.g. user@jetson.local. ${1:-} avoids an
# "unbound variable" error under `set -u` when no argument is given.
TARGET="${1:-}"
[ -n "$TARGET" ] || { echo "usage: $0 user@jetson-host" >&2; exit 1; }

# --- Locate this script ------------------------------------------------------
# Resolve symlinks by hand (no `readlink -f`, which is not portable to macOS),
# so HERE always points at the real deployment/ folder even when the script is
# invoked through a symlink.
SELF="$0"
while [ -L "$SELF" ]; do
  LINK="$(readlink "$SELF")"
  case "$LINK" in
    /*) SELF="$LINK" ;;                        # absolute link target
    *)  SELF="$(dirname "$SELF")/$LINK" ;;     # relative to the link's folder
  esac
done
HERE="$(cd "$(dirname "$SELF")" && pwd)"   # the deployment/ folder
APP_DIR="$(cd "${HERE}/.." && pwd)"        # the tensort/ folder
cd "$APP_DIR"

# --- Settings (each can be overridden from the environment) ------------------
MODEL="${MODEL:-yolo26m}"
MAX_BATCH="${MAX_BATCH:-8}"
OPT_BATCH="${OPT_BATCH:-${MAX_BATCH}}"
IMG_SZ="${IMG_SZ:-640}"
WORKSPACE_MB="${WORKSPACE_MB:-3072}"
# Deliberately no leading "~/": tilde is not expanded inside ${VAR:-...}, and
# scp's handling of "~" varies by OpenSSH version. A path relative to the
# remote home directory works the same for both ssh and scp.
REMOTE_DIR="${REMOTE_DIR:-1886NOENTRY/Backend/tensort}"
ONNX="models/${MODEL}.onnx"

# --- Local preflight ---------------------------------------------------------
# Fail early, with a useful message, if the inputs are missing.
[ -f "$ONNX" ] || {
  echo "ERROR: ${ONNX} not found locally. Export it first:" >&2
  echo "  yolo export model=models/${MODEL}.pt format=onnx dynamic=True imgsz=${IMG_SZ} simplify=True nms=False" >&2
  exit 1
}
[ -f "${HERE}/build_engine.sh" ] || {
  echo "ERROR: ${HERE}/build_engine.sh not found." >&2
  exit 1
}

# --- Ship the inputs to the Jetson -------------------------------------------
# Create BOTH target folders first; on a fresh Jetson deployment/ may not exist
# yet, which would make the second scp fail.
echo "==> Copying ${ONNX} to ${TARGET}:${REMOTE_DIR}/models/"
ssh "$TARGET" "mkdir -p ${REMOTE_DIR}/models ${REMOTE_DIR}/deployment"
scp "$ONNX" "${TARGET}:${REMOTE_DIR}/models/"
scp "${HERE}/build_engine.sh" "${TARGET}:${REMOTE_DIR}/deployment/"

# --- Build remotely ----------------------------------------------------------
# `ssh -t` allocates a terminal so the build's progress output streams live.
# ssh does not forward local environment variables, so every setting the
# remote build_engine.sh reads must be passed explicitly on the command line.
echo "==> Building on the Jetson (this takes ~10-20 min on an Orin Nano)"
ssh -t "$TARGET" "cd ${REMOTE_DIR} && chmod +x deployment/build_engine.sh && \
  MODEL=${MODEL} MAX_BATCH=${MAX_BATCH} OPT_BATCH=${OPT_BATCH} IMG_SZ=${IMG_SZ} \
  WORKSPACE_MB=${WORKSPACE_MB} ./deployment/build_engine.sh"

# --- Optional: archive the engine locally ------------------------------------
# The copy is suffixed with the Jetson's hostname/user (text after the last "@")
# so engines from different boards don't overwrite each other. It is for
# archiving only: it cannot run on this machine (see WHY THIS EXISTS).
if [ "${FETCH:-0}" = "1" ]; then
  echo "==> Fetching the built engine back (for archiving only — it runs on the Jetson)"
  scp "${TARGET}:${REMOTE_DIR}/models/${MODEL}.engine" "models/${MODEL}.engine.${TARGET##*@}"
fi

# --- Next steps (printed, not executed) --------------------------------------
echo
echo "Done. On the Jetson, point the service at it:"
echo "  cd ${REMOTE_DIR}"
echo "  sed -i 's|^DET_ENGINE=.*|DET_ENGINE=./models/${MODEL}.engine|' .env"
echo "  python3 tests/diag_batch.py <a_frame.jpg> ${MAX_BATCH}"
echo "  sudo systemctl restart jetson-cameras"