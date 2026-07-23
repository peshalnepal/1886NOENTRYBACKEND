#!/usr/bin/env bash
# =============================================================================
# common.sh — shared helpers for the tensort deployment scripts.
#
# Sourced by setup_orin.sh and setup_nano.sh. Not meant to be run directly.
#
# Provides: logging, board/OS detection, preflight resource checks,
#           systemd unit installation, and the post-install health check.
# =============================================================================

# --- Logging -----------------------------------------------------------------
# Colour only when stdout is a terminal, so `./setup_orin.sh > install.log`
# produces a clean readable log instead of escape-code soup.
if [ -t 1 ]; then
  _C_INFO=$'\033[1;36m'; _C_WARN=$'\033[1;33m'; _C_ERR=$'\033[1;31m'
  _C_OK=$'\033[1;32m';   _C_OFF=$'\033[0m'
else
  _C_INFO=''; _C_WARN=''; _C_ERR=''; _C_OK=''; _C_OFF=''
fi

LOG_PREFIX="${LOG_PREFIX:-setup}"

log()  { printf '%s[%s]%s %s\n' "$_C_INFO" "$LOG_PREFIX" "$_C_OFF" "$*"; }
ok()   { printf '%s[%s] OK:%s %s\n' "$_C_OK" "$LOG_PREFIX" "$_C_OFF" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$_C_WARN" "$LOG_PREFIX" "$_C_OFF" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$_C_ERR" "$LOG_PREFIX" "$_C_OFF" "$*" >&2; exit 1; }

# Track non-fatal problems so the script can print a single summary at the end
# instead of the operator having to scroll back through the whole run.
WARN_COUNT=0
warn_track() { WARN_COUNT=$((WARN_COUNT + 1)); warn "$@"; }

# --- Board / OS detection -----------------------------------------------------

detect_board() {
  # The redirect itself fails on non-Jetson hosts, so guard on the file first —
  # `2>/dev/null` on tr would not suppress the shell's own redirection error.
  if [ -r /proc/device-tree/model ]; then
    tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "unknown"
  else
    echo "unknown"
  fi
}

detect_l4t() {
  if [ -f /etc/nv_tegra_release ]; then
    head -n1 /etc/nv_tegra_release
  else
    echo "unknown"
  fi
}

detect_jetpack_major() {
  # R32 -> JetPack 4.x (original Nano); R35/R36 -> 5.x/6.x; R39 -> 7.x (Orin)
  local rel
  rel="$(detect_l4t)"
  case "$rel" in
    *R32*) echo 4 ;;
    *R35*) echo 5 ;;
    *R36*) echo 6 ;;
    *R39*) echo 7 ;;
    *)     echo 0 ;;
  esac
}

py_version() {
  python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "0.0"
}

total_mem_mb() {
  awk '/^MemTotal:/ {printf "%d", $2/1024}' /proc/meminfo 2>/dev/null || echo 0
}

free_disk_mb() {
  # Free space on the filesystem holding the tensort folder.
  df -Pm "$1" 2>/dev/null | awk 'NR==2 {print $4}' || echo 0
}

# --- Preflight ----------------------------------------------------------------
# Fails fast on the conditions that otherwise waste 45 minutes and then break:
# wrong board, not enough disk for the JetPack download, no swap on a 4GB board.

preflight() {
  local need_disk_mb="$1"     # required free disk (MB)
  local min_mem_mb="$2"       # minimum RAM (MB)
  local here="$3"             # tensort folder

  log "Board:   $(detect_board)"
  log "L4T:     $(detect_l4t)"
  log "Python:  $(py_version)"
  log "RAM:     $(total_mem_mb) MB"
  log "Disk:    $(free_disk_mb "$here") MB free at ${here}"

  local mem disk
  mem="$(total_mem_mb)"
  disk="$(free_disk_mb "$here")"

  if [ "$mem" -gt 0 ] && [ "$mem" -lt "$min_mem_mb" ]; then
    warn_track "RAM is ${mem} MB; ${min_mem_mb} MB expected for this board."
  fi

  if [ "$disk" -gt 0 ] && [ "$disk" -lt "$need_disk_mb" ]; then
    die "Only ${disk} MB free, need at least ${need_disk_mb} MB. \
Free space (sudo apt-get clean) or use a larger storage device, then re-run."
  fi

  # The engine build and pip compiles are the memory spikes. On a 4 GB board
  # without swap, trtexec is the classic silent OOM-kill.
  local swap
  swap="$(awk '/^SwapTotal:/ {printf "%d", $2/1024}' /proc/meminfo 2>/dev/null || echo 0)"
  if [ "$mem" -lt 6000 ] && [ "$swap" -lt 2000 ]; then
    warn_track "Only ${swap} MB swap on a ${mem} MB board. The TensorRT engine build \
may be killed by the kernel. Consider: sudo fallocate -l 4G /swapfile && \
sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile"
  fi

  if ! ping -c1 -W3 8.8.8.8 >/dev/null 2>&1 && ! ping -c1 -W3 github.com >/dev/null 2>&1; then
    warn_track "No network connectivity detected — the apt/pip steps will fail."
  fi
}

# --- Verification helpers -----------------------------------------------------

# Confirm the system OpenCV was built WITH GStreamer. A pip `opencv-python`
# wheel shadowing it is the single most common cause of "camera never connects":
# the pip wheel has no GStreamer, so every hardware-decode pipeline fails.
check_opencv_gstreamer() {
  python3 - <<'PY' 2>/dev/null
import re, sys
try:
    import cv2
except Exception as e:
    print("IMPORT_FAIL %s" % e)
    sys.exit(2)
info = cv2.getBuildInformation()
gst = bool(re.search(r"GStreamer:\s+YES", info))
print("%s %s %s" % ("YES" if gst else "NO", cv2.__version__, getattr(cv2, "__file__", "?")))
sys.exit(0 if gst else 1)
PY
}

# --- systemd ------------------------------------------------------------------

install_systemd_unit() {
  local svc_name="$1"      # e.g. jetson-cameras
  local workdir="$2"       # /home/user/tensort
  local venv="$3"          # .venv_trt
  local run_user="$4"
  local unit="/etc/systemd/system/${svc_name}.service"

  log "Installing systemd unit ${unit}…"

  # Written via a temp file + sudo install so the heredoc is not run as root.
  local tmp
  tmp="$(mktemp)"
  cat > "$tmp" <<EOF
[Unit]
Description=Jetson Camera Detection Service (tensort)
Documentation=file://${workdir}/deployment/README.md
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${run_user}
WorkingDirectory=${workdir}
EnvironmentFile=-${workdir}/.env
Environment=PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64
# '+' runs these as root regardless of User=, to restore max clocks after boot.
ExecStartPre=+/usr/sbin/nvpmodel -m 0
ExecStartPre=+/usr/bin/jetson_clocks
ExecStart=${workdir}/${venv}/bin/python3 main.py
Restart=on-failure
RestartSec=5
# The service holds camera state in SQLite; give it time to close cleanly.
TimeoutStopSec=20
KillSignal=SIGINT

[Install]
WantedBy=multi-user.target
EOF

  sudo install -m 0644 "$tmp" "$unit" || { rm -f "$tmp"; die "failed to write ${unit}"; }
  rm -f "$tmp"

  sudo systemctl daemon-reload
  sudo systemctl enable "${svc_name}" >/dev/null 2>&1 || warn_track "systemctl enable failed"
  ok "systemd unit installed and enabled: ${svc_name}"
}

# --- Health check -------------------------------------------------------------
# Polls /health until pipeline_ready is true. This is what turns "the script
# finished" into "the service actually works".

wait_for_health() {
  local port="${1:-8080}"
  local timeout_s="${2:-90}"
  local url="http://127.0.0.1:${port}/health"
  local deadline=$((SECONDS + timeout_s))

  log "Waiting for ${url} (up to ${timeout_s}s)…"

  while [ $SECONDS -lt $deadline ]; do
    local body
    body="$(curl -fsS --max-time 5 "$url" 2>/dev/null || true)"
    if [ -n "$body" ]; then
      if printf '%s' "$body" | grep -q '"pipeline_ready":[[:space:]]*true'; then
        ok "Service healthy: pipeline_ready=true"
        printf '%s\n' "$body"
        return 0
      fi
      # Reachable but not ready — keep the last body for the failure message.
      LAST_HEALTH_BODY="$body"
    fi
    sleep 3
  done

  warn_track "Service did not report pipeline_ready=true within ${timeout_s}s."
  [ -n "${LAST_HEALTH_BODY:-}" ] && printf 'Last /health response: %s\n' "$LAST_HEALTH_BODY"
  return 1
}

# --- Summary ------------------------------------------------------------------

print_warn_summary() {
  if [ "$WARN_COUNT" -gt 0 ]; then
    printf '\n%s[%s] Completed with %d warning(s) — review the output above.%s\n' \
      "$_C_WARN" "$LOG_PREFIX" "$WARN_COUNT" "$_C_OFF"
  fi
}
