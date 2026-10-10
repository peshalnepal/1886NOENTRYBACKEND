#!/usr/bin/env bash
# common.sh — helpers shared by setup_orin.sh and setup_nano.sh (source it, don't run it).

# No colour codes: logs piped through `tee` stay readable.
_C_INFO=''; _C_WARN=''; _C_ERR=''; _C_OK=''; _C_OFF=''
LOG_PREFIX="${LOG_PREFIX:-setup}"
WARN_COUNT=0

log()        { printf '[%s] %s\n' "$LOG_PREFIX" "$*"; }
ok()         { printf '[%s] OK: %s\n' "$LOG_PREFIX" "$*"; }
warn()       { printf '[%s] WARN: %s\n' "$LOG_PREFIX" "$*" >&2; }
warn_track() { WARN_COUNT=$((WARN_COUNT + 1)); warn "$@"; }
die()        { printf '[%s] ERROR: %s\n' "$LOG_PREFIX" "$*" >&2; exit 1; }

print_warn_summary() {
  [ "$WARN_COUNT" -gt 0 ] && printf '\n[%s] Completed with %d warning(s) — review the output above.\n' "$LOG_PREFIX" "$WARN_COUNT"
  return 0
}

# --- Board / OS ---------------------------------------------------------------
detect_board() { [ -r /proc/device-tree/model ] && tr -d '\0' < /proc/device-tree/model || echo "unknown"; }
detect_l4t()   { [ -f /etc/nv_tegra_release ] && head -n1 /etc/nv_tegra_release || echo "unknown"; }

# L4T R32 -> JetPack 4, R35 -> 5, R36 -> 6, R39 -> 7; 0 if unknown.
detect_jetpack_major() {
  case "$(detect_l4t)" in
    *R32*) echo 4 ;; *R35*) echo 5 ;; *R36*) echo 6 ;; *R39*) echo 7 ;; *) echo 0 ;;
  esac
}

py_version()   { python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "0.0"; }
total_mem_mb() { awk '/^MemTotal:/ {printf "%d", $2/1024}' /proc/meminfo 2>/dev/null || echo 0; }
free_disk_mb() { df -Pm "$1" 2>/dev/null | awk 'NR==2 {print $4}' || echo 0; }

# preflight NEED_DISK_MB MIN_MEM_MB DIR — log the board and stop early on
# conditions that would otherwise waste 45 minutes (no disk, no network).
preflight() {
  local need_disk_mb="$1" min_mem_mb="$2" dir="$3" mem disk swap
  mem="$(total_mem_mb)"; disk="$(free_disk_mb "$dir")"
  swap="$(awk '/^SwapTotal:/ {printf "%d", $2/1024}' /proc/meminfo 2>/dev/null || echo 0)"

  log "Board: $(detect_board) | L4T: $(detect_l4t) | Python: $(py_version)"
  log "RAM: ${mem} MB | Swap: ${swap} MB | Disk free at ${dir}: ${disk:-?} MB"

  [ "$mem" -gt 0 ] && [ "$mem" -lt "$min_mem_mb" ] && warn_track "RAM is ${mem} MB; ${min_mem_mb} MB expected."
  [ "${disk:-0}" -gt 0 ] && [ "$disk" -lt "$need_disk_mb" ] \
    && die "Only ${disk} MB free, need ${need_disk_mb} MB. Run 'sudo apt-get clean' or use a larger disk, then re-run."
  # trtexec on a small board without swap is the classic silent OOM-kill.
  [ "$mem" -lt 6000 ] && [ "$swap" -lt 2000 ] && warn_track "Only ${swap} MB swap on a ${mem} MB board; the engine build may be \
OOM-killed. Add swap: sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile"
  if ! ping -c1 -W3 8.8.8.8 >/dev/null 2>&1 && ! ping -c1 -W3 github.com >/dev/null 2>&1; then
    warn_track "No network connectivity — the apt/pip steps will fail."
  fi
  return 0
}

# --- Checks -------------------------------------------------------------------
# Prints "YES|NO <cv2 version> <path>" (exit 0/1) or "IMPORT_FAIL <err>" (exit 2).
# A pip opencv-python wheel has no GStreamer and silently kills hardware decode.
check_opencv_gstreamer() {
  python3 - <<'PY' 2>/dev/null
import re, sys
try:
    import cv2
except Exception as e:
    print("IMPORT_FAIL %s" % e); sys.exit(2)
gst = bool(re.search(r"GStreamer:\s+YES", cv2.getBuildInformation()))
print("%s %s %s" % ("YES" if gst else "NO", cv2.__version__, getattr(cv2, "__file__", "?")))
sys.exit(0 if gst else 1)
PY
}

# set_env_var FILE KEY VALUE — replace the active "KEY=…" line or append one.
set_env_var() {
  local file="$1" key="$2" val="$3" esc
  esc="$(printf '%s' "$val" | sed -e 's/[\\&|]/\\&/g')"
  if grep -qE "^${key}=" "$file"; then
    sed -i -E "s|^${key}=.*|${key}=${esc}|" "$file"
  else
    printf '%s=%s\n' "$key" "$val" >> "$file"
  fi
}

# --- systemd ------------------------------------------------------------------
# install_systemd_unit NAME WORKDIR VENV USER — write, reload and enable the unit.
install_systemd_unit() {
  local svc="$1" workdir="$2" venv="$3" user="$4" unit="/etc/systemd/system/$1.service" tmp
  log "Installing systemd unit ${unit}"
  tmp="$(mktemp)"
  cat > "$tmp" <<EOF
[Unit]
Description=Jetson Camera Detection Service (tensort)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${user}
WorkingDirectory=${workdir}
EnvironmentFile=-${workdir}/.env
Environment=PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=LD_LIBRARY_PATH=/usr/local/cuda/lib64
# '+' = run as root even though User= is set: restore max clocks after every boot.
ExecStartPre=-+/usr/sbin/nvpmodel -m 0
ExecStartPre=-+/usr/bin/jetson_clocks
ExecStart=${workdir}/${venv}/bin/python3 main.py
Restart=on-failure
RestartSec=5
TimeoutStopSec=20
KillSignal=SIGINT

[Install]
WantedBy=multi-user.target
EOF
  sudo install -m 0644 "$tmp" "$unit" || { rm -f "$tmp"; die "failed to write ${unit}"; }
  rm -f "$tmp"
  sudo systemctl daemon-reload
  sudo systemctl enable "$svc" >/dev/null 2>&1 || warn_track "systemctl enable ${svc} failed"
  ok "systemd unit installed and enabled: ${svc}"
}

# wait_for_health [PORT=8080] [TIMEOUT_S=90] — poll /health until pipeline_ready is true.
wait_for_health() {
  local port="${1:-8080}" timeout_s="${2:-90}" url body="" deadline
  url="http://127.0.0.1:${port}/health"; deadline=$((SECONDS + timeout_s))
  log "Waiting for ${url} (up to ${timeout_s}s)"
  while [ $SECONDS -lt $deadline ]; do
    body="$(curl -fsS --max-time 5 "$url" 2>/dev/null || true)"
    if printf '%s' "$body" | grep -q '"pipeline_ready":[[:space:]]*true'; then
      ok "Service healthy: pipeline_ready=true"
      printf '%s\n' "$body"
      return 0
    fi
    sleep 3
  done
  warn_track "Service did not report pipeline_ready=true within ${timeout_s}s."
  [ -n "$body" ] && printf 'Last /health response: %s\n' "$body"
  return 1
}
