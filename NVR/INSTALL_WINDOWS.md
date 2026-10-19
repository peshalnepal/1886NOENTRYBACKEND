# Install the NVR gateway on a Windows MiniPC

This guide installs the tower's **NVR gateway** (`Backend/NVR`) and its
**MediaMTX relay** on a Windows 10/11 x64 MiniPC. The MiniPC is the tower's
only public host. It discovers cameras, records them, relays them to the
Jetson and the cloud, and proxies the Jetson's detection API.

```
 Cameras / Hikvision NVR ──RTSP──► MediaMTX (MiniPC :8554) ──LAN RTSP──► Jetson (tensort :8080)
            ▲                          ▲   │                                 │
            │ discovery + ffmpeg probe │   └──public/VPN RTSP──► Cloud MediaMTX
            │                          │ loopback API (:9997 / :9996)
            └──────────── NVR gateway (MiniPC :8000) ◄──HTTP──► Jetson API
                                   ▲
                                   └── Cloud backend (HTTP / SSE)
```

Install the Jetson first, or at least know its LAN IP. See
[tensort/deployment/INSTALL_ORIN.md](../tensort/deployment/INSTALL_ORIN.md).

Throughout this guide:

| Placeholder | Meaning | Example |
|---|---|---|
| `<MINIPC_LAN_IP>` | MiniPC address on the camera LAN | `192.168.1.10` |
| `<JETSON_LAN_IP>` | Jetson address on the camera LAN | `192.168.1.20` |
| `<PUBLIC_HOST>` | Name/IP the cloud uses to reach this MiniPC (VPN or port-forward) | `tower01.example.com` |
| `<READER_PASSWORD>` | MediaMTX relay reader password (you choose it) | letters + digits only |

Run every command below in **PowerShell as Administrator**. In Windows
PowerShell 5.1, `curl` is an alias for `Invoke-WebRequest`, so this guide calls
`curl.exe` explicitly.

---

## 1. Prepare Windows

1. **Static IP.** Give the MiniPC a fixed address on the camera LAN, either a
   DHCP reservation or a static address in *Settings → Network*. The Jetson and
   cloud are configured with this address.
2. **Mark the LAN as Private.** Otherwise Windows Firewall applies the stricter
   *Public* profile:
   ```powershell
   Get-NetConnectionProfile
   Set-NetConnectionProfile -InterfaceAlias "Ethernet" -NetworkCategory Private
   ```
3. **Never sleep.** A sleeping MiniPC stops recording:
   ```powershell
   powercfg /change standby-timeout-ac 0
   powercfg /change hibernate-timeout-ac 0
   powercfg /change monitor-timeout-ac 0
   ```
4. **Accurate clock.** Event clips are cut by timestamp, so the clock must be right:
   ```powershell
   w32tm /config /manualpeerlist:"pool.ntp.org" /syncfromflags:manual /update
   Restart-Service w32time; w32tm /resync; w32tm /query /status
   ```
5. **Windows Update.** Set *Active hours* so updates don't reboot the MiniPC
   mid-shift. Both services in step 8 restart automatically after a reboot.

## 2. Install the prerequisites

| Software | Version | Why |
|---|---|---|
| Python | **3.12 x64** (3.10 minimum) | Runs the gateway. `recordings.py` uses `str \| None`, which needs 3.10+ |
| FFmpeg + ffprobe | A "full" build that includes **libx264** | Frame verification before adoption, and event-clip export |
| MediaMTX | **1.16.1**, `windows_amd64` | RTSP relay and recorder (needs the v3 control API) |
| NSSM | 2.24 | Runs both programs as Windows services |

```powershell
winget install -e --id Python.Python.3.12
winget install -e --id Gyan.FFmpeg          # "full" build, includes libx264
winget install -e --id NSSM.NSSM
```

If `winget` isn't available, download the same tools by hand:
`python.org` for Python (tick *Add to PATH*), the `ffmpeg-release-full.7z`
build from `gyan.dev` (unzip it to `C:\ffmpeg`), and NSSM from `nssm.cc`.

**MediaMTX:** download `mediamtx_v1.16.1_windows_amd64.zip` from
`github.com/bluenviron/mediamtx/releases` and extract it to `C:\mediamtx`.

Open a **new** PowerShell window so it picks up the updated PATH, then verify:

```powershell
py -3.12 --version
ffmpeg -hide_banner -encoders | Select-String libx264   # must print a line
(Get-Command ffmpeg).Source; (Get-Command ffprobe).Source  # note these paths
C:\mediamtx\mediamtx.exe --version
```

## 3. Copy the code

The gateway is a Python package that must run from the folder **above**
`NVR`. Copy only the `NVR` folder. It has no dependency on `tensort` or the
cloud backend.

```
C:\noentry\
└── Backend\
    ├── .venv-nvr\          (created in step 4)
    └── NVR\                (copied from the repo)
        ├── main.py, config.py, mediamtx.yml, requirements.txt, .env.example …
```

Copy it with Git or a zip. Don't copy `NVR\.venv`, `__pycache__` or the
`tests` folder. Then create the data folders:

```powershell
New-Item -ItemType Directory -Force C:\nvr\data, C:\nvr\recordings, C:\nvr\clips, C:\nvr\logs
```

## 4. Create the Python environment

```powershell
cd C:\noentry\Backend
py -3.12 -m venv .venv-nvr
.\.venv-nvr\Scripts\python -m pip install --upgrade pip
.\.venv-nvr\Scripts\pip install -r NVR\requirements.txt tzdata
```

> **`tzdata` is required on Windows** and is not in `requirements.txt`. Windows
> has no IANA timezone database, so without it a clip request such as
> `{"event_time": "...", "timezone": "America/Toronto"}` fails with HTTP 422.
> The cloud backend sends UTC, so it is unaffected, but manual requests are not.

## 5. Configure MediaMTX

Edit `C:\noentry\Backend\NVR\mediamtx.yml`:

1. **RTSP listener.** Change it from loopback so the Jetson and cloud can read
   streams:
   ```yaml
   rtspAddress: :8554            # or <MINIPC_LAN_IP>:8554 to bind only the LAN NIC
   ```
2. **Reader password.** Replace it. The shipped value is committed to git, so
   treat it as public:
   ```yaml
   authInternalUsers:
     - user: any
       ips: [127.0.0.1, '::1']
       permissions:
         - action: api
         - action: playback
     - user: nvr-reader
       pass: <READER_PASSWORD>
       permissions:
         - action: read
           path: '~^nvr-'
   ```
   Use letters and digits only. The password also goes into URLs in `.env`, and
   characters such as `@ : / # ? %` would need percent-encoding there.
3. **Leave `apiAddress: 127.0.0.1:9997` and `playbackAddress: 127.0.0.1:9996`
   unchanged.** The gateway reaches them over loopback, and they must never be
   exposed.

Do not add camera paths. The gateway creates them through the API on every
sweep: `nvr-<hash>` serves live/AI on demand, and `record-<hash>` records
continuously. MediaMTX forgets API-created paths when it restarts, and the next
sweep (within `DISCOVERY_INTERVAL_S`) recreates them.

## 6. Configure the gateway (`NVR\.env`)

```powershell
cd C:\noentry\Backend\NVR
Copy-Item .env.example .env
notepad .env
```

Use **forward slashes** in Windows paths. They work in both Python and
MediaMTX, and avoid escaping problems. Set these values:

```dotenv
# --- This service --------------------------------------------------------------
MINIPC_DB_PATH=C:/nvr/data/nvr.sqlite3
MINIPC_BIND_HOST=0.0.0.0
MINIPC_PORT=8000

# --- Jetson (its HTTP API) -----------------------------------------------------
JETSON_BASE_URL=http://<JETSON_LAN_IP>:8080
JETSON_TIMEOUT_S=30

# --- MediaMTX relay ------------------------------------------------------------
MEDIAMTX_API_URL=http://127.0.0.1:9997
# Not needed: mediamtx.yml grants API access to any localhost caller.
MEDIAMTX_API_USER=
MEDIAMTX_API_PASSWORD=
# What the CLOUD pulls. Must be reachable from the cloud (VPN or forwarded TCP 8554).
NVR_PUBLIC_RTSP_BASE_URL=rtsp://nvr-reader:<READER_PASSWORD>@<PUBLIC_HOST>:8554
# What the JETSON pulls, on the LAN.
MEDIAMTX_LAN_RTSP_BASE_URL=rtsp://nvr-reader:<READER_PASSWORD>@<MINIPC_LAN_IP>:8554

# --- Recording -----------------------------------------------------------------
RECORDING_ENABLED=true
RECORDING_PLAYBACK_URL=http://127.0.0.1:9996
# Interpreted by MediaMTX (it writes the files), not by Python.
RECORDING_DIRECTORY=C:/nvr/recordings
RECORDING_RETENTION=168h
RECORDING_CLIP_DIRECTORY=C:/nvr/clips
RECORDING_CLIP_RETENTION_S=604800
# Browser-reachable HTTPS base for clip links (see step 9).
RECORDING_PUBLIC_BASE_URL=https://<PUBLIC_HOST>/api
# 32+ random characters; the cloud's NVR_RECORDING_API_KEY must match.
RECORDING_API_KEY=<paste output of the command below>

# --- Discovery -----------------------------------------------------------------
DISCOVERY_ENABLED=true
DISCOVERY_INTERVAL_S=60
DISCOVERY_MISS_THRESHOLD=2
DISCOVERY_LOCAL_ENABLED=true
# Set this explicitly on Windows (see note below).
DISCOVERY_SUBNETS=192.168.1.0/24

HIK_USERNAME=admin
HIK_PASSWORD=<camera password>
HIK_RTSP_STREAM=2

# One NVR role per tower: leave NVR_HOST empty if this MiniPC is the NVR,
# or set it to the ONE physical Hikvision recorder.
NVR_HOST=
NVR_CHANNELS=
NVR_USERNAME=
NVR_PASSWORD=

# --- ffmpeg: full paths, so the service works even if PATH differs -------------
FFMPEG_BIN=C:/ffmpeg/bin/ffmpeg.exe
FFPROBE_BIN=C:/ffmpeg/bin/ffprobe.exe
```

Generate the recording key:

```powershell
-join ((48..57)+(65..90)+(97..122) | Get-Random -Count 48 | ForEach-Object {[char]$_})
```

Notes:

- **`DISCOVERY_SUBNETS`:** when it's empty, the gateway scans the /24 of
  whichever interface routes to `8.8.8.8`. On a Windows box with a VPN client,
  Hyper-V or Wi-Fi adapter, that can be the wrong network. Set the camera LAN
  explicitly (a /16 at most).
- **Physical NVR:** set `NVR_HOST=<nvr ip>`. If the recorder's ISAPI (HTTP) is
  blocked, also set `NVR_CHANNELS=1-16` so those channels stay candidates over
  RTSP. `NVR_CHANNELS` without `NVR_HOST` is a startup error.
- **Camera support:** only cameras that expose Hikvision-style
  `/Streaming/Channels/<n>` RTSP paths can be relayed and recorded. Other URL
  shapes are verified, but relay provisioning rejects them ("Cannot select
  stream profile").
- **Startup fails** if `NVR_PUBLIC_RTSP_BASE_URL` is missing, or if either RTSP
  base includes a path.

## 7. Open the firewall

Inbound only. Scope each rule to the networks that actually need it.

```powershell
# NVR HTTP API: cloud (via VPN/reverse proxy) + LAN admins
New-NetFirewallRule -DisplayName "NOENTRY NVR API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 `
  -Action Allow -Profile Private,Domain -RemoteAddress LocalSubnet,<VPN_SUBNET>

# MediaMTX RTSP: Jetson (LAN) + cloud MediaMTX (VPN / forwarded port)
New-NetFirewallRule -DisplayName "NOENTRY MediaMTX RTSP 8554" -Direction Inbound -Protocol TCP -LocalPort 8554 `
  -Action Allow -Profile Private,Domain -RemoteAddress LocalSubnet,<VPN_SUBNET>

# WS-Discovery replies from cameras (unicast UDP back to python.exe)
New-NetFirewallRule -DisplayName "NOENTRY WS-Discovery replies" -Direction Inbound -Protocol UDP `
  -Program "C:\noentry\Backend\.venv-nvr\Scripts\python.exe" -Action Allow -Profile Private,Domain -RemoteAddress LocalSubnet
```

Leave the `-RemoteAddress` scope out only if you use a port-forward from a
known cloud IP, and then use that IP instead. **Never** open 9996/9997. If the
WS-Discovery rule is blocked, the subnet sweep still finds cameras.

## 8. First run (foreground), then install as services

### 8a. Foreground test

Terminal 1:

```powershell
cd C:\mediamtx
.\mediamtx.exe C:\noentry\Backend\NVR\mediamtx.yml
```

Terminal 2:

```powershell
cd C:\noentry\Backend          # the folder ABOVE NVR
.\.venv-nvr\Scripts\python -m NVR.main
```

Expected log lines: `NVR database ready at C:/nvr/data/nvr.sqlite3`, then
Uvicorn on `0.0.0.0:8000`. If you see `ffmpeg not found: no camera can be
verified or adopted`, fix `FFMPEG_BIN`.

Check it from a third terminal:

```powershell
curl.exe http://127.0.0.1:8000/health                 # Jetson health, proxied
curl.exe http://127.0.0.1:8000/discovery/status
curl.exe -X POST http://127.0.0.1:8000/discovery/scan  # waits for a full sweep
curl.exe http://127.0.0.1:8000/discovery               # roster: relay_state / recording_state / jetson_state
curl.exe http://127.0.0.1:9997/v3/paths/list           # nvr-* and record-* paths
```

Check one relayed stream from a LAN PC, using a `source_url` from `/discovery`
with the LAN host substituted:

```powershell
ffprobe -rtsp_transport tcp "rtsp://nvr-reader:<READER_PASSWORD>@<MINIPC_LAN_IP>:8554/nvr-<hash>"
```

After a minute, `.mp4` segments should appear under `C:\nvr\recordings\record-<hash>\`.

Stop both programs with **Ctrl+C**.

### 8b. Windows services (NSSM)

```powershell
# MediaMTX
nssm install noentry-mediamtx C:\mediamtx\mediamtx.exe C:\noentry\Backend\NVR\mediamtx.yml
nssm set noentry-mediamtx AppDirectory C:\mediamtx
nssm set noentry-mediamtx AppStdout C:\nvr\logs\mediamtx.log
nssm set noentry-mediamtx AppStderr C:\nvr\logs\mediamtx.log
nssm set noentry-mediamtx AppRotateFiles 1
nssm set noentry-mediamtx AppRotateBytes 10485760
nssm set noentry-mediamtx Start SERVICE_AUTO_START

# NVR gateway (starts after MediaMTX)
nssm install noentry-nvr C:\noentry\Backend\.venv-nvr\Scripts\python.exe "-m NVR.main"
nssm set noentry-nvr AppDirectory C:\noentry\Backend
nssm set noentry-nvr AppEnvironmentExtra PYTHONUNBUFFERED=1
nssm set noentry-nvr AppStdout C:\nvr\logs\nvr.log
nssm set noentry-nvr AppStderr C:\nvr\logs\nvr.log
nssm set noentry-nvr AppRotateFiles 1
nssm set noentry-nvr AppRotateBytes 10485760
nssm set noentry-nvr DependOnService noentry-mediamtx
nssm set noentry-nvr AppStopMethodSkip 6        # stop with Ctrl+C (graceful shutdown)
nssm set noentry-nvr AppStopMethodConsole 20000
nssm set noentry-nvr Start SERVICE_AUTO_START

Start-Service noentry-mediamtx, noentry-nvr
Get-Service noentry-*
Get-Content C:\nvr\logs\nvr.log -Tail 50 -Wait
```

Rules:

- Run **exactly one** gateway process. Sweep locking and export queues live in
  that process. Never add `--workers` or `--reload`; on Windows those also
  switch Uvicorn to an event loop that can't start ffmpeg subprocesses.
- Services run as *LocalSystem* by default. LocalSystem can write to `C:\nvr`
  without changes. If you use a dedicated account instead, grant it Modify
  rights on `C:\nvr`.

## 9. Connect the cloud

1. **RTSP for live view.** The cloud MediaMTX pulls `NVR_PUBLIC_RTSP_BASE_URL`.
   Provide a VPN to the MiniPC (preferred) or forward TCP 8554 from the site
   router, restricted to the cloud's IP. RTSP is unencrypted, so prefer the VPN.
2. **HTTPS for the API and clip links.** Put a TLS reverse proxy in front of
   port 8000 that forwards `/api/*` unchanged, including the `Authorization` and
   `Range` headers, with a timeout of at least 180 s. On Windows,
   [Caddy](https://caddyserver.com) is the simplest option (`winget install
   CaddyServer.Caddy`):
   ```
   <PUBLIC_HOST> {
       reverse_proxy /api/* 127.0.0.1:8000
   }
   ```
   Caddy has no response timeout by default, so long clip exports aren't cut
   off, and it obtains the TLS certificate automatically (ports 80/443 must
   reach it).
   Install it as a service with NSSM the same way, and open TCP 443 instead of 8000.
3. **Cloud settings.** The tower Device's `device_url` must point to this
   gateway (`https://<PUBLIC_HOST>/api`), **not** to the Jetson. Set
   `NVR_RECORDING_API_KEY` in the cloud to the same value as `RECORDING_API_KEY`,
   plus `VIDEO_CLIP_PROVIDER=nvr`. [RECORDINGS.md](RECORDINGS.md) has the full
   cloud checklist.
4. **Jetson.** Run the TensorRT service, which accepts cameras through its HTTP
   API. This gateway is the only component that discovers and adopts cameras.

## 10. Storage sizing

Recording uses the camera's bitrate × camera count × retention. For example,
8 cameras at 4 Mbps for 7 days is about 2.4 TB. Retention is time-based only,
with no disk quota, so pick `RECORDING_RETENTION` to fit the disk and keep the
recordings on a separate data drive if possible (for example
`RECORDING_DIRECTORY=D:/nvr/recordings`). Check free space regularly:

```powershell
Get-PSDrive C, D | Select-Object Name, @{n='FreeGB';e={[int]($_.Free/1GB)}}
```

## 11. Upgrades

```powershell
Stop-Service noentry-nvr
# replace C:\noentry\Backend\NVR\* with the new version (keep NVR\.env and mediamtx.yml edits)
.\.venv-nvr\Scripts\pip install -r NVR\requirements.txt tzdata
Start-Service noentry-nvr
```

The SQLite database is kept. Startup creates missing tables but does **not**
migrate changed columns, so read the release notes before upgrading across a
schema change.

## 12. Troubleshooting

| Symptom | Check |
|---|---|
| Service stops immediately | `C:\nvr\logs\nvr.log`. Usually a `.env` validation error: a missing `NVR_PUBLIC_RTSP_BASE_URL`, an RTSP base with a path, `NVR_CHANNELS` without `NVR_HOST`, or a bad `RECORDING_RETENTION` (must be like `168h`) |
| `/health` returns 502 | Jetson unreachable: check `JETSON_BASE_URL`, then run `curl.exe http://<JETSON_LAN_IP>:8080/health` |
| No cameras found | Check `DISCOVERY_SUBNETS`, `HIK_PASSWORD`, and that the network profile is Private. Run `curl.exe -X POST .../discovery/scan` and read the `errors` in the report |
| Camera found but never adopted | ffmpeg can't decode a frame. Run `ffprobe -rtsp_transport tcp "<camera url>"` by hand; check `VERIFY_PER_HOST` on busy recorders |
| `relay_state: error` | MediaMTX isn't running or the API isn't on `127.0.0.1:9997`. Run `Get-Service noentry-mediamtx` and read `mediamtx.log` |
| "Cannot select stream profile" in the log | The camera URL isn't `/Streaming/Channels/<n>` (non-Hikvision), so it can't be relayed |
| `jetson_state: capacity_pending` | The Jetson is at `MAX_CAMERAS`; adoption retries on every sweep |
| Jetson can't play the stream | `rtspAddress` is still `127.0.0.1`, the 8554 firewall rule is missing, or the password in `.env` doesn't match `mediamtx.yml` |
| Clip request returns 409 | Less than 90 s of recording before the event, or a gap in the window. Check the files in `C:\nvr\recordings` and the clock (`w32tm /query /status`) |
| Clip request returns 503 | `RECORDING_API_KEY` or `RECORDING_PUBLIC_BASE_URL` is empty, or the export queue is full (retry) |
| Timezone request returns 422 | `tzdata` isn't installed in the venv |
