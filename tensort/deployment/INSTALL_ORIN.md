# Install tensort on a Jetson Orin Nano

This guide installs the tower's **detection service** (`Backend/tensort`) on a
**Jetson Orin Nano 8 GB**. In a camera tower the Jetson is passive and LAN-only:
the Windows MiniPC ([NVR gateway](../../NVR/INSTALL_WINDOWS.md)) discovers
cameras, relays their RTSP streams, and registers each camera on the Jetson
over HTTP. The Jetson decodes the streams in hardware, runs YOLO26m on
TensorRT, and serves detections on port 8080.

| Item | Value |
|---|---|
| Board | Jetson Orin Nano 8 GB (`cat /proc/device-tree/model` must say *Orin*) |
| OS | JetPack 6.x or 7.x with **TensorRT 10** (verified target: JetPack 7.2 / L4T R39 / Ubuntu 24.04 / Python 3.12) |
| Code | the current `main` branch (its `trt_infer.py` uses the TensorRT 10 API) |
| Model | `models/yolo26m.onnx`: dynamic `images[batch,3,H,W]` input, raw `output0[batch,84,anchors]` output, no NMS in the graph |
| Engine | built **on the Jetson**, FP16, 640×640, batch profile **min 1 / opt 8 / max 8** |
| Service | systemd `jetson-cameras`, Flask on `0.0.0.0:8080`, SQLite `jetson_cameras.db` |

> A TensorRT `.engine` can't be built on a PC and copied over. It is tied to the
> CPU architecture, the GPU (Orin is SM 8.7) and the exact TensorRT version.
> Only the `.onnx` is portable.

---

## 1. Flash and prepare the board

1. Flash JetPack with NVIDIA SDK Manager or the SD-card image. Booting from an
   NVMe SSD is strongly recommended; an SD card is slow for the engine build
   and the database.
2. Complete the first-boot setup, then update:
   ```bash
   sudo apt-get update && sudo apt-get -y upgrade && sudo reboot
   ```
3. Give the Jetson a **fixed LAN IP** (a DHCP reservation, or with `nmcli`).
   The MiniPC's `JETSON_BASE_URL` points at it:
   ```bash
   nmcli con show
   sudo nmcli con mod "Wired connection 1" ipv4.method manual \
       ipv4.addresses 192.168.1.20/24 ipv4.gateway 192.168.1.1 ipv4.dns 192.168.1.1
   sudo nmcli con up "Wired connection 1"
   ```
4. Enable SSH key login from your dev machine, so the remote engine build in
   section 4 doesn't prompt repeatedly:
   ```bash
   ssh-copy-id <user>@<JETSON_LAN_IP>        # run on the dev machine
   ```
5. Confirm the board and release:
   ```bash
   cat /proc/device-tree/model; echo
   head -n1 /etc/nv_tegra_release
   python3 --version
   df -h ~          # at least 12 GB free (JetPack packages + engine build)
   ```

## 2. Copy the code to the Jetson

Copy only `Backend/tensort`. The default location,
`~/1886NOENTRY/Backend/tensort`, is also the path `build_engine_remote.sh`
uses. From the repo root on the dev machine:

```bash
ssh <user>@<JETSON_LAN_IP> "mkdir -p 1886NOENTRY/Backend"
rsync -av --progress \
  --exclude '.venv*' --exclude '__pycache__' --exclude '*.engine' --exclude '*.pt' \
  --exclude '.env' --exclude 'jetson_cameras.db*' \
  Backend/tensort/ <user>@<JETSON_LAN_IP>:1886NOENTRY/Backend/tensort/
```

`.env` is excluded deliberately. Setup creates a clean one from `.env.example`
instead of shipping your dev configuration. Confirm the model arrived intact:

```bash
ls -l ~/1886NOENTRY/Backend/tensort/models/yolo26m.onnx     # 82,612,270 bytes
```

## 3. Run the setup script (on the Jetson)

```bash
cd ~/1886NOENTRY/Backend/tensort
chmod +x deployment/*.sh
MAX_BATCH=8 OPT_BATCH=4 ./deployment/setup_orin.sh 2>&1 | tee ~/tensort-setup.log
```

Run it as your normal user (it refuses to run as root and asks for sudo once,
up front). It is safe to re-run: finished work is detected and skipped, so a
second run takes about a minute. The seven steps:

| Step | What it does |
|---|---|
| 1 Preflight | Logs board/L4T/Python/RAM/disk; **stops** if < 12 GB is free or the ONNX is missing; warns about low swap and missing network |
| 2 Packages | `apt install nvidia-jetpack` (30+ min) **only if CUDA/TensorRT/trtexec are not already present**; always ensures build tools, GStreamer + `python3-gi` |
| 3 Verify | `nvcc`, `import tensorrt` (**stops unless TensorRT ≥ 10**), OpenCV **with GStreamer**, then `nvpmodel -m 0` + `jetson_clocks` |
| 4 Python | `.venv_trt` created with `--system-site-packages`, `pip install -r requirements-jetson.txt` (pycuda compiles from source on the first run) |
| 5 Engine | Runs `build_engine.sh` (10–20 min) **unless `models/yolo26m.engine.stamp` already records the same model/batch/size/TensorRT**. The build verifies the engine's real 1/opt/max profile and fails if the ONNX was static |
| 6 `.env` | Creates it from `.env.example` if missing; sets `DET_ENGINE`, `IMG_SZ`, `INFER_MAX_BATCH`, `PORT` |
| 7 Service | Installs and enables the `jetson-cameras` systemd unit, restarts it, and waits up to 180 s for `/health` → `pipeline_ready: true` |

The engine command it runs for the defaults (verified by a dry run of the
scripts):

```
trtexec --onnx=models/yolo26m.onnx --saveEngine=models/yolo26m.engine --fp16 \
        --memPoolSize=workspace:3072 \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:8x3x640x640 \
        --maxShapes=images:8x3x640x640
```

You can override these settings through environment variables:

| Variable | Default | Notes |
|---|---|---|
| `MODEL` | `yolo26m` | basename under `models/`; must be a **dynamic-batch** ONNX (`yolo26n.onnx` is static, batch 1) |
| `MAX_BATCH` | `8` | 1–8. The service never batches more than 8 |
| `OPT_BATCH` | `MAX_BATCH` | 1–`MAX_BATCH`. The batch TensorRT tunes kernels for |
| `IMG_SZ` | `640` | must equal `IMG_SZ` in `.env` |
| `WORKSPACE_MB` | `3072` (`1024` if RAM < 6 GB) | trtexec scratch memory |
| `FORCE_ENGINE=1` | — | rebuild even if an identical engine exists |
| `SKIP_APT=1` / `SKIP_ENGINE=1` / `SKIP_SERVICE=1` | — | skip a step outright (rarely needed: the script already skips finished work) |

**Choosing the batch profile:** the minimum is always 1 (a single camera's
frame must still run). The pipeline pools frames while the GPU is busy, so with
8 cameras the steady-state batch is 8, and opt = max = 8 is right. For a tower
with fewer cameras, set both to the camera count (e.g.
`MAX_BATCH=4 OPT_BATCH=4`). That uses a smaller engine with kernels tuned for
the real batch.

## 4. (Re)building only the engine from your dev machine

`build_engine_remote.sh` copies the ONNX and `build_engine.sh` to the Jetson
over SSH, builds there, and runs a throughput benchmark. Use it to rebuild
after changing the model or batch profile. It needs JetPack (`trtexec`) on the
Jetson already, which step 3 installs.

```bash
cd Backend/tensort
./deployment/build_engine_remote.sh <user>@<JETSON_LAN_IP>                    # yolo26m, 1/8/8
OPT_BATCH=4 MAX_BATCH=8 ./deployment/build_engine_remote.sh <user>@<JETSON_LAN_IP>
FETCH=1 ./deployment/build_engine_remote.sh <user>@<JETSON_LAN_IP>            # also archive the .engine locally
REMOTE_DIR=/opt/tensort ./deployment/build_engine_remote.sh <user>@<JETSON_LAN_IP>
```

It asks for the Jetson's sudo password once (to pin the clocks). Building on
the Jetson itself is equivalent:

```bash
cd ~/1886NOENTRY/Backend/tensort
MODEL=yolo26m MAX_BATCH=8 OPT_BATCH=8 ./deployment/build_engine.sh
```

Neither command edits `.env` or restarts the service. Afterwards:

```bash
cd ~/1886NOENTRY/Backend/tensort
grep -E '^(DET_ENGINE|IMG_SZ|INFER_MAX_BATCH)=' .env   # INFER_MAX_BATCH must be <= MAX_BATCH
sudo systemctl restart jetson-cameras
```

**Read the throughput gate** printed at the end of the build:

```
images/sec      = MAX_BATCH × 1000 / (GPU Compute Time mean, ms)
per-camera FPS  = images/sec / camera_count
```

Set `DEFAULT_SAMPLE_FPS` in `.env` to the per-camera FPS, rounded down. The
shipped value of 3 is an estimate for yolo26m on 8 cameras.

## 5. Configure for the camera tower

Edit `~/1886NOENTRY/Backend/tensort/.env`:

```dotenv
# The MiniPC owns discovery. The Jetson only accepts cameras pushed over HTTP.

DET_ENGINE=./models/yolo26m.engine
IMG_SZ=640
INFER_MAX_BATCH=8          # <= engine max batch
MAX_CAMERAS=8              # enabled-camera admission limit; a 9th POST returns 409
DEFAULT_SAMPLE_FPS=3       # from the throughput gate (section 4)
CONF=0.18                  # keep BELOW the cloud tracker's low_th (0.20)
PORT=8080
```

Camera discovery runs exclusively on the MiniPC. The Jetson accepts camera
configurations over HTTP and restores them from its database after restart.

Apply the change:

```bash
sudo systemctl restart jetson-cameras
```

Restrict the API to the MiniPC. It has no authentication:

```bash
sudo ufw allow OpenSSH
sudo ufw allow from <MINIPC_LAN_IP> to any port 8080 proto tcp
sudo ufw enable
```

## 6. Verify

```bash
cd ~/1886NOENTRY/Backend/tensort

# 1. The engine really has the 1/opt/max profile
.venv_trt/bin/python3 - <<'PY'
import tensorrt as trt
e = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(open("models/yolo26m.engine","rb").read())
print("TensorRT", trt.__version__, "| min/opt/max:", e.get_tensor_profile_shape("images", 0))
PY
# expect: [(1, 3, 640, 640), (8, 3, 640, 640), (8, 3, 640, 640)]

# 2. Service is up and batching is on
curl -s http://localhost:8080/health | python3 -m json.tool | grep -E 'pipeline_ready|engine_max_batch|"max_batch"|inference_ready'
# expect: pipeline_ready true, engine_max_batch 8, max_batch 8

# 3. Batched results match single-frame results (needs a JPEG with people/cars in it)
.venv_trt/bin/python3 tests/diag_batch.py frame.jpg 8 --seconds 15

# 4. Hardware capture path
.venv_trt/bin/python3 tests/test_native_capture.py
```

Once the MiniPC has adopted the cameras (`curl -s localhost:8080/cameras`),
check the live load after everything warms up:

```bash
.venv_trt/bin/python3 deployment/check_capacity.py --cameras 8 --fps 3 --seconds 120
tegrastats      # GPU (GR3D_FREQ) load, RAM, temperatures
```

`check_capacity.py` exits non-zero if rates, latency, connectivity or drop
counters fail its checks. If `pool_evicted_total` in `/health` keeps climbing,
`DEFAULT_SAMPLE_FPS` is too high.

## 7. Day-to-day operation

```bash
sudo systemctl status jetson-cameras
sudo journalctl -u jetson-cameras -f
sudo systemctl restart jetson-cameras
curl -s http://localhost:8080/cameras
```

- **Configuration:** the unit loads `.env` through `EnvironmentFile`, so restart
  after every edit. When running `main.py` by hand, load it first:
  `set -a; . ./.env; set +a; .venv_trt/bin/python3 main.py`.
- **Power mode:** every service start runs `nvpmodel -m 0` and `jetson_clocks`.
  Mode numbers differ between boards and JetPack releases. Run
  `sudo nvpmodel -q` and see `/etc/nvpmodel.conf` for the list. If you want a
  different mode, such as MAXN SUPER, change the `ExecStartPre=+/usr/sbin/nvpmodel -m 0`
  line in `/etc/systemd/system/jetson-cameras.service`, then
  `sudo systemctl daemon-reload`. Re-running `setup_orin.sh` regenerates the
  unit with mode 0. Provide proper cooling and power for sustained load.
- **One process only:** a second copy would open the cameras and the GPU twice.
- **Upgrading the code:** rsync as in section 2, then re-run
  `./deployment/setup_orin.sh` with the same `MAX_BATCH`/`OPT_BATCH`. It skips
  JetPack and the engine when nothing changed, and rebuilds the engine only when
  the model, batch profile, `IMG_SZ` or TensorRT version differs from
  `models/yolo26m.engine.stamp`. **A JetPack upgrade always triggers a rebuild.**

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Setup stops with "TensorRT 8.x found" | Wrong JetPack. This code needs TensorRT 10 (JetPack 6+) |
| Setup stops with "OpenCV … GStreamer:NO" | A pip `opencv-python` wheel is shadowing JetPack's OpenCV, which kills hardware decode. Run `.venv_trt/bin/pip uninstall -y opencv-python opencv-python-headless` and **never** pip-install opencv or numpy here |
| Build stops with "profile … != requested" or "FIXED batch axis" | The ONNX is static (e.g. `yolo26n.onnx`). Re-export with `dynamic=True` or use `MODEL=yolo26m` |
| Setup stops with "run as your normal user" | You used `sudo ./setup_orin.sh`. Run it without sudo; it asks for the password itself |
| trtexec killed during the build | Out of memory. Add swap (`sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`) or use `WORKSPACE_MB=2048` |
| Engine fails to deserialize at startup | It was built on another device or another TensorRT version. Rebuild on this Jetson |
| `/health` shows `max_batch: 1` | The engine came from a static ONNX (e.g. yolo26n), or `INFER_MAX_BATCH=1`. Rebuild from `yolo26m.onnx` |
| `pipeline_ready` stays false | `journalctl -u jetson-cameras -n 80`. Usually a wrong `DET_ENGINE` path or an `IMG_SZ` mismatch |
| Cameras never connect | The MiniPC's `MEDIAMTX_LAN_RTSP_BASE_URL` must use the MiniPC LAN IP, and MediaMTX must listen on `:8554` rather than `127.0.0.1`. Test with `gst-launch-1.0 rtspsrc location=rtsp://nvr-reader:…@<MINIPC_LAN_IP>:8554/nvr-<hash> ! fakesink` |
| A 9th camera returns 409 | `MAX_CAMERAS` admission limit; raise it only after benchmarking |
| Settings seem ignored | `.env` wasn't loaded. Run under systemd, or source it before `main.py` |

Further reading: [README.md](README.md) (capacity tuning),
[../ARCHITECTURE.md](../ARCHITECTURE.md) (frame path, `/health` fields),
[../doc/07_TROUBLESHOOTING.md](../doc/07_TROUBLESHOOTING.md).
