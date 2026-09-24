# Deploy on Jetson Orin Nano 8 GB

This revision defaults to **eight enabled cameras**, configurable with
`MAX_CAMERAS`, and requires the
TensorRT 10 API. Use JetPack for the Orin Nano that supplies TensorRT 10; verify
`python3 -c 'import tensorrt; print(tensorrt.__version__)'`. Earlier JetPack
releases with TensorRT 8 need a compatible older code revision.

Start with camera substreams at 640x360 (16:9) or 640x480 (4:3), 5–10 source FPS,
and **3 detection FPS per camera**. The default model is `yolo26m`, FP16 at 640 —
the accuracy-first option, roughly 2–2.5x the compute of `yolo26s`. Eight cameras
at three detection FPS request 24 images/sec. The device must be benchmarked with
all eight actual streams before accepting that rate; use `MODEL=yolo26n` if the
measured throughput is short.

## Install

Copy the service and ONNX model to the device, then run on the Jetson:

```bash
cd tensort/deployment
./setup_orin.sh
```

The script installs dependencies, creates a virtualenv using system packages,
builds an engine on this device, writes `.env` if absent, and starts the
`jetson-cameras` systemd service. The model build requires a dynamic-batch ONNX:

```bash
# Export in a model-development environment; copy the ONNX to the Jetson.
yolo export model=yolo26m.pt format=onnx dynamic=True imgsz=640 simplify=True nms=False
```

An engine compiled for another device or TensorRT version must be rebuilt.
Never install `opencv-python` in the Jetson environment: use JetPack's OpenCV
with GStreamer enabled. The setup checks for a shadowing wheel.

| Override | Purpose |
|---|---|
| `MODEL=yolo26m` | Model basename under `models/` |
| `MAX_BATCH=8` | Maximum batch size, 1–8 |
| `OPT_BATCH=8` | Optimization batch, no larger than MAX_BATCH |
| `IMG_SZ=640` | Model input size |
| `SKIP_APT=1` | Skip packages already installed |
| `SKIP_ENGINE=1` | Retain an engine already built on this device |
| `SKIP_SERVICE=1` | Skip systemd changes |

For just the engine: `MODEL=yolo26m MAX_BATCH=8 ./deployment/build_engine.sh` from
`tensort/`. Select the board's appropriate power mode; numeric `nvpmodel` mode
IDs are board-specific. Setup retains its existing mode-0/clocks commands, so
verify the active mode using `sudo nvpmodel -q`. Supply adequate power and cooling.

## Upgrade configuration

Existing `.env` files and database camera settings are retained. Review them
against `.env.example`; setup only reconciles engine path, image size, batch size
and port. Recommended starting values:

```dotenv
MAX_CAMERAS=8
INFER_NUM_WORKERS=1
INFER_MAX_BATCH=8
FRAME_POOL_CAP=16
DEFAULT_SAMPLE_FPS=5
DEFAULT_RESIZE_W=640
DEFAULT_RESIZE_H=360
CONF=0.20
```

`INFER_NUM_WORKERS` values above one are reduced to one to keep results ordered.
`MAX_CAMERAS` is the enabled-camera admission limit; eight is a starting profile,
not a hardcoded ceiling. A limit of 20 and eight configured channels still
provides at most eight candidates. `INFER_MAX_BATCH` separately limits batches
and is bounded by the engine's actual capacity.
Stored per-camera FPS and resize settings take precedence over defaults; update
them with PATCH if you are reducing an existing installation's load.

Native GStreamer capture requires system packages `python3-gi`,
`gir1.2-gstreamer-1.0` and `gir1.2-gst-plugins-base-1.0`, visible through the
service's `--system-site-packages` environment. Setup installs these packages.
The supported data path is appsink → owned BGR NumPy frame → inference;
OpenCV's GStreamer video-capture bridge is not used.

Discovery confirms a candidate only after receiving a decoded frame, including
static NVR channels. Edge and cloud upgrades add nullable `first_frame_at`
columns without deleting existing cameras. Old rows remain unverified until
frames arrive. Deploy the matching cloud changes to retain verification state
in inventory and avoid adopting cameras the edge rejected for capacity.
Sync returns the last completed report while a new scan runs in the background.

Validate on the Jetson using `python3 tests/test_native_capture.py`, then inspect
`/health`: capture entries should report frame ages and increasing frame counts.
Run `python3 deployment/check_capacity.py --cameras 8 --fps 3 --seconds 120`
after all eight streams have warmed up. The native tests exercise synthetic
video; only this live check can validate NVR stability and hardware throughput.

## Cameras and health

```bash
curl -X POST http://localhost:8080/cameras \
  -H 'Content-Type: application/json' \
  -d '{"camera_uuid":"front-door","source_url":"rtsp://user:pass@192.168.1.50:554/stream2"}'
curl -X PATCH http://localhost:8080/cameras/front-door \
  -H 'Content-Type: application/json' \
  -d '{"sample_fps":5,"resize":[640,360]}'
curl http://localhost:8080/health
```

A ninth enabled camera returns HTTP 409. Disabled camera configurations do not
consume a slot. When an old database contains more than eight enabled cameras,
excess rows are retained on disk and skipped during restore. Discovery follows
the same limit and retries cameras awaiting admission.

`/health` must show `pipeline_ready: true`. Within `stats`, check
`engine_max_batch`, `max_batch`, `capture`, `cameras`, `pool_evicted_total`,
`pool_expired_total`, and `infer_fail`. `channel_count` counts enabled channels;
`capture.<id>.connected` indicates whether each is reading frames.

Both `/cameras/detections/stream` and per-camera SSE streams are supported.
Other endpoints include DELETE `/cameras/<id>`, GET
`/cameras/<id>/latest`, and GET `/cameras/<id>/snapshot.jpg`. All routes also have
an `/api` prefix alias. GET `/cameras` lists configurations.

## Commission all eight cameras

After the service warms up:

```bash
cd tensort
.venv_trt/bin/python tests/diag_batch.py frame.jpg 8 --seconds 15
.venv_trt/bin/python deployment/check_capacity.py --cameras 8 --fps 5 --seconds 120
```

The first command needs an image with detectable objects. It checks single,
partial and full batches, then times preprocessing, inference and postprocessing.
The second measures actual capture-to-result operation through `/health` and
returns a nonzero status if rates, latency, connectivity or drop counters fail
its checks. It does not verify cloud SSE consumption. Repeat under busy scenes
for at least 30 minutes and inspect the cloud overlays as well as `tegrastats`.

If results lag, first lower stored camera `sample_fps`, select lower-resolution
substreams, and check whether capture fell back to CPU decoding. Larger queues
retain older frames; they do not add GPU throughput.

```bash
sudo systemctl status jetson-cameras
sudo journalctl -u jetson-cameras -f
sudo systemctl restart jetson-cameras
tegrastats
```

The HTTP service has no authentication. Keep deployment behind the existing
camera LAN/VPN access controls. Run one service process per GPU; a multi-process
web-server configuration would start duplicate camera capture and inference.
For a WSGI server use the `main:create_app()` factory with one worker.

See [ARCHITECTURE.md](../ARCHITECTURE.md) for the frame path, health fields and
regression-test commands.
