# Jetson Orin Nano 8 GB inference service

This service decodes camera streams, runs YOLO detection, and sends detection
metadata to the cloud over SSE. Tracking, recordings, and alerts live in the
cloud backend.

The supported ceiling is **eight enabled cameras**. Start at **3 detection FPS
per camera**, using `yolo26m`, FP16, 640-pixel model input, and low-resolution
camera substreams. This is a commissioning target, not a measured throughput
claim. `yolo26m` is the accuracy-first model — roughly 2–2.5x the compute of
`yolo26s` — so the per-camera frame rate is traded for detection quality; drop
to `yolo26n` if a site needs the frame rate back. Resolution, codec, scene density, power mode, cooling, and JetPack all
influence capacity. Eight 1080p or 4K main streams cost much more to decode even
when inference samples only five frames per second.

## Frame path

```text
8 capture threads -> latest-frame handoff -> per-camera FIFO FramePool
                  -> one ordered GPU worker -> detections -> SSE
                                             -> bounded JPEG worker
```

Each capture thread retains at most one frame awaiting its event-loop callback;
the asyncio output queue also defaults to one frame. The shared FramePool holds
16 frames, discards frames older than 700 ms, and evicts the oldest frame from
the camera holding the largest backlog when full. Batches draw one frame from
each ready camera before taking a second from any camera. A missing camera never
holds up the others.

The dispatcher waits for worker queue capacity before collecting a batch. One
batch can execute and one can wait. There is one TensorRT worker regardless of
legacy `INFER_NUM_WORKERS` settings: this keeps a camera's results ordered and
avoids duplicating engine memory and CUDA contexts on one GPU. Camera replacement
invalidates old results so a removed/reconfigured camera cannot refill the cache.

TensorRT resolves output shapes at the profile's maximum batch and the configured
image size before allocating reusable pinned host/device buffers. Frames are
letterboxed directly into the input buffer. The batch parser reads output buffer
views before the next inference overwrites them, avoiding another output copy.
Initialization must succeed before the runtime becomes ready. A static batch-one
engine is supported, but `/health` reports that actual limit.

Snapshots use one encoding thread and at most one pending job per camera. Slow
SSE clients discard their oldest queued message when full. Both paths stay
bounded when consumers cannot keep up.

## Files

| File | Responsibility |
|---|---|
| `main.py` | Load environment and create the Flask application |
| `runtime.py` | Serialize camera mutations, enforce capacity, persist/restore cameras |
| `limits.py` | Shared ceiling of eight enabled cameras |
| `pipeline.py` | Frame pool, ordered inference queue, results and snapshot cache |
| `channels/channel.py` | Capture, GStreamer pipelines, reconnect and frame handoff |
| `trt_infer.py` | CUDA ownership, TensorRT buffers, YOLO preprocessing/postprocessing |
| `service.py`, `discovery.py` | Camera discovery and adoption |
| `deployment/setup_orin.sh` | Device setup and local engine build |

## Capture and deployment

Use JetPack's OpenCV with GStreamer support. Do not install `opencv-python` on
the Jetson: its wheel can shadow that build. The runtime sets OpenCV's CPU thread
count to one to avoid nested thread pools across eight cameras.

RTSP tries hardware decode before CPU fallbacks. Compressed packets are never
intentionally dropped before the decoder; decoded frames can be dropped safely.
GStreamer `videorate` limits BGR conversion to the sample rate. Hardware scaling
still happens before that gate, and the original stream is still decoded in
full, so camera substreams are the main way to reduce decode load. Generic
`uridecodebin` selects its decoder automatically; the `*-hw` backend name reports
the selected pipeline, not proof of which decoder it auto-plugged.

`DEFAULT_RESIZE_W/H` defaults to 640x360. GStreamer scales to those exact dimensions;
choose dimensions matching the camera aspect ratio (640x480 for 4:3 sources), or
set per-camera `resize` explicitly. The OpenCV fallback fits inside those bounds.
Detection coordinates always refer to the emitted frame size.

Use a JetPack release for this board that provides **TensorRT 10**; check the
installed version rather than assuming every JetPack 6 release does. Build the
engine on the actual device. Engines supplied in `models/` must be rebuilt for
the target GPU and TensorRT version. Export ONNX with a dynamic batch axis, then:

```bash
cd Backend/tensort
MODEL=yolo26m MAX_BATCH=8 ./deployment/build_engine.sh
# Full install instead:
# MODEL=yolo26m MAX_BATCH=8 ./deployment/setup_orin.sh
```

`.env.example` contains the recommended starting settings. On upgrade, existing
`.env` and stored per-camera FPS/resize settings are retained: update both as
needed. The setup script updates engine path, input size and batch size to match
its build. `MAX_CAMERAS` can lower the ceiling but cannot raise it above eight.
POST or PATCH enabling a ninth camera returns HTTP 409. Disabled configurations
do not consume a slot. On restore, excess enabled rows stay on disk and are
logged as skipped. Discovery retries unadopted cameras after capacity becomes free.

## Verify eight-camera capacity on the device

1. Select the appropriate power mode using `sudo nvpmodel -q` and the board's
   documented modes; use adequate cooling. Observe `tegrastats` during testing.
2. Run the engine/preprocessing diagnostic using a representative camera image:

   ```bash
   .venv_trt/bin/python tests/diag_batch.py frame.jpg 8 --seconds 15
   ```

   This checks batch rows and reports preprocessing + inference + parsing FPS.
   It excludes stream decoding and network delivery. Leave at least 25% headroom
   over the desired aggregate rate (40 FPS for eight cameras at five FPS).
3. Add all eight real streams, wait for initialization/warm-up, then run:

   ```bash
   .venv_trt/bin/python deployment/check_capacity.py --cameras 8 --fps 5 --seconds 120
   ```

   The check measures per-camera result rates, sampled frame age, failures,
   connectivity, and frame-pool drops through `/health`. A pass applies only to
   that run. Repeat for at least 30 minutes in representative busy scenes and
   verify SSE delivery/overlays from the cloud. Unplug one camera and confirm
   the other seven continue; reconnect it and confirm recovery.
4. If rates fall short or drops grow, reduce each camera's `sample_fps`, reduce
   source resolution/rate, and check decoder fallback and thermal throttling.
   Raising queue depth increases latency; it does not increase GPU throughput.

## Health fields

Statistics are under `stats` in `/health`.

| Field | Meaning |
|---|---|
| `inference_ready` | Engine initialized and pipeline running |
| `max_cameras`, `channel_count` | Configured ceiling and enabled channels |
| `engine_max_batch`, `max_batch` | Engine capacity and effective dispatch batch limit |
| `capture` | Per-camera connection, selected pipeline, sample rate, capture/handoff counters |
| `cameras` | Per-camera successful results, last result time, sampled frame age and sequence |
| `pool_depth`, `pool_capacity` | Bounded backlog |
| `pool_evicted_total`, `pool_expired_total` | Capacity and stale-frame drops |
| `infer_fail`, `infer_dropped` | Failed inference and refused worker jobs |
| `snapshot_pending` | At most one pending JPEG per active camera generation |

Empty detection events remain enabled so the cloud tracker can age tracks.
Inference failures are counted and normally withheld from SSE. Keep `CONF` low
enough for the cloud tracker's low-confidence association (default 0.20).

## Off-device verification

The test files run separately because older suites install import stubs globally.
No GPU is required. Discovery's integration tests need permission to bind a local
HTTP server; route tests additionally need Flask.

```bash
python3 tests/test_frame_pool.py
python3 tests/test_dispatch.py
python3 tests/test_cross_class_dedupe.py
python3 tests/test_orin_capacity.py
python3 tests/test_trt_shapes.py
python3 tests/test_discovery.py
```

The code uses the TensorRT 10 API; the legacy `setup_nano.sh` does not make this
revision compatible with TensorRT 8. Use a compatible legacy revision for the
original Jetson Nano.
