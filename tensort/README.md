# Camera capture and detection

This Jetson service accepts camera configurations from the [NVR gateway](../NVR/ARCHITECTURE.md), persists them, captures frames, runs TensorRT inference, and publishes detections and JPEG snapshots. NVR owns network discovery, verification, camera identity, and adoption. The cloud handles tracking and alerts.

## Read the code in this order

| File | Responsibility |
| --- | --- |
| `main.py` | Load `.env`, create the runtime, and register HTTP routes |
| `runtime.py` | Own the event loop; validate, restore, add, update, and remove cameras |
| `pipeline.py` | Connect capture, buffering, inference results, snapshots, and subscribers |
| `frame_pool.py` | Keep a bounded, fair queue of recent frames |
| `inference_worker.py` | Run ordered batches on the GPU's dedicated thread |
| `trt_infer.py` | Prepare model input, execute TensorRT, and parse detections |
| `channels/channel.py` | Open streams, sample frames, and reconnect after failures |
| `channels/gstreamer_capture.py` | Read native GStreamer samples into owned BGR arrays |
| `repositories/` | Read and write saved camera configurations |
| `routes/` | Translate HTTP requests into runtime calls |
| `lifecycle.py` | Drain workers when the standalone server exits |

`env_utils.py` provides shared environment parsing. `limits.py` defines camera
admission policy. `database_orm.py` defines the camera configuration table; `database.py`
and the migration files preserve older installations.

## Startup

1. `main.py` loads `.env` before importing modules that read configuration.
2. `create_app()` constructs `PipelineRuntime` unless a caller supplies one.
3. The runtime initializes SQLite and starts an asyncio loop on a background thread.
4. `SimpleInferencePipeline.start()` starts the GPU worker and waits for TensorRT
   initialization. A failed engine load fails startup instead of reporting ready.
5. The runtime restores saved camera configurations. Disabled cameras remain
   saved without starting capture; enabled cameras must fit `MAX_CAMERAS`.
6. Flask routes use the runtime stored in
   `app.extensions`, so separate application instances keep separate state.

Run one service process per GPU. The development reloader is disabled because
it would start a second runtime.

## Camera registration

NVR verifies a stream and sends its `camera_uuid`, relay `source_url`, and settings to `POST /cameras`. TensorRT validates capacity, starts capture, and saves the configuration. It restores saved configurations on every restart. Disabled cameras remain saved without consuming capture capacity. NVR can update a source using `PATCH /cameras/<camera_uuid>`.

Use NVR for `/discovery*` and `POST /sync`; these routes do not exist on TensorRT. Cloud tower devices must point to the NVR gateway, which proxies camera and detection APIs.

Older Jetson databases may retain an unused `discovered_cameras` table; this service no longer reads, creates, or migrates it. Saved camera configurations and UUIDs remain intact.

## Detection, step by step

```mermaid
flowchart LR
    A[Capture thread per camera] --> B[Latest-frame handoff]
    B --> C[FramePool]
    C --> D[Ordered GPU worker]
    D --> E[Result callback]
    E --> F[SSE detection subscribers]
    E --> G[Latest result cache]
    E --> H[Bounded JPEG encoder]
```

1. `VideoChannel` opens the configured stream, preferring hardware decode where
   supported. Its capture thread samples frames at `sample_fps` and reconnects
   with backoff after failures.
2. The handoff keeps only the latest pending frame. `_pump_channel()` puts
   detection-enabled frames into `FramePool` on the asyncio thread.
3. `FramePool` discards stale frames and evicts the oldest frame from the camera
   with the largest backlog when full. Batches take one frame from each ready
   camera before taking a second. A silent camera never blocks others.
4. `_pump_inference()` waits for worker queue capacity before drawing a batch.
   `_prepare_frame_job()` attaches camera identity, sequence, timestamp, and a
   completion callback to each frame.
5. `InferenceWorker` runs the batch on its CUDA-owning thread. `TRTInfer` resizes
   and letterboxes images, normalizes RGB input, executes TensorRT, and maps
   filtered detections back to each frame's dimensions.
6. `_handle_result()` updates metrics and the latest result, broadcasts the event,
   and optionally schedules a JPEG. One encoder thread keeps snapshot work bounded.
7. Slow SSE clients discard their oldest queued event. Empty detection events
   normally remain enabled so the cloud tracker can age objects that disappeared.

The single GPU worker preserves per-camera ordering and avoids duplicate engines.
The dispatcher respects the engine's actual batch limit. Timeout failures are
counted; successful late results can still be delivered. Camera replacement
changes a generation token, preventing old results from repopulating its cache.

Do not remove the queue bounds, cancellation handling, generation checks, or
reconnect logic merely to shorten the code. They keep a busy or disconnected
camera from disrupting other cameras.

## Configuration and running

Use [`.env.example`](.env.example) as the configuration reference. Keep actual
credentials in `.env`. The main groups are:

| Settings | Purpose |
| --- | --- |
| `DET_ENGINE`, `IMG_SZ`, `CONF` | Model engine, input size, confidence threshold |
| `MAX_CAMERAS`, `DEFAULT_SAMPLE_FPS` | Admission limit and default sampling rate |
| `DEFAULT_RESIZE_W`, `DEFAULT_RESIZE_H` | Capture resize bounds |
| `FRAME_POOL_CAP`, `FRAME_MAX_AGE_MS`, `INFER_MAX_BATCH` | Bounded inference workload |
| `ENABLE_SNAPSHOT_CACHE`, `SNAPSHOT_*` | JPEG cache behavior |

Saved per-camera settings survive restart. Editing a default in `.env` does not
rewrite existing camera configurations.

Follow [deployment/README.md](deployment/README.md) to install JetPack dependencies
and build the engine on the target device. Then run from this directory:

```bash
.venv_trt/bin/python main.py
```

`requirements-jetson.txt` is for the current TensorRT 10 service. The legacy Nano
setup and `requirements.txt` are retained for historical deployment reference;
they do not make this revision compatible with TensorRT 8.

## Verification and troubleshooting

Run all off-device regression suites with one command:

```bash
python3 tests/run_tests.py
# Or selected suites:
python3 tests/run_tests.py test_frame_pool.py test_dispatch.py
# Camera registration, restart, removed routes, and database compatibility:
python3 ../tests/test_tensort_registration.py
```

Use a Python environment with Flask, NumPy, OpenCV, SQLAlchemy, and aiosqlite.
On the Jetson, use JetPack's OpenCV and NumPy. The runner starts each suite in a
separate process because tests stub GPU and capture imports. They do not scan
or modify real cameras.

The tests cover API registration, startup restoration, queue fairness, late results, capacity, reconnect behavior, native sample handling,
route isolation, and shutdown. They cannot prove GPU throughput or real-camera
compatibility. Use `tests/diag_batch.py` and `deployment/check_capacity.py` on the
Jetson as described in [ARCHITECTURE.md](ARCHITECTURE.md).

Start debugging with `/health`: check inference readiness, capture status, frame
age, dropped frames, and per-camera results. Use `deployment/audit_cameras.py`
to inspect saved configurations; optionally provide `--expected-urls streams.json`
with a JSON list of expected relay URLs. Diagnose discovery on the NVR gateway.
Keep the model assets and migration scripts until the deployment no longer needs
them; the current default model is not enough evidence that alternatives are unused.
