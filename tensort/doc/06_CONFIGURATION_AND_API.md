# 06 — Configuration, APIs, persistence, and deployment

## Which value actually runs?

There are four layers to distinguish: code fallback, example environment file, deployed process environment, and saved per-camera settings. Editing `.env.example` changes no running service. Editing `DEFAULT_SAMPLE_FPS` does not rewrite existing camera configurations.

Example:

```text
runtime.py fallback DEFAULT_SAMPLE_FPS = 5
.env.example DEFAULT_SAMPLE_FPS = 3
Saved camera config sample_fps = 6
Restored camera sample_fps = 6, subject to current normalization/clamp
```

Some values are read at module import; others at object construction. Treat changes as requiring an orderly restart unless an explicit per-camera API handles them. The service does not implement a universal dynamic environment reload.

## Inference and buffering settings

| Setting | Code fallback | `.env.example` | Meaning |
| --- | --- | --- | --- |
| `DET_ENGINE` | Module-relative `models/yolo26m.engine` | `./models/yolo26m.engine` | Actual compiled model artifact |
| `IMG_SZ` | 640 | 640 | Square model input side; must fit engine |
| `CONF` | 0.20 in `build_default` | 0.18 | Edge confidence cutoff |
| `IOU` | 0.45 | 0.45 | Raw-output NMS threshold, not the YOLO26 six-column branch |
| `NMS_TOPK` | 50 | 50 | Maximum final detections per image |
| `ALLOWED_CLASSES` | `person,car,motorcycle,truck` | Same | COCO labels retained |
| `CUDA_DEVICE` | 0 | 0 | CUDA device ordinal |
| `DEDUPE_CROSS_CLASS` | true | true | Custom duplicate suppression |
| `DEDUPE_IOU` | 0.55 | 0.55 | IoU duplicate threshold |
| `DEDUPE_OVERLAP` | 0.70 | 0.70 | Intersection relative to smaller box |
| `DEDUPE_SIZE_RATIO` | 0.65 | 0.65 | Similar-size requirement for overlap rule |
| `MAX_CAMERAS` | 8, at least 1 | 8 | Selection/admission policy; runtime has no hard upper ceiling |
| `INFER_NUM_WORKERS` | 1 | 1 | Values above 1 are overridden to one worker |
| `INFER_MAX_BATCH` | Camera limit | 8 | Further capped by camera policy and engine profile |
| `INFER_QUEUE_MAX` | 1 | 1 | Waiting **batches**, excluding executing batch |
| `INFER_BATCH_LINGER_MS` | 10 | 10 | Brief top-up wait for an underfilled pool |
| `INFER_RESULT_TIMEOUT_S` | 3 | 4 | Future leak/late-result guard, not a GPU deadline |
| `INFER_ERROR_LOG_INTERVAL_S` | 10 | Not set | Repeated failure log throttling |
| `FRAME_POOL_CAP` | Configured max batch × 2, minimum camera limit | 16 | Waiting images in pool |
| `FRAME_MAX_AGE_MS` | 700 | 1000 | Pool expiry age; zero disables expiry |
| `CHANNEL_OUT_Q_MAX` | 1 | 1 | Async output queue per capture |
| `EMIT_EMPTY_DETECTIONS` | true | true | Publish successful zero-object frames |
| `EMIT_FAILED_EVENTS` | false | false | Publish failed inference events if enabled |
| `OPENCV_NUM_THREADS` | 1, minimum 1 | 1 | OpenCV thread setting in runtime |

Constructors such as `TRTInfer(...)` and `YoloV8DetTRT(...)` have their own defaults, but normal service startup uses `build_default()`. Documentation should name that entry point when quoting production fallbacks.

## Capture and snapshot settings

| Setting | Code fallback | Example file | Interpretation |
| --- | --- | --- | --- |
| `DEFAULT_SAMPLE_FPS` | 5 | 3 | New-camera default |
| `MAX_SAMPLE_FPS` | 12 | 8 | Normalization ceiling, not measured throughput |
| `DEFAULT_RESIZE_W/H` | 640/360 | 640/360 | Capture resize; nonpositive defaults disable it |
| `GST_LATENCY_MS` | 200 | Not set | RTSP jitter buffering request |
| `RECONNECT_BASE_MS` | 1000 | Not set | Initial reconnect delay before random factor |
| `RECONNECT_MAX_MS` | 30000 | Not set | Backoff ceiling before random factor |
| `CAMERA_START_STAGGER_S` | 1.5 | Not set | Process-wide spacing between capture opens |
| `CAPTURE_PROBE_S` | 12 | Not set | First candidate verification budget |
| `GRAB_MISS_TOLERANCE` | 15 | Not set | Consecutive unsuccessful grabs before reconnect |
| `OPENCV_FFMPEG_CAPTURE_OPTIONS` | TCP plus timeout/delay options set with `setdefault` | Not set | Existing process value wins |
| `ENABLE_SNAPSHOT_CACHE` | true | true | Enable async JPEG caching |
| `SNAPSHOT_ON_DETECTION_ONLY` | true | true | Empty successful frames do not refresh JPEG |
| `SNAPSHOT_MIN_INTERVAL_MS` | 1000 | 1000 | Minimum frame-timestamp gap for cache refresh |
| `SNAPSHOT_MAX_EDGE` | 640 | 640 | JPEG resize bound |
| `SNAPSHOT_JPEG_QUALITY` | 65 | 65 | JPEG quality, capped at 100 |

Per-camera fields include `sample_fps`, `resize`, `decode_backend`, `gst_latency_ms`, `rtsp_transport`, `gst_decoder`, reconnect bounds, `enabled`, and `detection_enabled`. Unknown fields are deliberately tolerated by `VideoChannelConfig` because cloud payloads carry more metadata.

`notification_enabled` is stored on the edge but is not a filter applied by `_pump_channel()`; the cloud applies notification policy. `enabled=false` removes the live edge channel while keeping the configuration through the add/patch flow. `detection_enabled=false` leaves capture running but keeps its events out of the inference pool.

## Discovery settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `DISCOVERY_ENABLED` | true | Enable scanner |
| `DISCOVERY_LOCAL_ENABLED` | true | Include LAN scan, beyond configured NVRs |
| `DISCOVERY_AUTO_ADD` | true | Provision verified selected candidates |
| `DISCOVERY_INTERVAL_S` | 150, minimum 60 | Scheduler gap after a sweep; also the `/sync` quiet period after a completed sweep |
| `DISCOVERY_MISS_THRESHOLD` | 2 | Intended missing transition threshold |
| `DISCOVERY_SUBNETS` | Empty | Derive default-route `/24` if absent |
| `DISCOVERY_WSD_TIMEOUT_S` | 3 | Multicast receive window |
| `DISCOVERY_HTTP_TIMEOUT_S` | 2 | Per-ISAPI request timeout |
| `DISCOVERY_PROBE_WORKERS` | 32, clamped 1..128 | LAN host probe concurrency |
| `DISCOVERY_SWEEP_ONLY_IF_WSD_EMPTY` | false | Whether responders suppress subnet scanning |
| `HIK_USERNAME` / `HIK_PASSWORD` | `admin` / empty | Direct-camera account; example password is a placeholder |
| `NVR_USERNAME` / `NVR_PASSWORD` | HIK account | Recorder account |
| `HIK_RTSP_PORT` / `HIK_HTTP_PORT` | 554 / 80 | Default video/control ports |
| `HIK_RTSP_CHANNEL` / `HIK_RTSP_STREAM` | 1 / 2 | Direct channel / chosen stream |
| `DISCOVERY_NVRS` | Empty | Recorder ISAPI enumeration targets |
| `STATIC_NVRS` | Empty | Declared input candidates with frame verification |
| `NVR_CHANNEL_OFFSET` | 0 | Stream-address offset for recorders only |

Changing source credentials, stream number, or offset changes the selected URL and can trigger the source-transition defect described in document 08. Treat these as explicit migrations until that behavior is fixed.

## Edge HTTP reference

All listed routes also have `/api` aliases, except root uses `/api` as its alias. Base URL examples assume the command runs **on the Jetson**.

| Method/path | Behavior |
| --- | --- |
| `GET /` | Readiness/discovery summary |
| `GET /health` | Summary plus counters and per-camera capture/result metrics |
| `GET /cameras` | In-memory configured camera list, not every saved/historical row |
| `POST /cameras` | Add/replace camera with caller-supplied UUID |
| `PATCH /cameras/<uuid>` | Merge patch and replace channel through runtime |
| `DELETE /cameras/<uuid>` | Stop live channel and delete saved camera configuration |
| `GET /cameras/<uuid>/latest` | Cached event, or 404 before one is cached |
| `GET /detection/<uuid>` or `/detections/<uuid>` | Latest-event compatibility aliases |
| `GET /cameras/<uuid>/snapshot.jpg` | Last cached JPEG, or 404 |
| `GET /cameras/<uuid>/detections/stream` | Camera-scoped edge SSE |
| `GET /cameras/detections/stream` | All-camera edge SSE, including discovery deltas |
| `GET /discovery` | Currently selected roster and status |
| `GET /discovery/status` | Scheduler/quiet-period/in-progress information |
| `GET /discovery/report` | Last completed report; 404 before first completion |
| `POST /sync` | Nonblocking request/coalescing and current/partial report |
| `POST /discovery/scan` | Blocking scan; may wait on an existing sweep |
| `DELETE /discovery/<identity>` | Forget roster history, not camera deletion |

Read-only examples:

```bash
curl --fail --silent --show-error http://127.0.0.1:8080/health
curl --fail --silent --show-error http://127.0.0.1:8080/discovery/status
curl --no-buffer --max-time 20 \
  http://127.0.0.1:8080/cameras/11111111-1111-4111-8111-111111111111/detections/stream
```

The final command intentionally stops after 20 seconds and can exit with curl's timeout status even after receiving useful data. SSE has no natural completion point.

Illustrative manual-mode create payload:

```json
{
  "source_url": "rtsp://viewer:PLACEHOLDER@192.168.50.21:554/Streaming/Channels/102",
  "config": {
    "camera_uuid": "11111111-1111-4111-8111-111111111111",
    "sample_fps": 3,
    "resize": [640, 360],
    "enabled": true,
    "detection_enabled": true,
    "rtsp_transport": "tcp"
  }
}
```

In discovery-managed mode, the runtime rejects sources outside its selected URL set and rejects another UUID claiming the same selected URL. A 409 can therefore mean selection/duplicate identity policy, not simply “eight cameras are running.” Avoid replaying POSTs as a read-only health check: they restart capture.

The edge health route returns JSON with `pipeline_ready`; do not rely only on HTTP 200. Its readiness calculation also does not currently prove dispatcher/worker liveness after startup.

## Relevant cloud endpoints

The FastAPI app prefixes routes with `/api`. These routes require their normal authorization; the edge service and cloud service are separate servers even when both use port 8080 in examples.

| Path | Purpose |
| --- | --- |
| `GET /api/cameras?site_uuid=<uuid>` | Application cameras visible to authorized user |
| `GET /api/cameras/<uuid>/detections/latest` | Cloud latest detection cache |
| `GET /api/cameras/<uuid>/detections/latest?refresh=true` | Fetch from edge **if cache is empty**; does not force-refresh a nonempty stale cache |
| `GET /api/cameras/<uuid>/detections/stream` | Authorized browser detection SSE |
| `POST /api/devices/<uuid>/edge/reconcile` | Explicit device reconcile |
| `POST /api/devices/<uuid>/inventory/refresh` | Import cached completed edge roster |
| `GET /api/devices/<uuid>/inventory` | Stored cloud inventory |
| `GET /api/sites/<uuid>/inventory` | Site inventory |
| `POST /api/sites/<uuid>/inventory/add` | Explicit site adoption; see UUID defect |

Consult [`device_routes.py`](../../routes/device_routes.py), [`site_routes.py`](../../routes/site_routes.py), and [`camera_routes.py`](../../routes/camera_routes.py) for request schemas and permissions. The reconciler's `dry_run` currently still performs some external mutations before reaching its guard; it is not a safe preview until B7 is addressed.

## Database schema and migrations

The edge database is `Backend/tensort/jetson_cameras.db`. `camera_configs` stores UUID/channel/user/source/config JSON. `discovered_cameras` stores identity, metadata, source URL, linked UUID, first-frame proof, and presence/history fields. `to_dict()` reports unverified legacy rows as not present until `first_frame_at` exists.

Startup calls `create_all` plus specific migrations for legacy source naming and frame verification. `create_all` creates missing tables; it does not perform arbitrary schema upgrades on existing tables. The standalone [`migrate_camera_configs_schema.py`](../migrate_camera_configs_schema.py) implements a more extensive normalization/rebuild path with backup support. Read it and back up an actual deployment database before using it; it is not a routine latency fix.

Do not copy a production SQLite file while a writer is mid-transaction and assume the copy is a consistent backup. Use a planned database backup procedure. The existing audit tool opens a read-only database path and reports configuration concerns without being a migration tool.

## Deployment tools and what they prove

| Tool | Use | Limitation |
| --- | --- | --- |
| [`setup_orin.sh`](../deployment/setup_orin.sh) | Install stack/environment, build engine, create service | Changes system packages/power/service state; review on target |
| [`build_engine.sh`](../deployment/build_engine.sh) | Build dynamic 1..8 FP16 TensorRT engine | Overwrites selected artifact; synthetic throughput output is informational |
| [`check_capacity.py`](../deployment/check_capacity.py) | Measure live per-camera result FPS/age and counter changes | Samples metrics; not a full latency histogram |
| [`diagnose_discovery.py`](../deployment/diagnose_discovery.py) | Inspect settings, ports, ISAPI identification | ISAPI success does not prove decoded video |
| [`probe_nvr_paths.py`](../deployment/probe_nvr_paths.py) | Explore recorder RTSP path responses | RTSP DESCRIBE is not a full frame-decode test |
| [`audit_cameras.py`](../deployment/audit_cameras.py) | Compare saved configurations against expected endpoints | Does not measure GPU speed |
| [`tests/diag_batch.py`](../tests/diag_batch.py) | Compare repeated-image outputs across batch rows | Needs target engine/CUDA/image; not an end-to-end camera test |

For this branch, use Python compatible with its current requirements and TensorRT 10 APIs. The legacy Nano/Python 3.6 comments and setup script are not compatibility guarantees. Build the engine on the target stack as the practical supported workflow; do not assume a copied engine is portable across TensorRT/GPU configurations. Specialized TensorRT compatibility features exist, but these scripts do not establish that they were enabled.

The setup scripts call `nvpmodel -m 0` and `jetson_clocks`. Power-mode numbers depend on the installed board/software configuration; mode 0 is not a universal “maximum performance” promise. Inspect the target's available modes, power supply, and cooling before benchmarking.
