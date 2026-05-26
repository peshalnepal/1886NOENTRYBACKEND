# Tensort — Jetson RTSP Inference Service

A Flask-based service that runs on a Jetson (Nano/Orin) device. It pulls RTSP video from cameras, runs **YOLOv8 TensorRT** object detection on the frames, and exposes detection results over HTTP + Server-Sent Events (SSE).

This document explains the architecture, the data flow, and what every class & function does inside [Backend/tensort/](Backend/tensort/).

---

## 1. High-Level Picture

```
                ┌────────────────────────────────────────────────────┐
                │ Azure Backend (control plane)                      │
                │  - registers cameras, sends config to Jetson       │
                └───────────────┬────────────────────────────────────┘
                                │ REST  (/cameras add/patch/delete)
                                ▼
   ┌────────────────────────────────────────────────────────────────┐
   │  Flask app  (main.py)                                          │
   │   - Receives /cameras requests                                 │
   │   - Owns PipelineRuntime  ───────────────────────┐             │
   │     (background thread that hosts asyncio loop)  │             │
   └───────────────────────────────────────────────── │ ────────────┘
                                                      │
                ┌─────────────────────────────────────▼──────────────┐
                │  SimpleInferencePipeline  (pipeline.py)            │
                │                                                    │
                │   VideoChannel  ──►  CoalescingBuffer  ──►  Pump   │
                │   (one per cam)      (latest frame only)    │      │
                │                                             ▼      │
                │                                  InferenceWorkerPool│
                │                                   (N TRT threads)  │
                │                                             │      │
                │                                             ▼      │
                │                                       _handle_result│
                │                                      ┌──────┴──────┐│
                │                                      ▼             ▼│
                │                               Broadcaster   _latest │
                │                               (SSE feed)   (snapshot│
                │                                            cache)   │
                └────────────────────────────────────────────────────┘
                                                      │
                                                      ▼
                                           SQLite (jetson_cameras.db)
                                           — persists camera configs
```

### Three threading domains (important)
1. **Flask worker threads** – handle HTTP requests; never touch CUDA.
2. **`pipeline-loop` thread** – hosts the single asyncio event loop. Owns all camera coroutines and dispatches inference jobs.
3. **`trt-infer-worker-N` threads** – each owns its **own CUDA context + TRT engine**. Inference must happen here (CUDA contexts are not safely shareable).

CUDA / PyCUDA / TensorRT are **never imported on the main thread** — they are only imported inside the worker thread that uses them. This is critical on Jetson Nano.

---

## 2. Data Flow (a single frame)

1. **VideoChannel** worker thread reads RTSP via GStreamer (`nvv4l2decoder` on Jetson) → produces `RTSPEvent(frame=BGR)`.
2. Frame is pushed into `_out_q` (asyncio.Queue) for the channel; pipeline pumps it into the **CoalescingBuffer** (newest-only-per-camera).
3. `_pump_inference` pulls one frame at a time, picks a worker by hashing `camera_uuid`, and submits via `InferenceWorkerPool.submit()`.
4. The TRT worker runs `preprocess` → `TRTEngine.infer()` → YOLO post-process → returns a `DetectionsProducedEvent` dict.
5. `_handle_result` stores the result in `self._latest[cam]`, optionally caches a JPEG snapshot, and broadcasts via the Broadcaster.
6. Consumers receive results through:
   - `GET /cameras/{uuid}/latest` – polls `self._latest`
   - `GET /cameras/{uuid}/detections/stream` – SSE via `Broadcaster.subscribe()`
   - `GET /cameras/{uuid}/snapshot.jpg` – JPEG from `_latest_snapshots`

---

## 3. Files at a Glance

| File | Purpose |
|------|---------|
| [main.py](Backend/tensort/main.py) | Flask HTTP API + `PipelineRuntime` (bridges Flask ↔ asyncio loop) |
| [pipeline.py](Backend/tensort/pipeline.py) | Asyncio pipeline: buffer, worker pool, dispatch, broadcast |
| [trt_infer.py](Backend/tensort/trt_infer.py) | TensorRT engine wrapper + YOLOv8 detector + preprocess/NMS |
| [channels/channel.py](Backend/tensort/channels/channel.py) | Single RTSP capture channel (GStreamer/OpenCV) |
| [channels/channel_config.py](Backend/tensort/channels/channel_config.py) | Plain config object for a channel |
| [database.py](Backend/tensort/database.py) | Async + sync SQLite engines (sqlalchemy) |
| [database_orm.py](Backend/tensort/database_orm.py) | `CameraConfig` SQLAlchemy ORM model |
| [schemas.py](Backend/tensort/schemas.py) | Pydantic request/response schemas (used by Azure side) |
| [migrate_camera_configs_schema.py](Backend/tensort/migrate_camera_configs_schema.py) | One-shot SQLite migration script |
| [models/](Backend/tensort/models/) | YOLOv8 `.engine` (TRT) + `.onnx` weights |

---

## 4. [main.py](Backend/tensort/main.py) — Flask + PipelineRuntime

### Module-level constants
- `DEFAULT_SAMPLE_FPS` (5.0) — how often a frame is sampled per camera.
- `DEFAULT_RESIZE_W/H` (640×480) — pre-resize before inference. Cuts per-frame memory ~6 MB (1080p) → ~700 KB. Critical on Jetson Nano.
- `DEFAULT_JPEG_QUALITY` (70), `MAX_SAMPLE_FPS` (25.0).

### `class PipelineRuntime`
The **bridge** between synchronous Flask handlers and the async pipeline loop running in another thread. Singleton — one instance lives at module scope as `runtime`.

**Construction** ([main.py:38-60](Backend/tensort/main.py#L38)):
- Creates `_cameras` dict (uuid → cfg).
- Initializes SQLite tables.
- Spawns a daemon thread (`_run_loop`) that hosts an asyncio loop.
- Waits up to 30s for pipeline to be `_ready`.
- Restores cameras from DB.

**Methods**:
- `_normalize_camera_cfg(cfg)` ([main.py:63](Backend/tensort/main.py#L63)) — clamps `sample_fps` to `[0.1, MAX]`, applies `resize`, clamps `jpeg_quality` to `[30, 95]`, forces `emit_format="raw"`.
- `_run_loop()` ([main.py:92](Backend/tensort/main.py#L92)) — the background thread entry point. Imports TRT modules **here** so CUDA init happens in this thread, builds the pipeline, calls `pipeline.start()`, then `loop.run_forever()`.
- `_call(coro, timeout_s)` ([main.py:118](Backend/tensort/main.py#L118)) — submits a coroutine to the pipeline loop from a Flask thread via `asyncio.run_coroutine_threadsafe` and waits for the result.
- `_require_pipeline()` — raises if pipeline isn't ready yet.
- **DB ops**:
  - `_save_camera_to_db_async / _save_camera_to_db` — upsert into `camera_configs` table.
  - `_delete_camera_from_db_async / _delete_camera_from_db` — delete row.
  - `_restore_cameras_from_db()` — on startup, reads all rows, builds `VideoChannelConfig`, registers in pipeline. Important: only adds to `_cameras` **after** the config validates, so a bad row doesn't leave a "ghost" camera that confuses sync.
- **Camera ops**:
  - `add_camera(rtsp_url, cfg_patch)` — requires `camera_uuid` (the Jetson never generates IDs — Azure does). Builds full config, persists to DB, calls `pipeline.add_channel`.
  - `remove_camera(camera_uuid)` — stops pipeline channel **first** (avoids "still inferencing after delete"), then deletes DB row.
  - `list_cameras()` — returns the in-memory `_cameras` view.
  - `patch_camera(uuid, patch)` — updates config in memory, persists, then calls `pipeline.add_channel` (which replaces the channel).
  - `get_latest(uuid)` — most recent detection result for a camera (uses `peek_latest`, no event-loop hop).
  - `get_snapshot(uuid)` — most recent JPEG snapshot bytes.
  - `get_stats()` — pipeline counters.

### Helper functions
- `_json()` — safely parses request body as a dict.
- `_require_rtsp(url)` — checks `rtsp://` / `rtsps://` prefix.
- `_runtime_status(include_stats)` — health payload.
- `_sse_generator(target_camera_uuid=None)` ([main.py:556](Backend/tensort/main.py#L556)) — yields SSE-formatted strings. Subscribes to broadcaster (optionally per-camera), uses `run_coroutine_threadsafe(q.get())` with a 1 s timeout to allow the Flask thread to detect client disconnect and emit keepalive comments.

### HTTP routes
| Method | Path | Description |
|--------|------|-------------|
| GET | `/`, `/api`, `/health` | Liveness / stats |
| GET | `/cameras` | List cameras |
| POST | `/cameras` | Add camera (needs `rtsp_url`, `camera_uuid`) |
| PATCH | `/cameras/<uuid>` | Patch fields |
| DELETE | `/cameras/<uuid>` | Remove camera |
| GET | `/cameras/<uuid>/latest`, `/detection/<uuid>` | Latest detection JSON |
| GET | `/cameras/<uuid>/snapshot.jpg` | Latest annotated JPEG |
| GET | `/cameras/detections/stream` | SSE stream — all cameras |
| GET | `/cameras/<uuid>/detections/stream` | SSE stream — one camera |

All routes also have an `/api/...` alias for the reverse proxy.

---

## 5. [pipeline.py](Backend/tensort/pipeline.py) — The Async Inference Pipeline

This is where most of the cleverness lives. Six explicit fixes are documented in the file header — they target multi-camera throughput and event-loop responsiveness.

### Helpers (top of file)
- `_encode_jpeg_bytes(frame, max_edge, jpeg_quality)` — resizes frame so longest edge ≤ `max_edge`, then `cv2.imencode('.jpg', ...)`.
- `_env_int / _env_float / _env_bool(name, default, ...)` — typed env-var readers with floors.
- `_detect_total_memory_mb()` — reads `/proc/meminfo` `MemTotal:`.
- `_default_auto_worker_cap(mem_mb)` — picks max workers based on RAM (1 worker ≤ 4.5 GB, 2 ≤ 8 GB, 3 ≤ 16 GB, else 4).
- `_default_infer_result_timeout_s(mem_mb)` — wider timeout on smaller boxes (slower CPU).

### `class Broadcaster`  (FIX 5: lock-free hot path)
Pub/sub for detection events. The hot `broadcast()` path **must not** await.

- `__init__` — `_subscribers: list[(asyncio.Queue, Optional[camera_uuid])]`.
- `async subscribe(camera_uuid=None)` — creates a queue (size 200), atomically replaces `_subscribers` with a new list (CPython atomic), returns the queue.
- `async unsubscribe(q)` — same atomic-replace pattern.
- `async broadcast(msg)` — single attribute read of `_subscribers` is GIL-atomic, so no lock. Iterates list and `put_nowait` to each queue, optionally filtering by `camera_uuid`.

### `class CoalescingBuffer`  (FIX 3)
Keeps **only the newest frame per camera**. Stale frames are silently overwritten — old frames waste GPU time.

- `_latest: dict[camera_uuid → RTSPEvent]` — newest frame.
- `_pending: asyncio.Queue` — keys awaiting dispatch.
- `_in_queue: set` — guards against duplicate enqueues.
- `async put(ev)` — overwrites `_latest[key]`; if key isn't already queued, enqueues it.
- `async get()` — pops a key, looks up + removes from `_latest`. Skips if `_latest` no longer has it (already consumed).

### `class InferenceWorker`  (FIX 1)
One thread, one CUDA context, one TRT engine. Never share across threads.

- Constructor builds a daemon thread that runs `_run`. Starts it immediately.
- `submit(bgr, meta, fut)` — non-blocking `Queue.put_nowait`. If full, immediately resolves `fut` with an `InferenceFailedEvent` via `loop.call_soon_threadsafe`.
- `_run()` — imports `trt_infer.build_default()` (this constructs the TRT engine **inside this thread** — required for CUDA). Loop: pull `(bgr, meta, fut)` from queue, call `self._infer.infer_multitask`, post result back to event loop with `call_soon_threadsafe`.
- `stop()/join()` — clean shutdown.

### `class InferenceWorkerPool`
Owns N `InferenceWorker`s.

- `_pick(camera_uuid)` — `hash(uuid) % N`. **Same camera always lands on the same worker** — preserves per-camera frame ordering & avoids cross-worker queue contention.
- `ensure_size(num)` — grows the pool when new cameras are added.
- `submit(...)` — picks worker, forwards.
- `stop()/join()`.

### `class SimpleInferencePipeline`
The pipeline orchestrator. One per process.

#### Constructor key fields
- `_channels` — uuid → `VideoChannel`.
- `_channel_tasks` — uuid → `asyncio.Task` running `_pump_channel`.
- `_buffer` — `CoalescingBuffer`.
- `_out_q` — asyncio.Queue exposed via `events()`.
- `_latest` — uuid → most recent result dict.
- `_latest_snapshots` / `_latest_snapshot_ts_ms` — JPEG cache + last-cache timestamp.
- `_inflight` — `(camera_uuid, seq) → Future` currently being inferred.
- `_max_inflight` — back-pressure cap (env `MAX_INFLIGHT_FRAMES`, default 8).
- `_num_workers` — 0 means auto-size (= number of cameras, capped).
- `broadcaster` — public `Broadcaster` instance.

#### Channel management
- `async add_channel(cfg)` — atomically swaps in a new `VideoChannel`, cancels old task, stops old channel, starts new task. Auto-grows pool if `_num_workers==0`.
- `async remove_channel(uuid)` — cancels task, stops channel, clears `_latest` / snapshot caches.
- `list_channels()` — list of camera_uuids.

#### Lifecycle
- `async start()` — creates `InferenceWorkerPool`, launches `_pump_inference`, launches `_pump_channel` tasks for any pre-loaded channels.
- `async shutdown()` — cancels all tasks, stops channels, joins worker pool, sends `_DONE` sentinel.

#### Internal pumps
- `async _put_out(ev)` — queues an event for the public `events()` stream; drops oldest if full.
- `_start_channel_task(key)` — creates a task for `_pump_channel`.
- `async _pump_channel(key, ch)` — iterates `ch.stream()`. Frames go into the buffer; meta-events (`Connected`, `Disconnected`, etc.) go into `_out_q`.

#### Dispatch (FIX 2: fire-and-forget)
- `async _pump_inference()` — the heart of throughput. Loop:
  1. `await self._buffer.get()` — newest frame for some camera.
  2. Build `meta`, allocate a `loop.create_future()`.
  3. Track in `_inflight`.
  4. `_infer_pool.submit(bgr, meta, fut)` — fire and continue (no await).
  5. Attach `fut.add_done_callback(_on_done)` which schedules `_handle_result` on the loop.
  6. Spawn `_watchdog_future` — if the future doesn't resolve within `_infer_result_timeout_s`, force a timeout result.
  - Back-pressure: if `len(_inflight) >= _max_inflight`, drop the frame.
  - Worker queue is the **real** memory guard (`INFER_QUEUE_MAX=1`).

- `_drop_inflight(key)` — pops from `_inflight`.
- `async _watchdog_future(fut, key, meta)` — sleeps `_infer_result_timeout_s`, if not done sets a timeout `InferenceFailedEvent`.

#### Result handling
- `async _handle_result(result, meta)` — runs concurrently for different cameras (one camera never serializes another):
  - On failure: increments stats, log once-per-reason via `_should_log_infer_failure` (rate-limited per camera+reason).
  - On success: increments stats, optionally schedules `_cache_snapshot`.
  - Stores `_latest[uuid] = result` (atomic dict write).
  - If detections (or `EMIT_EMPTY_DETECTIONS=true`): `broadcaster.broadcast(result)` and `_put_out(result)`.

- `_should_log_infer_failure(uuid, reason)` — rate-limits identical errors to once every `INFER_ERROR_LOG_INTERVAL_S` (default 10 s).

#### Public API
- `async events()` — async generator over `_out_q`.
- `peek_latest(uuid)` / `async get_latest(uuid)` — most recent inference result.
- `peek_latest_snapshot(uuid)` / `async get_latest_snapshot(uuid)` — JPEG bytes.
- `peek_stats()` / `async get_stats()` — counters + queue sizes.

#### Snapshot cache (FIX 4: encode in executor)
- `async _cache_snapshot(camera_uuid, frame_bgr, ts_ms)` — throttled by `SNAPSHOT_MIN_INTERVAL_MS` (default 500 ms). Offloads `cv2.imencode` to the default thread executor so the event loop is never stalled. On success stores bytes in `_latest_snapshots`.

---

## 6. [trt_infer.py](Backend/tensort/trt_infer.py) — TensorRT + YOLOv8

### CUDA context helpers
- `cuda.init()` — runs once at import.
- `_tls = threading.local()` — per-thread CUDA context store.
- `ensure_cuda_context(device_id)` — lazily creates a context on **this thread**; pops it after creation so it isn't current by default.
- `class CudaContext` — a context manager (`with CudaContext(...)`) that pushes the per-thread context for a critical section, pops on exit. Used inside `TRTEngine.infer`.
- `release_cuda_context()` — detaches when shutting down.

### CPU preprocessing
- `letterbox_bgr(img, new_shape, color)` — scales the image to fit `new_shape×new_shape` while preserving aspect ratio, padding the rest with gray (114,114,114). Returns the padded image, the scale ratio `r`, and the `(left, top)` padding offsets (needed to map detections back to original coordinates).
- `_prepare_input_tensor(img_lb)` — converts to NCHW float32, BGR→RGB, scaled `1/255`. Uses `cv2.dnn.blobFromImage` if available (faster), else manual numpy.
- `preprocess(bgr, imgsz)` — one-shot wrapper returning `(tensor, r, (padx,pady))`.

### Detection helpers
- `nms_xyxy(boxes, scores, iou_thr, topk)` — pure-numpy non-max-suppression. Returns indices to keep.
- `clamp_xyxy(...)` — keep box inside `[0, W-1]×[0, H-1]`; reorder if `x2<x1`.
- `box_norm_xyxy(...)` — convert pixel box to `{x, y, w, h}` normalized to `[0, 1]`.

### `class TRTEngine`  (FIX 1: double-buffered streams)
Wraps a TensorRT engine + execution context. Uses ping-pong buffers so memcpy can overlap execution.

- Constructor:
  - Loads `.engine` file under a CUDA context.
  - Walks bindings, validates **fixed shapes** (rejects dynamic).
  - Allocates **two** pinned host buffer slots (`_host_bufs[0]`, `_host_bufs[1]`) but **shares** device buffers (we sync the slot before reuse).
  - Allocates two `cuda.Stream()`.
- `infer(input_chw)`:
  1. `b = self._buf_idx` — current ping/pong slot.
  2. `np.copyto(...)` into pinned host input buffer for slot `b`.
  3. `memcpy_htod_async` on `streams[b]`.
  4. `execute_async_v2`.
  5. `memcpy_dtoh_async` for each output.
  6. `streams[b].synchronize()` — wait for **this slot only**; the other slot can be uploading the next frame.
  7. Copy outputs into reshaped numpy arrays.
  8. Flip `_buf_idx`.

### `class YoloV8DetTRT`  (FIX 2: pre-process before TRT)
- `COCO_NAMES` — id → label, restricted to `{person, car, motorcycle, truck}`.
- Constructor builds the underlying `TRTEngine`, stores `imgsz, conf, iou, topk, allowed`.
- `run(bgr)`:
  1. `preprocess(bgr, imgsz)` — pure CPU, can overlap a previous frame's GPU run.
  2. `self.trt.infer(x)` — only memcpy + execute + memcpy + sync.
  3. Parse YOLOv8 raw output `(C, N)` → boxes XYWH + class scores. Auto-handles transposed shape.
  4. Filter by `conf` threshold and `allowed` classes.
  5. Convert XYWH → XYXY in letterboxed coordinates.
  6. Run `nms_xyxy`.
  7. Reverse the letterbox transform `(box - pad) / scale` → original image coords.
  8. Build the result list `[{cls_name, conf, box, box_norm}, ...]`.

### `class TRTInfer`
The public entry point used by an `InferenceWorker`.

- Constructor — calls `ensure_cuda_context`, builds a `YoloV8DetTRT` with config from env.
- `infer_multitask(bgr, meta) → dict` — wraps `det_runner.run(bgr)`, returns either a `DetectionsProducedEvent` (with `detections`, `inference_ms`, `frame_w/h`) or an `InferenceFailedEvent`.

### `build_default()`
Reads env (`DET_ENGINE`, `IMG_SZ=640`, `CONF=0.350`, `IOU=0.45`, `CUDA_DEVICE=0`, `NMS_TOPK=50`), defaults the engine path to `tensort/models/yolov8n.engine`, and returns a `TRTInfer`.

---

## 7. [channels/channel.py](Backend/tensort/channels/channel.py) — RTSP Capture

### Event classes
All inherit from `ChannelEvent(type, channel_id, camera_uuid, ts_ms)`.

- `ChannelConnectedEvent(rtsp_url)` — after the cv2 capture opens.
- `ChannelDisconnectedEvent(reason)` — on grab/retrieve failure or shutdown.
- `FrameDroppedEvent(reason, dropped_count)` — backpressure drop.
- `RTSPEvent(seq, format, frame, width, height, fps_hint, detection_enabled)` — a real frame, what the pipeline consumes.

`RTSPEvent` uses `__slots__` for memory efficiency (frames are large).

### `class VideoChannel`
One per RTSP camera. Owns its own background `threading.Thread` (because OpenCV's `VideoCapture.grab()` is blocking).

- **`_out_q`** — asyncio.Queue with maxsize 1 (env `CHANNEL_OUT_Q_MAX`). Old frames are dropped (leaky).
- **`_build_gst_pipeline(rtsp_url, decoder)`** — builds GStreamer pipeline string:
  - `rtspsrc → rtph264depay → h264parse → nvv4l2decoder` (Jetson HW) or `avdec_h264` (CPU fallback).
  - Optional `nvvidconv` resize before BGR conversion.
  - `appsink drop=true sync=false max-buffers=1` so the latest frame wins.
- **`_open_capture()`** — tries `gst_decoder` then `avdec_h264`; falls back to plain OpenCV/FFmpeg if both fail.
- **`_maybe_resize(frame)`** — aspect-preserving downscale (never upscale).
- **`_put_latest(ev)`** — leaky put into `_out_q`. Special-cases `_DONE` sentinel: clears queue and inserts a single `_DONE` (sticky).
- **`_handle_in_loop(ev)`** — runs on the asyncio thread; pushes to `_out_q` and to optional `event_queue`.
- **`_push_from_thread(ev)`** — from the worker thread, schedules `_handle_in_loop` via `call_soon_threadsafe`.
- **`_worker()`** — the capture thread:
  1. Open capture, push `ChannelConnectedEvent`.
  2. Loop: `cap.grab()`; if interval since last emit < `1000/sample_fps` ms, drop. Else `cap.retrieve()`, optionally resize, build `RTSPEvent`, push.
  3. On any exception: push `ChannelDisconnectedEvent`, sleep with exponential backoff (`reconnect_base_ms` doubling up to `reconnect_max_ms`), reconnect.
  4. Always release the capture in `finally:`.
- **`async stream(event_queue=None)`** — async generator that the pipeline awaits. Starts the worker thread on first call, yields events from `_out_q` until `_DONE`.
- **`async stop()`** — idempotent. Sets stop flag, pushes sticky `_DONE`, joins the thread in an executor (never blocks the event loop, because RTSP reads can hang for several seconds).

---

## 8. [channels/channel_config.py](Backend/tensort/channels/channel_config.py) — `VideoChannelConfig`

A plain (no-Pydantic) data class with `__slots__`. Fields:

| Field | Default | Notes |
|-------|---------|-------|
| `channel_id` | None | Defaults to `camera_uuid` if missing. |
| `camera_uuid` | None | Required for known cameras; the Jetson never invents IDs. |
| `rtsp_url` | None | Required when adding a new camera. |
| `enabled` | True | If False, `stream()` returns immediately. |
| `detection_enabled` | True | If False, frames are still captured but not inferred. |
| `notification_enabled` | True | Used downstream (alerts). |
| `sample_fps` | 5.0 | Frame sampling rate. |
| `decode_backend` | "gstreamer" | "gstreamer" or "opencv". |
| `resize` | None | `(W, H)` tuple. |
| `reconnect_base_ms / _max_ms` | 1000 / 8000 | Exponential backoff. |
| `emit_format` | "raw" | Forced to "raw" by `_normalize_camera_cfg`. |
| `jpeg_quality` | 80 | |
| `gst_latency_ms` | 5 | rtspsrc latency. |
| `rtsp_transport` | "tcp" | "tcp" / "udp". |
| `gst_decoder` | "nvv4l2decoder" | Jetson HW decode. |

- `_normalize()` — coerces types, falls back to safe defaults for invalid values.
- `validate()` — raises `ValueError` for impossible combinations (no rtsp_url + no camera_uuid; sample_fps ≤ 0; reconnect_max < base; jpeg_quality outside `[1,100]`).

---

## 9. [database.py](Backend/tensort/database.py) — DB engines

Single `DatabaseManager` class:

- Sets up **both** engines pointing at `tensort/jetson_cameras.db`:
  - **Async** — `sqlite+aiosqlite`, used by `_save_camera_to_db_async`/`_delete_camera_from_db_async`.
  - **Sync** — `sqlite:///`, used during startup `_restore_cameras_from_db` (before the loop is fully running).
- `expire_on_commit=False` so ORM objects keep their attributes after commit.
- `initialize_tables()` — calls `Base.metadata.create_all(...)`. **Note**: does not alter existing tables. Schema migrations need a separate script.
- `get_session() / get_async_session()` — factory methods.
- Module-level globals `db_manager`, `SessionLocal`, `AsyncSessionLocal`, `engine`, `async_engine` for convenience.

---

## 10. [database_orm.py](Backend/tensort/database_orm.py) — `CameraConfig` model

A single SQLAlchemy table `camera_configs`:

| Column | Type | Notes |
|--------|------|-------|
| `id` | Integer PK auto-increment | |
| `channel_id` | String(64), unique, indexed | UUID string |
| `camera_uuid` | String(64), unique, indexed | UUID string |
| `user_id` | Integer, indexed | Multi-tenant key (default 1) |
| `rtsp_url` | Text | Required |
| `config_json` | JSON | Full snapshot of channel config |
| `created_at / updated_at` | DateTime | `datetime.utcnow` defaults |

Methods:
- `__repr__()` — debugging string.
- `to_dict()` — serializable dict (handles None datetimes).

---

## 11. [schemas.py](Backend/tensort/schemas.py) — Pydantic Schemas

Used on the **Azure backend** side for validating camera CRUD payloads (also imported here for shared types).

- `ROISchema` — polygon of `[x, y]` points; `normalized` toggles 0–1 vs pixel.
- `CameraBaseSchema` — common fields with cross-validator that strips blank strings.
- `CameraCreateSchema` — POST `/cameras` payload. Forbids `webrtc_url`, requires `device_uuid`.
- `CameraEditSchema` — PATCH `/cameras/{uuid}` payload — all fields optional.
- `CameraSchema` — DB output for GET `/cameras`.
- `CameraWithConfigSchema` — extends with `configuration` and `timezone`.
- `CameraPlaybackSchema` — minimal `{camera_uuid, webrtc_url}` for playback URLs.

`ConfigDict(extra="forbid")` is applied everywhere — unknown fields raise immediately.

---

## 12. [migrate_camera_configs_schema.py](Backend/tensort/migrate_camera_configs_schema.py) — One-shot migration

Standalone CLI script (`python migrate_camera_configs_schema.py [--db PATH]`). It rebuilds an old `camera_configs` table that only had `(channel_id, camera_uuid, user_id, rtsp_url, config_json)` into the expanded schema with top-level columns (`site_uuid`, `device_uuid`, `name`, `location`, `webrtc_url`, `is_enabled`, `is_detection_enabled`, `is_notification_enabled`, `roi`, …).

Strategy:
1. Backup DB file via `shutil.copy2`.
2. `CREATE TABLE camera_configs_new (…)` with target schema.
3. Copy rows, lifting fields out of `config_json` into proper columns.
4. Rename old table to `camera_configs_legacy`, rename new table to `camera_configs`.
5. Recreate indexes.

---

## 13. Environment Variables Cheat-Sheet

| Var | Default | Where used |
|-----|---------|------------|
| `PORT` | 8080 | Flask bind |
| `DEFAULT_SAMPLE_FPS` | 5.0 | main.py |
| `DEFAULT_RESIZE_W/H` | 640 / 480 | main.py |
| `DEFAULT_JPEG_QUALITY` | 70 | main.py |
| `MAX_SAMPLE_FPS` | 25.0 | main.py |
| `INFER_NUM_WORKERS` | 0 (auto) | pipeline.py |
| `INFER_NUM_WORKERS_MAX` | from RAM | pipeline.py |
| `INFER_QUEUE_MAX` | 1 | per-worker queue |
| `MAX_INFLIGHT_FRAMES` | 8 | pipeline.py back-pressure |
| `INFER_RESULT_TIMEOUT_S` | from RAM | watchdog |
| `INFER_ERROR_LOG_INTERVAL_S` | 10 | rate-limit logs |
| `EMIT_EMPTY_DETECTIONS` | false | broadcast empty results? |
| `ENABLE_SNAPSHOT_CACHE` | true | cache JPEGs |
| `SNAPSHOT_ON_DETECTION_ONLY` | true | only when objects detected |
| `SNAPSHOT_MIN_INTERVAL_MS` | 500 | throttle |
| `SNAPSHOT_MAX_EDGE` | 960 | resize before JPEG |
| `SNAPSHOT_JPEG_QUALITY` | 75 | |
| `PIPELINE_OUT_QUEUE_MAX` | 500 | events() queue |
| `PIPELINE_LOG_EVERY_N_FRAMES` | 0 (off) | debug log |
| `PENDING_KEY_MAX` | 1000 | CoalescingBuffer |
| `CHANNEL_OUT_Q_MAX` | 1 | per-channel queue |
| `DET_ENGINE` | `models/yolov8n.engine` | trt_infer.py |
| `IMG_SZ` | 640 | TRT input size |
| `CONF` | 0.350 | confidence threshold |
| `IOU` | 0.45 | NMS IoU |
| `NMS_TOPK` | 50 | max boxes after NMS |
| `CUDA_DEVICE` | 0 | GPU id |

---

## 14. Putting It Together — Adding a Camera Step-By-Step

1. Azure POSTs `/api/cameras` with `rtsp_url + camera_uuid + config`.
2. `add_camera` in [main.py:421](Backend/tensort/main.py#L421) builds defaults, applies the patch, and calls `runtime.add_camera`.
3. `PipelineRuntime.add_camera` validates, saves to SQLite, and submits `pipeline.add_channel(cfg)` to the asyncio loop via `_call`.
4. `SimpleInferencePipeline.add_channel` swaps in a new `VideoChannel`, cancels the old task, and starts `_pump_channel`.
5. `_pump_channel` iterates `channel.stream()`. The channel's worker thread opens an RTSP capture and pushes `RTSPEvent` frames.
6. Frames land in `CoalescingBuffer`; only the newest per-camera survives.
7. `_pump_inference` picks them up, hashes uuid → worker, submits to a TRT thread.
8. The TRT thread: `preprocess` → `TRTEngine.infer` → YOLO post-process → returns event dict.
9. `_handle_result` stores in `_latest`, optionally caches a JPEG snapshot, and broadcasts.
10. Clients poll `/cameras/<uuid>/latest` or hold open `/cameras/<uuid>/detections/stream` (SSE).

---

## 15. Common Operational Gotchas

- **Never call CUDA on the Flask thread.** Always go through `runtime._call(coro)` → asyncio loop → worker thread.
- **`emit_format` is forced to "raw"** — JPEG mode would defeat zero-copy frame handoff to TRT.
- **Camera UUIDs come from Azure** — `add_camera` raises if missing; this prevents drift between Azure DB and Jetson DB.
- **Schema migrations**: SQLAlchemy's `create_all` does not `ALTER TABLE` — use [migrate_camera_configs_schema.py](Backend/tensort/migrate_camera_configs_schema.py) when columns change.
- **One camera per worker** is the natural sweet spot — don't manually set `INFER_NUM_WORKERS` higher than the number of cameras unless you also raise the GPU memory budget.
- **SSE clients** must tolerate keepalive comments (`: keepalive\n\n`) — Flask emits them every second when no events arrive.
