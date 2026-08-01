# tensort — Jetson edge inference service

This is the service that runs **on the Jetson**. It pulls RTSP (and WHEP/SRT/RTMP/HTTP)
video from cameras, runs YOLO object detection on a TensorRT engine, and streams
detection results to the cloud backend over Server-Sent Events.

It does **not** track objects, store clips, or decide what an alert is — the cloud
does all of that. This box's only job is: *frames in, boxes out, as steadily as
possible.*

---

## 1. The whole picture in one diagram

```
 CAMERA 1 ──┐
 CAMERA 2 ──┤   [capture thread per camera]        [event loop thread]
 CAMERA N ──┘    GStreamer/OpenCV decode                  │
                 sample_fps gate                          │
                        │                                 │
                        └── asyncio.Queue(1) ──────────────┤
                            (keep newest)                  │
                                                           ▼
                                                    ╔════════════╗
                                                    ║ FramePool  ║  one deque
                                                    ║            ║  per camera
                                                    ╚════════════╝
                                                           │
                                          get_batch(10)  ◄──┤ waits for a free
                                          round-robin       │ worker slot FIRST
                                                           ▼
                                             [inference worker thread]
                                              letterbox → pinned buffer
                                              ONE TensorRT call (B,3,640,640)
                                              parse detections
                                                           │
                                       future resolved ────┤
                                                           ▼
                                                    Broadcaster
                                                           │
                                              SSE  ────────┴──────► CLOUD
                                                                    (tracker,
                                                                     alerts)
```

Threads that exist at runtime:

| Thread | Count | What it does |
|---|---|---|
| Flask/Werkzeug | several | HTTP requests + holds SSE connections open |
| `pipeline-loop` | 1 | the asyncio event loop; owns the FramePool and dispatch |
| `VideoChannel-<id>` | one per camera | blocking decode loop, cannot be async |
| `trt-infer-worker-N` | 1 (default) | owns the CUDA context + TensorRT engine |

The GPU work happens on a worker thread, not the event loop, because TensorRT and
PyCUDA calls block. The CUDA context is **thread-local** — this is why the engine
is built inside the worker thread and must never be shared across threads.

---

## 2. Files

| File | Role |
|---|---|
| `main.py` | Flask app + HTTP routes; starts the pipeline in a background thread; restores cameras from the DB on boot |
| `pipeline.py` | `FramePool`, `SimpleInferencePipeline` (dispatch), `InferenceWorker`, `Broadcaster` |
| `trt_infer.py` | `TRTEngine` (TensorRT bindings), `YoloV8DetTRT` (letterbox + parse), `build_default()` |
| `channels/channel.py` | `VideoChannel` — one camera's decode thread and reconnect logic |
| `channels/channel_config.py` | per-camera settings object |
| `database.py`, `database_orm.py` | local SQLite; remembers cameras across restarts |
| `deployment/setup_orin.sh` | builds the TensorRT engine **on the device** and installs the systemd unit |
| `models/*.engine` | the compiled engine — device-specific, never copy between machines |

---

## 3. Frame path, step by step

### 3.1 Capture (`channels/channel.py`)

Each camera gets a thread running `_worker()`. It tries a list of GStreamer
pipelines in order and keeps the first that opens — hardware decode
(`nvv4l2decoder`) first, then software (`avdec_h264`), then a plain
`cv2.VideoCapture` fallback.

The important detail is the FPS gate:

```python
grabbed = cap.grab()                       # decode, but do not convert
if (ts_ms - last_emit_ms) < emit_interval_ms:
    continue                               # skip: no BGR conversion
ok, frame = cap.retrieve()                 # only now pay the conversion cost
```

`grab()`-without-`retrieve()` is what makes `DEFAULT_SAMPLE_FPS` cheap: the stream
is still decoded at its native rate (you cannot skip that with RTSP), but frames
you do not want never get converted to a numpy array.

The frame is handed to the event loop via `call_soon_threadsafe` into an
`asyncio.Queue(maxsize=1)` that **keeps the newest** frame. A camera that outruns
the loop drops its own stale frames here rather than backing up.

### 3.2 FramePool (`pipeline.py`) — the part that was redesigned

The pool holds a FIFO deque of pending frames **per camera**.

```
_frames = {
  "cam-1": deque([f1, f2, f3, f4]),   # busy camera, several frames waiting
  "cam-2": deque([f9]),
  "cam-3": deque([f7]),
}
```

**Adding a frame** (`put`): drop anything older than `FRAME_MAX_AGE_MS`, evict if
the pool is at `FRAME_POOL_CAP`, then append. Cameras are created lazily, so a
newly added camera starts contributing immediately with no registration step.

**Eviction rule** when the pool is full:

1. Frames older than `FRAME_MAX_AGE_MS` are always dropped first. A stale
   detection is rejected by the cloud tracker anyway, so inferring it wastes a
   GPU slot.
2. Otherwise the camera holding the **most** frames loses its **oldest** frame.
   A camera that has hogged the pool pays for the newcomer's frame; quiet
   cameras are never charged.
3. If every camera holds exactly one frame, nobody is over-represented, so the
   **globally oldest** frame goes.

**Drawing a batch** (`get_batch`): round-robin, oldest camera first. Every camera
with a pending frame contributes one before any camera contributes a second.
This is what makes per-camera detection FPS roughly equal under load instead of
first-come-first-served. It returns whatever is pooled — **it never waits for all
cameras**, so one dead camera cannot stall the other nine.

The linger (`INFER_BATCH_LINGER_MS`, 10 ms) is only a top-up window used when the
pool is underfilled — idle or cold start. In steady state the pool has already
accumulated frames while the GPU was busy with the previous batch.

### 3.3 Dispatch (`_pump_inference`)

```python
while not closing:
    while not pool.has_capacity():         # ← wait for a worker slot FIRST
        await worker_free_event
    events = await frame_pool.get_batch(max_batch, linger)
    ... build futures ...
    worker_pool.submit_batch(job)
```

Waiting for capacity *before* drawing frames is the key inversion. It means the
FramePool — not the dispatcher — decides what to drop, so shedding is per-camera
and fair.

### 3.4 Inference worker (`InferenceWorker._run` → `trt_infer.py`)

One thread, one CUDA context, one engine. For each batch:

1. **Letterbox each frame directly into the engine's pinned input buffer.**
   `TRTEngine.input_view` is a `(max_batch, 3, H, W)` view over pinned memory;
   `_fill_row()` writes frame *i* into row *i*. One copy per frame.
2. `infer_prepared(b)` — set the dynamic batch shape, one H2D copy, one
   `execute_async_v3`, one D2H copy, synchronise.
3. Parse each row back to its own camera's coordinates (each frame keeps its own
   scale + padding, so cameras of different resolutions batch together fine).

Results are pushed back to the loop with `call_soon_threadsafe`, which resolves
each frame's future in order.

### 3.5 Output

`_handle_result` runs synchronously in the future's done-callback: it updates
`_latest`, counts stats, and calls `Broadcaster.broadcast()`, which is a
`put_nowait` into each SSE subscriber's queue. Only the JPEG snapshot encode is
offloaded to an executor, because `cv2.imencode` is genuinely slow.

---

## 4. Why detections used to flicker

Five separate causes, all fixed:

| Cause | Effect | Fix |
|---|---|---|
| Whole-batch drops when overloaded | Every camera's box vanished at the same instant | Dispatcher waits for capacity; the pool sheds per-camera instead |
| Watchdog discarded late results | GPU finished the work, the result was thrown away, camera got a gap | A late result is now still delivered (`_deliver_result`) |
| Failure events broadcast to the cloud | Tracker read "no objects" and aged every track on that camera | Failures are counted and logged, not broadcast (`EMIT_FAILED_EVENTS=false`) |
| Edge `CONF=0.30` vs tracker `low_th=0.3` | A dimming object fell out of the tracker's rescue band entirely and lost its ID | Edge filters at `0.20`, below the tracker's `low_th` |
| `EMIT_EMPTY_DETECTIONS=false` | Tracker never saw "object gone" frames, so tracks coasted and boxes lingered | Defaults to `true` everywhere |

The last two live on **both** sides of the wire — the edge's `CONF` and the cloud
tracker's `low_th` are coupled. If you raise `CONF` above `low_th`, the flicker
comes back. There is a regression test for this
(`Backend/tests/test_tracker_flicker_recovery.py`).

---

## 5. Configuration

`.env` is read two ways: systemd passes it via `EnvironmentFile`, and `main.py`
also calls `load_dotenv()` so `python3 main.py` behaves identically. Variables
already set in the environment always win.

The knobs that matter most, in order:

| Variable | Meaning |
|---|---|
| `DEFAULT_SAMPLE_FPS` | Frames analysed per camera per second. **The main lever.** Never set it above what the GPU can drain. |
| `INFER_MAX_BATCH` | Frames per GPU call. Must be ≤ the engine's `maxShapes` batch. |
| `DET_ENGINE` / `IMG_SZ` | Which engine, and its input size. `IMG_SZ` must match what the engine was built with — the service now refuses to start on a mismatch. |
| `CONF` | Detection threshold. Keep **≤ the cloud tracker's `low_th` (0.20)**. |
| `FRAME_POOL_CAP` / `FRAME_MAX_AGE_MS` | Pool depth and staleness cutoff. |
| `EMIT_EMPTY_DETECTIONS` | Keep `true` — the tracker needs empty frames to age tracks. |
| `INFER_NUM_WORKERS` | Keep at 1. On one GPU, several CUDA contexts time-slice and fragment batches. |

See `.env.example` for the complete annotated list.

### Reading `/health`

| Field | What it tells you |
|---|---|
| `infer_ok`, `infer_fail` | successful vs failed frames |
| `pool_depth` | frames currently waiting for the GPU |
| `pool_evicted_total` | **climbing steadily = cameras outrunning the GPU.** Lower `DEFAULT_SAMPLE_FPS`. |
| `pool_expired_total` | frames dropped for being older than `FRAME_MAX_AGE_MS` |
| `max_batch` | what the **engine** actually accepts — if this says 1, batching is off and you need a dynamic-batch engine |
| `inflight_count` | frames handed to the GPU and not yet returned |

---

## 6. The engine

A TensorRT engine is **built for one device and one TensorRT version**. It cannot
be copied from another machine. `deployment/setup_orin.sh` builds it on the box.

The ONNX must be exported with a **dynamic batch axis** or batching silently
does nothing (`max_batch` in `/health` will read 1):

```bash
yolo export model=yolo26s.pt format=onnx dynamic=True imgsz=640 simplify=True
```

Then the script runs `trtexec` with `--fp16` and min/opt/max shapes of
1 / `OPT_BATCH` / `MAX_BATCH`.

### Throughput gate — do this before trusting a target FPS

```bash
sudo /usr/src/tensorrt/bin/trtexec \
  --loadEngine=models/yolo26s.engine --shapes=images:10x3x640x640
```

Aggregate images/sec = `10 × 1000 / (GPU Compute Mean ms)`. Divide by your camera
count to get the per-camera FPS the hardware can actually sustain, then set
`DEFAULT_SAMPLE_FPS` at or below it. yolo26s at 640 on an Orin Nano is expected
around 65–100 img/s, i.e. roughly 6.5–10 FPS across 10 cameras — **measure yours**
rather than assuming. If it falls short, either lower the FPS or switch
`DET_ENGINE` back to `yolo26n.engine`.

### Two boards, two branches

`main` targets the original Jetson Nano (TensorRT 8, fixed batch=1, ~25–40 fps
*total* across all cameras). `orin-nano` targets the Orin Nano (TensorRT 10,
dynamic batching). The engines and code paths are **not interchangeable**.

---

## 7. Testing

Off-device (no GPU, runs on any dev machine):

```bash
cd Backend/tensort
python3 tests/test_frame_pool.py      # eviction, fairness, staleness
python3 tests/test_dispatch.py        # late results, failure gating, sweeper
```

Cloud-side, from `Backend/`:

```bash
PYTHONPATH=$PWD python3 tests/test_tracker_flicker_recovery.py
PYTHONPATH=$PWD python3 tests/test_tracker_fast_motion.py
```

On-device, in this order:

1. Build the engine, run the throughput gate above.
2. `python3 tests/diag_batch.py <image> 10` — all 10 rows must produce identical
   detections, which proves dynamic batching and the pinned-buffer preprocessing
   are both correct.
3. Start with one camera; confirm `infer_ok` climbs and `pool_evicted_total` ≈ 0.
4. Scale to ten; confirm per-camera result rates are roughly **equal** (that is
   the fairness guarantee) and `pool_evicted_total` is stable.
5. Watch `tegrastats` for GPU utilisation and memory headroom.

---

## 8. Gotchas worth knowing

- **Never `pip install opencv-python` on a Jetson.** The wheel has no GStreamer
  support and silently kills hardware decode. Use the JetPack system OpenCV
  (`requirements-jetson.txt` explains this).
- **The engine is not portable.** Rebuild on the target device after any JetPack
  or TensorRT upgrade.
- **`IMG_SZ` must match the engine.** The service now fails fast at startup
  instead of raising on every frame.
- **One worker is correct on one GPU.** More workers means more CUDA contexts
  time-slicing the same silicon and smaller batches.
- **Per-camera frame order is load-bearing.** The cloud drops out-of-order
  `frame_seq`, so the pool's FIFO-per-camera property must be preserved by
  anything that touches batching.
- **`self._seq` resets to 0 when a channel restarts**, which the cloud reads as a
  Jetson restart. Expect a brief tracker adjustment after a camera reconnects.
