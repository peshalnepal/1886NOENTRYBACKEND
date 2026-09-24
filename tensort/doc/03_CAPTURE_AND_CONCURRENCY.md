# 03 — Capture, threads, asyncio, and queues

## What a thread and a coroutine mean here

A **thread** is an operating-system execution path. A blocking RTSP read can wait inside one capture thread while other threads continue. A **coroutine** is a cooperatively scheduled function on an asyncio event loop. It lets other coroutines run when it awaits an operation that yields. Writing `async def` does not make a blocking function nonblocking.

```python
# Bad inside the pipeline loop: blocks every coroutine on that loop.
time.sleep(2)

# Cooperative wait: the loop can handle other cameras while this task waits.
await asyncio.sleep(2)

# For an appropriate blocking CPU/I/O operation, use an executor.
result = await loop.run_in_executor(executor, blocking_function, argument)
```

These are teaching fragments, not changes applied to the service. Do not move CUDA calls into arbitrary executor threads: their context ownership must remain coherent. Python documents `run_coroutine_threadsafe()` for submitting a coroutine to a loop from another thread. See [Python's asyncio task documentation](https://docs.python.org/3.10/library/asyncio-task.html).

In the usual CPython build, the GIL restricts simultaneous Python bytecode execution. Native OpenCV/NumPy/GStreamer/CUDA work and network waits have their own execution behavior. Eight Python capture threads do not imply eight independent GPUs or eightfold inference speed.

## Execution owners

| Owner | Main work | Communication |
| --- | --- | --- |
| Main/Flask request threads | HTTP server and synchronous route handlers | `runtime._call()` submits coroutines and waits on a concurrent future |
| `discovery-scheduler` / `discovery-refresh` | Blocking probes and sequential verification | Runtime methods and `call_soon_threadsafe` |
| Temporary probe pool | Parallel ISAPI host probes | Return values collected by scanner |
| `VideoChannel-<id>` per camera | Native capture, sampling, reconnect | Latest-frame handoff to loop |
| `pipeline-loop` | Channel coroutines, frame pool, batch dispatch, futures, broadcaster | Asyncio queues, events, tasks |
| `trt-infer-worker-0` | Preprocessing, TensorRT execution, postprocessing | Thread-safe `queue.Queue` and callbacks to loop |
| `snapshot_*` executor, one worker | JPEG encoding | `run_in_executor()` |
| SQLite worker threads | aiosqlite work | Async database interface |
| GStreamer internal threads | Transport and media elements | Native pipeline internals |

This is an ownership map, not a fixed total thread count. Media libraries, Flask clients, database drivers, and executors can add threads.

`runtime._call(coro)` uses `asyncio.run_coroutine_threadsafe(coro, loop)` and `future.result(timeout=...)`. The caller thread blocks; the pipeline loop keeps running. Calling this same blocking bridge from the pipeline loop would deadlock it. Native capture does the reverse with `loop.call_soon_threadsafe(...)`: it asks the loop to run a short callback without waiting.

## Decode pipeline for RTSP

[`VideoChannel._gst_candidates()`](../channels/channel.py) tries:

1. RTSP `uridecodebin` with a hardware conversion tail.
2. Explicit H.265 with the configured hardware decoder (default `nvv4l2decoder`).
3. Explicit H.264 hardware path.
4. CPU H.265 (`avdec_h265`).
5. CPU H.264 (`avdec_h264`).
6. Software `uridecodebin`.
7. OpenCV/FFmpeg if the GStreamer ladder fails or native GI bindings are unavailable.

An illustrative H.264 path is:

```text
rtspsrc
  -> rtph264depay
  -> h264parse
  -> nvv4l2decoder
  -> nvvidconv / resize
  -> videorate drop-only=true
  -> videoconvert to BGR
  -> appsink drop=true sync=false max-buffers=1
```

Depayloading removes RTP packaging; parsing organizes compressed video data; decoding reconstructs pixels; scaling changes dimensions; color conversion provides the BGR layout expected by preprocessing. `appsink` is the boundary where Python pulls decoded images.

The primary path uses native GStreamer GI bindings through [`GstCapture`](../channels/gstreamer_capture.py), not `cv2.VideoCapture(..., CAP_GSTREAMER)`. This matters because the deployment preflight still insists on OpenCV reporting GStreamer support, although the current runtime's primary capture bridge is independent of OpenCV videoio. Required GStreamer plugins and GI imports still must work.

The `hw` backend label is an attempted pipeline name, not a measured assertion about every negotiated element. Inspect native plugins and utilization when validating hardware decode. The explicit NVIDIA path requests `nvv4l2decoder`; CPU fallbacks are visible in `/health` and logs.

NVIDIA distinguishes hardware video decode from other engines on Orin Nano; see its [Orin power/performance documentation](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html). Decoding H.264/H.265 and running a neural network are separate workloads.

## FPS is not one number

```text
Camera encoded stream:       example 1920×1080 at 25 FPS
Requested edge sampling:     example 3 FPS
Actual inference delivery:   measured, perhaps 2.7 FPS per camera
Browser video playback:      separate MediaMTX stream cadence
```

For GStreamer, `videorate` drops decoded output down to the requested sample rate. It does not make an inter-frame codec stop decoding reference pictures before that element. Reducing the camera substream's FPS/resolution/bitrate can reduce upstream work; reducing only `sample_fps` primarily reduces work downstream of decoding.

For OpenCV/FFmpeg, the thread repeatedly grabs frames and uses a monotonic scheduling condition before retrieving/emitting a sampled frame. Requesting a sample FPS is a target, not a guarantee.

`resize=(640,360)` follows different geometry in the two paths: GStreamer caps request a fixed width/height, while `_maybe_resize()` preserves aspect ratio and avoids upscaling. A 4:3 source can be stretched to 16:9 in the former path. Later letterboxing cannot undo a distortion already introduced at capture. See the optimization report before standardizing aspect-ratio handling.

## A frame's values as it moves

Assume a decoded image has `shape=(360,640,3)` and `dtype=uint8`. The axes mean height, width, and BGR color channels. It occupies `360 × 640 × 3 = 691200` bytes, about 0.659 MiB.

```python
RTSPEvent(
    channel_id="11111111-1111-4111-8111-111111111111",
    camera_uuid="11111111-1111-4111-8111-111111111111",
    ts_ms=1800000000000,  # illustrative wall-clock timestamp, milliseconds
    seq=42,
    frame=frame_bgr,     # numpy array, not a JPEG or JSON string
    detection_enabled=True,
)
```

`ts_ms` is assigned at the Jetson after a successful `grab()`, before retrieval/resizing. It is **not the camera sensor exposure timestamp**. Time accumulated inside the camera, NVR, encoder, network jitter buffer, or decoder before that point is absent from the edge's reported frame age.

`seq` increments for emitted frames. Dropped frames or camera replacement produce gaps/resets; it is not a globally persistent frame number.

## Why two small handoff stages exist

The capture thread first stores one `_pending_frame` under `_handoff_lock`. It schedules at most one `_drain_handoff` callback. If it captures ten frames before the event loop runs, nine pending frames can be replaced instead of scheduling ten callbacks that retain ten image arrays.

Then `_drain_handoff()` puts the latest event into `_out_q`, an asyncio queue whose default maximum size is 1. If full, it discards the oldest queued event. `_pump_channel()` consumes that queue and sends detection-enabled frames to `FramePool`.

Example:

```text
Capture emits seq 40 -> one callback scheduled
Capture emits seq 41 -> replaces pending seq 40
Capture emits seq 42 -> replaces pending seq 41
Loop runs callback  -> receives seq 42
handoff_dropped increases by 2
```

An asyncio queue is owned by its loop. The capture thread does not call its `put_nowait()` directly. The additional handoff bounds the event-loop callback backlog, which `Queue(maxsize=1)` alone would not bound.

## FramePool: keep freshness and share capacity

[`FramePool`](../frame_pool.py) has one FIFO deque per camera, a global capacity, and an age limit. It has no mutex because accesses happen on the single pipeline loop.

For a small teaching example, suppose capacity is 4:

```text
cam-A: [A1 at 100 ms, A2 at 200 ms, A3 at 300 ms]
cam-B: [B1 at 150 ms]
new arrival: B2 at 350 ms
```

The pool is full. A has the largest backlog, so it evicts A1. Now A and B each hold two frames. A batch of size 2 takes B1 and A2 in oldest-head order, one per camera. A larger batch can take another round. A camera that stops sending frames never makes the scheduler wait for a complete set.

Expiration is checked on puts and batch draws using `ts_ms` and wall time. With a 1000-ms limit and current time 2000 ms, frames older than 1000 ms are discarded. Setting age to zero disables expiry. This limit applies to the **pool**, not to a batch already queued or executing.

Counters have different meanings:

- `handoff_dropped`: capture replaced a pending handoff before the loop consumed it.
- `pool_evicted_total`: pool made room under capacity pressure.
- `pool_expired_total`: frames were too old before dispatch.
- `infer_dropped`: inference submission/failure paths counted dropped work; not all upstream drops.
- `frames_in`: frames drawn for inference; not all decoded camera frames.

## Batch dispatch and completion

The dispatcher checks `InferenceWorkerPool.has_capacity()` **before** drawing frames. The default worker queue has room for one batch, in addition to the batch currently executing. `INFER_MAX_BATCH=8` means up to eight images, not necessarily eight different cameras.

`_prepare_frame_job()` turns each event into `(bgr, meta, future)`:

```python
meta = {
    "camera_uuid": "11111111-1111-4111-8111-111111111111",
    "channel_id": "11111111-1111-4111-8111-111111111111",
    "frame_ts_ms": 1800000000000,
    "frame_seq": 42,
    "_bgr_ref": frame_bgr,
    "_generation": generation_token,
}
```

The image reference lets snapshot encoding use the correct frame. `event.frame=None` releases the event's ownership, not the worker's image data. The future is a promise that a result will arrive; it is not itself a thread.

The worker removes a batch from its thread-safe queue, signals capacity back to the loop, runs inference, and posts completion callbacks with `call_soon_threadsafe`. Each callback resolves its frame's future on the loop. Metadata is per frame, so camera A's result cannot accidentally inherit camera B's final loop variable.

A generation token changes when a channel is replaced. Late results from its old capture generation are ignored, preventing removed cameras from repopulating caches. The single ordered worker is deliberately retained; configuring `INFER_NUM_WORKERS>1` logs a warning and still uses one worker.

## Timeouts do not cancel CUDA

`_sweep_inflight()` periodically resolves overdue futures as `InferenceFailedEvent`. It does not terminate a GPU kernel or remove the worker's batch. A successful late result is still delivered through `late_result_cb`.

With timeout 4 seconds, the sweep interval is `max(0.25, 4/4) = 1` second. A result can therefore be declared late on a later sweep rather than at exactly 4.000 seconds. Failed events are normally not broadcast; genuine successful empty detections are broadcast so trackers can age disappeared objects.

Increasing `INFER_RESULT_TIMEOUT_S` can reduce timeout counters. It does not make inference faster. A timeout can consume `_bgr_ref`, so a later successful result can have detections without a new snapshot.

## Reconnect and probe costs

Every top-level capture-open attempt passes the process-wide gate, default 1.5 seconds. The gate is shared by persistent captures and temporary discovery verification. It spaces starts; it does not serialize an entire handshake or each fallback candidate.

Reconnect starts at 1000 ms, doubles up to 30000 ms, and applies a random factor between 0.75 and 1.25. A 4000-ms base therefore waits about 3–5 seconds. Backoff resets only after an actual frame is decoded.

The first GStreamer candidate has a default 12-second frame budget. Later candidates use `min(CAPTURE_PROBE_S, max(3, latency_seconds+2))`. At ordinary latency, six unsuccessful candidates can spend roughly `12 + 5×3 = 27` seconds in frame probes before FFmpeg fallback, plus construction/teardown/network costs. These budgets are not absolute end-to-end deadlines.

`GstCapture.grab()` normally has a five-second read budget and polls in short intervals for cancellation. `GRAB_MISS_TOLERANCE=15` means a silent stream can spend roughly 75 seconds in unsuccessful reads before reconnect, plus waits, unless EOS/errors make those reads return earlier. This is a deployment hypothesis worth checking when a camera looks connected but stops producing frames.

## Safe resource handling

`GstCapture.retrieve()` maps a native buffer, reads width/height/stride metadata, copies to an owned NumPy array, and unmaps the native memory. Stride is bytes between rows; it can exceed `width*3` because rows are padded. For width 641, visible BGR bytes are 1923 per row but a native stride can be 1924. Blindly reshaping mapped bytes corrupts such images.

Do not remove the `.copy()` without replacing ownership/lifetime handling. After native `unmap`, a view into borrowed memory can become invalid. Similarly, do not release a capture handle from a different thread while `grab()` is reading it. The capture worker releases its own handle in cleanup.
