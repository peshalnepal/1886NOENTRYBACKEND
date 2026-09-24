# Camera tower developer handbook

This handbook explains the checked-out implementation for a **Jetson Orin Nano 8GB running `yolo26m`**, connected either to cameras directly or to a Hikvision NVR. It follows discovery, video capture, inference, cloud registration, tracking, overlays, and alerts. It also documents defects found during this review. Application code has not been changed.

Review dates: **2026-09-21–22**. The installed JetPack, TensorRT, CUDA, engine profile, camera count, codecs, and production environment values were not supplied. Values called “example” are teaching examples, not measurements from your tower. Values in `.env.example` are a suggested configuration, not proof of what production loads.

## Reading order

| Document | What you will learn |
| --- | --- |
| [01 — Architecture and repository map](01_ARCHITECTURE.md) | What runs on the tower, cloud, media server, and browser; startup and ownership |
| [02 — Camera discovery and Hikvision](02_CAMERA_DISCOVERY.md) | Networks, authentication, direct cameras, NVR channels, identities, selection, verification, and timing |
| [03 — Capture, threads, asyncio, and queues](03_CAPTURE_AND_CONCURRENCY.md) | How a frame travels safely from a decoder thread to the GPU worker |
| [04 — YOLO, TensorRT, and CUDA](04_DETECTION_AND_CUDA.md) | Image shapes, letterboxing, GPU memory, batches, output parsing, and numerical examples |
| [05 — Cloud registration, tracking, and alerts](05_CLOUD_PIPELINE.md) | SQLAlchemy data flow, identity alignment, SSE, MediaMTX, ROI, notifications, and browser behavior |
| [06 — Configuration and API reference](06_CONFIGURATION_AND_API.md) | Actual defaults versus example values, endpoints, persistence, and deployment tools |
| [07 — Diagnose missing or delayed detections](07_TROUBLESHOOTING.md) | A stage-by-stage investigation of your current symptoms |
| [08 — Optimization, cleanup, and bug report](08_OPTIMIZATION_AND_BUGS.md) | Configuration findings (C1–C6), prioritized defects, evidence, proposed fixes, and measurable Orin tuning |
| [09 — The discovery flow end to end](09_DISCOVERY_FLOW_END_TO_END.md) | One camera traced from the Jetson's scheduler to the browser, with the real deployed values at every hop |
| [10 — Camera discovery pipeline](10_CAMERA_DISCOVERY_PIPELINE.md) | Every edge and cloud stage (E0–E11, C1–C7) with its trigger, input, output, and example payloads |

If you are responding to an incident, begin with document 07. If you want to understand how a camera actually becomes visible — or why one has not — read document 09, which follows the whole chain with literal input/output values. If you are implementing changes, read document 08 and its acceptance criteria before changing a queue, timeout, or camera identity rule.

**Document 09 and the C1–C6 section of document 08 are based on the deployed `.env`**, not on `.env.example`. Where documents 02 and 06 quote example values that differ from production, document 09 is the authority for this tower.

## Five different meanings of “camera works”

1. **Candidate:** an address or NVR channel is known.
2. **Verified stream:** the Jetson decoded a real frame.
3. **Edge inference:** successful inference results arrive for the edge camera UUID.
4. **Cloud registration and ingestion:** the cloud has a Camera row and consumes that same UUID's results.
5. **Visible outcome:** the browser shows a live video/overlay, or an alert satisfies its separate notification rules.

Passing one stage does not prove the next. A valid ISAPI response does not prove RTSP works. A working RTSP stream does not prove TensorRT works. A working edge detector does not prove the cloud is listening to the correct UUID. An overlay does not prove an ROI alert should fire.

## Conventions

- `cam-A` and `cam-B` are explanatory aliases. Where an API requires a real UUID, use values such as `11111111-1111-4111-8111-111111111111`.
- Private addresses such as `192.168.50.20` are examples. They are not addresses read from your deployment.
- Credentials in examples are placeholders. Existing secrets are deliberately not reproduced.
- **Verified behavior** means source tracing or an isolated reproduction supports the statement. **Deployment hypothesis** means production evidence is still needed. A confirmed code defect is not automatically the cause of today's incident.
- Source links are relative to this folder. Symbol names are supplied so references remain useful as line numbers change.

## Most relevant findings

**Start with configuration, not code.** Review of the deployed `.env` (2026-09-22) found several settings that explain the reported symptoms on their own: a 4-second RTSP jitter buffer (`GST_LATENCY_MS=4000`) that dominates detection lag and is invisible to the edge's own age metric; a capture resize of 640×**480** that distorts 16:9 video before inference ever sees it; a 6-second connect stagger that costs ~96 seconds of pure waiting at cold start; and a declared channel range (`1-6`) narrower than the 15 channels the adjacent comment records as verified. These are C1–C6 in document 08.

Separately, the code contains several independent ways for cameras or detections to disappear: sequential video verification, missing-camera aging that stops after one absent selection (**reproduced** — two failing tests), source changes that retire a camera before reattachment, repeated upserts that restart capture, and a cloud stream client whose shared connection pool can be occupied by long-lived SSE streams.

**Fixed 2026-09-22:** inventory adoption ("add to site") dropped the edge's camera UUID, so the cloud subscribed to a UUID the Jetson never emitted — the camera showed live video but produced no detections at all. See [B3 in document 08](08_OPTIMIZATION_AND_BUGS.md). Cameras added *before* that fix are not migrated and still need auditing.

These are explained separately from possible GPU overload. Do not infer that `yolo26m` is too slow until capture, identity, delivery, and per-camera result rates have been measured. The repository's throughput estimates are provisional — the deployed `.env` itself flags `DEFAULT_SAMPLE_FPS=3` as an unmeasured placeholder.
