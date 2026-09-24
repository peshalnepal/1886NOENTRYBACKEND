# 07 — Missing/delayed detections and slow cloud camera appearance

## Begin with one camera and one UUID

Your reported symptoms cross two paths: camera registration and detection delivery. Start with a single known camera and record its edge UUID, discovery identity, cloud UUID, selected endpoint, and device URL. Use redacted URLs in shared notes.

Do not restart everything before collecting evidence. A restart clears the latest reports/counters and can temporarily hide an identity or reconnect problem. No production tower or cloud endpoint was accessed during this review; the steps below are a runbook for that environment.

## First evidence to collect on the Jetson

Run these read-only commands on the deployed device, in its existing environment:

```bash
cat /etc/nv_tegra_release
python3 --version
python3 -c "import tensorrt; print(tensorrt.__version__)"
python3 -c "import cv2; print(cv2.__version__)"
timedatectl status
ip -brief address
ip route
systemctl status jetson-cameras --no-pager
curl --silent --show-error http://127.0.0.1:8080/health
curl --silent --show-error http://127.0.0.1:8080/discovery/status
```

Use the service's Python executable, such as `.venv_trt/bin/python`, if `python3` resolves to a different environment. Read the service unit locally to establish its working directory and environment-file path; do not publish credential-bearing environment values. Keep unredacted diagnostics private.

Collect a bounded log window:

```bash
journalctl -u jetson-cameras --since '15 minutes ago' --no-pager
```

On the Jetson, `tegrastats` can show RAM, CPU, GPU, clocks, and temperatures while the real workload runs. Record the available field names from the installed release. Do not substitute a desktop GPU benchmark for this observation. Query the current power mode with `sudo nvpmodel -q`; changing it is a separate deployment operation.

## Branch A: camera is slow to appear in the cloud

### A1. Does the edge know the candidate?

Read `/discovery/status`, `/discovery/report`, `/discovery`, and `/cameras`. Interpret each separately:

| Observation | Next check |
| --- | --- |
| `pipeline_ready=false` or server never starts | Engine path/version/profile and startup logs |
| First report returns 404 | First sweep has not completed; inspect `scan_in_progress` and logs |
| Candidate not in selected roster | Correct subnet/interface, credentials, channel list, selection priority/limit |
| Identity in `unverified_candidates` | Video connection/codec/frame decode rather than only ISAPI |
| Roster has UUID but `/cameras` lacks it | Source-change/delete recovery defects or admission failure |
| `/cameras` has it and capture is fresh | Move to cloud connectivity/adoption |

A default-route `/24` may scan the tower's uplink rather than camera Ethernet. With an NVR, the recorder's HTTP port and RTSP port must not be confused. With eight static channels selected first, a direct camera can be excluded even if some selected channels are offline.

### A2. Distinguish scheduled scan from forced scan

The default quiet period is 150 seconds after completion. Repeated `/sync` calls do not bypass it. `discovery_pending=true` is the return value of `request_scan()`, which also returns true during a quiet period; inspect `status.scan_in_progress` before assuming a scan is active.

A single deliberate `POST /discovery/scan` can help during commissioning, but it performs network/video work and can contend with captures. It is blocking and should not be called repeatedly by a polling UI. Reducing scan delays without measuring recorder session pressure can recreate reconnect storms.

### A3. Can the cloud reach the edge?

From the cloud host/container, request the configured Jetson device URL's `/health` and `/cameras`. A successful `localhost:8080` request on the Jetson says nothing about cloud reachability. Check the actual VPN/router/firewall path and the hostname used by the cloud.

Use the same UUID when comparing layers:

```text
edge /cameras UUID                  A
edge roster camera_uuid             A
cloud Camera.camera_uuid            A
cloud subscription /cameras/A/...    A
```

If the cloud row is B while the edge emits A, record whether it was created through inventory-add, automatic adoption, or manual registration. The existing B3 defect specifically affects inventory-add.

### A4. Can adoption finish?

Check the cloud reconcile output/logs for:

- Device missing `device_url` or unreachable from the cloud container.
- No site linked to the device, or multiple sites making automatic adoption ambiguous.
- Camera identity marked removed by the user in inventory.
- MediaMTX admin failure during `ensure_stream`, before cloud row creation.
- Database transaction/permission/ownership failure.
- A 409 rejection because another UUID already owns the selected edge URL.

The MediaMTX API accepting a path proves configuration succeeded; it does not prove that MediaMTX can pull that private LAN address. Validate media-server-to-source routing separately.

### A5. Has the browser refreshed?

Read the cloud camera list directly. If the row exists there but not in the browser, inspect authorization/site filtering and frontend refresh. The normal frontend refresh timer is 30 seconds. Inventory refresh reads a completed edge report and is not synonymous with forcing discovery.

## Branch B: edge detection is absent or late

### B1. Capture freshness first

In `/health`, inspect `stats.capture[uuid]`:

```json
{
  "connected": true,
  "backend": "rtsp-uridecodebin-hw",
  "capture_interface": "native-gstreamer",
  "last_frame_age_ms": 120,
  "frames_emitted": 900,
  "handoff_dropped": 0,
  "sample_fps": 3
}
```

These are illustrative values. Compare two samples over time. A growing `frames_emitted` and low frame age suggest decoded frames are reaching capture. `connected=true` with rising age and no emitted frames means an open session is not delivering usable images.

Inspect `gstreamer_failures` and fallback logs. `opencv-ffmpeg` indicates the CPU fallback in this implementation. Native CPU decoder labels also consume CPU; `capture_interface='native-gstreamer'` alone does not prove hardware decode. Missing GI plugins, wrong codec, credentials, jitter/keyframes, and unsupported pipeline properties need separate investigation.

A 15-miss tolerance combined with five-second reads can delay reconnect detection. Conversely, reducing tolerances too far can cause healthy low-FPS or jittery streams to restart repeatedly. Measure actual no-frame intervals before tuning.

### B2. Is detection enabled and dispatched?

Check the saved/running config's `enabled`, `detection_enabled`, and `sample_fps`. A camera can capture with detection disabled. Compare `frames_in`, pool depth, and inference counters.

| Pattern | Interpretation |
| --- | --- |
| Capture counters stationary | Ingest/network/decoder problem first |
| Capture grows, no pool/inference growth | Detection disabled or dead channel/dispatcher task |
| Pool repeatedly full, evictions increase | Arrival load exceeds service capacity or loop/worker stalls |
| Expiry grows | Waiting frames exceed pool age threshold |
| `infer_fail` grows | Read failure reasons: timeout, engine execution, parser, or missing data |
| `infer_ok` grows, `detections_total` remains zero | Successful empty/filter-only outputs; inspect scene, classes, confidence, and parser compatibility |
| `infer_ok` grows, cloud shows nothing | Investigate SSE/UUID/cloud path |

Readiness remains a startup-derived flag. A dispatcher task that has exited can leave health claiming readiness; stable counters are stronger evidence than a single green `ok` field.

### B3. Quantify the real rate

After warm-up, run the existing read-only capacity tool from `Backend/tensort` on the device:

```bash
.venv_trt/bin/python deployment/check_capacity.py \
  --url http://127.0.0.1:8080 --cameras 4 --fps 3 --seconds 120 --max-age-ms 1000
```

Replace four with the number actually running. The tool checks each camera reaches at least 90% of target FPS, samples age/silence, and flags changed failure/drop counters. It exits nonzero if these checks fail. Its “max sampled age” is not a mathematically complete p99/max over every event.

Example calculation from two health readings:

```text
cam-A infer_ok at start = 1200
cam-A infer_ok after 60 s = 1350
delivered rate = (1350 - 1200)/60 = 2.5 FPS
```

Use per-camera deltas. A total device count can hide one starving camera. If an upsert restarts the channel, per-camera counters reset and that interval should be classified as a restart, not interpreted as negative throughput.

### B4. Check the actual engine

Verify `stats.engine_max_batch`, `stats.max_batch`, and `configured_max_batch`. Setting `INFER_MAX_BATCH=8` cannot turn a static batch-1 engine into a dynamic engine. Confirm that the file loaded is the target's `yolo26m` engine, built for its TensorRT stack and `IMG_SZ`.

Use `tests/diag_batch.py` on the Jetson with a non-sensitive test image to check per-row consistency; it requires CUDA/TensorRT and does not run on this review host. Rebuilds and service restarts belong in a planned deployment window, not a read-only triage sequence.

For 8×3 FPS, the requested load is 24 images/s. If live measurements cannot sustain it, reduce requested rate/camera-side stream work before expanding buffers. Large queues mostly trade dropping for older detections.

## Branch C: edge results exist, cloud/browser results are missing

1. Read edge `/cameras/<uuid>/latest` twice; confirm sequence/timestamps advance.
2. Open edge per-camera SSE from the cloud network for a bounded interval. Confirm actual data events, not just keepalive comments.
3. Compare edge/cloud UUIDs and cloud device URL.
4. Check cloud detection-enabled state and whether its background pipeline loaded that camera.
5. Inspect authentication failures. CRUD and detection HTTP clients do not currently send identical authentication headers.
6. Inspect shared HTTPX pool exhaustion at fleet scale. One SSE per camera holds one connection; snapshots/latest use the same 100-connection pool.
7. Inspect ROI database latency before cloud publication. A slow cached-ROI refresh can stop that camera's SSE consumer from draining promptly.
8. Read the cloud latest endpoint, then inspect browser SSE and overlay preferences. A `refresh=true` query only fetches if the cache is empty.

SSE proxies must pass data promptly rather than buffering it into large chunks. The repository's existence of an SSE route does not establish the production reverse proxy's settings. Compare event arrival times at the edge socket, cloud consumer, and browser to locate buffering.

The edge subscriber queue can retain 200 events. At 3 FPS for one camera, that is about 66.7 seconds of history if a consumer stalls and the queue fills. The drop-oldest policy prevents unlimited memory, but it is not an age-based latency bound. Resume tests should inspect whether the client drains old events before current ones.

## Branch D: detections exist but alerts are late/missing

Check raw detections, tracks, ROI state, and notification state separately. A score 0.30 may rescue an existing track but not start a new one. ROI needs confirmed tracks, valid dimensions, a suitable polygon/anchor, and entry/cooldown conditions. Site notification mode may be ROI-only.

Review notification enablement, schedule/timezone, temporary arm override, operator approval, and immediate persistence/media work. The current enqueue path commits directly; retained 60-second buffer settings do not impose a delay on it. Database/media failures or approval waiting can still delay stored/visible alerts, independently of the live overlay.

An edge snapshot can be older than the detection because JPEG caching is asynchronous and detection-only by default. A missing JPEG does not prove inference failed. Clip timestamps also depend on synchronized tower/cloud/media-server clocks; the code's clock-skew warning does not correct them.

## Build a latency timeline

For one event, record:

| Marker | Meaning | Available now? |
| --- | --- | --- |
| T0 | Camera exposure/capture time | Not provided by this event schema |
| T1 | Jetson timestamp after grab | `frame_ts_ms` |
| T2 | Frame admitted to pool | Needs added instrumentation for exact timing |
| T3 | Worker begins batch | Needs added instrumentation |
| T4 | Worker finishes/result handled | Approximate via inference time and health result time |
| T5 | Cloud receives event | Needs log/metric instrumentation |
| T6 | Cloud publishes response | Needs log/metric instrumentation |
| T7 | Browser receives/draws | Browser instrumentation |

`T4-T1` includes queueing and worker work but excludes pre-T1 camera/decoder buffering. `T5-T1` requires synchronized clocks. `inference_ms` measures worker batch work, not `T7-T0`.

Illustrative latency budget, not a measurement:

```text
Camera/network/decode before Jetson stamp:  300 ms
Pool + queued batch waiting:               450 ms
Worker batch processing:                   240 ms
Cloud transit + processing:                 90 ms
Browser transit/render:                     40 ms
Total camera-to-overlay:                 1120 ms
Edge frame_age_ms near completion:         690 ms
```

The edge age looks smaller because the timestamp starts after upstream video delay. That is why “frame age under 1 second” is not automatically “overlay aligns with live video.”

## Tower commissioning matrix

For each real deployment record camera count, topology, codec, source resolution/FPS/bitrate, selected stream, power mode, temperature, JetPack/TRT versions, model artifact, batch profile, sample FPS, and per-camera result rate/age.

Exercise direct cameras, NVR channels, mixed setups, a late camera connection, one disconnected input, NVR reboot, DHCP move, edge restart, cloud restart, cloud network outage, and media-server outage. Verify both UUID continuity and first-detection recovery time. Run source-change/identity scenarios again after fixes, since existing isolated tests do not cover all cross-component behavior.
