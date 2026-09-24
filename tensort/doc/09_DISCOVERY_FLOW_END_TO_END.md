# 09 — The discovery flow end to end, with real values

This document follows **one camera** from the moment the Jetson's scheduler wakes
up to the moment a user sees its video in the browser. At every hop it shows the
**input**, the **transformation**, and the **output** as literal values.

Documents 01–08 explain each subsystem. This one is the *connecting thread*: it
is the document to read when a camera does not appear, or appears late, because
it shows exactly which value is produced where and who reads it next.

Everything here was traced against the checked-out source on **2026-09-22**, and
the configuration values are read from the **deployed `Backend/tensort/.env`**
you are actually running, not from `.env.example`. Where the two disagree, this
document uses the deployed value and says so. Credentials are never reproduced.

---

## 0. The configuration this tower actually runs

Read this table first. Almost every timing number later in this document comes
from it, and several values differ sharply from `.env.example`, which is what
documents 02 and 06 quote.

| Setting | `.env.example` | **Your deployed `.env`** | Consequence |
| --- | --- | --- | --- |
| `STATIC_NVRS` | empty | `initialsecurity.dvrlists.com:16805:1-6;` `innovationdistrict.asuscomm.com:12050:1-2` | 8 declared channels, reached over the **public internet via DDNS**, not a LAN |
| `DISCOVERY_LOCAL_ENABLED` | `true` | **`false`** | No WS-Discovery, no subnet sweep. Only the declared channels exist |
| `DISCOVERY_INTERVAL_S` | 150 | **240** | Scheduler gap and quiet period after a sweep completes |
| `CAMERA_START_STAGGER_S` | unset (code default 1.5) | **6** | Global gap between *every* capture open, process-wide |
| `GST_LATENCY_MS` | unset (code default 200) | **4000** | 4-second RTSP jitter buffer on every stream |
| `DEFAULT_RESIZE_H` | 360 | **480** | Capture resizes to 640×480 |
| `DEFAULT_SAMPLE_FPS` | 3 | 3 | Frames requested per camera per second |
| `MAX_CAMERAS` | 8 | 8 | Selection ceiling |
| `HIK_PASSWORD` | placeholder | **empty** | Harmless here — see §1.2 |
| `DISCOVERY_SWEEP_ONLY_IF_WSD_EMPTY` | false | true | Inert while `DISCOVERY_LOCAL_ENABLED=false` |

> **The single most important structural fact:** your cameras are **not on the
> Jetson's LAN**. They are channels on two Hikvision recorders reached through
> DDNS hostnames on non-standard ports. Every timing assumption that documents
> 02 and 03 make about LAN round-trips is optimistic for this tower.

### 0.1 A declared-channel deployment skips most of discovery

Because `DISCOVERY_LOCAL_ENABLED=false` and `DISCOVERY_NVRS` is empty, only one
of the four candidate sources in `scan_network()` produces anything:

```text
_static_nvr_cameras(cfg)   -> 8 devices    <-- the ONLY live source for you
_scan_remote_nvrs(cfg)     -> []           (DISCOVERY_NVRS is empty)
_scan_local_network(cfg)   -> skipped      (DISCOVERY_LOCAL_ENABLED=false)
```

So `scan_network()` does **zero network I/O**. It returns in microseconds. This
is important and often misdiagnosed: *your sweep time is not spent scanning.* It
is spent entirely in **frame verification** (§3), which opens real RTSP sessions.

---

## 1. Stage 1 — The Jetson builds candidates

**Runs on:** thread `discovery-scheduler`
**Code:** [`discovery.py`](../discovery.py) `scan_network()` → `_static_nvr_cameras()`

### 1.1 Input: the `STATIC_NVRS` string

```dotenv
STATIC_NVRS=initialsecurity.dvrlists.com:16805:1-6;innovationdistrict.asuscomm.com:12050:1-2
```

`parse_remote_nvrs()` splits on `;`, then on `:`. The grammar is
`host:rtsp_port[:channels[:http_port]]`.

**Output — two `RemoteNvr` tuples:**

```python
RemoteNvr(host='initialsecurity.dvrlists.com',   rtsp_port=16805, channels=[1,2,3,4,5,6], http_port=80)
RemoteNvr(host='innovationdistrict.asuscomm.com', rtsp_port=12050, channels=[1,2],         http_port=80)
```

`_parse_channel_spec("1-6")` expands the range: `"1-6"` → `[1, 2, 3, 4, 5, 6]`.
A list form works too: `"1,3,5-7"` → `[1, 3, 5, 6, 7]`. A **malformed** spec
returns `[]`, and the entry is skipped entirely — it never silently widens to
"all channels".

> ⚠️ **Mismatch between your comment and your configuration.** The comment block
> above `STATIC_NVRS` in your `.env` says *"McCallum Office (Abbotsford):
> channels 1-15"* and records that all 15 answered `DESCRIBE 200 OK`. But the
> live setting declares **`1-6`**. Channels 7–15 are not candidates, are never
> verified, and can never reach the cloud. If you expect 15 cameras from that
> recorder and see 6, this line is the reason — not a bug in the code.
> Note that raising it to `1-15` gives 17 declared channels against
> `MAX_CAMERAS=8`; see §2.2 for what the truncation then does.

### 1.2 Output: eight `HikDevice` records

`_static_nvr_cameras()` builds one record per declared channel. Static channels
are **never probed over ISAPI**, so serial, MAC and firmware are all `None`:

```python
HikDevice(
    ip='initialsecurity.dvrlists.com',   # the hostname IS the "ip" field here
    serial_number=None,                   # never probed
    model='NVR-Static-Channel',           # a literal marker, not a real model
    firmware=None,
    device_name='initialsecurity ch1',    # host.split('.')[0] + ' ch' + channel
    mac=None,
    channel=1,
    rtsp_port=16805,
    is_nvr=True,
)
```

Because `serial_number` and `mac` are both `None`, `identity_of()` falls to its
third branch, and because `is_nvr=True` it appends the channel:

```python
identity_of(dev) == 'ip:initialsecurity.dvrlists.com#1'
```

**The eight identities for your tower, exactly:**

```text
ip:initialsecurity.dvrlists.com#1        ip:initialsecurity.dvrlists.com#4
ip:initialsecurity.dvrlists.com#2        ip:initialsecurity.dvrlists.com#5
ip:initialsecurity.dvrlists.com#3        ip:initialsecurity.dvrlists.com#6
ip:innovationdistrict.asuscomm.com#1     ip:innovationdistrict.asuscomm.com#2
```

> **Why this matters for a moving tower.** An `ip:` identity is the *weakest* of
> the three. A `serial:` identity survives a DHCP move; an `ip:` identity does
> not. Yours are keyed to **DDNS hostnames**, which is actually more stable than
> a raw IP — the hostname follows the recorder. But if you ever re-point a DDNS
> name at a different recorder, every camera behind it silently keeps its old
> identity and roster history. That is a rename, not a new camera, as far as
> this code is concerned.

`HIK_PASSWORD` being empty is harmless **in this deployment**: it is only used
for direct-camera ISAPI probes and direct-camera RTSP URLs, and you have neither.
`scan_network()` does log a warning about the empty password, but it is
suppressed when `STATIC_NVRS` is set — which it is.

### 1.3 Building the RTSP URL

`DiscoveryConfig.rtsp_url_for(dev)` turns a device into a connectable URL.

```python
# dev.is_nvr is True -> use the NVR account, not HIK_*
username, password = cfg.nvr_username, cfg.nvr_password   # 'monitor1', <secret>

channel = dev.channel + cfg.nvr_channel_offset            # 1 + 0 = 1
stream_id = channel * 100 + cfg.rtsp_stream               # 1*100 + 2 = 102
```

**Output:**

```text
rtsp://monitor1:<url-encoded-secret>@initialsecurity.dvrlists.com:16805/Streaming/Channels/102
```

The credentials are percent-encoded with `quote(..., safe='')`. Your NVR password
contains a `$`, which is safe in a URL, but the encoder also protects the
dangerous cases: a password `p@ss/word` becomes `p%40ss%2Fword`, so the `@` is
not mistaken for the credential/host separator.

**The full stream-id table for your two recorders** (`HIK_RTSP_STREAM=2`,
substream; `NVR_CHANNEL_OFFSET=0`):

| Identity | Channel | Stream id | Path |
| --- | --- | --- | --- |
| `ip:initialsecurity.dvrlists.com#1` | 1 | 102 | `/Streaming/Channels/102` |
| `ip:initialsecurity.dvrlists.com#2` | 2 | 202 | `/Streaming/Channels/202` |
| `ip:initialsecurity.dvrlists.com#6` | 6 | 602 | `/Streaming/Channels/602` |
| `ip:innovationdistrict.asuscomm.com#1` | 1 | 102 | `/Streaming/Channels/102` |
| `ip:innovationdistrict.asuscomm.com#2` | 2 | 202 | `/Streaming/Channels/202` |

Switching `HIK_RTSP_STREAM` to `1` would request the **main** stream: `101`,
`201`, … That is a much larger picture, far more decode cost and far more WAN
bandwidth — on a metered DDNS link that is a significant change, not a tweak.

---

## 2. Stage 2 — Selection reserves the slots

**Code:** [`service.py`](../service.py) `DiscoveryService._sweep()`

```python
devices = scan_network(self._cfg)          # your 8 static HikDevices
devices = devices[:camera_limit()]         # camera_limit() == MAX_CAMERAS == 8
selected_identities = {dev.identity() for dev in devices}
self._runtime.select_discovery_sources([cfg.rtsp_url_for(d) for d in devices])
```

### 2.1 What `select_discovery_sources` does to running cameras

This call happens **before any frame is verified**. It reserves the eight URLs
and, critically, **stops any running camera whose URL is not in the new set**:

```python
for camera in self.list_cameras():
    if camera["source_url"] not in selected:
        self._call(self._require_pipeline().remove_channel(camera["camera_uuid"]))
        self._cameras.pop(camera["camera_uuid"], None)
self._restore_cameras_from_db()
```

Since your URLs are rebuilt deterministically from `.env` every sweep, they are
byte-identical each time and nothing is retired. **But** the URL embeds the
credentials and the stream path. So changing `NVR_PASSWORD`, `HIK_RTSP_STREAM`
or `NVR_CHANNEL_OFFSET` changes every URL at once, retires all eight cameras,
and walks straight into the **B1** defect (document 08): the old camera is
removed before its identity is re-linked, and the roster keeps a `camera_uuid`
with no running capture behind it.

> **Operational rule:** treat any edit to `NVR_USERNAME`, `NVR_PASSWORD`,
> `HIK_RTSP_STREAM`, `NVR_CHANNEL_OFFSET` or `STATIC_NVRS` as a **migration**,
> and restart the edge service deliberately rather than letting a live sweep
> discover the change.

### 2.2 Truncation happens before verification, not after

`devices[:8]` runs **before** any stream is opened. The ordering inside
`scan_network()` is static → dynamic → local-NVR → local-camera, and
configuration order breaks ties within the static group.

Concretely, if you change `1-6` to `1-15` on the McCallum recorder you get 17
candidates, truncated to the **first 8 in declaration order**:

```text
kept:    McCallum ch1..ch8
dropped: McCallum ch9..ch15, and BOTH Penticton channels
```

Penticton would disappear entirely — not because it is offline, but because it
is declared second. If you need cameras from both sites on one tower, either
raise `MAX_CAMERAS` (and prove the GPU sustains it, see §7) or interleave the
declarations so each site gets slots.

---

## 3. Stage 3 — Frame verification: where your sweep time goes

**Code:** [`runtime.py`](../runtime.py) `verify_source()`, [`channels/channel.py`](../channels/channel.py) `_ConnectGate`

For each of the eight candidates, `_inspect_candidate()` calls
`verify_source(url, stop_evt)`. There are two very different paths.

### 3.1 Fast path — the camera is already running

```python
max_age = max(15000, int(3000 / sample_fps))    # max(15000, 3000/3=1000) = 15000 ms
if age is not None and age <= max_age:
    return True
```

With `DEFAULT_SAMPLE_FPS=3` this is a flat **15 seconds**. A running camera
whose last frame arrived within 15 s verifies instantly, with **no new RTSP
session**. In steady state, all eight cameras take this path and the whole
verification phase costs milliseconds.

> "Verified" therefore does **not** mean "a frame arrived just now". It means
> "a frame arrived within the last 15 seconds".

### 3.2 Slow path — a new or stopped camera

This is the cold-start and recovery path, and it is expensive:

```text
1. CONNECT_GATE.wait_turn()     <-- global 6-second reservation
2. VideoChannel._open_capture() <-- GStreamer ladder, up to 6 candidates
3. _probe_first_frame(cap)      <-- up to CAPTURE_PROBE_S = 12 s on candidate 1
4. cap.release()                <-- the probe session is thrown away
   ...then adoption opens a SECOND, persistent session (another gate turn)
```

**The connect gate is a reservation, not a rate limiter.** Reading
`_ConnectGate.wait_turn()`:

```python
with self._lock:
    now = time.monotonic()
    start_at = max(now, self._next_allowed_at)
    self._next_allowed_at = start_at + self._min_interval_s   # +6 s, unconditionally
```

Every caller pushes the next slot 6 seconds further out, **whether or not the
connection succeeds**. With `CAMERA_START_STAGGER_S=6`:

| Phase | Gate turns | Gate time |
| --- | --- | --- |
| Verification probe, 8 cameras | 8 | 48 s |
| Adoption's persistent open, 8 cameras | 8 | 48 s |
| **Cold-start total, gate alone** | **16** | **96 s** |

That is **before** any RTSP handshake, keyframe wait or decode. Add the probe
budget for any channel that does not produce a frame on its first GStreamer
candidate and a cold start legitimately runs into several minutes.

### 3.3 The 4-second jitter buffer compounds it

`GST_LATENCY_MS=4000` sets `rtspsrc latency=4000`. That buffer exists for good
reason on a WAN link — it absorbs jitter that would otherwise shred the stream —
but it has two costs:

1. **Startup:** the pipeline will not emit its first frame until the buffer has
   filled. Four seconds of the 12-second `CAPTURE_PROBE_S` budget is spent
   before a frame can appear at all.
2. **Steady-state latency:** every frame is deliberately held ~4 s before it
   reaches the decoder.

This is the **largest single contributor to your "detections are delayed"
symptom**, and it is invisible to the edge's own metrics. `ts_ms` is stamped
*after* `grab()` succeeds, i.e. **after** the jitter buffer. So
`last_frame_age_ms` in `/health` can read a healthy `120 ms` while the image
content is already ~4 s old.

```text
Real timeline for one frame on your tower (illustrative magnitudes):

  camera sensor ─┬─ NVR encode/mux ─┬─ WAN transit ─┬─ rtspsrc buffer ─┬─ decode
                 │                  │               │    ~4000 ms      │
                 └──────── invisible to the edge ────────────────────┘ │
                                                          ts_ms stamped ┘
                                                          age_ms counts from HERE
```

**To measure the true end-to-end lag**, put a visible clock in front of a camera
and compare it with the browser overlay. Do not trust `last_frame_age_ms` for
this; it structurally cannot see the buffer.

---

## 4. Stage 4 — The roster row in SQLite

**Code:** [`repositories/discovery_repository.py`](../repositories/discovery_repository.py) `mark_seen()`
**Table:** `discovered_cameras` in `Backend/tensort/jetson_cameras.db`

A verified candidate is written to the roster. **Input** is the `HikDevice` plus
the URL; **output** is a row whose `to_dict()` shape is consumed by the cloud.

```json
{
  "identity": "ip:initialsecurity.dvrlists.com#1",
  "ip_address": "initialsecurity.dvrlists.com",
  "mac_address": null,
  "serial_number": null,
  "model": "NVR-Static-Channel",
  "firmware": null,
  "device_name": "initialsecurity ch1",
  "source_url": "rtsp://monitor1:***@initialsecurity.dvrlists.com:16805/Streaming/Channels/102",
  "public_rtsp_url": "rtsp://monitor1:***@initialsecurity.dvrlists.com:16805/Streaming/Channels/102",
  "rtsp_port": 16805,
  "camera_uuid": "11111111-1111-4111-8111-111111111111",
  "is_present": true,
  "first_frame_at": "2026-09-22T09:14:03.120000",
  "verification_state": "online",
  "consecutive_misses": 0,
  "alerted": false,
  "first_seen_at": "2026-09-20T11:02:44.900000",
  "last_seen_at": "2026-09-22T09:14:03.120000",
  "missing_since": null
}
```

Three fields deserve a junior developer's attention:

- **`public_rtsp_url` is a lie of a name.** `mark_seen()` assigns it the *same*
  value as `source_url`. Nothing makes it publicly reachable.
- **`is_present` is computed, not stored.** `to_dict()` returns
  `bool(self.is_present) and self.first_frame_at is not None`. A legacy row that
  never proved a frame reports `is_present: false` even if the column says true.
- **`verification_state`** is derived the same way: `unverified` when
  `first_frame_at is None`, else `online`/`offline`. The cloud **refuses to
  adopt** anything `unverified` (§6.1).

### 4.1 The missing-camera state machine (and its confirmed bug)

`mark_missing(present_identities, miss_threshold=2)` ages every row **not** in
the present list:

```text
Sweep 1: absent -> consecutive_misses = 1, missing_since = now
         1 < 2, so no alert yet  (grace window)
Sweep 2: absent -> consecutive_misses = 2
         2 >= 2 -> is_present = False, alerted = True, RETURNED as newly missing
Sweep 3: absent -> consecutive_misses = 3, but alerted is already True
         -> NOT returned again (no repeat alerting)
```

At `DISCOVERY_INTERVAL_S=240` (gap and quiet period after each sweep),
two sweeps is **8+ minutes** before a dead camera is reported missing.

> ### ⛔ Verified defect (B2) — reproduced on this checkout, 2026-09-22
>
> ```
> FAIL: test_missing_respects_grace_window_and_alerts_once   AssertionError: 0 != 1
> FAIL: test_recovery_after_missing                          AssertionError: 0 != 1
> ```
>
> In `_sweep()`, rows whose identity is in neither the current nor the previous
> selection are classified as `excluded` **historical** rows and passed to
> `mark_missing()` *as if they were present*:
>
> ```python
> excluded = [row["identity"] for row in self._repo.list_all()
>             if row["identity"] not in (selected_identities | previous_selection)]
> report["missing_cameras"] = self._repo.mark_missing(present_identities + excluded, ...)
> ```
>
> A camera that drops out of the candidate list stops aging after one miss and
> **never reaches the threshold**, so it never alerts and never "recovers".
>
> **Does this affect your tower?** Your candidate list is static — the eight
> identities are rebuilt from `.env` every sweep and never vary. So a camera that
> merely *fails verification* stays in `selected_identities`, is correctly absent
> from `present_identities`, and **does** age normally. The bug bites when the
> candidate set itself changes: editing `STATIC_NVRS`, or any future move to
> ISAPI/local discovery. Fix it before you rely on missing-camera alerts.

---

## 5. Stage 5 — Adoption on the edge mints the UUID

**Code:** [`service.py`](../service.py) `_adopt()` → [`runtime.py`](../runtime.py) `add_camera()`

`_inspect_candidate()` decides whether the camera needs adopting:

```python
needs_adoption = state.get("is_new") or (row is not None and not row.get("camera_uuid"))
```

So adoption happens on **first sight**, and again on any later sweep if the row
has no `camera_uuid` (for instance because a previous adoption hit the capacity
limit). Before minting, `_existing_camera_uuid_for()` tries four matches in
strength order:

| # | Match | Applies to your tower? |
| --- | --- | --- |
| 1 | Roster's linked `camera_uuid` is an active camera | Yes — the normal steady-state hit |
| 2 | Running camera's `config.discovery_identity` equals this identity | Yes |
| 2b | Same `discovery_serial` **and** same channel | No — your serials are `None` |
| 3 | Credential-free stream endpoint equality | Yes |
| 4 | Host-only fallback, **direct cameras only** | **No** — guarded by `if not is_nvr` |

Match 4 being NVR-guarded is deliberate and important: eight channels share one
host, so a host-only match would collapse all of them into one camera.

If nothing matches, the edge mints a UUID **itself**:

```python
camera_uuid = str(uuid_mod.uuid4())     # e.g. '11111111-1111-4111-8111-111111111111'
result = self._runtime.add_camera(source_url, {
    "camera_uuid": camera_uuid,
    "channel_id": camera_uuid,
    "enabled": True, "detection_enabled": True, "notification_enabled": True,
    "discovered": True,
    "discovery_identity": 'ip:initialsecurity.dvrlists.com#1',
    "discovery_ip": 'initialsecurity.dvrlists.com',
    "discovery_model": 'NVR-Static-Channel',
    "discovery_serial": None,
    "discovery_name": 'initialsecurity ch1',
    "discovery_channel": 1,
    "discovery_is_nvr": True,
})
self._repo.attach_camera_uuid(identity, camera_uuid, source_url=source_url)
```

> **A contradiction worth knowing.** `add_camera()` raises with the message
> *"camera_uuid must be provided by backend (do not let Jetson generate IDs)"* —
> yet the Jetson's own discovery service generates them right here. The message
> is stale; edge-minted UUIDs are the intended design for discovered cameras, and
> the cloud is built to adopt them (§6.2). Do not "fix" this by removing the
> edge's UUID generation.

### 5.1 The saved camera config

`_apply_camera()` normalises, starts the capture, and persists to `camera_configs`:

```python
{
  "camera_uuid": "11111111-1111-4111-8111-111111111111",
  "channel_id":  "11111111-1111-4111-8111-111111111111",
  "source_url":  "rtsp://monitor1:***@initialsecurity.dvrlists.com:16805/Streaming/Channels/102",
  "enabled": True, "detection_enabled": True, "notification_enabled": True,
  "sample_fps": 3.0,                  # clamped to max(0.1, min(fps, MAX_SAMPLE_FPS=8))
  "decode_backend": "gstreamer",
  "resize": (640, 480),               # from DEFAULT_RESIZE_W/H
  "gst_latency_ms": 4000,
  "reconnect_base_ms": 1000, "reconnect_max_ms": 30000,
  "discovered": True, "discovery_identity": "ip:initialsecurity.dvrlists.com#1", ...
}
```

> ⚠️ **`resize=(640, 480)` distorts a 16:9 stream — and the two capture paths
> disagree with each other.**
>
> In the **GStreamer** path (`_gst_appsink_tail`, the one you actually run) the
> resize becomes a hard caps filter:
>
> ```text
> nvvidconv interpolation-method=1 ! video/x-raw,width=640,height=480,format=BGRx !
> ```
>
> There is no `add-borders`, no pixel-aspect preservation. A 1280×720 substream
> is **stretched** into 4:3: every object becomes ~33 % narrower than reality.
>
> In the **OpenCV/FFmpeg fallback** path, `_maybe_resize()` does the opposite —
> it scales to *fit* and never upscales:
>
> ```python
> scale = min(target_w / fw, target_h / fh)   # min(640/1280, 480/720) = 0.5
> new_w, new_h = 640, 360                     # aspect PRESERVED
> ```
>
> So the same camera produces geometrically different images depending on which
> decoder won — which also makes fallback events hard to compare.
>
> The letterbox in `trt_infer.py` cannot undo distortion introduced at capture;
> it pads an already-squashed image. This costs accuracy on exactly the tall,
> thin objects (standing people at distance) that matter most here.
> **Set `DEFAULT_RESIZE_H=360`**: it matches 16:9, makes both paths agree, and
> decodes ~25 % fewer pixels per frame.

---

## 6. Stage 6 — The cloud pulls, adopts, and files the camera

**The cloud always initiates.** The Jetson never calls the cloud.

```mermaid
sequenceDiagram
    participant L as Cloud reconcile loop (every 30 s)
    participant R as DeviceReconciler
    participant J as Jetson /sync
    participant A as CameraAdopter
    participant D as Cloud MySQL
    participant M as MediaMTX
    L->>R: reconcile_all_devices_edge()
    R->>M: list + ensure WHEP streams
    R->>J: POST /sync            (timeout 45 s)
    J-->>R: {cameras, discovery{roster}, status}
    R->>A: adopt_discovered_cameras(roster)
    A->>D: resolve exactly ONE site for the device
    A->>D: create Camera + ChannelConfiguration (reusing edge UUID)
    R->>J: GET /cameras          (timeout 5 s)
    R->>R: to_add / to_remove diff
    R->>J: POST /cameras (upserts)
```

### 6.1 `POST /sync` — what the cloud actually receives

`sync()` in [`routes/discovery_routes.py`](../routes/discovery_routes.py) does
**not** block for a sweep. It schedules one and returns immediately:

```python
pending = svc.request_scan()      # returns True if scheduled OR in quiet period
report  = svc.current_report()    # last COMPLETE report, or a partial one
```

```json
{
  "cameras": [ { "camera_uuid": "...", "source_url": "...", "config": {...} } ],
  "discovery": {
    "type": "discovery_report",
    "generated_at": "2026-09-22T09:14:07.880000Z",
    "duration_s": 96.4,
    "scanned": 8,
    "present": [ ... ], "roster": [ ... ],
    "new_cameras": [], "missing_cameras": [], "recovered_cameras": [],
    "unverified_candidates": [], "capacity_rejected": [], "ip_changes": [],
    "errors": []
  },
  "status": { "in_quiet_period": true, "scan_in_progress": false, ... },
  "discovery_pending": true
}
```

Two traps here, both of which produce "the camera takes ages to appear":

1. **`discovery_pending: true` does not mean a scan is running.** `request_scan()`
   returns `True` when it schedules a sweep *and* when it declines because of the
   quiet period. Read `status.scan_in_progress` for the truth.
2. **`current_report()` can return a PARTIAL report.** While a sweep holds the
   lock, it returns a synthetic report whose roster is filtered to rows that are
   *both* selected *and* currently active:

   ```python
   report["partial"] = True
   active = {c["camera_uuid"] for c in self._runtime.list_cameras()}
   report["roster"] = [row for row in self._repo.list_all()
                       if row["identity"] in self._selected_identities
                       and row.get("camera_uuid") in active]
   ```

   During a cold start — precisely when you are waiting for cameras to appear —
   most rows are not yet active, so the roster is nearly empty and the cloud
   adopts nothing on that pass.

### 6.2 Adoption into the cloud

`CameraAdopter._roster_entries()` filters the roster down to adoptable rows,
skipping anything that is absent, has no URL, is `unverified`, or was
capacity-rejected. Then it requires **exactly one site**:

```text
0 sites -> error "Device is not linked to any site..."   -> NOTHING is adopted
1 site  -> proceed
2 sites -> error "...target site is ambiguous."          -> NOTHING is adopted
```

> **The most common cause of "my camera never shows up in the cloud" is that the
> device is linked to zero sites, or to two.** Check this before investigating
> the network. It fails silently as a warning in the reconcile output.

`_create_camera()` then **reuses the edge UUID** — this is the line that keeps
both sides in agreement:

```python
camera_uuid = entry.get("camera_uuid")
cam_uuid = uuid.UUID(str(camera_uuid)) if camera_uuid else uuid.uuid4()
```

and creates the camera through the *same* path a manually created camera takes,
producing `camera_code = 'cam-' + uuid.hex[:8]` → `cam-11111111`, a MediaMTX
path, a `ChannelConfiguration`, and an edge upsert.

### 6.3 ✅ Fixed defect (B3) — the "Add to site" button used to lose the UUID

There are **two** ways a camera becomes a cloud row. Until 2026-09-22 only one
of them preserved the edge's UUID.

| Path | Entry built by | Carries edge UUID? |
| --- | --- | --- |
| Automatic reconcile adoption | roster entry (has `camera_uuid`) | ✅ Always did |
| **User clicks "add to site"** | `InventoryService._adoption_entry(row)` | ✅ **Now does** (was ❌) |

[`inventory_service.py`](../../application/services/inventory_service.py) stores
the edge UUID as `edge_camera_uuid`, but the adoption entry used to omit it
entirely, so `_create_camera()` saw no `camera_uuid` and minted a fresh one:

```text
Edge is decoding and emitting as   UUID A
Cloud creates the camera as        UUID B
Cloud pushes POST /cameras with    UUID B
  -> edge rejects 409: "Selected source already has a camera UUID"
Cloud subscribes to SSE for        UUID B
  -> edge only ever emits          UUID A
  -> the camera exists, video plays, but NO detections ever arrive
```

Video still worked because MediaMTX pulls the RTSP source directly and never
needs the two UUIDs to agree — which is exactly what made this so confusing to
diagnose from the UI.

**The fix now in place:**

```python
edge_uuid = str(row.edge_camera_uuid or "").strip()
if edge_uuid:
    try:
        entry["camera_uuid"] = str(uuid.UUID(edge_uuid))
    except ValueError:
        logger.warning(...)   # drop it rather than fail the whole add
```

`edge_camera_uuid` is free-text `String(64)`, so a malformed value is dropped
with a warning instead of raising — otherwise `uuid.UUID(...)` in the adopter
would fail the entire add. `NULL`/blank omits the key and the adopter mints one,
as before. Reusing an existing UUID is safe because the channel-create path is an
**upsert**, not an insert.

Covered by `AdoptionEntryTests` in
[`tests/test_camera_inventory.py`](../../tests/test_camera_inventory.py) — five
tests that fail against the old code and pass against the fix.

> **Cameras added before this fix are not migrated**, and they will not announce
> themselves: they still play video and still emit nothing. Find them with the
> read-only audit:
>
> ```bash
> python3 Backend/scripts/audit_inventory_camera_uuids.py
> ```
>
> It exits `1` when any camera's cloud UUID disagrees with the edge UUID stored
> in inventory, and prints the remediation options for each one.

### 6.4 The reconcile diff and why discovered cameras survive it

```python
to_add    = sorted(desired_set - edge_set)
to_remove = sorted(edge_set - desired_set)
```

An edge-discovered camera the cloud has not adopted yet would land in
`to_remove` — and deleting it would start an add/delete loop with the next sweep.
Two guards prevent that:

1. UUIDs present in the roster are carved out of `to_remove` and reported as
   `discovered` (left running).
2. If the edge reports `camera_selection == "discovery"` — which yours does,
   since `DISCOVERY_ENABLED` and `DISCOVERY_AUTO_ADD` are both true — **both
   lists are cleared entirely**:

```python
if selection_managed:
    to_add = []
    to_remove = []
```

> **Consequence you must understand:** on your tower the cloud **cannot** push
> or remove cameras on the edge through the normal diff. The edge owns its
> camera set completely. This also means a disarmed site's cameras are **not**
> torn down on the Jetson (defect B5): the GPU keeps working, only cloud-side
> alerting stops.

---

## 7. Stage 7 — Detections flow back

Once the cloud has a Camera row, `ModelPipeline` opens one long-lived SSE
connection per camera to the edge:

```text
GET http://<device_url>/api/cameras/11111111-1111-4111-8111-111111111111/detections/stream
```

```text
data: {"type":"DetectionsProducedEvent","camera_uuid":"1111...","frame_seq":42,
       "frame_ts_ms":1800000000000,"frame_w":640,"frame_h":480,
       "detections":[{"cls_name":"person","conf":0.83,
                      "box":{"x1":100,"y1":50,"x2":300,"y2":250}}],
       "inference_ms":240,"batch_size":4}

: keepalive
```

A `: keepalive` comment proves the socket is alive; it proves nothing about
frames. `"detections": []` means inference **succeeded** and found nothing —
which is different from `InferenceFailedEvent`.

### 7.1 The batching arithmetic for your tower

Your load is `8 cameras × 3 FPS = 24 images/s`. The dispatcher waits for a free
worker slot *before* drawing frames, then:

```python
events = await self._buffer.get_batch(self._max_batch, self._batch_linger_s)
```

with `INFER_MAX_BATCH=8` and `INFER_BATCH_LINGER_MS=10`. The linger only applies
when the pool is underfilled:

```python
if self._total < max_n and linger_s > 0.0:
    await asyncio.sleep(linger_s)     # 10 ms top-up window
```

At 24 images/s, frames arrive one every ~42 ms on average. A 10 ms linger is
**too short to gather a second frame** most of the time, so at low load the GPU
frequently runs batches of 1–3 instead of 8, paying the per-batch fixed cost
(host→device copy, context setup, synchronize) several times more often than
necessary. This is a throughput inefficiency, not a correctness bug — and it only
matters if `/health` shows `pool_evicted_total` climbing. Measure before tuning:
raising the linger trades a little latency for larger batches.

**Capacity check, done honestly:**

```text
arrival load        = 8 cameras × 3 fps            = 24 images/s
if measured 8-wide batch time = 400 ms  -> 8/0.4   = 20 images/s   -> BACKLOG
if measured 8-wide batch time = 200 ms  -> 8/0.2   = 40 images/s   -> headroom
```

Measure the real number on the device with `tests/diag_batch.py` and
`deployment/check_capacity.py`. Do not assume yolo26m's throughput; your `.env`
header itself flags `DEFAULT_SAMPLE_FPS=3` as an unverified placeholder.

---

## 8. The complete timeline, with your numbers

**Cold start — Jetson boots with nothing in its database:**

| Stage | Cost with your config |
| --- | --- |
| Engine load + worker ready | up to 60 s (65 s outer timeout) |
| `scan_network()` | ~0 s (static only, no network I/O) |
| Verification gate, 8 × 6 s | 48 s |
| Probe/handshake/4 s jitter buffer per camera | 5–15 s each, serialised behind the gate |
| Adoption re-open, 8 × 6 s | 48 s |
| **Edge has all 8 cameras** | **typically 2–4 minutes** |
| Cloud reconcile pass picks it up | 0–30 s |
| Adoption + `ensure_stream` + DB writes | seconds |
| Browser list refresh timer | 0–30 s |
| **Camera visible to the user** | **~3–5 minutes after boot** |

**Steady state — a camera is unplugged and returns:**

| Stage | Cost |
| --- | --- |
| Sweep start-to-start | `sweep duration + max(240, 240)` ≈ **4+ minutes** |
| Two sweeps to mark missing | **~8–10 minutes** |
| Recovery detected on next sweep after it returns | **up to 4 minutes** |

If those windows are too slow for your operators, the levers are
`DISCOVERY_INTERVAL_S` and `DISCOVERY_MISS_THRESHOLD`
— but shortening them means more RTSP sessions against two remote recorders over
DDNS, which is exactly the session pressure the connect gate exists to prevent.
Change one at a time and watch `gstreamer_failures` in `/health`.

---

## 9. Reading the flow from a live tower

One command per stage, in order. Run on the Jetson:

```bash
# 1. Is the scheduler even running, and is it sleeping?
curl -s http://127.0.0.1:8080/discovery/status

# 2. What did the last COMPLETE sweep find? (404 = none finished yet)
curl -s http://127.0.0.1:8080/discovery/report

# 3. What does the roster remember, including absent cameras?
curl -s http://127.0.0.1:8080/discovery

# 4. What is actually running right now?
curl -s http://127.0.0.1:8080/cameras

# 5. Per-camera capture health and counters
curl -s http://127.0.0.1:8080/health
```

Then from the **cloud host** (reachability from the Jetson proves nothing):

```bash
curl -s http://<device_url>/health
curl -s -X POST http://<device_url>/api/sync
```

The UUID must be identical at every layer:

```text
edge /cameras[].camera_uuid          A
edge roster[].camera_uuid            A
cloud Camera.camera_uuid             A
cloud SSE subscription path          A
```

If the cloud shows **B** while the edge emits **A**, you are looking at §6.3 —
that camera was added through the site inventory screen before the fix. Rather
than checking cameras one at a time, run the audit from the cloud host to find
every affected camera at once:

```bash
python3 Backend/scripts/audit_inventory_camera_uuids.py
```

---

## 10. Summary of what to fix first

| Priority | Item | Where | Effort |
| --- | --- | --- | --- |
| ~~1~~ | ~~`_adoption_entry()` drops `edge_camera_uuid`~~ (**B3**) | `inventory_service.py` | ✅ **Fixed 2026-09-22** |
| 1 | `STATIC_NVRS` declares `1-6`, your notes say `1-15` | `.env` | Config |
| 2 | `resize` 640×**480** distorts 16:9 → use 640×360 | `.env` | Config |
| 3 | `GST_LATENCY_MS=4000` dominates detection lag | `.env` | Measure, then lower cautiously |
| 4 | `CAMERA_START_STAGGER_S=6` → 96 s of pure gate time at cold start | `.env` | Measure recorder tolerance first |
| 5 | Audit cameras added *before* the B3 fix for UUID mismatch | `scripts/audit_inventory_camera_uuids.py` | Run it, then re-point |
| 6 | Missing-camera aging stops after one miss (**B2**) | `service.py:214` | Real fix, see doc 08 |
| 7 | Source change retires camera before re-link (**B1**) | `service.py` / `runtime.py` | Real fix, see doc 08 |

Items 1–4 are configuration and you can change them today. Item 5 is the cleanup
left behind by B3: the fix restores continuity for new adds but does not migrate
cameras already filed under a mismatched UUID. Items 6 and 7 are code changes
with acceptance criteria in [document 08](08_OPTIMIZATION_AND_BUGS.md).

> Verification note: B3 was reproduced, **fixed, and covered by five regression
> tests** that fail against the old code (2026-09-22). B2 is **reproduced** (two
> failing tests on this checkout). B1 is **source-confirmed** by direct code
> trace. The configuration findings are read from the `.env` you supplied. The
> timing figures are arithmetic from those settings, not measurements from your
> tower — measure before and after any change.
