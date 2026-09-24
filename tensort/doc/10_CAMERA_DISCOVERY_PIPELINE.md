# 10 — Camera discovery pipeline: every stage, its input and its output

This document describes how a camera gets from "a box on the network" to "a
Camera row in the cloud that receives detections". It covers two codebases:

- **Edge** — `Backend/tensort/` (Flask, runs on the Jetson). Finds cameras,
  checks that they deliver video, and starts decoding them.
- **Cloud** — `Backend/` (FastAPI + MySQL). Asks the edge what it found,
  registers those cameras under a site, and subscribes to their detections.

Each stage below lists its **trigger**, **input**, **what it does**, and
**output**, with an example value for each. The examples use placeholder
credentials (`<hik_pass>`, `<nvr_user>`, `<nvr_pass>`) and made-up serials and
UUIDs. Code references use symbol names so they stay valid when line numbers
move.

Traced against the checked-out source on **2026-09-22**.

---

## 0. The whole pipeline on one page

```
 EDGE  (Backend/tensort, Flask on the Jetson)
 ───────────────────────────────────────────────────────────────────────────
  E1 Trigger ──► E2 Collect candidates ──► E3 Merge + identity ──► E4 Select (MAX_CAMERAS)
  (60 s loop,     static NVRs / NVR ISAPI /    HikDevice list,           reserve source URLs,
   /sync, /scan)  WS-Discovery + subnet sweep  deduped                   retire other captures
                                                                             │
      ┌──────────────────────────────────────────────────────────────────────┘
      ▼
  E5 Build RTSP URL ──► E6 Verify one decoded frame ──► E7 Roster row (discovered_cameras)
                                                               │
                         E8 Match existing / adopt new ◄───────┘
                         (mint uuid4 → camera_configs + live capture)
                                 │
                         E9 Repoint moved cameras ──► E10 Age missing cameras
                                                               │
                                              E11 Report: cache + SSE broadcast
                                                               │
            HTTP surface: POST /sync · GET /discovery/report · GET /cameras · GET /health
 ─────────────────────────────────────────────────────────────┼─────────────
 CLOUD  (Backend, FastAPI + MySQL)                             ▼
  C1 Trigger ──► C2 Fetch report ──► C3 Adopt into site ──► C4 Reconcile diff ──► C5 Response
  (30 s loop,    POST /sync or        Camera row + MediaMTX   edge list vs         EdgeReconcileOut
   Sync button,  GET /discovery/      stream + edge upsert    desired set
   link device)  report
                                     C6 Inventory (camera_inventory: available/added/removed)
                                     C7 Detections: per-camera SSE from the edge, same UUID
```

### Stage map

| # | Stage | Where | Output |
| --- | --- | --- | --- |
| E1 | Trigger a sweep | `service.py` `DiscoveryService._loop` / `request_scan` / `run_once` | one call to `_sweep()` |
| E2 | Collect candidates | `discovery.py` `_static_nvr_cameras`, `_scan_remote_nvrs`, `_scan_local_network` | `HikDevice` lists |
| E3 | Merge + identity | `discovery.py` `scan_network`, `identity_of` | deduped `List[HikDevice]` |
| E4 | Select | `service.py` `_sweep`, `runtime.py` `select_discovery_sources` | at most `MAX_CAMERAS` devices; reserved URL set |
| E5 | Build URL | `discovery.py` `DiscoveryConfig.rtsp_url_for` | `rtsp://…/Streaming/Channels/NNN` |
| E6 | Verify video | `runtime.py` `verify_source` | `True` / `False` |
| E7 | Roster row | `repositories/discovery_repository.py` `mark_seen` | state dict (`is_new`, `recovered`, …) |
| E8 | Match / adopt | `service.py` `_existing_camera_uuid_for`, `_adopt`; `runtime.py` `add_camera` | `new_cameras[]` entry, running channel |
| E9 | Repoint | `service.py` `_repoint` | `ip_changes[]` entry |
| E10 | Age missing | `discovery_repository.py` `mark_missing` | `missing_cameras[]` |
| E11 | Report | `service.py` `_publish_report`, `_broadcast` | cached report; SSE event |
| C1 | Trigger | `main.py` `_edge_reconcile_loop`; `routes/device_routes.py`; `routes/site_routes.py` | reconcile or adoption call |
| C2 | Fetch report | `application/services/edgeinference.py` `sync_discovery`, `fetch_discovery_report` | report dict or `None` |
| C3 | Adopt | `application/services/manager/controllers/adopt.py` `CameraAdopter` | `adopted` / `linked` / `repointed` / `skipped` |
| C4 | Reconcile diff | `application/services/manager/controllers/reconcile.py` `reconcile_device_edge_simple` | `to_add`, `to_remove`, warnings |
| C5 | Response | `core/schemas.py` `EdgeReconcileOut` | JSON to the caller |
| C6 | Inventory | `application/services/inventory_service.py`, `inventory_repository.py` | `camera_inventory` rows |
| C7 | Detections | `application/channels/channel.py` | SSE subscription per camera UUID |

---

## Part 1 — Edge (`Backend/tensort`)

### E0. Startup: what exists before the first sweep

**Trigger:** `python3 main.py`.

`main.py` loads `.env` (only if `python-dotenv` is installed), then
`create_app()` builds `PipelineRuntime()` and calls
`runtime.start_discovery()`.

`PipelineRuntime.__init__`:

1. Creates the SQLite tables (`camera_configs`, `discovered_cameras`).
2. Starts the asyncio pipeline loop on its own thread (`pipeline-loop`).
3. Sets `discovery_managed = DISCOVERY_ENABLED and DISCOVERY_AUTO_ADD`
   (both default `true`).
4. **Only if `discovery_managed` is false** does it restore every saved camera
   from `camera_configs` right away. When discovery manages cameras, the
   restore is postponed: `select_discovery_sources()` (E4) calls
   `_restore_cameras_from_db()` later, restoring only the saved cameras whose
   `source_url` the sweep selected.

`start_discovery()` builds a `DiscoveryService` and starts the
`discovery-scheduler` daemon thread. If that fails, the error is logged and the
Jetson keeps serving without discovery.

**Output:** a running pipeline with zero or more restored channels, plus a
scheduler thread whose first sweep starts at once.

#### E0 input: configuration (`DiscoveryConfig`, read once)

| Variable | Code default | Your `.env` today | Used for |
| --- | --- | --- | --- |
| `DISCOVERY_ENABLED` | `true` | `true` | Master switch |
| `DISCOVERY_LOCAL_ENABLED` | `true` | *not set → `true`* | WS-Discovery + subnet sweep |
| `DISCOVERY_INTERVAL_S` | 150 (min 60) | 60 | Scheduler gap and quiet period after a sweep |
| `DISCOVERY_MISS_THRESHOLD` | 2 | 2 | Sweeps absent before alerting |
| `DISCOVERY_AUTO_ADD` | `true` | `true` | Adopt into the pipeline |
| `HIK_USERNAME` / `HIK_PASSWORD` | `admin` / empty | `admin` / empty | Direct cameras |
| `NVR_USERNAME` / `NVR_PASSWORD` | fall back to `HIK_*` | set | NVR channels |
| `STATIC_NVRS` | empty | `initialsecurity.dvrlists.com:16805:1-6;innovationdistrict.asuscomm.com:12050:1-2` | Declared channels, never probed over HTTP |
| `DISCOVERY_NVRS` | empty | empty | NVRs probed over ISAPI |
| `DISCOVERY_SUBNETS` | empty → own `/24` | empty | Subnet sweep range |
| `DISCOVERY_SWEEP_ONLY_IF_WSD_EMPTY` | `false` | `true` | Skip subnet sweep if ONVIF answered |
| `HIK_RTSP_PORT` / `HIK_HTTP_PORT` | 554 / 80 | 554 / 80 | Direct camera ports |
| `HIK_RTSP_CHANNEL` / `HIK_RTSP_STREAM` | 1 / 2 | 1 / 2 | URL path (`102` = ch 1 substream) |
| `NVR_CHANNEL_OFFSET` | 0 | 0 | Added to NVR channel in the URL |
| `MAX_CAMERAS` | 8 | 8 | Selection and admission cap |
| `CAMERA_START_STAGGER_S` | 1.5 | 6 | Gap between capture opens (all cameras) |
| `CAPTURE_PROBE_S` | 12 | 12 | Seconds to wait for a first frame |

> Your `.env` does not set `DISCOVERY_LOCAL_ENABLED`, so it is `true`. Every
> sweep sends WS-Discovery and, if no ONVIF device answers, probes the
> Jetson's own `/24`, **in addition to** the two static NVRs. Document 09's
> configuration table lists `false`, `240`, and `4000 ms` for
> `DISCOVERY_LOCAL_ENABLED`, `DISCOVERY_INTERVAL_S` and `GST_LATENCY_MS`. The
> current `.env` has `true` (by default), `60`, and `200`.

---

### E1. Trigger a sweep

**Trigger (three ways):**

| Caller | Method | Blocks the caller? |
| --- | --- | --- |
| Scheduler thread | `_loop()` → `run_once()` every `max(DISCOVERY_INTERVAL_S, remaining quiet period)` | n/a |
| Cloud `POST /sync` | `request_scan()` → starts a `discovery-refresh` thread | **No.** Returns at once |
| Operator `POST /discovery/scan` | `run_once(force=True)` | **Yes.** Waits for the whole sweep |

**Guards:**

- `_sweep_lock` allows one sweep at a time. A second `run_once` waits for the
  running sweep and returns that sweep's report with `"coalesced": true`.
- `request_scan()` does nothing if a sweep is running, a refresh thread is
  alive, or the last sweep finished less than `DISCOVERY_INTERVAL_S` ago.
  It still returns `True`.

**Input:** nothing (configuration only).
**Output:** a call to `_sweep()`, which runs stages E2–E11 in order.

---

### E2. Collect candidates

`scan_network(cfg)` collects devices from three sources.

#### E2a. Static NVR channels (`STATIC_NVRS`) — no network traffic

**Input:** the env string, parsed by `parse_remote_nvrs` into
`RemoteNvr(host, rtsp_port, channels, http_port)`.

```
STATIC_NVRS=initialsecurity.dvrlists.com:16805:1-6;innovationdistrict.asuscomm.com:12050:1-2
→ RemoteNvr(host="initialsecurity.dvrlists.com", rtsp_port=16805, channels=[1,2,3,4,5,6], http_port=80)
  RemoteNvr(host="innovationdistrict.asuscomm.com", rtsp_port=12050, channels=[1,2], http_port=80)
```

**Processing:** one `HikDevice` per channel. The NVR is never contacted over
HTTP. The only check is frame verification in E6. An entry with no channel
range is skipped with a warning.

**Output (one of 8):**

```python
HikDevice(ip="initialsecurity.dvrlists.com", serial_number=None,
          model="NVR-Static-Channel", firmware=None,
          device_name="initialsecurity ch3", mac=None,
          channel=3, rtsp_port=16805, is_nvr=True)
```

#### E2b. Probed NVRs (`DISCOVERY_NVRS`) — ISAPI over HTTP

This source is empty in your `.env`. It is described here because it is the
path for NVRs whose HTTP port is reachable.

**Input:** `host:rtsp_port[:channels[:http_port]]`, for example
`192.168.1.200:554:1-8:80`.

**Processing:** `probe_device(host, …, credentials=(NVR_USERNAME, NVR_PASSWORD))`:

1. `GET http://192.168.1.200:80/ISAPI/System/deviceInfo` with Basic/Digest
   auth. No proxy is used.
2. The response must be `<DeviceInfo>` and contain a `<serialNumber>`.
3. If `deviceType` contains NVR/DVR, or the model starts with `DS-7`, `DS-8` or
   `DS-9`, it is treated as a recorder →
   `GET /ISAPI/ContentMgmt/InputProxy/channels`.

**Input XML (trimmed):**

```xml
<InputProxyChannelList>
  <InputProxyChannel>
    <id>3</id>
    <name>Loading Bay</name>
    <enabled>true</enabled>
    <sourceInputPortDescriptor>
      <model>DS-2CD2143G2-I</model>
      <serialNumber>DS-2CD2143G2-I20230101AAWRL1234567</serialNumber>
      <macAddress>a4:14:37:aa:bb:cc</macAddress>
      <firmwareVersion>V5.7.3</firmwareVersion>
    </sourceInputPortDescriptor>
  </InputProxyChannel>
</InputProxyChannelList>
```

**Output:** one `HikDevice` per enabled channel that passes the filter. The
`ip` is the **NVR** address, because video is streamed through the recorder:

```python
HikDevice(ip="192.168.1.200", serial_number="DS-2CD2143G2-I20230101AAWRL1234567",
          model="DS-2CD2143G2-I", firmware="V5.7.3", device_name="Loading Bay",
          mac="a4:14:37:aa:bb:cc", channel=3, rtsp_port=554, is_nvr=True)
```

If the NVR returns no cameras, a warning suggests checking the credentials and
confirming that the 4th field is the HTTP port rather than the RTSP port.

#### E2c. Local network (`DISCOVERY_LOCAL_ENABLED=true`)

**Step 1 — WS-Discovery (`wsdiscover`).**
*Input:* an ONVIF `Probe` SOAP message, sent three times over UDP to
`239.255.255.250:3702`.
*Processing:* listens for `DISCOVERY_WSD_TIMEOUT_S` (3 s) and collects each
sender IP plus every IPv4 address inside `<XAddrs>`.
*Output:* `{"192.168.1.64", "192.168.1.65"}` (empty if multicast is blocked).

**Step 2 — subnet sweep (`_candidate_hosts`).**
The subnet sweep runs when WS-Discovery found nothing, or when
`DISCOVERY_SWEEP_ONLY_IF_WSD_EMPTY=false`. The range is `DISCOVERY_SUBNETS`,
or the Jetson's own `/24` if that is empty. Only `/16` to `/32` are accepted.
The Jetson's own IP and any host listed in `STATIC_NVRS`/`DISCOVERY_NVRS` are
removed.
*Output:* a sorted list of candidate IPs, e.g. 254 addresses for
`192.168.1.0/24`.

**Step 3 — ISAPI identification (`probe_device`, 32 threads).**
*Input:* one host.

```xml
<!-- GET http://192.168.1.64/ISAPI/System/deviceInfo -->
<DeviceInfo xmlns="http://www.hikvision.com/ver20/XMLSchema">
  <deviceName>Front Gate</deviceName>
  <model>DS-2CD2143G0-I</model>
  <serialNumber>DS-2CD2143G0-I20210315AAWRE12345678</serialNumber>
  <macAddress>44:19:b6:11:22:33</macAddress>
  <firmwareVersion>V5.6.3</firmwareVersion>
  <deviceType>IPCamera</deviceType>
</DeviceInfo>
```

*Processing:*
- HTTP 401 with camera credentials, when NVR credentials differ → retry with
  the NVR account. A device that answers on that retry is accepted only if it
  is a recorder.
- No `<DeviceInfo>`, no serial, a timeout, or a non-Hikvision device → `[]`.
- A recorder → enumerate channels (E2b step 3).

*Output:*

```python
HikDevice(ip="192.168.1.64", serial_number="DS-2CD2143G0-I20210315AAWRE12345678",
          model="DS-2CD2143G0-I", firmware="V5.6.3", device_name="Front Gate",
          mac="44:19:b6:11:22:33", channel=1, rtsp_port=554, is_nvr=False)
```

> ISAPI proves the device is a reachable Hikvision box with working HTTP
> credentials. It does **not** prove RTSP access, that the substream is
> enabled, or that video decodes. E6 checks those.

---

### E3. Merge, deduplicate, assign identity

**Input:** the three lists from E2.

**Processing (`scan_network`):**

1. Priority order: static channels → probed NVR channels → local NVR
   channels → local direct cameras. When a probed NVR reports the same endpoint
   as a static entry, the static position is kept but the probed details
   (serial, model) replace it.
2. Endpoint dedupe on `(ip, rtsp_port, channel, is_nvr)`. The first
   occurrence wins.
3. Identity dedupe with `identity_of(dev)`. The first occurrence wins:

| Device has | Identity | Example |
| --- | --- | --- |
| serial | `serial:<serial>` (+ `#<ch>` if NVR) | `serial:DS-2CD2143G0-I20210315AAWRE12345678` |
| MAC only | `mac:<mac lowercased>` (+ `#<ch>`) | `mac:44:19:b6:11:22:33` |
| neither | `ip:<host>` (+ `#<ch>`) | `ip:initialsecurity.dvrlists.com#3` |

The `#<channel>` suffix matters. Every channel of one NVR shares the NVR's
host, and sometimes its serial. Without the suffix, all channels would collapse
into one camera. The cloud relies on the same rule (`_is_channel_identity` in
`adopt.py`).

**Output:** an ordered, deduplicated `List[HikDevice]`, e.g. 8 static channels
followed by `Front Gate`.

---

### E4. Select at most `MAX_CAMERAS`

**Input:** the ordered device list from E3.

**Processing (`_sweep` → `runtime.select_discovery_sources`):**

1. `devices = devices[:MAX_CAMERAS]`. With 8 static channels and
   `MAX_CAMERAS=8`, `Front Gate` is cut here. **A device cut at this step does
   not appear anywhere in the report**: not as unverified, not as
   capacity-rejected.
2. The selected RTSP URLs become the reserved set `_selected_sources`.
3. Every running capture whose `source_url` is not in the set is removed from
   the pipeline and from memory. Its `camera_configs` row is kept.
4. `_restore_cameras_from_db()` restores saved cameras whose URL is now
   selected (one UUID per URL).

**Output:** `_selected_identities` (a set of identities) and a pipeline whose
running captures are a subset of the selection.

> Because selection is authoritative, `POST /cameras` from the cloud with a URL
> outside the selection fails with **409 "Camera is not selected by
> discovery"** (`_apply_camera`). On a discovery-managed Jetson, only
> discovered cameras can run.

---

### E5. Build the RTSP URL

**Input:** a `HikDevice`.

**Processing (`rtsp_url_for`):** stream id = `(channel + offset) * 100 +
HIK_RTSP_STREAM`. NVR devices use `NVR_*` credentials, direct cameras use
`HIK_*`. Both are percent-encoded.

**Output:**

```
direct   rtsp://admin:<hik_pass>@192.168.1.64:554/Streaming/Channels/102
NVR ch3  rtsp://<nvr_user>:<nvr_pass>@initialsecurity.dvrlists.com:16805/Streaming/Channels/302
```

The credentials stay in the URL because the GStreamer/OpenCV capture takes a
single URL string.

---

### E6. Verify that the stream delivers a decoded frame

Devices are verified **one at a time**, in order, on the discovery thread.

**Input:** `source_url` and the service stop event.

**Processing (`PipelineRuntime.verify_source`):**

- **The URL is already running as an enabled channel:** no new connection is
  opened. The check reads `stats.capture[<uuid>].last_frame_age_ms`. Verified
  if the age is ≤ `max(15000, 3000 / sample_fps)` ms.
- **The URL is new:** it waits for `CONNECT_GATE` (one capture open every
  `CAMERA_START_STAGGER_S`, shared with all reconnects), opens a temporary
  `VideoChannel` capture, waits up to `CAPTURE_PROBE_S` for one real frame,
  then releases it.

**Output:** `True` → continue to E7. `False` → the identity is appended to
`report["unverified_candidates"]` and the device stops here. No roster row is
written.

**Timing example:** 8 new static channels, stagger 6 s, probe up to 12 s →
between about 48 s and 144 s before the last channel is verified. That is why
`/sync` does not wait for the sweep.

---

### E7. Record the camera in the roster (`discovered_cameras`)

**Input:** the verified `HikDevice` and its `source_url`.

**Processing (`DiscoveryRepository.mark_seen`, synchronous SQLite, commits):**
it upserts the row by `identity`, copies the device fields, and sets
`is_present=True`, `consecutive_misses=0`, `alerted=False`,
`missing_since=NULL`, `last_seen_at=now`, and `first_frame_at` (on first
sight).

**Output (state dict):**

```python
{
  "identity": "serial:DS-2CD2143G0-I20210315AAWRE12345678",
  "is_new": True,          # no row existed → adoption candidate
  "recovered": False,      # row existed with is_present=False → "recovered_cameras"
  "ip_changed": False,
  "row": { ...DiscoveredCamera.to_dict()... }
}
```

**Roster row (`to_dict`)** — this shape goes to the cloud in `report.roster`:

```json
{
  "identity": "serial:DS-2CD2143G0-I20210315AAWRE12345678",
  "ip_address": "192.168.1.64",
  "mac_address": "44:19:b6:11:22:33",
  "serial_number": "DS-2CD2143G0-I20210315AAWRE12345678",
  "model": "DS-2CD2143G0-I",
  "firmware": "V5.6.3",
  "device_name": "Front Gate",
  "source_url": "rtsp://admin:<hik_pass>@192.168.1.64:554/Streaming/Channels/102",
  "public_rtsp_url": "rtsp://admin:<hik_pass>@192.168.1.64:554/Streaming/Channels/102",
  "rtsp_port": 554,
  "camera_uuid": null,
  "is_present": true,
  "first_frame_at": "2026-09-22T14:02:11.402913",
  "verification_state": "online",
  "consecutive_misses": 0,
  "alerted": false,
  "first_seen_at": "2026-09-22T14:02:11.402913",
  "last_seen_at": "2026-09-22T14:02:11.402913",
  "missing_since": null
}
```

`verification_state` is `unverified` (never got a frame), `online`, or
`offline`. The roster timestamps are naive UTC with no `Z`.

---

### E8. Match an existing camera, or adopt a new one

**Input:** the device, its identity, `source_url`, and the E7 state.

**Step 1 — find an existing pipeline UUID (`_existing_camera_uuid_for`).**
Checks from strongest to weakest, each across all running cameras:

1. The roster row's linked `camera_uuid` is running.
2. `config.discovery_identity == identity`, **or** same `discovery_serial`
   (and same `discovery_channel` for NVRs).
3. Same stream endpoint `(scheme, host, port, path, query)`, ignoring
   credentials.
4. Direct cameras only: same host.

If the match is running with a different `source_url`, a repoint is queued for
E9.

**Step 2 — adopt if needed.** Adoption runs when `is_new`, or when the row
exists but has no `camera_uuid` (for example, it failed earlier because the
Jetson was full).

- `DISCOVERY_AUTO_ADD=false` → reported under `new_cameras` with
  `adopted: false`. Nothing starts.
- Step 1 found a UUID → the roster row is linked to it (`already_present: true`).
- Otherwise `_adopt()` generates `camera_uuid = uuid4()` and calls
  `runtime.add_camera(source_url, cfg)`:

```python
{
  "camera_uuid": "3f0c1e9a-5b7d-4c2e-9a61-0d4b8e7f2a10",
  "channel_id":  "3f0c1e9a-5b7d-4c2e-9a61-0d4b8e7f2a10",
  "enabled": True, "detection_enabled": True, "notification_enabled": True,
  "discovered": True,
  "discovery_identity": "serial:DS-2CD2143G0-I20210315AAWRE12345678",
  "discovery_ip": "192.168.1.64",
  "discovery_model": "DS-2CD2143G0-I",
  "discovery_serial": "DS-2CD2143G0-I20210315AAWRE12345678",
  "discovery_name": "Front Gate",
  "discovery_channel": 1,
  "discovery_is_nvr": False
}
```

`add_camera` fills in defaults (`sample_fps`, `resize`, `decode_backend`,
reconnect backoff, `gst_latency_ms`), then `_apply_camera` checks admission:
the URL must be selected, must not already belong to another UUID, and must
fit under `MAX_CAMERAS`. It then starts the channel on the pipeline loop,
saves the row to `camera_configs`, and stores the config in memory. The roster
row is linked with `attach_camera_uuid`.

The UUID is generated on the Jetson because the cloud's `list_cameras` drops
any `camera_uuid` that is not a valid UUID, and the cloud has no row for this
camera yet. The cloud reuses this UUID when it adopts the camera (C3).

**Output (`report["new_cameras"]` entry):**

```json
{
  "identity": "serial:DS-2CD2143G0-I20210315AAWRE12345678",
  "ip_address": "192.168.1.64",
  "source_url": "rtsp://admin:<hik_pass>@192.168.1.64:554/Streaming/Channels/102",
  "camera_uuid": "3f0c1e9a-5b7d-4c2e-9a61-0d4b8e7f2a10",
  "adopted": true,
  "model": "DS-2CD2143G0-I",
  "serial_number": "DS-2CD2143G0-I20210315AAWRE12345678",
  "config": { "...": "full normalized channel config" }
}
```

On `CameraCapacityError` the identity goes to `capacity_rejected` and
`errors: ["At capacity, not added: <identity>"]`.

---

### E9. Repoint a camera that moved

**Input:** the queued changes from E8.

```python
{"identity": "serial:DS-2CD…5678", "ip_address": "192.168.1.77",
 "source_url": "rtsp://admin:<hik_pass>@192.168.1.77:554/Streaming/Channels/102",
 "camera_uuid": "3f0c1e9a-…"}
```

**Processing:** `runtime.patch_camera(uuid, {"source_url": …, "discovery_ip": …})`
restarts the channel on the new URL and saves it.

**Output:** the change is appended to `report["ip_changes"]`. On failure it is
appended to `errors`, and the next sweep tries again.

---

### E10. Age cameras that were not seen

**Input:** the identities verified this sweep, plus roster rows outside both
the current and previous selection. Those rows are historical and are
deliberately skipped.

**Processing (`mark_missing`):** for every other row,
`consecutive_misses += 1` and `missing_since` is set once. When misses reach
`DISCOVERY_MISS_THRESHOLD` (2), `is_present=False`. The row is **returned only
once**, when `alerted` flips from false to true. Rows are never deleted. Only
`DELETE /discovery/<identity>` removes one.

**Output (`report["missing_cameras"]`):** roster dicts with
`"is_present": false`, `"verification_state": "offline"`, and `missing_since`
set.

Example: a camera unplugged at 14:00, with sweeps at 14:01 and 14:03 (the
quiet period makes the gap about 150 s), is reported missing on the 14:03
sweep. It is not reported again until it recovers and goes missing again.

---

### E11. Build, cache and broadcast the report

**Output — the full sweep report** (cached in memory, returned by
`/discovery/report`, embedded in `/sync`):

```json
{
  "type": "discovery_report",
  "generated_at": "2026-09-22T14:03:40.118220Z",
  "duration_s": 71.4,
  "scanned": 8,
  "unverified_candidates": ["ip:innovationdistrict.asuscomm.com#2"],
  "capacity_rejected": [],
  "present":           [ { "...HikDevice.to_dict() + identity" : "" } ],
  "new_cameras":       [ { "...E8 entry" : "" } ],
  "missing_cameras":   [ { "...roster row, is_present=false" : "" } ],
  "recovered_cameras": [],
  "ip_changes":        [],
  "roster":            [ { "...E7 roster row, selected identities only" : "" } ],
  "errors": []
}
```

`scanned` counts **selected** devices, after the `MAX_CAMERAS` cut.

If any of `new_cameras`, `missing_cameras`, `recovered_cameras` or
`ip_changes` is non-empty, a trimmed event (no `present`, no `roster`) is sent
through `pipeline.broadcaster` on the all-cameras stream
`GET /cameras/detections/stream`. The event has no `camera_uuid`, so
per-camera streams never receive it.

---

### Edge HTTP surface (what the cloud can call)

Every route is also available under `/api/…`.

| Method + path | Input | Output |
| --- | --- | --- |
| `POST /sync` | `{}` | `{cameras, discovery, status, discovery_pending[, warnings]}`; schedules a sweep **without** waiting |
| `GET /discovery/report` | — | last complete report, or 404 before the first sweep |
| `POST /discovery/scan` | — | a fresh report (blocks for the whole sweep) |
| `GET /discovery` | — | `{discovered, present, missing, status}` (selected identities only) |
| `GET /discovery/status` | — | scheduler state |
| `DELETE /discovery/<identity>` | identity in path | `{forgotten, identity, existed}` |
| `GET /cameras` | — | `{"cameras": [{camera_uuid, source_url, config}]}` |
| `POST /cameras` | `{camera_uuid, source_url, …}` | 201 config, 409 capacity/not selected, 400 invalid |
| `GET /health` | — | `{ok, pipeline_ready, stats{max_cameras, camera_selection, capture{…}}, discovery{…}}` |
| `GET /cameras/<uuid>/detections/stream` | — | SSE detections for one camera |

**`POST /sync` example response:**

```json
{
  "cameras": [
    { "camera_uuid": "3f0c1e9a-5b7d-4c2e-9a61-0d4b8e7f2a10",
      "source_url": "rtsp://admin:<hik_pass>@192.168.1.64:554/Streaming/Channels/102",
      "config": { "discovered": true, "discovery_identity": "serial:DS-…5678", "sample_fps": 3.0, "...": "" } }
  ],
  "discovery": { "...last complete report (E11)": "" },
  "status": {
    "enabled": true, "running": true, "interval_s": 60, "miss_threshold": 2,
    "auto_add": true, "in_quiet_period": true,
    "scan_in_progress": false, "last_sweep_at": "2026-09-22T14:03:40.118220Z",
    "last_sweep_duration_s": 71.4, "present_count": 7, "missing_count": 0
  },
  "discovery_pending": true
}
```

**While a sweep is running**, `discovery` is a *partial* report from
`current_report()`: `"reason": "discovery_pending"`, `"partial": true`, empty
delta lists, and a `roster` limited to selected identities that already have
a running camera. The cloud treats a partial report as valid because it
contains `roster`, so it can adopt what is already running. Missing-camera
alerts, however, only come from complete reports.

---

## Part 2 — Cloud (`Backend/`)

### C1. What triggers the cloud to look

| Trigger | Code | Calls |
| --- | --- | --- |
| Background loop every `EDGE_RECONCILE_INTERVAL_S` (30 s) | `main.py` `_edge_reconcile_loop` | `reconcile_all_devices_edge(delete_unknown=EDGE_RECONCILE_DELETE_UNKNOWN)` → one `reconcile_device_edge_simple` per unique `device_url` |
| **Sync** button in the Devices view | `Frontend/…/views/devices.js` → `POST /api/devices/{device_uuid}/edge/reconcile?dry_run=false&delete_unknown=true` | `reconcile_device_edge_simple` |
| Link a device to a site | `POST /api/sites/{site_uuid}/devices` (`discover=true` by default) | `InventoryService.refresh_from_device` then `manager.adopt_discovered_cameras` (120 s timeout) |
| Refresh inventory | `POST /api/devices/{device_uuid}/inventory/refresh` | `InventoryService.refresh_from_device` |
| Add from inventory | `POST /api/sites/{site_uuid}/inventory/add` | `InventoryService.add_to_site` |
| Site schedule / camera edits | `reconcile_devices_best_effort` | `reconcile_device_edge_simple(delete_unknown=True)` |

Access control: reconcile and inventory refresh need
`ORG_MANAGE_DEVICES`; linking needs `ORG_MANAGE_SITES`; adding from inventory
needs `ORG_MANAGE_CAMERAS`; listing inventory needs `ORG_READ`.

---

### C2. Fetch the discovery report from the edge (`EdgeInferenceClient`)

Every call tries `path` and then `/api` + `path`, with `EDGE_HTTP_RETRIES`
attempts (3) and exponential backoff. 404/405 moves on to the next path.
400/409/422 are final and are not retried.

| Method | Edge call | Timeout | Returns |
| --- | --- | --- | --- |
| `sync_discovery(device_url)` | `POST {device_url}/sync` | `EDGE_SYNC_TIMEOUT_S` (≥ 45 s) | `response["discovery"]` or `None` |
| `fetch_discovery_report(device_url)` | `GET {device_url}/discovery/report` | `EDGE_LIST_TIMEOUT_S` (5 s) | the whole report, or `None` |
| `list_cameras(device_url)` | `GET {device_url}/cameras` | 5 s | `Set[str]` of valid UUIDs |
| `get_health(device_url)` | `GET /health` then `/api/health` | 2 s | health dict or `None` |

**Input:** `device_url`, e.g. `http://10.8.0.12:8080` (normalized by
`normalize_device_url`).
**Output:** the report dict shown in E11. `None` means an older firmware
(404 on every path) or an unreachable device. That is never fatal.

---

### C3. Adopt discovered cameras into a site (`CameraAdopter`)

**Input:** `device_uuid`, `device_url`, `user_id` (the device owner),
the discovery report, and the peer device UUIDs (logical device rows that share
the same URL inside the org).

**Step 1 — filter the roster (`_roster_entries`).** Keeps rows that have
`is_present: true` and a `source_url`, whose `verification_state` is not
`unverified`, and whose identity is not in `capacity_rejected`.

**Step 2 — find the site (`_sites_for_device`).** Uses `site_devices` joined to
non-deleted sites.
- 0 sites → error `"Device is not linked to any site…"`. Nothing is adopted.
- More than 1 site → error `"…target site is ambiguous"`. Nothing is adopted.

**Step 3 — build indexes.**
- `by_identity`: `ChannelConfiguration.configuration["discovery_identity"]` →
  Camera.
- `by_host`: host of `Camera.source_url` → Camera. **Skipped for `#channel`
  identities**, because NVR channels share one host.
- `blocked`: identities whose `camera_inventory.state` is `removed` or `added`.

**Step 4 — decide per entry:**

| Condition | Result |
| --- | --- |
| identity is `blocked` and has no live Camera | `skipped` (`reason: removed_by_user`) |
| match in `by_identity` (or `by_host` for non-channel identities) | `linked`; `_sync_existing_camera` backfills provenance and, if the **host** changed, updates `Camera.source_url` + the MediaMTX path → `repointed` |
| `dry_run` | `skipped` (`reason: dry_run`) |
| otherwise | `_create_camera` → `adopted` |

**Step 5 — create the camera (`_create_camera`).** Builds a
`ChannelCreateEvent` and sends it through `update_pipeline` →
`ChannelController.add_channel`. This is the same path the manual "add camera"
route uses:

```python
configs = {
  "camera_uuid": UUID("3f0c1e9a-5b7d-4c2e-9a61-0d4b8e7f2a10"),   # reused from the edge
  "site_uuid":   UUID("…site…"),
  "device_uuid": UUID("…device…"),
  "user_id": 7,
  "source_url": "rtsp://admin:<hik_pass>@192.168.1.64:554/Streaming/Channels/102",
  "name": "Front Gate (192.168.1.64)",
  "location": "Front Gate",
  "enabled": True, "detection_enabled": True, "notification_enabled": True,
  "discovered": True,
  "discovery_identity": "serial:DS-2CD2143G0-I20210315AAWRE12345678",
  "discovery_ip": "192.168.1.64", "discovery_mac": "44:19:b6:11:22:33",
  "discovery_model": "DS-2CD2143G0-I",
  "discovery_serial": "DS-2CD2143G0-I20210315AAWRE12345678",
  "discovery_name": "Front Gate",
  "discovered_at": "2026-09-22T14:04:02.551000+00:00"
}
```

`add_channel` then:

1. Checks that the site belongs to the user and that the device is linked to it.
2. Sets `camera_code = "cam-" + uuid.hex[:8]` → `cam-3f0c1e9a`.
3. `webrtc.ensure_stream(stream_key="cam-3f0c1e9a", source_url=…)` creates the
   MediaMTX path that live view uses.
4. Upserts the `Camera` + `ChannelConfiguration` rows. If that fails, the
   MediaMTX stream is deleted.
5. Starts a background edge upsert (`POST {device_url}/cameras` with the
   **same UUID and URL**). The edge accepts it because the URL is selected and
   belongs to that UUID.
6. Adds the channel to the cloud's model pipeline so it subscribes to
   detections (C7).

**Output:**

```json
{ "adopted": ["3f0c1e9a-5b7d-4c2e-9a61-0d4b8e7f2a10"],
  "linked": ["8d2b…", "a41f…"],
  "repointed": [],
  "skipped": [],
  "errors": [] }
```

Reusing the edge's UUID matters. If the cloud created a different UUID, it
would subscribe to detections that the Jetson never sends: live video would
work but no detections would arrive. The edge would also reject the upsert
with 409.

---

### C4. Reconcile: cloud desired state vs edge actual state

`reconcile_device_edge_simple(device_uuid, user_id, dry_run, delete_unknown)`:

| Step | Input | Output |
| --- | --- | --- |
| 1. Load device + peers | `device_uuid` | `device_url`, owner id, `reconcile_device_uuids` |
| 2. `_recompute_desired` | cameras of those devices + schedule | `desired_set` = UUIDs that are detection-enabled **and** armed; `active_streams` = `camera_code` of enabled cameras |
| 3. WebRTC diff | MediaMTX list vs `active_streams` | `to_add_stream`, `to_remove_stream` (applied right away) |
| 4. `sync_discovery` | `device_url` | report (C2) or `None` |
| 5. Adopt (C3) | report | `adoption` dict; if anything was adopted or repointed, step 2 runs again |
| 6. `list_cameras` | `device_url` | `edge_set` |
| 7. Diff | `desired_set`, `edge_set` | `to_add = desired − edge`, `to_remove = edge − desired` |
| 8. Protect discovered | report `roster` + `new_cameras` UUIDs | those UUIDs are removed from `to_remove` → `discovered` + warning |
| 9. Missing warnings | report `missing_cameras` | `"Camera went missing from the network: Front Gate (last seen …)"` |
| 10. Selection policy | `/health` `stats.camera_selection` | if `"discovery"` → **`to_add = to_remove = []`** |
| 11. Capacity | `/health` `stats.max_cameras` | `to_add` trimmed; the extra UUIDs go to `capacity_overflow` + warning |
| 12. Apply (unless `dry_run`) | `to_add`, repoints, `to_remove` | edge `POST /cameras`, `PATCH /cameras/{uuid}`, `DELETE /cameras/{uuid}` |

If the edge cannot be reached (and `/health` does not say it is ready),
reconcile raises `EdgeDeviceUnavailableError`. The route turns that into
**503**.

Step 10 means that on a discovery-managed Jetson (the default), reconcile
never adds or removes edge cameras. The cloud's influence goes through
adoption (C3), the per-camera upsert in `add_channel`, and repoint patches.

---

### C5. Reconcile response (`EdgeReconcileOut`)

```json
{
  "device_uuid": "b7e1…",
  "device_url": "http://10.8.0.12:8080",
  "to_add": [], "to_remove": [],
  "to_add_stream": ["cam-3f0c1e9a"], "to_remove_stream": [],
  "added": ["3f0c1e9a-5b7d-4c2e-9a61-0d4b8e7f2a10"],
  "removed": [],
  "errors": [],
  "warnings": [
    "Camera went missing from the network: Loading Bay (last seen 2026-09-22T13:58:02.114000)"
  ],
  "discovered": [],
  "missing_cameras": [ { "identity": "serial:DS-…4567#3", "device_name": "Loading Bay", "is_present": false, "missing_since": "…", "...": "" } ],
  "capacity_overflow": [],
  "adopted": ["3f0c1e9a-5b7d-4c2e-9a61-0d4b8e7f2a10"],
  "linked": ["8d2b…"],
  "repointed": []
}
```

`added` includes WHEP streams created in step 3 as well as edge upserts from
step 12.

---

### C6. Inventory (`camera_inventory`) — the cloud's record of reported cameras

**Why it exists:** deleting a camera deletes its `Camera` row, and the next
sweep would find the hardware and adopt it again. The inventory row outlives
the Camera row and records that the user removed it.

**Key:** `(device_uuid, discovery_identity)`. It is never keyed by URL.

**States:**

```
             refresh_from_device                add_to_site
  (report) ─────────────────────► available ───────────────────► added
                                      ▲                            │
                                      │  add_to_site (explicit)    │ camera deleted
                                      └──────────── removed ◄──────┘  (remove_camera)
```

**Refresh (`refresh_from_device`).**
*Input:* `GET /discovery/report` (the cached report, fast).
*Processing:* `entries_from_roster` copies only the allowlisted fields
(`camera_uuid` becomes `edge_camera_uuid`). `upsert_reported` writes only
device-owned columns, so `state`, `site_uuid` and `camera_uuid` are never
overwritten by a sweep. `mark_absent` sets `is_present=False` on rows the
report did not include.
*Output:* `InventoryRefreshOut`:

```json
{ "fetched": true, "created": 1, "updated": 7, "marked_offline": 0, "detail": null }
```

**Listing (`GET /api/sites/{site_uuid}/inventory` or
`/api/devices/{device_uuid}/inventory`).** Returns `InventoryCameraOut`. The
raw `source_url` is never returned; it is replaced by a credential-free
`source`:

```json
{
  "discovery_identity": "serial:DS-2CD2143G0-I20210315AAWRE12345678",
  "device_uuid": "b7e1…", "site_uuid": null, "camera_uuid": null,
  "state": "available",
  "display_name": "Front Gate",
  "ip_address": "192.168.1.64", "mac_address": "44:19:b6:11:22:33",
  "serial_number": "DS-2CD2143G0-I20210315AAWRE12345678",
  "model": "DS-2CD2143G0-I", "firmware": "V5.6.3",
  "source": "rtsp://192.168.1.64:554/Streaming/Channels/102",
  "is_present": true, "verification_state": "online",
  "first_seen_at": "…", "last_seen_at": "…", "missing_since": null
}
```

**Add (`POST /api/sites/{site_uuid}/inventory/add`).**
*Input:* `{"device_uuid": "b7e1…", "discovery_identities": ["serial:DS-…5678"]}`
(the device must be linked to the site, or the call returns 409).
*Processing:* per identity: `not_in_inventory` / `already_added` /
`no_source_url` / `unverified`, or `create_camera_from_inventory` (the C3
step 5 path, reusing `edge_camera_uuid`) followed by `mark_added`.
*Output:*

```json
{ "site_uuid": "…",
  "results": [ { "discovery_identity": "serial:DS-…5678", "ok": true,
                 "outcome": "added", "camera_uuid": "3f0c1e9a-…", "detail": null } ] }
```

**Remove (camera delete route).** `InventoryService.remove_camera` →
`state=removed`, `site_uuid=NULL`, `camera_uuid=NULL`. From then on,
`blocked_identities` stops C3 from adopting it again. Only an explicit
inventory add brings it back.

---

### C7. How the backend gets detections for a discovered camera

Once the Camera row exists, its `VideoChannel` in the cloud pipeline
(`application/channels/channel.py`) subscribes to the edge's per-camera
endpoints using the **same UUID**:

```
GET {device_url}/api/cameras/{camera_uuid}/detections/stream   (then /cameras/…)
GET {device_url}/api/detections/{camera_uuid}                  (latest, fallback)
```

The edge's `SimpleInferencePipeline` broadcasts each result to subscribers of
that UUID. The cloud tracks, applies ROI rules, sends notifications, and
re-exposes the results to the browser through
`GET /api/cameras/{camera_uuid}/detections/stream`. Live video comes
separately from MediaMTX (WHEP) at the `camera_code` path created in C3.

---

## Part 3 — One camera end to end (timeline)

Scenario: `Front Gate` (192.168.1.64) is plugged into a Jetson that is linked
to exactly one site, and the Jetson has a free slot.

| t | Where | What happens | Visible result |
| --- | --- | --- | --- |
| 0 s | edge E1 | scheduled sweep starts | — |
| ~3 s | edge E2c | WS-Discovery reply from 192.168.1.64; ISAPI `deviceInfo` returns serial | `HikDevice(…is_nvr=False)` |
| ~3 s | edge E3–E5 | identity `serial:DS-…5678`; URL `…/Channels/102` | — |
| 3–21 s | edge E6 | waits for the connect gate, opens a temporary capture, gets a frame | verified |
| ~21 s | edge E7–E8 | roster row `is_new`; generates `3f0c1e9a-…`; `add_camera`; saves to `camera_configs` | edge `GET /cameras` includes it; detections start on the edge |
| end of sweep | edge E11 | report cached; SSE `discovery_report` with `new_cameras` | — |
| ≤ 30 s later | cloud C1–C2 | reconcile loop → `POST /sync` → last complete report | — |
| same pass | cloud C3 | `adopted: ["3f0c1e9a-…"]`; `Camera` row `cam-3f0c1e9a`; MediaMTX path; edge upsert | camera appears in the site; live view works |
| same pass | cloud C7 | cloud channel subscribes to `/cameras/3f0c1e9a-…/detections/stream` | overlays and alerts |

If the device is not linked to a site, C3 stops with *"Device is not linked to
any site"*. The camera keeps running on the edge, and reconcile keeps it out of
`to_remove`. Linking the device later (`POST /sites/{site}/devices`) adopts it
at that point.

---

## Part 4 — The IDs and where each one lives

| ID | Example | Created by | Stored in |
| --- | --- | --- | --- |
| `identity` / `discovery_identity` | `serial:DS-…5678`, `ip:host#3` | edge `identity_of` | edge `discovered_cameras.identity`; edge `camera_configs.config_json.discovery_identity`; cloud `ChannelConfiguration.configuration.discovery_identity`; cloud `camera_inventory.discovery_identity` |
| `camera_uuid` | `3f0c1e9a-5b7d-…` | edge `_adopt` (`uuid4`), reused by the cloud | edge `camera_configs.camera_uuid`, `discovered_cameras.camera_uuid`; cloud `Camera.camera_uuid`, `camera_inventory.camera_uuid` / `edge_camera_uuid` |
| `camera_code` | `cam-3f0c1e9a` | cloud `add_channel` | cloud `Camera.camera_code`; MediaMTX path name |
| `source_url` | `rtsp://…/Streaming/Channels/102` | edge `rtsp_url_for` | both sides; the cloud compares **only the host** when deciding whether to repoint |
| `device_uuid` / `device_url` | `b7e1…` / `http://10.8.0.12:8080` | cloud device registration | cloud `devices`, `site_devices` |

---

## Part 5 — Things the trace shows that are easy to miss

These come from reading the code. None of them were changed.

1. **Missing-camera alerts reach users only through the reconcile response.**
   The edge's SSE `discovery_report` event goes to the all-cameras stream. The
   cloud subscribes per camera (C7), so it never receives that event. Nothing
   in `Backend/` or the frontend reads `discovery_report`. A missing camera
   shows up as a `warnings` line when someone presses **Sync**, and in the
   background reconcile log.
2. **The frontend never calls the inventory endpoints.** No frontend code
   calls `/inventory`. Inventory is filled when a device is linked to a site
   and by `POST /devices/{uuid}/inventory/refresh`. The background reconcile
   adopts cameras without updating inventory, so a camera adopted that way can
   still show `state: available` in inventory while it has a live Camera row.
   A later `inventory/add` for it returns `already_added` only if the row says
   `added`. Otherwise it goes through `create_camera_from_inventory`.
3. **`POST /sync` no longer waits for the sweep.** `DISCOVERY.md` says "the
   sweep runs before the camera list is read". The current code schedules the
   sweep and returns the last complete report (or a partial one). A camera
   found during this sweep is adopted by the cloud on a *later* reconcile.
4. **Cameras past `MAX_CAMERAS` disappear without a trace.** Devices cut in E4
   are not listed under `unverified_candidates` or `capacity_rejected`, so
   they cannot be seen from the API. With 8 static channels and
   `MAX_CAMERAS=8`, no LAN camera can ever be selected.
5. **A discovery-managed Jetson ignores cloud-only cameras.** Reconcile clears
   `to_add` (C4 step 10), and the edge returns 409 for any URL that discovery
   did not select. A camera created by hand in the cloud with a URL the edge
   did not discover will have live view (MediaMTX) but no detections.
6. **Verification is sequential and shares the connect gate.** Eight new
   channels at a 6 s stagger and a 12 s probe can take about 2.5 minutes to
   verify. Reconnecting live cameras use the same gate, so a sweep and a
   reconnect storm slow each other down.
7. **Document 09's configuration table is out of date** for
   `DISCOVERY_LOCAL_ENABLED`, `DISCOVERY_INTERVAL_S` and `GST_LATENCY_MS` (see E0).

---

## Quick commands

```sh
# On the Jetson (tensort directory)
curl -s localhost:8080/discovery/status | jq
curl -s localhost:8080/discovery/report | jq '{scanned, unverified_candidates, capacity_rejected, new: [.new_cameras[].identity], missing: [.missing_cameras[].identity]}'
curl -s -X POST localhost:8080/discovery/scan | jq '.present | length'      # blocking
curl -s localhost:8080/cameras | jq '.cameras[] | {camera_uuid, id: .config.discovery_identity}'
python3 deployment/diagnose_discovery.py --full-sweep

# Against the cloud (JWT required)
curl -s -X POST -H "Authorization: Bearer $TOKEN" \
  "$API/api/devices/$DEVICE/edge/reconcile?dry_run=true" | jq '{adopted, linked, discovered, missing_cameras, warnings}'
curl -s -X POST -H "Authorization: Bearer $TOKEN" "$API/api/devices/$DEVICE/inventory/refresh" | jq
curl -s -H "Authorization: Bearer $TOKEN" "$API/api/sites/$SITE/inventory" | jq '.[] | {discovery_identity, state, verification_state}'
```
