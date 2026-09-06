# Camera Auto-Discovery — Jetson Edge

The Jetson is now the **source of truth** for which cameras physically exist.
A scheduled service sweeps the network every minute, adopts Hikvision cameras
it has never seen into the detection pipeline, and raises an alert when a
camera that used to be there stops answering.

---

## Files

| File | What it does |
| --- | --- |
| `discovery.py` | **New.** Network probing. ONVIF WS-Discovery + Hikvision ISAPI identification. Stdlib only. |
| `env_utils.py` | **New.** Shared `env_str/int/float/bool` getters used by `discovery.py` and `service.py`. |
| `tests/test_discovery.py` | **New.** The suite described under *Verification status*. Fake camera + in-memory repo; no hardware or DB. |
| `service.py` | **New.** The scheduler. Runs a sweep every `DISCOVERY_INTERVAL_S`, adopts new cameras, ages out missing ones, builds the report. |
| `repositories/discovery_repository.py` | **New.** Persistence for the discovered-camera roster. |
| `routes/discovery_routes.py` | **New.** `/discovery*` endpoints and `POST /sync`. |
| `database_orm.py` | **Changed.** Added the `DiscoveredCamera` model (`discovered_cameras` table). |
| `repositories/__init__.py` | **Changed.** Exports `DiscoveryRepository`. |
| `routes/__init__.py` | **Changed.** Registers the discovery blueprint. |
| `routes/helpers.py` | **Changed.** `/` and `/health` now include a `discovery` block. |
| `main.py` | **Changed.** `runtime.discovery` attribute + `start_discovery()` called after boot. |
| `.env`, `.env.example` | **Changed.** Discovery configuration block appended. |

Cloud side (`Backend/`):

| File | What changed |
| --- | --- |
| `application/services/edgeinference.py` | Added `sync_discovery()`, `sync_path`, `sync_timeout_s`. |
| `application/services/manager/controllers/reconcile.py` | Reconcile now triggers an edge sweep first, protects edge-discovered cameras from deletion, and surfaces missing-camera warnings. |
| `core/schemas.py` | `EdgeReconcileOut` gained `discovered` and `missing_cameras`. |

---

## How a camera is found

Two probes run per sweep, merged by IP:

1. **ONVIF WS-Discovery** — UDP multicast to `239.255.255.250:3702`. Every
   camera with ONVIF enabled answers. Needs no credentials, but does not
   reliably identify the vendor. The probe is sent 3× because a single lost
   datagram would silently hide every camera on the segment.

2. **Hikvision ISAPI** — `GET /ISAPI/System/deviceInfo` with HTTP digest auth.
   This is the authoritative check: only a Hikvision box returns a
   `<DeviceInfo>` document with a `<serialNumber>`. It also confirms the
   credentials work.

**A camera is only adopted if ISAPI confirms it.** That matters: an RTSP URL
built with the wrong password produces a camera that is "discovered" but never
decodes a frame. Wrong credentials → not adopted.

If WS-Discovery finds nothing (multicast is often blocked on managed switches),
the service falls back to sweeping the configured subnet directly. Only `/16`
through `/32` are accepted — anything wider is refused as a port scan.

### Identity

Cameras are tracked by a stable identity, preferred in this order:

```
serial:<serialNumber>   →   mac:<macAddress>   →   ip:<address>
```

The serial is what makes "the camera in the loading bay" recognisable after a
DHCP lease change. A camera with no serial that moves IP will be reported as
one gone plus one new — that is a known and accepted limitation.

---

## The sweep lifecycle

```
scan_network()
      │
      ├── for each confirmed camera → repo.mark_seen()
      │        ├── never seen before  → ADOPT into pipeline (mint UUID, add channel)
      │        ├── was absent         → RECOVERED (clears the alert)
      │        └── IP changed         → REPOINT the existing camera's source_url
      │
      ├── repo.mark_missing()  → rows not seen this sweep age towards missing
      │        └── past DISCOVERY_MISS_THRESHOLD → one-shot MISSING alert
      │
      └── report → SSE broadcast + cached for the next /sync
```

### Why UUIDs are minted on the edge

`PipelineRuntime.add_camera` normally refuses to generate IDs — the cloud owns
them. But an auto-discovered camera has no cloud row yet, and the cloud's
`EdgeInferenceClient.list_cameras` **discards any entry whose `camera_uuid` is
not a parseable UUID**. So the edge mints a real `uuid4` and the cloud adopts it
on the next sync. Every such config carries `discovered: True` plus
`discovery_identity` / `discovery_serial` provenance.

### Guards against duplicate adoption

Three things prevent one physical camera becoming two decoding channels:

- The roster row is the primary memory — a camera with a row is never "new".
- `_existing_camera_uuid_for()` cross-checks the live pipeline by discovery
  identity, then serial, then RTSP host. This catches the case where the roster
  table was recreated but `camera_configs` survived, and the case where the
  cloud pushed the camera down before discovery ever ran.
- `_sweep_lock` serialises sweeps. A `/sync`-triggered sweep and the scheduled
  one can never interleave; the second caller waits and receives the first
  one's report (flagged `coalesced: true`).

### Missing-camera alerting

`DISCOVERY_MISS_THRESHOLD` (default 2) is a grace window: a camera must be
absent for two consecutive sweeps — about two minutes — before it alerts. One
dropped UDP probe or a camera rebooting must not page anyone.

The alert fires **once**. A camera unplugged for a week does not re-alert every
minute; it stays in the roster as absent (`is_present: false`) and shows up in
the roster listing. It re-arms if the camera comes back and disappears again.

Rows are **never** deleted by the scanner. A decommissioned camera is removed
explicitly via `DELETE /discovery/<identity>` — an unplugged camera should keep
alerting, a decommissioned one should be forgotten, and only a human knows
which is which.

---

## Threading

Discovery does blocking socket work — a UDP probe window plus up to 254 HTTP
probes — that takes seconds. Running that on the pipeline's asyncio loop would
stall every camera's decode task.

So the scheduler owns a **dedicated daemon thread** and reaches the pipeline
only through `PipelineRuntime`'s existing thread-safe wrappers, the same ones
the Flask request threads use. The SSE broadcast hops back onto the pipeline
loop via `loop.call_soon_threadsafe`.

The repository is deliberately **synchronous** (unlike `CameraRepository`'s
async writes) because the scanner thread has no event loop to await on.

---

## Endpoints

### Existing — unchanged

`GET /cameras` already returns the cameras attached to this Jetson, and the
cloud's `EdgeInferenceClient` is already pointed at it (`EDGE_LIST_PATH`
defaults to `/cameras`). **No new listing route was added** — a second one would
give the cloud two disagreeing sources of truth.

### New

Every route is also mounted under `/api/…`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/discovery` | The roster: every camera ever seen, split into `present` / `missing`. |
| `GET` | `/discovery/status` | Scheduler state — running, interval, last sweep, counts. |
| `GET` | `/discovery/report` | The last sweep report, **without** rescanning. 404 before the first sweep. |
| `POST` | `/discovery/scan` | Force a sweep now, return the fresh report. Slow — use a generous timeout. |
| `DELETE` | `/discovery/<identity>` | Forget a decommissioned camera so it stops alerting. |
| `POST` | `/sync` | **The cloud sync entry point.** Rescans, then returns `cameras` + `discovery` report. |

`GET /` and `GET /health` now carry a `discovery` block so the cloud can tell an
actively-scanning Jetson from one whose scanner died, without a second round trip.

### `POST /sync` response

```json
{
  "cameras": [ { "camera_uuid": "...", "source_url": "...", "config": {...} } ],
  "discovery": {
    "type": "discovery_report",
    "generated_at": "2026-08-27T19:37:16Z",
    "duration_s": 4.21,
    "scanned": 8,
    "present": [ { "identity": "serial:...", "ip": "192.168.1.64", "model": "DS-2CD2143G0-I", ... } ],
    "new_cameras": [ { "identity": "...", "camera_uuid": "...", "adopted": true, ... } ],
    "missing_cameras": [ { "identity": "...", "ip_address": "...", "missing_since": "...", ... } ],
    "recovered_cameras": [ ... ],
    "ip_changes": [ ... ],
    "roster": [ ... ],
    "errors": []
  },
  "status": { "enabled": true, "running": true, "interval_s": 60, ... }
}
```

The sweep runs **before** the camera list is read, so a camera adopted during
that very sweep is already in `cameras`.

A failed sweep does **not** fail the sync — the cloud still needs the camera
list, and stale discovery data beats no response. The failure appears in
`warnings` and `discovery` falls back to the last good report.

---

## Frontend alerting

The missing-camera alert reaches the frontend two ways:

**1. Live, via SSE.** The report is pushed onto the existing detection stream at
`GET /cameras/detections/stream`. The event carries **no `camera_uuid`**, so
per-camera subscribers correctly ignore it — this is a fleet-level event and
only the all-cameras stream receives it.

```json
{
  "type": "discovery_report",
  "generated_at": "2026-08-27T19:37:16Z",
  "scanned": 8,
  "new_cameras": [...],
  "missing_cameras": [...],
  "recovered_cameras": [...],
  "ip_changes": [...],
  "errors": []
}
```

Filter on `type === "discovery_report"` to separate these from detection frames.
The payload is trimmed to deltas — the full roster can be hundreds of rows and
the broadcaster's queues are bounded at 200 messages. Nothing is broadcast when
a sweep finds no changes.

**2. On sync.** `POST /devices/{device_uuid}/edge/reconcile` now returns:

- `missing_cameras` — full detail per missing camera
- `warnings` — human-readable lines: `"Camera went missing from the network: Loading Bay (last seen 2026-08-27T18:02:00)"`
- `discovered` — cameras the edge adopted that the cloud has no row for yet

---

## Cloud reconcile behaviour

Two changes to `reconcile_device_edge_simple`:

**It triggers a sweep first.** Because the edge is the source of truth, the
reconcile asks the Jetson to re-scan before reading its camera list. Otherwise
a camera plugged in since the last sweep would be missing from `edge_set`.

This is best-effort: an edge that predates discovery returns `None` (404 on all
candidate paths) and reconcile proceeds exactly as before.

**Edge-discovered cameras are never deleted.** A camera the Jetson adopted has
no cloud row, so it would normally land in `to_remove` as an unknown stray.
Deleting it would tear down a camera the Jetson just adopted — and the next
sweep would re-add it. That is a permanent add/delete loop. Those UUIDs are
filtered out of `to_remove`, reported in `discovered`, and surfaced as a
warning so the frontend can offer to register them.

---

## Configuration

All in `Backend/tensort/.env`.

| Variable | Default | Notes |
| --- | --- | --- |
| `DISCOVERY_ENABLED` | `true` | Master switch. |
| `DISCOVERY_INTERVAL_S` | `60` | Sweep interval. Floor is 10s. |
| `DISCOVERY_MISS_THRESHOLD` | `2` | Sweeps absent before alerting. **Do not set to 1.** |
| `DISCOVERY_AUTO_ADD` | `true` | `false` = discover and report only, never touch the pipeline. |
| `HIK_USERNAME` | `admin` | Applied to every discovered camera. |
| `HIK_PASSWORD` | *(empty)* | **Must be set** or no camera will authenticate. |
| `DISCOVERY_SUBNETS` | *(empty)* | Comma-separated CIDRs. Empty = derive the `/24` from this host's own IPv4. Only `/16`–`/32`. |
| `DISCOVERY_SWEEP_ONLY_IF_WSD_EMPTY` | `true` | Set `false` if some cameras have ONVIF disabled. |
| `DISCOVERY_WSD_TIMEOUT_S` | `3.0` | Multicast listen window. |
| `DISCOVERY_HTTP_TIMEOUT_S` | `2.0` | Per-host ISAPI probe timeout. |
| `DISCOVERY_PROBE_WORKERS` | `32` | Concurrent probe threads. |
| `HIK_RTSP_PORT` | `554` | |
| `HIK_RTSP_CHANNEL` | `1` | |
| `HIK_RTSP_STREAM` | `2` | 1 = main, 2 = sub. Sub is right for detection. |
| `HIK_HTTP_PORT` | `80` | ISAPI port. |

Cloud side (`Backend/.env`):

| Variable | Default | Notes |
| --- | --- | --- |
| `EDGE_SYNC_PATH` | `/sync` | |
| `EDGE_SYNC_TIMEOUT_S` | `45` | A sweep scans the network; it needs far more than a normal edge call. |

### RTSP URL construction

Hikvision encodes the path segment as `channel * 100 + stream`:

```
rtsp://<user>:<pass>@<ip>:554/Streaming/Channels/102
                                                 ^^^  channel 1, sub stream
```

101 = ch1 main, 102 = ch1 sub. Credentials are percent-encoded and embedded
because the GStreamer/OpenCV decode path in `channels/channel.py` takes a single
URL string and has nowhere else to put them.

---

## Database

New table `discovered_cameras`, created automatically by
`Base.metadata.create_all` on the next boot — **no manual migration needed**,
because this is a brand-new table, not an alteration of an existing one.

Key columns: `identity` (unique), `camera_uuid` (link to `camera_configs`),
`is_present`, `consecutive_misses`, `alerted`, `first_seen_at`, `last_seen_at`,
`missing_since`, plus `ip_address` / `mac_address` / `serial_number` / `model` /
`firmware` / `device_name`.

---

## Deployment notes

- **Python 3.6 / stdlib only.** No new dependency was added to
  `requirements.txt`. Discovery uses raw sockets, `urllib`, and
  `xml.etree.ElementTree`. This is deliberate — pip-installing on a Jetson risks
  shadowing the JetPack `cv2`/`numpy` and silently killing GStreamer HW decode.
- Discovery failing at startup **never** stops the inference service. It is
  wrapped so a Jetson with a broken scanner still serves its existing cameras.
- Set `HIK_PASSWORD` before enabling, or every probe will fail auth and nothing
  will be adopted.
- On a network where multicast is blocked, set `DISCOVERY_SUBNETS` explicitly
  rather than relying on the derived `/24`.

## Verification status

Logic is verified by `tests/test_discovery.py` (35 tests, `python3
tests/test_discovery.py` from `Backend/tensort`) covering RTSP URL
construction, CIDR expansion and rejection, digest auth against a live HTTP
server, ISAPI identification (including wrong-credential and non-Hikvision
rejection), the full sweep lifecycle (adopt → idempotent → grace window →
missing → alert once → recover → IP repoint → dedupe on wiped roster), sweep
coalescing under concurrency, all HTTP routes, and sync resilience to a failing
sweep.

The cloud client's `sync_discovery` and its 404/500/unreachable fallbacks are
**not** covered here — that code lives on the cloud side.

**Not yet run against real Hikvision hardware.** WS-Discovery multicast
behaviour and real camera ISAPI/firmware quirks are the untested surface.
