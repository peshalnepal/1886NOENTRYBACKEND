# Task: Build `Backend/NVR/`, the camera-tower MiniPC service

Work in **plan mode**. Complete one phase at a time. At the end of each phase, stop and wait for my approval before writing any code for the next phase. Every code change must be discussed first. For each phase, present:
- the architecture decision and why you made it,
- a file-by-file change list (new/modified, with the purpose of each), and
- the risks and what you will test.

Do not edit a file I have not approved in the current phase.

## Decisions already made (do not re-ask)

| Topic | Decision |
|---|---|
| Physical layout | One **camera tower** = one cloud `Device` (`Backend/core/database_orm.py:270`). A tower always holds one **MiniPC** and one **Jetson**, plus IP cameras, all on the same LAN. **A tower has exactly one NVR role, never two or more NVRs.** |
| NVR mode | `NVR_MODE=minipc`: the MiniPC **is** the NVR (0 physical NVRs) and **records video locally via MediaMTX**. `NVR_MODE=hikvision`: exactly **one** physical Hikvision NVR is a camera source and does the recording; the MiniPC does not record. In both modes the MiniPC discovers, keeps the roster, adopts cameras, and relays detections and live video. Startup fails if the configured NVR count does not match the mode (0 for `minipc`, 1 for `hikvision`). A second recorder found by the LAN sweep is reported as an error, and its channels are not adopted. |
| Direct + NVR cameras | Cameras may be **directly on the LAN even when a physical NVR exists**. The same camera can therefore be seen both directly and through an NVR channel. That camera must be deduped into one roster row. |
| Public exposure | **Only the MiniPC is public.** The Jetson, the cameras and the NVRs are LAN-only. |
| Discovery owner | The MiniPC is the **single source of truth**. The Jetson becomes passive: `DISCOVERY_ENABLED=false` in its `.env`, with no code deleted. It only accepts cameras pushed to its existing `POST /cameras`. |
| Discovery code | **Reuse `Backend/tensort/discovery.py`** (stdlib-only: ONVIF WS-Discovery + ISAPI, `probe_device`, `_probe_nvr_channels`, `parse_remote_nvrs`, `STATIC_NVRS`, identity `serial→mac→ip`). Don't fork its logic. Port the sweep lifecycle from `tensort/service.py`: adopt / miss-threshold / one-shot alert / recover / IP-repoint / `_sweep_lock` coalescing. |
| NVR roster | `nvr_devices` holds **at most one row**. `cameras` has nullable `nvr_id` + `channel_no`, which mirror the winning source. `camera_sources` holds every route (direct and/or NVR channel) to a camera. Direct LAN cameras have `nvr_id = NULL`. **Dedupe** by the camera's serial, then its MAC, not by `identity_of()`, which adds `#<ch>` to NVR identities. A serial shared by several channels of the NVR is the recorder's own serial and is never used for dedupe. **Winning source:** a verified direct route wins, with fallback to the NVR channel (`SOURCE_PREFERENCE=direct|nvr`). The public `identity` is sticky: it is the first identity seen and is never rewritten. |
| Frame verification | The **MiniPC verifies** with ffmpeg on x86 (decode one frame, with a timeout and a session semaphore per host) **before** adoption. Jetson `/health` `capture.<id>.connected` is recorded afterwards as `jetson_connected`, a health signal, never a gate. |
| discovery.py import | Namespace import `from tensort import discovery`, with `Backend/` on `sys.path` (`python -m NVR.main`). `tensort/` has no `__init__.py`, so no Jetson change is needed. The MiniPC `.env` uses the **same variable names** (`HIK_*`, `NVR_*`, `DISCOVERY_NVRS`, `STATIC_NVRS`, …). |
| Secrets at rest | Stream URLs are stored **without credentials** and rendered from settings when used. |
| Detection flow | The MiniPC adopts a camera, mints the `camera_uuid` (uuid4), and pushes it to the Jetson as a channel. The Jetson runs inference. The MiniPC **subscribes to the Jetson's existing SSE `GET /cameras/detections/stream`**, so the Jetson needs no code change. The MiniPC enriches each event (camera_uuid, NVR/source, channel no., name) and re-exposes it publicly. |
| MiniPC public API | (1) detections: SSE for all cameras and per camera, plus REST for the latest detections per camera; (2) the camera/NVR roster with rescan and forget; (3) a **reverse proxy of the Jetson control API**; (4) **video relay via MediaMTX** on the MiniPC. |
| Stack | FastAPI + asyncio + SQLAlchemy 2.x async + **SQLite** (aiosqlite), Pydantic v2. The MiniPC is x86, so the Jetson pip restrictions don't apply here. Run blocking discovery probes in a thread executor, never on the event loop. |
| Auth | **Per-tower API key** (`X-API-Key`) on every public route. The cloud stores the key per Device. TLS will come from a reverse proxy later, so don't build it. |
| Scope | `Backend/NVR/` + Jetson config change + **minimal cloud wiring** (Device points at the MiniPC). **No frontend.** |

## Facts to rely on (verify, don't rediscover from scratch)

- `tensort/DISCOVERY.md` and `tensort/ARCHITECTURE.md` describe the current edge. Read these first.
- The cloud talks to the edge through `Backend/application/services/edgeinference.py` (`EDGE_ADD_PATH=/cameras`, `EDGE_SYNC_PATH=/sync`, discovery-report path, `_headers()`), and reconciles in `application/services/manager/controllers/reconcile.py`. The cloud drops any camera whose `camera_uuid` is not a valid UUID. Reconcile must **not** delete edge-discovered cameras (it filters them out of `to_remove`), or you get a permanent add/delete loop.
- The cloud detection consumer is `Backend/application/channels/channel.py:65` (`/api/cameras/{camera_uuid}/detections/stream` and `/cameras/...`). **Serve the same paths on the MiniPC** so the cloud consumer works unchanged.
- The Jetson caps enabled cameras at `MAX_CAMERAS` (default 8). `POST /cameras` above that returns **409**. The MiniPC must treat 409 as "pending capacity" and retry, not as an error.
- Discovery currently confirms a camera only after a **decoded frame** arrives. Decide in Phase 0 where that verification happens now. The options are: MiniPC ffprobe/OpenCV on x86, or confirm via Jetson `/health` `stats.capture.<id>.connected` after the push. Present both with their trade-offs.
- Hikvision RTSP path: `/Streaming/Channels/{ch*100+stream}`, using substream `2` for detection. NVR channels go through the NVR (`NVR_USERNAME/PASSWORD`, `NVR_CHANNEL_OFFSET`). A remote NVR often exposes only RTSP; that case is what `STATIC_NVRS` handles.
- MediaMTX serves WHEP **anonymously**. A public MiniPC MediaMTX therefore needs auth (MediaMTX `authInternalUsers` or a proxy in front). Flag this explicitly.
- `create_all` never alters existing tables. Any cloud column change needs migration code in `Backend/core/database.py::initialize_tables_and_data()`, detected with `information_schema.COLUMNS`.
- Cloud conventions: `Backend/CLAUDE.md` (route → schema → DTO → repository; repositories don't commit). Mirror them inside `NVR/` where it's sensible.
- **Never pip-install anything on the Jetson.**

## Phases (stop after each one)

**Phase 0: Explore and design (no code).** ✅ Done (2026-09-29). The outcomes are recorded in the decisions table above and under "Open decisions" below.
Read: `tensort/discovery.py`, `service.py`, `repositories/discovery_repository.py`, `routes/discovery_routes.py`, `routes/detection_routes.py`, `routes/camera_routes.py`, `pipeline.py` (SSE event shape), `database_orm.py`; and on the cloud side, `edgeinference.py`, `reconcile.py`, `application/channels/channel.py`, and where MediaMTX paths are patched today.

Deliver:
- an ASCII tower diagram (LAN vs public, ports), drawn for both **0 NVRs** and **N NVRs**,
- a sequence diagram of camera found → adopted → pushed to the Jetson → detection → MiniPC → cloud,
- the proposed `NVR/` folder tree,
- the SQLite schema (`nvr_devices`, `cameras` with nullable `nvr_id` + `channel_no`, presence/miss/alert columns),
- the public API table (method, path, auth, request/response),
- how `discovery.py` is imported without forking it (copy vs. shared package vs. `sys.path`), with a recommendation,
- the frame-verification decision, and
- open questions.

**Phase 1: Skeleton, config, DB, roster.** `NVR/` app factory, `.env.example`, settings, SQLite models + repository, and a discovery adapter + sweep scheduler that fills the roster (physical NVRs and their channels, plus direct cameras). No Jetson interaction yet. Include unit tests using the fakes pattern from `tensort/tests/test_discovery*.py`. Cover: `minipc` mode (direct cameras only); `hikvision` mode (NVR channels only); `hikvision` mode plus direct cameras, including the same camera seen directly and via the NVR; a recorder serial shared across channels; and a config with 2 NVRs being rejected. `camera_uuid` stays NULL and `jetson_state=pending` until Phase 2.

**Phase 2: Adoption into the Jetson.** Async Jetson client (add/patch/delete/list/health). Mint the UUID, push the camera, handle 409 retries, repoint on IP change, and handle missing/recovered. It must be idempotent across MiniPC restarts: reconcile the roster against Jetson `GET /cameras` on boot.

**Phase 3: Detection relay.** A single long-lived SSE subscription to the Jetson with backoff reconnect. Enrich each event, keep a latest-per-camera cache, and fan out to bounded per-client queues (drop oldest). Pass through `discovery_report` fleet events generated by the MiniPC. Include a test with a fake Jetson SSE.

**Phase 4: Public API, proxy and auth.** API-key dependency; roster routes; `/sync` and `/discovery*` served **by the MiniPC with the same response shape as the Jetson's** (see DISCOVERY.md, "`POST /sync` response"), so cloud reconcile works unchanged; reverse proxy for the remaining Jetson control routes (`/cameras*`, `/health`, snapshots) with a timeout and error mapping.

**Phase 5: Video relay + recording.** MediaMTX config on the MiniPC: source-on-demand paths per `camera_uuid`, auth enabled, with the anonymous `any` user removed. The path lifecycle is tied to adoption/forget. In `NVR_MODE=minipc`, record locally: recorded paths must be **always-on** (`sourceOnDemand: false`); fMP4 segments go under `/var/lib/nvr/recordings`; `recordDeleteAfter` handles retention; a disk watchdog deletes the oldest segments above `RECORD_MAX_DISK_PCT`. The MediaMTX playback server (:9996) requires auth and is not public.

**Phase 6: Jetson + cloud wiring.** Jetson: `.env`/`.env.example` set `DISCOVERY_ENABLED=false`; bind to the LAN only. Cloud: `Device.device_url` = MiniPC public URL; the per-device API key is sent by `edgeinference._headers()` (propose the storage: new column + migration vs. env); verify that reconcile, the detection consumer and live view work through the MiniPC. Change as little as possible.

**Phase 7: Docs and deploy.** `NVR/README.md` (architecture, config table, runbook, troubleshooting), a systemd unit, and a smoke-test script for the full chain.

## Open decisions (from Phase 0; settle each one before its phase)

| # | Phase | Issue | Proposed default |
|---|---|---|---|
| C1/Q1 | 6 | The cloud detection consumer (`channel.py:17`) sends **no API key**, so "key on every route" and "consumer unchanged" conflict. | Add the key header in `channel.py` (about 5 lines) |
| Q2 | 6 | Where the per-device key is stored | New `devices.api_key` column + migration; `_headers()` falls back to `EDGE_API_KEY` |
| C2/Q10 | 6 | The Jetson hard-codes `host="0.0.0.0"` (`tensort/main.py`), so binding to the LAN cannot be done through `.env`. | Firewall: port 8080 accepts only the MiniPC's IP. No code change. |
| C5/Q5 | 6 | Reconcile never removes roster cameras from the edge (`reconcile.py:433`), so arm/disarm has no effect on discovered cameras. | Accept for now; propose a fix in Phase 6 |
| Q3 | 4/5 | Which `source_url` is reported to the cloud | The MiniPC relay URL `rtsp://…@minipc:8554/<uuid>`. The Jetson pulls the LAN relay. |
| Q4 | 4 | Cloud `POST`/`DELETE /cameras` through the MiniPC | Intercept (store `detection_wanted`), then push to the Jetson; pass a 409 through |
| Q6 | 4 | MiniPC `/health` `camera_selection` | `"manual"` |
| Q7 | 2 | Priority when the roster has more cameras than `MAX_CAMERAS` | `detection_wanted`, then NVR config order, then `first_seen_at` |
| Q9 | 2 | Towers where the Jetson already adopted cameras | At boot, match the Jetson's `GET /cameras` by `discovery_identity`, then serial+channel, then endpoint, and reuse those UUIDs |
| R1 | 5 | Which stream to record | Main stream on a separate always-on path `<uuid>_rec` |
| R2 | 5 | Retention | 14 days and ≤ 90% disk |
| R3 | 5 | The cloud's MediaMTX also records (`record: yes`) | Keep it; no cloud change |
| R4 | 5 | Recording in `hikvision` mode | No (the NVR records) |

## Rules
- Keep the phase report short: tables and diagrams, not prose.
- Run the tests you add and paste the actual pass/fail output. Nothing has been tested on real hardware, so say that explicitly.
- Don't create commits unless I ask.
- If something in the code contradicts this prompt, stop and tell me. Don't work around it silently.
