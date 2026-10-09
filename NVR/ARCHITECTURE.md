# NVR gateway and camera relay

NVR runs on the MiniPC. It owns camera discovery, verification, the local roster,
and MediaMTX relay provisioning. TensorRT runs separately on Jetson and owns
inference. Their only runtime integration is HTTP; NVR can be deployed without
the `tensort` folder. Neither service imports the other's Python code.

```mermaid
flowchart LR
    Cloud[Cloud backend] <-->|HTTP camera API / JSON / JPEG / SSE| NVR[NVR gateway on MiniPC]
    NVR <-->|HTTP API| Jetson[TensorRT on Jetson]
    NVR -->|WS-Discovery / ISAPI / frame probes| Cameras[Cameras or physical recorder]
    NVR -->|Loopback control API| Relay[MediaMTX on MiniPC]
    Cameras -->|RTSP| Relay
    Relay -->|LAN RTSP| Jetson
    Relay -->|Public or VPN RTSP| CloudMTX[Cloud MediaMTX]
    CloudMTX -->|WebRTC live view| Users[Cloud consumers]
```

## Code ownership

- `config.py`: the single `Settings`, read from the environment and `NVR/.env`.
  Every module, `discovery.py` included, receives this object; nothing else
  reads the environment.
- `discovery.py`: WS-Discovery, Hikvision ISAPI, bounded subnet probes and
  credential-free stream URL construction.
- `application/services/discovery_adapter.py`: worker-thread execution, recorder
  policy, physical-camera deduplication and credential rendering.
- `frame_verifier.py`: bounded ffmpeg probes; one decoded frame is required.
- `sweep_service.py`: discovery scheduling, presence transitions and reporting.
- `relay.py`: MiniPC MediaMTX HTTP control client, stable stream paths and public/LAN
  relay URL translation.
- `adoption_service.py`: register or repoint verified cameras through Jetson HTTP.
- `jetson.py` and `routes.py`: Jetson HTTP boundary and external API proxy.
- `core/` and `application/repositories/`: SQLite camera and NVR roster.

Discovery candidates contain an immutable, ordered tuple of sources. Their dedupe
key is derived from those sources (lowest serial, then MAC, then route identity).
Verification returns the first playable source, or no winner. The sweep passes
that result to the camera repository inside one transaction; repositories never
commit independently.

`CameraRepository.find_for_candidate` compares identifiers from every candidate
source with `Camera.serial_number`, `mac_address`, `identity` and `dedupe_key`.
Serials and MACs are normalized by discovery and stored in that form. Matches on
device identifiers take precedence over IP-based identities. Multiple matching
camera rows, or an IP-only match with contradictory known serial/MAC metadata,
produce a report error and skip that candidate rather than reassigning a camera.
Other candidates continue; unmatched cameras follow the normal missing grace period.

`CameraRepository.save_verified` owns camera creation, verified-route changes,
metadata selection and successful-sighting timestamps. `Camera` holds the stable
public identity, last verified route, best known metadata, camera presence and
Jetson adoption state. If DHCP changes an address but playback fails, the camera's
verified route remains unchanged.

Only cameras and the physical NVR are persisted. Alternate routes exist in each
scan's candidate DTOs and are tried in source-preference order. There is no stored
route inventory or historical alias lookup: if discovery loses every identifier
that matches a saved camera, the candidate cannot be recognized as that camera.
An IP-only camera whose address changes therefore becomes a new camera. Failed
playback does not trigger retries of routes absent from the current scan.

A tower has exactly one NVR role. With `NVR_HOST` empty the MiniPC is the NVR;
with `NVR_HOST` set, that one physical Hikvision recorder is. Direct LAN cameras
are discovered in both cases, and any other recorder found on the LAN is reported
and ignored. Discovery is Hikvision-oriented; WS-Discovery alone does not
negotiate arbitrary vendors' video profiles.

## Discovery and relay lifecycle

1. Discover direct cameras and the NVR's channels. Channels come from the NVR's
   ISAPI; `NVR_CHANNELS` restricts them and keeps them as RTSP candidates when
   ISAPI is unreachable.
2. Group alternate routes by camera serial/MAC. Repeated recorder serials do not
   identify individual cameras. Rank routes with `SOURCE_PREFERENCE`.
3. Decode a frame from a candidate route with ffmpeg. Only verified cameras enter
   the roster. Store LAN URLs without credentials in SQLite.
4. Reconcile a MediaMTX path named `nvr-<sha256>` from
   the camera's sticky identity. The source uses the authenticated LAN camera URL,
   RTSP over TCP and on-demand pulling. Existing identical paths are left alone;
   changed sources are patched; absent paths are added. Each verified sweep
   checks the API again, restoring paths after MediaMTX restarts.
5. Register the camera on Jetson through `GET /cameras`, `POST /cameras` or
   `PATCH /cameras/{uuid}`. Jetson receives the MiniPC's LAN relay URL. Persist its
   UUID before sending mutations so lost responses do not create duplicate cameras.
   The cloud backend requires this UUID for camera creation and never generates
   a replacement. Missing/invalid UUIDs reject adoption; an existing discovery
   identity with a different cloud UUID reports a conflict for reconciliation.
   Relay provisioning failure prevents new adoption/repointing that sweep.
6. Export the public relay URL as `source_url` in the discovery roster and report
   entries. Camera IP changes update MediaMTX's upstream source while leaving
   the public URL stable. The cloud sees no alternate LAN stream URLs.

`relay_state=configured` means MediaMTX accepted its configuration; it does not
prove public reachability or successful playback through the relay. `pending`
means this process has not configured it yet; `error` means the latest attempt
failed. The report still supplies the stable URL so cloud consumers can retry.
The `first_frame_at` evidence comes from the direct LAN probe, not a cloud probe.

Jetson capacity failures remain `capacity_pending` and retry on subsequent
verified sweeps. Jetson failures do not mark a locally verified camera missing.
At least two unsuccessful sweeps are required to mark an existing camera missing.
Missing cameras, their Jetson channels and configured relay paths are retained.
Forgetting a local roster entry is not decommissioning: it does not delete Jetson
channels or MediaMTX paths, and connected cameras can be rediscovered.

## Cloud contract

The existing cloud code requires no changes:

- `application/services/inventory_service.py::entries_from_roster` consumes
  `/discovery/report.roster[*].source_url`.
- `application/services/webrtcgateway.py::ensure_stream` uses that source as the
  input to cloud MediaMTX, which retains its WebRTC responsibility. Stream 1 is
  recorded locally by tower MediaMTX; see [RECORDINGS.md](RECORDINGS.md) for
  timestamp-based event exports, viewing/download links, and deployment.
- The cloud's device URL points to the MiniPC NVR HTTP service.

The public URL includes only the relay reader credentials, never camera login
credentials. These API responses belong behind the existing authenticated device
network/ingress. Cloud MediaMTX needs the full URL to authenticate to the relay.

Camera API JSON translates known relay `source_url` values in both directions:
public URL to LAN URL for Jetson requests, and LAN URL to public URL in responses.
This also applies to nested camera configuration and `/sync` inventory. Other
camera settings remain unchanged. Unknown/manual camera URLs pass through.
JPEG, detection JSON and SSE continue through the streaming HTTP proxy.

## API routes

Both bare and `/api`-prefixed routes are supported.

| Method | Path | Behavior |
|---|---|---|
| GET | `/discovery` | Local roster, present/missing lists and status |
| GET | `/discovery/status` | Scheduler and last sweep counters |
| GET | `/discovery/report` | Last complete report, or 404 before the first sweep |
| POST | `/discovery/scan` | Await a forced sweep; concurrent callers share it |
| DELETE | `/discovery/{identity}` | Forget local roster entry |
| POST | `/sync` | Schedule a sweep, return Jetson inventory and previous report |
| GET | `/`, `/health` | Jetson status and health |
| GET, POST | `/cameras` | Jetson inventory / camera registration |
| PATCH, DELETE | `/cameras/{uuid}` | Jetson camera controls |
| GET | `/cameras/{uuid}/latest`, `/cameras/{uuid}/snapshot.jpg` | Detection JSON / JPEG |
| GET | `/detection/{uuid}`, `/detections/{uuid}` | Detection aliases |
| GET | `/cameras/detections/stream`, `/cameras/{uuid}/detections/stream` | SSE |

The FastAPI process handles HTTP. MediaMTX is a separate process on the same
MiniPC and serves RTSP; opening the NVR HTTP port alone does not expose video.

## Deployment

1. Copy this `NVR` directory to the MiniPC. Install `requirements.txt` and system
   ffmpeg. Run one NVR worker; sweep locking is process-local.
2. Install MediaMTX with v3 control API support on that MiniPC. Use
   [mediamtx.yml](mediamtx.yml), replacing its reader password and changing
   `rtspAddress` from its loopback default to the MiniPC LAN address on port 8554.
   Keep its control API bound to loopback. Start it with `mediamtx NVR/mediamtx.yml`
   from the parent directory; manage it with your normal service supervisor.
3. Copy [.env.example](.env.example) to `NVR/.env`. Set real camera credentials,
   `NVR_HOST` if the tower has a physical NVR, the Jetson HTTP address and these
   relay values:

   ```dotenv
   MEDIAMTX_API_URL=http://127.0.0.1:9997
   NVR_PUBLIC_RTSP_BASE_URL=rtsp://nvr-reader:URL_ENCODED_PASSWORD@tower.example.com:8554
   MEDIAMTX_LAN_RTSP_BASE_URL=rtsp://nvr-reader:URL_ENCODED_PASSWORD@192.168.1.10:8554
   ```

   `tower.example.com` is a placeholder for this MiniPC's cloud-reachable DNS/IP.
   `192.168.1.10` is the MiniPC LAN address where MediaMTX serves video to Jetson.
   `JETSON_BASE_URL` separately points to the Jetson HTTP API. The
   passwords must match MediaMTX's reader account. Percent-encode special characters
   in URL credentials. Public and LAN bases must be RTSP(S) origins without paths.
   The LAN base falls back to the public base when omitted. Startup fails when
   `NVR_PUBLIC_RTSP_BASE_URL` is missing.
4. Provide cloud-to-MiniPC connectivity using a VPN or a forwarded TCP RTSP port.
   DNS, NAT, TLS and firewall provisioning are deployment work, not something a
   generated URL creates. Restrict ingress to the intended consumers. RTSP is
   unencrypted; use a private tunnel or configure MediaMTX RTSPS for public transit.
5. On Jetson, set `DISCOVERY_ENABLED=false` so NVR owns discovery and Jetson accepts
   API-managed cameras. No TensorRT code change or NVR package install is needed there.
6. Start NVR from its parent directory:

   ```bash
   python3 -m venv .venv-nvr
   .venv-nvr/bin/pip install -r NVR/requirements.txt
   .venv-nvr/bin/python -m NVR.main
   ```

Use a writable `MINIPC_DB_PATH`. Startup creates missing tables but does not migrate
existing table definitions.

## Validation

```bash
PYTHONPATH=Backend python3 -m pytest Backend/NVR/tests -q
```

Tests cover settings loading, scanner normalization and simulated ISAPI, isolated imports with all
`tensort` imports blocked, SQLite transitions, frame verification, Jetson adoption,
HTTP/SSE proxy behavior, MediaMTX failures/retries, stable URLs after camera moves,
MediaMTX restart recovery and bidirectional relay URL translation. Network tests
use simulated HTTP peers; live camera, Jetson and public RTSP playback must be
validated on the target deployment.

MediaMTX references: [control API](https://mediamtx.org/docs/references/control-api)
and [configuration](https://mediamtx.org/docs/references/configuration-file).
