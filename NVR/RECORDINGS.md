# Tower recordings and event clips

Each tower records **stream 1** locally and uses **stream 2** for live view and
Jetson detection. The cloud stores clip metadata and URLs; it does not ingest
continuous main-stream video or upload these clips to Azure Blob storage.

This records new footage through the tower's MediaMTX service. It does not read
older recordings from a physical Hikvision recorder's proprietary archive.
The current discovery adapter supports Hikvision `/Streaming/Channels/<id>`
URLs: channel 1 uses `101` for recording and `102` for live view; channel 17 uses
`1701` and `1702`. The channel offset and credentials are preserved.

## Tower setup

1. Install the NVR Python requirements, MediaMTX **1.16.1**, and FFmpeg/ffprobe
   with the `libx264` encoder. Run **one NVR API worker** per tower. Export jobs
   and their concurrency limit are process-local.
2. Run MediaMTX with `NVR/mediamtx.yml`. Keep its control API (`9997`) and
   recording playback API (`9996`) on loopback. Bind RTSP to the tower LAN
   interface for Jetson/cloud readers, as documented in `ARCHITECTURE.md`.
3. Create persistent directories `/var/lib/nvr/recordings` and
   `/var/lib/nvr/clips`. Give MediaMTX write access to the recordings directory
   and the NVR API write access to the clips directory. With containers, mount
   persistent tower disks at these paths; the configured `RECORDING_DIRECTORY`
   is interpreted inside **MediaMTX**, not inside the Python process.
4. Configure `NVR/.env` using `.env.example`:

   ```dotenv
   RECORDING_ENABLED=true
   RECORDING_PLAYBACK_URL=http://127.0.0.1:9996
   RECORDING_DIRECTORY=/var/lib/nvr/recordings
   RECORDING_RETENTION=168h
   RECORDING_CLIP_DIRECTORY=/var/lib/nvr/clips
   RECORDING_CLIP_RETENTION_S=604800
   RECORDING_PUBLIC_BASE_URL=https://tower.example.com/api
   RECORDING_API_KEY=<random-secret-at-least-32-characters>
   ```

5. Configure the tower HTTPS reverse proxy to forward `/api/recordings/*` to
   this NVR API, preserving the path, Authorization and Range headers. Allow
   long export requests (at least 180 seconds; increase for busy towers). The
   public base must be reachable by **viewers**, not only by the cloud backend.
   It can use a private VPN address if the viewers also have VPN access.
6. Start `python -m NVR.main` from `Backend/`. Discovery provisions two paths
   per camera: `nvr-<identity-hash>` (stream 2, on demand, no recording), and
   `record-<identity-hash>` (stream 1, always pulling and recording).
   Recording works independently of the Jetson's detection capacity. Check
   `recording_state` in `/api/discovery`; `configured` confirms configuration,
   not that camera video is arriving. Check MediaMTX readiness and recording
   files to confirm healthy ingest. Synchronize tower clocks with NTP.

The main stream is saved as fMP4 without continuous transcoding. Raw segments
are 15 seconds, with seven-day retention by default. Extracted event clips are
saved separately for seven days, independent of raw segment deletion. Event
exports alone are encoded to H.264/AAC MP4 with fast-start for browser playback
and seeking. Concurrency defaults to two encodes and sixteen pending requests.
Requests exceeding the pending limit receive `503` with `Retry-After`.

Size storage using camera **main-stream bitrates**. For example, eight cameras
at 4 Mbps need about **345.6 GB/day**, or **2.42 TB for seven days**, before
filesystem overhead and extracted clips. Set retention for the installed disk
and monitor free space; there is no automatic disk-size quota/oldest-first
eviction beyond time-based retention. The default export size cap is 512 MiB.

## Cloud setup

For each camera, its assigned Device's `device_url` must point to the **tower
NVR HTTP gateway**, e.g. `https://tower.example.com/api`, not directly to Jetson.
The existing camera UUID must match the UUID in that tower's roster.

```dotenv
VIDEO_CLIP_PROVIDER=nvr
VIDEO_CLIP_CAPTURE_ENABLED=true
NVR_RECORDING_API_KEY=<same-secret-as-the-tower>
VIDEO_CLIP_HTTP_TIMEOUT_S=180
SITE_PRERECORD_TIMEOUT_S=180
```

The current deployment uses one shared service key across managed towers;
give it only to trusted backend/tower processes. `deploy.sh` passes the key as
the secure Bicep parameter `nvrRecordingApiKey`. No schema migration is needed:
camera context now includes the existing Device URL. Cloud clip records retain
the existing approval and overlay behavior and contain no blob storage key.

The cloud YAML, Bicep/JSON templates, and newly provisioned cloud paths disable
recording. **Existing runtime cloud paths may still have `record: true`:** after
towers have been validated, restart/reprovision cloud MediaMTX paths or patch
their `record` setting to `false`. Disabling cloud recording before tower disks,
secrets and endpoints are ready causes a recording gap. This code change does
not deploy or restart existing services.

The legacy cloud playback implementation remains selectable with
`VIDEO_CLIP_PROVIDER=mediamtx`; it requires its previous recording configuration.

## Request a two-minute event clip

```http
POST /api/recordings/cameras/<camera_uuid>/events
Authorization: Bearer <RECORDING_API_KEY>
Content-Type: application/json

{
  "event_time": "2026-10-5-15-18",
  "timezone": "America/Toronto"
}
```

Alternatively send `{"event_time":"2026-10-05T15:18:00-04:00"}`. The backend
sends UTC timestamps automatically. Timezone-less inputs require an IANA
timezone; ambiguous/nonexistent DST times require an explicit offset.
Minute-only inputs mean `:00` seconds. This example selects **15:16:30 through
15:18:30 Toronto time**, returned as `19:16:30Z` through `19:18:30Z`.

The request waits until the 30-second post-event tail and up to 20 seconds of
segment flush slack have elapsed, then exports the clip. An HTTP timeout does
not cancel an in-process export; repeat the same camera/timestamp to obtain the
cached result. An API restart interrupts unfinished exports; retry the request
after restart. Completed exports and metadata survive restarts on disk.

```json
{
  "clip_id": "<stable-id>",
  "camera_uuid": "<camera_uuid>",
  "event_time": "2026-10-05T19:18:00Z",
  "start_time": "2026-10-05T19:16:30Z",
  "end_time": "2026-10-05T19:18:30Z",
  "duration": 120,
  "status": "completed",
  "expires_at": 1791832800,
  "recording_url": "https://tower.example.com/api/recordings/clips/<id>/video?expires=...&signature=...",
  "view_url": "https://tower.example.com/api/recordings/clips/<id>/video?expires=...&signature=...",
  "download_url": "https://tower.example.com/api/recordings/clips/<id>/video?expires=...&signature=...&download=true"
}
```

Use `view_url` as a video source; use `download_url` for an attachment download.
Both support HTTP byte ranges. Links are bearer capabilities: anyone holding
one can use it until expiry. Avoid logging query strings on this route.
`GET /api/recordings/clips/<clip_id>` with the service Authorization header
returns metadata/links for a saved clip. Expired links return `403`; expired
metadata returns `410` until cleanup removes it, then `404`.

No historical footage is fabricated: missing cameras return `404`, incomplete
120-second windows return `409`, invalid timestamps return `422`, and recorder
failures return `502`. A new installation needs at least 90 seconds of footage
before it can satisfy a new event's pre-roll. Exports crossing recording gaps
or incompatible codec changes are rejected rather than silently shortened.

## Verification

From `Backend/`, run:

```sh
python -m pytest tests/test_nvr_recordings.py tests/test_clip_capture.py -q
```

The tests cover tower provisioning, timestamps/DST, complete-window checks,
export caching and limits, signed links, Range downloads, the cloud-to-tower
request/metadata flow, and a real synthetic two-minute FFmpeg MP4 export.
Validate a real tower's camera clock, disk retention, public HTTPS playback,
and simultaneous eight-camera event exports before switching production.
