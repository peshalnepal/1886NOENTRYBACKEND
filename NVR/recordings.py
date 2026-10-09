"""Tower-local event exports. Video bytes never pass through the cloud API."""

import asyncio
import hashlib
import hmac
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from starlette.responses import FileResponse

from .core.database_orm import Camera
from .relay import MediaRelay, RelayError

logger = logging.getLogger(__name__)
router = APIRouter()
UTC = timezone.utc


class EventRequest(BaseModel):
    event_time: str
    timezone: str | None = None


def parse_event_time(value: str, zone: str | None = None) -> datetime:
    """Require explicit time semantics; compact minute precision means :00."""
    try:
        if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}", value):
            value_dt = datetime.strptime(value, "%Y-%m-%d-%H-%M")
        else:
            value_dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if value_dt.tzinfo is None:
            if not zone:
                raise ValueError("timezone required")
            tz = ZoneInfo(zone)
            first, second = value_dt.replace(tzinfo=tz, fold=0), value_dt.replace(tzinfo=tz, fold=1)
            if first.utcoffset() != second.utcoffset():
                raise ValueError("ambiguous or nonexistent local time; use an explicit UTC offset")
            value_dt = first
        return value_dt.astimezone(UTC)
    except (ValueError, ZoneInfoNotFoundError, OverflowError):
        raise HTTPException(422, "Use an ISO timestamp with UTC offset, or a local timestamp plus an unambiguous IANA timezone") from None


def iso(value):
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def covers_window(spans, start, end):
    # MediaMTX reports contiguous, codec-compatible spans. Never label a
    # partial recording or a window crossing a gap as a complete event clip.
    for span in spans:
        try:
            first = datetime.fromisoformat(span["start"].replace("Z", "+00:00"))
            last = first + timedelta(seconds=float(span["duration"]))
            if first <= start and last >= end:
                return True
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return False


class RecordingService:
    def __init__(self, settings, sessions, relay=None):
        self.settings, self.sessions = settings, sessions
        self.relay = relay
        self.directory = Path(settings.recording_clip_directory)
        self.http = httpx.AsyncClient(base_url=str(settings.recording_playback_url).rstrip("/"),
                                      timeout=120, trust_env=False)
        self.exports = asyncio.Semaphore(settings.recording_max_exports)
        self.pending = {}
        self.cleaner = None

    async def start(self):
        if self.settings.recording_enabled:
            await asyncio.to_thread(self.directory.mkdir, parents=True, exist_ok=True)
            self.cleaner = asyncio.create_task(self._cleanup_loop())

    async def close(self):
        tasks = list(self.pending.values()) + ([self.cleaner] if self.cleaner else [])
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.http.aclose()

    def authorize(self, request):
        key = self.settings.recording_api_key
        if not key or not self.settings.recording_enabled:
            raise HTTPException(503, "Tower recording API is not configured")
        if not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + key):
            raise HTTPException(401, "Recording API authentication required")

    def _cleanup(self):
        now = time.time()
        for path in self.directory.glob("*"):
            if path.suffix not in {".json", ".mp4", ".part"}:
                continue
            try:
                # Temporary exports are removed only after a day (active
                # exports have a much shorter timeout).
                ttl = 86400 if path.suffix == ".part" else self.settings.recording_clip_retention_s
                if now - path.stat().st_mtime > ttl:
                    path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass

    async def _cleanup_loop(self):
        while True:
            try:
                await asyncio.to_thread(self._cleanup)
            except OSError:
                logger.warning("Unable to clean expired tower clips")
            await asyncio.sleep(60)

    def _manifest(self, clip_id):
        if not re.fullmatch(r"[a-f0-9]{64}", clip_id):
            raise HTTPException(404, "Clip not found")
        try:
            data = json.loads((self.directory / (clip_id + ".json")).read_text())
            if not (self.directory / (clip_id + ".mp4")).is_file():
                raise FileNotFoundError
            if data["expires_at"] <= time.time():
                raise HTTPException(410, "Clip has expired")
            return data
        except (FileNotFoundError, ValueError, KeyError):
            raise HTTPException(404, "Clip not found") from None

    def _signature(self, clip_id, expires):
        return hmac.new(self.settings.recording_api_key.encode(),
                        f"{clip_id}:{expires}".encode(), hashlib.sha256).hexdigest()

    def describe(self, data):
        base = self.settings.recording_public_base_url
        if not base:
            raise HTTPException(503, "RECORDING_PUBLIC_BASE_URL must be configured")
        query = urlencode({"expires": data["expires_at"],
                           "signature": self._signature(data["clip_id"], data["expires_at"])})
        url = f"{base}/recordings/clips/{data['clip_id']}/video?{query}"
        return {**data, "recording_url": url, "view_url": url, "download_url": url + "&download=true"}

    async def capture(self, camera_uuid, event):
        if not self.settings.recording_public_base_url:
            raise HTTPException(503, "RECORDING_PUBLIC_BASE_URL must be configured")
        if event > datetime.now(UTC) + timedelta(seconds=5):
            raise HTTPException(422, "Event timestamp is in the future")
        async with self.sessions() as db:
            camera = (await db.execute(select(Camera).where(Camera.camera_uuid == camera_uuid))).scalar_one_or_none()
        if camera is None:
            raise HTTPException(404, "Camera is not registered on this tower")
        clip_id = hashlib.sha256(f"{camera_uuid}:{iso(event)}".encode()).hexdigest()
        try:
            return self.describe(self._manifest(clip_id))
            # {
            # "clip_id": "<64-character-clip-id>",
            # "camera_uuid": "92a11140-ed7a-4471-9f72-fba7bbd596de",
            # "event_time": "2026-10-05T19:18:00Z",
            # "start_time": "2026-10-05T19:16:30Z",
            # "end_time": "2026-10-05T19:18:30Z",
            # "duration": 120,
            # "status": "completed",
            # "expires_at": 1791832730,
            # "recording_url": "https://tower.example.com/api/recordings/clips/<clip-id>/video?expires=1791832730&signature=<signature>",
            # "view_url": "https://tower.example.com/api/recordings/clips/<clip-id>/video?expires=1791832730&signature=<signature>",
            # "download_url": "https://tower.example.com/api/recordings/clips/<clip-id>/video?expires=1791832730&signature=<signature>&download=true"
            # }
        except HTTPException as exc:
            if exc.status_code not in {404, 410}:
                raise
        task = self.pending.get(clip_id)
        if task is None:
            if len(self.pending) >= self.settings.recording_max_pending:
                raise HTTPException(503, "Tower export queue is full; retry later", headers={"Retry-After": "10"})
            task = asyncio.create_task(self._export(clip_id, camera_uuid, camera.identity, event,
                                       source_kind=camera.source_kind, source_url=camera.source_url))
            self.pending[clip_id] = task
            def finished(done):
                self.pending.pop(clip_id, None)
                if not done.cancelled():
                    done.exception()  # retrieve errors even if requesting client disconnected
            task.add_done_callback(finished)
        return self.describe(await asyncio.shield(task))

    async def _export(self, clip_id, camera_uuid, identity, event, *, source_kind=None, source_url=None):
        start, end = event - timedelta(seconds=90), event + timedelta(seconds=30)
        # Allow a full 15-second recording segment to close, plus I/O slack.
        delay = (end + timedelta(seconds=20) - datetime.now(UTC)).total_seconds()
        if delay > 0:
            await asyncio.sleep(delay)
        path = MediaRelay.recording_path(identity)
        temporary = self.directory / (clip_id + ".part")
        try:
            async with self.exports:
                # API-created MediaMTX paths disappear after its restart. Restore
                # the stored path even for an offline camera so its historical
                # files can still be found by the playback server.
                if self.relay is not None:
                    await self.relay.ensure_recording(identity, self.relay.adapter.render(source_kind, source_url))
                response = await self.http.get("/list", params={"path": path, "start": iso(start), "end": iso(end)})
                if response.status_code == 404:
                    raise HTTPException(404, "No tower recordings exist for this event")
                response.raise_for_status()
                if not covers_window(response.json(), start, end):
                    raise HTTPException(409, "The full 90-second pre-event and 30-second post-event recording is unavailable")
                size = 0
                async with self.http.stream("GET", "/get", params={"path": path, "start": iso(start),
                                              "duration": "120", "format": "mp4"}) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as output:
                        async for chunk in response.aiter_bytes(65536):
                            size += len(chunk)
                            if size > self.settings.recording_max_clip_bytes:
                                raise HTTPException(413, "Event clip exceeds tower export size limit")
                            await asyncio.to_thread(output.write, chunk)
                if not size:
                    raise HTTPException(502, "Recorder returned an empty clip")
                await self._finalize_video(temporary)
                temporary.replace(self.directory / (clip_id + ".mp4"))
                data = {"clip_id": clip_id, "camera_uuid": camera_uuid, "event_time": iso(event),
                        "start_time": iso(start), "end_time": iso(end), "duration": 120,
                        "status": "completed", "expires_at": int(time.time()) + self.settings.recording_clip_retention_s}
                temporary.write_text(json.dumps(data))
                temporary.replace(self.directory / (clip_id + ".json"))
                return data
        except (RelayError, httpx.HTTPError, ValueError, KeyError, TypeError, OSError, asyncio.TimeoutError):
            raise HTTPException(502, "Tower recording export failed") from None
        finally:
            temporary.unlink(missing_ok=True)

    async def _run(self, *args, timeout=120):
        process = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE,
                                                       stderr=asyncio.subprocess.DEVNULL)
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout)
            if process.returncode:
                raise HTTPException(502, "Tower could not prepare a playable event clip")
            return stdout
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    async def _finalize_video(self, source):
        # Validate MediaMTX's export, which can be shorter if a camera changed
        # codec mid-window. Encode only event clips for portable browser playback
        # and frame-accurate duration; continuous recordings remain stream copies.
        probe = await self._run(self.settings.ffprobe_bin, "-v", "error", "-show_entries",
                                "format=duration", "-of", "json", str(source), timeout=15)
        duration = float(json.loads(probe)["format"]["duration"])
        if not 119.5 <= duration <= 125:
            raise HTTPException(409, "Recorder export does not cover the complete event window")
        output = source.with_suffix(".encoding.part")
        try:
            await self._run(self.settings.ffmpeg_bin, "-nostdin", "-v", "error", "-y", "-i", str(source),
                "-map", "0:v:0", "-map", "0:a?", "-t", "120", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "23", "-pix_fmt", "yuv420p", "-threads", "2", "-c:a", "aac",
                "-movflags", "+faststart", "-f", "mp4", str(output))
            if output.stat().st_size > self.settings.recording_max_clip_bytes:
                raise HTTPException(413, "Encoded clip exceeds tower export size limit")
            output.replace(source)
        finally:
            output.unlink(missing_ok=True)


@router.post("/recordings/cameras/{camera_uuid}/events")
async def capture_event(camera_uuid: str, body: EventRequest, request: Request):
    service = request.app.state.recordings
    service.authorize(request)
    return await service.capture(camera_uuid, parse_event_time(body.event_time, body.timezone))


@router.get("/recordings/clips/{clip_id}")
async def clip_metadata(clip_id: str, request: Request):
    service = request.app.state.recordings
    service.authorize(request)
    return service.describe(service._manifest(clip_id))


@router.get("/recordings/clips/{clip_id}/video")
async def clip_video(clip_id: str, request: Request, expires: int, signature: str, download: bool = False):
    service = request.app.state.recordings
    if not service.settings.recording_enabled or not service.settings.recording_api_key:
        raise HTTPException(503, "Tower recording API is not configured")
    if expires <= time.time() or not hmac.compare_digest(signature, service._signature(clip_id, expires)):
        raise HTTPException(403, "Invalid or expired clip link")
    service._manifest(clip_id)
    return FileResponse(service.directory / (clip_id + ".mp4"), media_type="video/mp4",
                        filename=clip_id + ".mp4", content_disposition_type="attachment" if download else "inline",
                        headers={"Cache-Control": "private, no-store", "Referrer-Policy": "no-referrer"})
