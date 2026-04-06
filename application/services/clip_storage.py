import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobSasPermissions, ContentSettings, generate_blob_sas
from azure.storage.blob.aio import BlobServiceClient
from sqlalchemy.ext.asyncio import AsyncSession

from application.repositories.notification_repository import CameraContext
from core.database_orm import VideoRecord


logger = logging.getLogger(__name__)

SessionFactory = Callable[[], AsyncSession]


def _parse_connection_string(raw: str) -> Dict[str, str]:
    parts: Dict[str, str] = {}
    for item in str(raw or "").split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parts[key.strip().lower()] = value.strip()
    return parts


def _parse_ts(raw: Any) -> Optional[datetime]:
    if not isinstance(raw, str):
        return None

    text = raw.strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def _truncate_message(raw: Any, limit: int = 240) -> str:
    text = str(raw or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


class PlaybackUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClipCaptureResult:
    external_id: str
    storage_key: str
    recording_url: str
    status: str
    start_time: datetime
    end_time: datetime
    duration: int
    path: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "external_id": self.external_id,
            "storage_key": self.storage_key,
            "recording_url": self.recording_url,
            "status": self.status,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration": self.duration,
            "path": self.path,
        }


class EventClipService:
    CLIP_DURATION_S = 120
    COOLDOWN_S = 120.0
    MINIMUM_DURATION_S = 10
    SAS_TTL_HOURS = 168
    DOWNLOAD_FORMAT = "mp4"
    HTTP_TIMEOUT_S = 180.0
    PLAYBACK_WARN_INTERVAL_S = 60.0
    PLAYBACK_BACKOFF_S = 60.0

    def __init__(self) -> None:
        self.playback_base_url = (os.getenv("MEDIAMTX_PLAYBACK_BASE_URL") or "").strip().rstrip("/")
        self.connection_string = (os.getenv("VIDEO_CLIP_BLOB_CONNECTION_STRING") or "").strip()
        self.container_name = (os.getenv("VIDEO_CLIP_BLOB_CONTAINER") or "event-clips").strip() or "event-clips"

        # Read optional overrides from env (fallback to class constants)
        capture_enabled_raw = os.getenv("VIDEO_CLIP_CAPTURE_ENABLED", "").strip().lower()
        capture_enabled = capture_enabled_raw not in {"0", "false", "no", "off"} if capture_enabled_raw else True
        self.enabled = bool(capture_enabled and self.playback_base_url and self.connection_string)

        try:
            self.CLIP_DURATION_S = int(os.getenv("VIDEO_CLIP_DURATION_S") or self.CLIP_DURATION_S)
        except (ValueError, TypeError):
            pass
        try:
            self.COOLDOWN_S = float(os.getenv("VIDEO_CLIP_COOLDOWN_S") or self.COOLDOWN_S)
        except (ValueError, TypeError):
            pass
        try:
            self.MINIMUM_DURATION_S = int(os.getenv("VIDEO_CLIP_MIN_DURATION_S") or self.MINIMUM_DURATION_S)
        except (ValueError, TypeError):
            pass
        try:
            self.SAS_TTL_HOURS = int(os.getenv("VIDEO_CLIP_SAS_TTL_HOURS") or self.SAS_TTL_HOURS)
        except (ValueError, TypeError):
            pass
        raw_fmt = (os.getenv("VIDEO_CLIP_DOWNLOAD_FORMAT") or "").strip().lower()
        if raw_fmt:
            self.DOWNLOAD_FORMAT = raw_fmt
        try:
            self.HTTP_TIMEOUT_S = float(os.getenv("VIDEO_CLIP_HTTP_TIMEOUT_S") or self.HTTP_TIMEOUT_S)
        except (ValueError, TypeError):
            pass

        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(self.HTTP_TIMEOUT_S, connect=min(10.0, self.HTTP_TIMEOUT_S))
        )
        self._blob_service: Optional[BlobServiceClient] = None
        self._session_factory: Optional[SessionFactory] = None
        self._camera_locks: Dict[str, asyncio.Lock] = {}
        self._recent_by_camera: Dict[str, Tuple[float, ClipCaptureResult]] = {}
        self._playback_last_warn_at = 0.0
        self._playback_unavailable_until = 0.0

        conn_parts = _parse_connection_string(self.connection_string)
        self._sas_account_name = conn_parts.get("accountname", "")
        self._sas_account_key = conn_parts.get("accountkey", "")

    def set_session_factory(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    async def close(self) -> None:
        await self._http.aclose()
        if self._blob_service is not None:
            await self._blob_service.close()
            self._blob_service = None

    def _camera_lock(self, camera_uuid: str) -> asyncio.Lock:
        key = str(camera_uuid)
        lock = self._camera_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._camera_locks[key] = lock
        return lock

    def _iso_utc(self, dt: datetime) -> str:
        return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _get_capture_window(self, event_ts_ms: Optional[int]) -> Tuple[datetime, datetime]:
        now = datetime.now(timezone.utc)

        if event_ts_ms is None:
            start_time = now
        else:
            try:
                event_time = datetime.fromtimestamp(float(event_ts_ms) / 1000.0, tz=timezone.utc)
            except (TypeError, ValueError, OSError, OverflowError):
                event_time = now

            # Capture POST-event footage: from event time onwards (with 5-second pre-buffer for context)
            # This ensures we capture the object's actions in the ROI, not stale pre-event footage
            start_time = event_time - timedelta(seconds=5)  # 5s pre-buffer for context

        # Ensure start_time is not in the future
        if start_time > now:
            start_time = now - timedelta(seconds=self.CLIP_DURATION_S)

        # End time should be CLIP_DURATION_S after start_time
        end_time = start_time + timedelta(seconds=self.CLIP_DURATION_S)

        # If calculated end_time is in the future, adjust both times to capture available footage
        if end_time > now:
            end_time = now
            start_time = end_time - timedelta(seconds=self.CLIP_DURATION_S)

        return start_time, end_time

    async def _get_blob_service(self) -> BlobServiceClient:
        if self._blob_service is None:
            self._blob_service = BlobServiceClient.from_connection_string(self.connection_string)
        return self._blob_service

    async def _playback_get(self, endpoint: str, *, params: Dict[str, Any]) -> httpx.Response:
        if not self.playback_base_url:
            raise PlaybackUnavailableError("Playback base URL is not configured")

        url = f"{self.playback_base_url}{endpoint}"

        try:
            resp = await self._http.get(url, params=params)
        except httpx.RequestError as exc:
            raise PlaybackUnavailableError(f"{url} -> {_truncate_message(exc)}") from exc

        if resp.status_code >= 500:
            raise PlaybackUnavailableError(f"{url} -> HTTP {resp.status_code}")

        return resp

    def _warn_playback_unavailable(
        self,
        *,
        camera_uuid: str,
        path: str,
        trigger: Optional[str],
        event_ts_ms: Optional[int],
        detail: str,
    ) -> None:
        now = time.monotonic()
        if (now - self._playback_last_warn_at) < self.PLAYBACK_WARN_INTERVAL_S:
            return

        self._playback_last_warn_at = now
        logger.warning(
            "Skipping clip capture; playback endpoint unavailable camera=%s path=%s trigger=%s event_ts_ms=%s playback_url=%s detail=%s",
            camera_uuid,
            path,
            trigger,
            event_ts_ms,
            self.playback_base_url,
            _truncate_message(detail, limit=600),
        )

    async def _fetch_recording_spans(
        self,
        *,
        path: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        params = {
            "path": path,
            "start": self._iso_utc(start_time),
            "end": self._iso_utc(end_time),
        }
        resp = await self._playback_get("/list", params=params)
        if resp.status_code in {400, 404}:
            return []

        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, list) else []

    def _choose_span(
        self,
        spans: List[Dict[str, Any]],
        *,
        desired_start: datetime,
        desired_end: datetime,
    ) -> Optional[Tuple[datetime, datetime]]:
        best: Optional[Tuple[float, float, datetime, datetime]] = None

        for span in spans:
            start_dt = _parse_ts(span.get("start"))
            try:
                span_duration_s = float(span.get("duration", 0) or 0)
            except (TypeError, ValueError):
                span_duration_s = 0.0

            if start_dt is None or span_duration_s <= 0.0:
                continue

            span_end = start_dt + timedelta(seconds=span_duration_s)
            clip_start = max(start_dt, desired_start)
            clip_end = min(span_end, desired_end)

            if clip_end <= clip_start:
                continue

            available_s = (clip_end - clip_start).total_seconds()
            candidate = (clip_end.timestamp(), available_s, clip_start, clip_end)

            if best is None or candidate[:2] > best[:2]:
                best = candidate

        if best is None:
            return None

        return best[2], best[3]

    async def _download_clip(
        self,
        *,
        path: str,
        start_time: datetime,
        duration_s: int,
    ) -> bytes:
        params = {
            "path": path,
            "start": self._iso_utc(start_time),
            "duration": f"{int(duration_s)}s",
            "format": self.DOWNLOAD_FORMAT,
        }
        resp = await self._playback_get("/get", params=params)
        if resp.status_code in {400, 404}:
            return b""

        resp.raise_for_status()
        return resp.content

    def _build_storage_key(self, *, camera_uuid: str, start_time: datetime, external_id: str) -> str:
        day = start_time.astimezone(timezone.utc).strftime("%Y/%m/%d")
        return f"clips/{camera_uuid}/{day}/{external_id}.mp4"

    def _signed_url(self, *, blob_name: str, blob_url: str) -> str:
        if not self._sas_account_name or not self._sas_account_key:
            return blob_url

        token = generate_blob_sas(
            account_name=self._sas_account_name,
            container_name=self.container_name,
            blob_name=blob_name,
            account_key=self._sas_account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=self.SAS_TTL_HOURS),
        )
        return f"{blob_url}?{token}" if token else blob_url

    async def _upload_blob(self, *, blob_name: str, payload: bytes) -> str:
        blob_service = await self._get_blob_service()
        blob = blob_service.get_blob_client(container=self.container_name, blob=blob_name)
        await blob.upload_blob(
            payload,
            overwrite=True,
            content_settings=ContentSettings(content_type="video/mp4"),
        )
        return self._signed_url(blob_name=blob_name, blob_url=blob.url)

    async def delete_blob(self, *, blob_name: str) -> bool:
        blob_key = str(blob_name or "").strip()
        if not blob_key or not self.connection_string:
            return False

        blob_service = await self._get_blob_service()
        blob = blob_service.get_blob_client(container=self.container_name, blob=blob_key)

        try:
            await blob.delete_blob(delete_snapshots="include")
        except ResourceNotFoundError:
            return False

        return True
    async def _save_video_record(
        self,
        *,
        camera_uuid: str,
        external_id: str,
        start_time: datetime,
        end_time: datetime,
        duration_s: int,
        status: str,
        storage_key: Optional[str] = None,
        recording_url: Optional[str] = None,
        overlay_payload: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        if self._session_factory is None:
            return

        try:
            camera_uuid_obj = uuid.UUID(str(camera_uuid))
        except Exception:
            return

        async with self._session_factory() as db:
            row = VideoRecord(
                camera_uuid=camera_uuid_obj,
                external_id=external_id,
                start_time=start_time,
                end_time=end_time,
                duration=int(duration_s),
                status=status,
                storage_key=storage_key,
                recording_url=recording_url,
                overlay_payload=overlay_payload,
                error=error,
            )
            db.add(row)
            await db.commit()
              
    async def capture_pre_event_clip(
        self,
        *,
        camera_uuid: str,
        ctx: CameraContext,
        event_ts_ms: Optional[int] = None,
        trigger: Optional[str] = None,
        overlay_payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        path = str(ctx.camera_code or "").strip()
        if not path:
            return None

        camera_key = str(camera_uuid)
        lock = self._camera_lock(camera_key)

        async with lock:
            if time.monotonic() < self._playback_unavailable_until:
                return None

            cached = self._recent_by_camera.get(camera_key)
            if cached and (time.monotonic() - cached[0]) <= self.COOLDOWN_S:
                return cached[1].to_payload()

            start_time, end_time = self._get_capture_window(event_ts_ms)
            external_id = f"{camera_key}-{int(end_time.timestamp())}-{uuid.uuid4().hex[:10]}"

            # ENHANCEMENT: Wait briefly to allow MediaMTX to buffer post-event segments
            # This ensures the playback endpoint has segments available before we request them
            # For ROI events, the event_ts_ms is when the object enters the ROI, and we capture
            # POST-event footage (T to T+120s), so a 1-2 second delay is reasonable to ensure
            # availability of the first few seconds of the recording window.
            if event_ts_ms is not None:
                await asyncio.sleep(1.5)

            try:
                spans = await self._fetch_recording_spans(
                    path=path,
                    start_time=start_time,
                    end_time=end_time,
                )

                window = self._choose_span(
                    spans,
                    desired_start=start_time,
                    desired_end=end_time,
                )
                if window is None:
                    logger.info(
                        "Skipping clip capture; no playback window available camera=%s path=%s",
                        camera_key,
                        path,
                    )
                    return None

                clip_start, clip_end = window
                duration_s = int(max(0.0, (clip_end - clip_start).total_seconds()))
                if duration_s < self.MINIMUM_DURATION_S:
                    logger.info(
                        "Skipping clip capture; available window too small camera=%s path=%s duration=%ss",
                        camera_key,
                        path,
                        duration_s,
                    )
                    return None

                payload = await self._download_clip(
                    path=path,
                    start_time=clip_start,
                    duration_s=duration_s,
                )
                if not payload:
                    return None

                storage_key = self._build_storage_key(
                    camera_uuid=camera_key,
                    start_time=clip_start,
                    external_id=external_id,
                )
                recording_url = await self._upload_blob(
                    blob_name=storage_key,
                    payload=payload,
                )

                try:
                    await self._save_video_record(
                        camera_uuid=camera_key,
                        external_id=external_id,
                        start_time=clip_start,
                        end_time=clip_end,
                        duration_s=duration_s,
                        status="completed",
                        storage_key=storage_key,
                        recording_url=recording_url,
                        overlay_payload=overlay_payload,
                    )
                except Exception:
                    logger.exception(
                        "Failed to persist successful video_record camera=%s external_id=%s",
                        camera_key,
                        external_id,
                    )

                result = ClipCaptureResult(
                    external_id=external_id,
                    storage_key=storage_key,
                    recording_url=recording_url,
                    status="completed",
                    start_time=clip_start,
                    end_time=clip_end,
                    duration=duration_s,
                    path=path,
                )
                self._recent_by_camera[camera_key] = (time.monotonic(), result)
                return result.to_payload()

            except PlaybackUnavailableError as exc:
                self._playback_unavailable_until = time.monotonic() + self.PLAYBACK_BACKOFF_S
                self._warn_playback_unavailable(
                    camera_uuid=camera_key,
                    path=path,
                    trigger=trigger,
                    event_ts_ms=event_ts_ms,
                    detail=str(exc),
                )
                return None

            except Exception as exc:
                logger.exception(
                    "Failed to capture pre-event clip camera=%s path=%s trigger=%s event_ts_ms=%s",
                    camera_key,
                    path,
                    trigger,
                    event_ts_ms,
                )
                try:
                    await self._save_video_record(
                        camera_uuid=camera_key,
                        external_id=external_id,
                        start_time=start_time,
                        end_time=end_time,
                        duration_s=self.CLIP_DURATION_S,
                        status="failed",
                        error=str(exc),
                    )
                except Exception:
                    logger.exception(
                        "Failed to persist failed video_record camera=%s external_id=%s",
                        camera_key,
                        external_id,
                    )
                return None
