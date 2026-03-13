import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from azure.storage.blob import BlobSasPermissions, ContentSettings, generate_blob_sas
from azure.storage.blob.aio import BlobServiceClient
from sqlalchemy.ext.asyncio import AsyncSession

from application.repositories.notification_repository import CameraContext
from core.database_orm import VideoRecord


logger = logging.getLogger(__name__)

SessionFactory = Callable[[], AsyncSession]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        value = float(default)
    return max(minimum, value)


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
    """
    Pulls the recent playback window from MediaMTX and stores it in Azure Blob Storage.
    """

    def __init__(self):
        playback_base = (os.getenv("MEDIAMTX_PLAYBACK_BASE_URL") or "").strip().rstrip("/")
        public_base = (os.getenv("WEBRTC_PUBLIC_BASE_URL") or "").strip().rstrip("/")
        conn_str = (os.getenv("VIDEO_CLIP_BLOB_CONNECTION_STRING") or "").strip()

        self.playback_base_url = playback_base or (f"{public_base}/playback" if public_base else "")
        self.connection_string = conn_str
        self.container_name = (os.getenv("VIDEO_CLIP_BLOB_CONTAINER") or "event-clips").strip() or "event-clips"
        self.duration_s = int(_env_float("VIDEO_CLIP_DURATION_S", 120.0, minimum=5.0))
        self.cooldown_s = _env_float("VIDEO_CLIP_COOLDOWN_S", float(self.duration_s), minimum=0.0)
        self.minimum_duration_s = int(_env_float("VIDEO_CLIP_MIN_DURATION_S", 10.0, minimum=1.0))
        self.sas_ttl_hours = int(_env_float("VIDEO_CLIP_SAS_TTL_HOURS", 168.0, minimum=1.0))
        self.download_format = (os.getenv("VIDEO_CLIP_DOWNLOAD_FORMAT") or "mp4").strip() or "mp4"
        self.enabled = _env_bool("VIDEO_CLIP_CAPTURE_ENABLED", True) and bool(self.playback_base_url and self.connection_string)

        timeout_s = _env_float("VIDEO_CLIP_HTTP_TIMEOUT_S", 180.0, minimum=10.0)
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)))
        self._blob_service: Optional[BlobServiceClient] = None
        self._session_factory: Optional[SessionFactory] = None
        self._camera_locks: Dict[str, asyncio.Lock] = {}
        self._recent_by_camera: Dict[str, Tuple[float, ClipCaptureResult]] = {}

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

    async def _get_blob_service(self) -> BlobServiceClient:
        if self._blob_service is None:
            self._blob_service = BlobServiceClient.from_connection_string(self.connection_string)
        return self._blob_service

    async def _fetch_recording_spans(
        self,
        *,
        path: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        url = f"{self.playback_base_url}/list"
        params = {
            "path": path,
            "start": self._iso_utc(start_time),
            "end": self._iso_utc(end_time),
        }
        resp = await self._http.get(url, params=params)
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
        url = f"{self.playback_base_url}/get"
        params = {
            "path": path,
            "start": self._iso_utc(start_time),
            "duration": f"{int(duration_s)}s",
            "format": self.download_format,
        }
        resp = await self._http.get(url, params=params)
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
            expiry=datetime.now(timezone.utc) + timedelta(hours=self.sas_ttl_hours),
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
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        path = str(ctx.camera_code or "").strip()
        if not path:
            return None

        camera_key = str(camera_uuid)
        lock = self._camera_lock(camera_key)
        async with lock:
            now_mono = time.monotonic()
            cached = self._recent_by_camera.get(camera_key)
            if cached and (now_mono - cached[0]) <= self.cooldown_s:
                return cached[1].to_payload()

            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(seconds=self.duration_s)
            external_id = f"{camera_key}-{int(end_time.timestamp())}-{uuid.uuid4().hex[:10]}"

            try:
                spans = await self._fetch_recording_spans(
                    path=path,
                    start_time=start_time,
                    end_time=end_time,
                )
                window = self._choose_span(spans, desired_start=start_time, desired_end=end_time)
                if window is None:
                    logger.info("Skipping clip capture; no playback window available camera=%s path=%s", camera_key, path)
                    return None

                clip_start, clip_end = window
                duration_s = int(max(0.0, (clip_end - clip_start).total_seconds()))
                if duration_s < self.minimum_duration_s:
                    logger.info(
                        "Skipping clip capture; available window too small camera=%s path=%s duration=%ss",
                        camera_key,
                        path,
                        duration_s,
                    )
                    return None

                payload = await self._download_clip(path=path, start_time=clip_start, duration_s=duration_s)
                if not payload:
                    return None

                storage_key = self._build_storage_key(camera_uuid=camera_key, start_time=clip_start, external_id=external_id)
                recording_url = await self._upload_blob(blob_name=storage_key, payload=payload)

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
                    )
                except Exception:
                    logger.exception("Failed to persist successful video_record camera=%s external_id=%s", camera_key, external_id)

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
                        duration_s=self.duration_s,
                        status="failed",
                        error=str(exc),
                    )
                except Exception:
                    logger.exception("Failed to persist failed video_record camera=%s external_id=%s", camera_key, external_id)
                return None
