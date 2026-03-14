import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

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


def _env_csv(name: str) -> List[str]:
    raw = os.getenv(name)
    if raw is None:
        return []
    items: List[str] = []
    for part in str(raw).replace("\n", ",").split(","):
        value = str(part or "").strip()
        if value:
            items.append(value)
    return items


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


def _normalize_url_base(raw: str) -> str:
    return str(raw or "").strip().rstrip("/")


def _append_unique_url(target: List[str], raw: str) -> None:
    value = _normalize_url_base(raw)
    if value and value not in target:
        target.append(value)


def _derive_direct_playback_base(raw: str) -> str:
    base = _normalize_url_base(raw)
    if not base:
        return ""
    parsed = urlsplit(base)
    if not parsed.hostname:
        return ""
    host = parsed.hostname
    port = 9996
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth = f"{auth}:{parsed.password}"
        host = f"{auth}@{host}"
    netloc = f"{host}:{port}"
    return urlunsplit(("http", netloc, "", "", ""))


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
    """
    Pulls the recent playback window from MediaMTX and stores it in Azure Blob Storage.
    """

    def __init__(self):
        playback_base = (os.getenv("MEDIAMTX_PLAYBACK_BASE_URL") or "").strip().rstrip("/")
        public_base = (os.getenv("WEBRTC_PUBLIC_BASE_URL") or "").strip().rstrip("/")
        conn_str = (os.getenv("VIDEO_CLIP_BLOB_CONNECTION_STRING") or "").strip()

        self.playback_base_urls = self._resolve_playback_base_urls(
            playback_base=playback_base,
            public_base=public_base,
        )
        self.playback_base_url = self.playback_base_urls[0] if self.playback_base_urls else ""
        self.connection_string = conn_str
        self.container_name = (os.getenv("VIDEO_CLIP_BLOB_CONTAINER") or "event-clips").strip() or "event-clips"
        self.duration_s = int(_env_float("VIDEO_CLIP_DURATION_S", 120.0, minimum=5.0))
        self.cooldown_s = _env_float("VIDEO_CLIP_COOLDOWN_S", float(self.duration_s), minimum=0.0)
        self.minimum_duration_s = int(_env_float("VIDEO_CLIP_MIN_DURATION_S", 10.0, minimum=1.0))
        self.sas_ttl_hours = int(_env_float("VIDEO_CLIP_SAS_TTL_HOURS", 168.0, minimum=1.0))
        self.download_format = (os.getenv("VIDEO_CLIP_DOWNLOAD_FORMAT") or "mp4").strip() or "mp4"
        self.enabled = _env_bool("VIDEO_CLIP_CAPTURE_ENABLED", True) and bool(self.playback_base_urls and self.connection_string)

        timeout_s = _env_float("VIDEO_CLIP_HTTP_TIMEOUT_S", 180.0, minimum=10.0)
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)))
        self._blob_service: Optional[BlobServiceClient] = None
        self._session_factory: Optional[SessionFactory] = None
        self._camera_locks: Dict[str, asyncio.Lock] = {}
        self._recent_by_camera: Dict[str, Tuple[float, ClipCaptureResult]] = {}
        self._playback_warn_interval_s = _env_float("VIDEO_CLIP_PLAYBACK_WARN_INTERVAL_S", 60.0, minimum=1.0)
        self._playback_backoff_s = _env_float("VIDEO_CLIP_PLAYBACK_BACKOFF_S", 60.0, minimum=1.0)
        self._playback_last_warn_at = 0.0
        self._playback_unavailable_until = 0.0

        conn_parts = _parse_connection_string(self.connection_string)
        self._sas_account_name = conn_parts.get("accountname", "")
        self._sas_account_key = conn_parts.get("accountkey", "")

    def _resolve_playback_base_urls(self, *, playback_base: str, public_base: str) -> Tuple[str, ...]:
        urls: List[str] = []
        for name in ("MEDIAMTX_PLAYBACK_BASE_URLS", "VIDEO_CLIP_PLAYBACK_BASE_URLS"):
            for value in _env_csv(name):
                _append_unique_url(urls, value)

        primary = playback_base or (f"{public_base}/playback" if public_base else "")
        _append_unique_url(urls, primary)
        _append_unique_url(urls, os.getenv("MEDIAMTX_PLAYBACK_FALLBACK_BASE_URL") or "")
        _append_unique_url(urls, os.getenv("VIDEO_CLIP_PLAYBACK_FALLBACK_BASE_URL") or "")

        if _env_bool("VIDEO_CLIP_ALLOW_DIRECT_PLAYBACK_FALLBACK", False):
            _append_unique_url(urls, _derive_direct_playback_base(primary or public_base))

        return tuple(urls)

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

    async def _playback_get(
        self,
        endpoint: str,
        *,
        params: Dict[str, Any],
    ) -> httpx.Response:
        errors: List[str] = []
        for base_url in self.playback_base_urls:
            url = f"{base_url}{endpoint}"
            try:
                resp = await self._http.get(url, params=params)
            except httpx.RequestError as exc:
                errors.append(f"{url} -> {_truncate_message(exc)}")
                continue

            if resp.status_code >= 500:
                errors.append(f"{url} -> HTTP {resp.status_code}")
                continue

            return resp

        detail = "; ".join(errors) if errors else "Playback base URL is not configured"
        raise PlaybackUnavailableError(_truncate_message(detail, limit=600))

    def _mark_playback_unavailable(self) -> None:
        self._playback_unavailable_until = time.monotonic() + self._playback_backoff_s

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
        if (now - self._playback_last_warn_at) < self._playback_warn_interval_s:
            return
        self._playback_last_warn_at = now
        logger.warning(
            "Skipping clip capture; playback endpoint unavailable camera=%s path=%s trigger=%s event_ts_ms=%s playback_urls=%s detail=%s",
            camera_uuid,
            path,
            trigger,
            event_ts_ms,
            ",".join(self.playback_base_urls),
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
            "format": self.download_format,
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
            if time.monotonic() < self._playback_unavailable_until:
                return None

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
            except PlaybackUnavailableError as exc:
                self._mark_playback_unavailable()
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
                        duration_s=self.duration_s,
                        status="failed",
                        error=str(exc),
                    )
                except Exception:
                    logger.exception("Failed to persist failed video_record camera=%s external_id=%s", camera_key, external_id)
                return None
