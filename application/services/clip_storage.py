import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import os
import time
import uuid
from urllib.parse import urlencode
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobSasPermissions, ContentSettings, generate_blob_sas
from azure.storage.blob.aio import BlobServiceClient
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from application.repositories.notification_repository import CameraContext
from application.services.overlay_normalize import (
    _append_overlay_detection,
    normalize_overlay_frame_dict as _normalize_overlay_frame,
)
from application.services.storage_common import parse_connection_string as _parse_connection_string
from core.coercions import coerce_positive_int as _coerce_positive_int
from core.database_orm import VideoRecord
from core.env import env_float, env_int

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], AsyncSession]


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


def _normalize_overlay_payload(
    raw_payload: Any,
    *,
    default_camera_uuid: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_payload, dict):
        return None

    camera_uuid = str(raw_payload.get("camera_uuid") or default_camera_uuid or "").strip() or None
    raw_frames = raw_payload.get("frames")
    frames: List[Dict[str, Any]] = []

    if isinstance(raw_frames, list):
        for raw_frame in raw_frames:
            normalized_frame = _normalize_overlay_frame(raw_frame, default_camera_uuid=camera_uuid)
            if normalized_frame is not None:
                frames.append(normalized_frame)

    if not frames:
        root_frame = _normalize_overlay_frame(
            {
                "camera_uuid": camera_uuid,
                "frame_ts_ms": raw_payload.get("frame_ts_ms"),
                "frame_seq": raw_payload.get("frame_seq"),
                "frame_w": raw_payload.get("frame_w"),
                "frame_h": raw_payload.get("frame_h"),
                "detections": raw_payload.get("detections"),
            },
            default_camera_uuid=camera_uuid,
        )
        if root_frame is not None:
            frames.append(root_frame)

    if not frames:
        return None

    merged_frames: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for frame in frames:
        merged_frames[(frame["frame_ts_ms"], frame["frame_seq"])] = frame
    ordered_frames = sorted(
        merged_frames.values(),
        key=lambda item: (int(item["frame_ts_ms"]), int(item["frame_seq"])),
    )
    reference = ordered_frames[-1]

    payload: Dict[str, Any] = {
        "camera_uuid": camera_uuid or str(reference.get("camera_uuid") or ""),
        "frame_ts_ms": int(reference["frame_ts_ms"]),
        "frame_seq": int(reference["frame_seq"]),
        "detections": list(reference.get("detections") or []),
        "frames": ordered_frames,
    }
    if reference.get("frame_w") is not None:
        payload["frame_w"] = int(reference["frame_w"])
    if reference.get("frame_h") is not None:
        payload["frame_h"] = int(reference["frame_h"])
    for key in ("clip_start_time", "clip_end_time", "timeline_source", "alert_type"):
        if raw_payload.get(key) is not None:
            payload[key] = raw_payload.get(key)
    return payload


def _merge_overlay_payloads(
    current_payload: Any,
    next_payload: Any,
    *,
    default_camera_uuid: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    normalized_current = _normalize_overlay_payload(
        current_payload,
        default_camera_uuid=default_camera_uuid,
    )
    normalized_next = _normalize_overlay_payload(
        next_payload,
        default_camera_uuid=default_camera_uuid,
    )

    if normalized_current is None:
        return normalized_next
    if normalized_next is None:
        return normalized_current

    frames_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for raw_frame in list(normalized_current.get("frames") or []) + list(normalized_next.get("frames") or []):
        normalized_frame = _normalize_overlay_frame(raw_frame, default_camera_uuid=default_camera_uuid)
        if normalized_frame is None:
            continue
        frames_by_key[(normalized_frame["frame_ts_ms"], normalized_frame["frame_seq"])] = normalized_frame

    ordered_frames = sorted(
        frames_by_key.values(),
        key=lambda item: (int(item["frame_ts_ms"]), int(item["frame_seq"])),
    )
    if not ordered_frames:
        return normalized_next

    reference = ordered_frames[-1]
    preserved_root = _normalize_overlay_frame(normalized_current, default_camera_uuid=default_camera_uuid)
    if preserved_root is None:
        preserved_root = _normalize_overlay_frame(normalized_next, default_camera_uuid=default_camera_uuid)
    if preserved_root is not None:
        reference = frames_by_key.get(
            (preserved_root["frame_ts_ms"], preserved_root["frame_seq"]),
            preserved_root,
        )

    merged: Dict[str, Any] = {
        "camera_uuid": str(
            normalized_current.get("camera_uuid")
            or normalized_next.get("camera_uuid")
            or default_camera_uuid
            or ""
        ),
        "frame_ts_ms": int(reference["frame_ts_ms"]),
        "frame_seq": int(reference["frame_seq"]),
        "detections": list(reference.get("detections") or []),
        "frames": ordered_frames,
    }
    if reference.get("frame_w") is not None:
        merged["frame_w"] = int(reference["frame_w"])
    if reference.get("frame_h") is not None:
        merged["frame_h"] = int(reference["frame_h"])
    for key in ("clip_start_time", "clip_end_time", "timeline_source"):
        value = normalized_next.get(key) if normalized_next.get(key) is not None else normalized_current.get(key)
        if value is not None:
            merged[key] = value
    return merged


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


def extract_notification_clip_storage_keys(payload: Any) -> List[str]:
    """Extract clip storage_key values embedded in a notification payload."""
    if not isinstance(payload, dict):
        return []
    keys: List[str] = []

    def _collect(raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        key = str(raw.get("storage_key") or "").strip()
        if key:
            keys.append(key)

    msg = payload.get("msg")
    if isinstance(msg, dict):
        _collect(msg)
        _collect(msg.get("clip"))

    extra = payload.get("extra")
    if isinstance(extra, dict):
        _collect(extra)
        _collect(extra.get("clip"))
        for item in list(extra.get("multi_camera_prerecordings") or []):
            _collect(item)

    _collect(payload.get("clip"))
    return list(dict.fromkeys(keys))


def extract_notification_clip_external_ids(payload: Any) -> List[str]:
    """Extract clip external_id values embedded in a notification payload.

    Used to flip the captured VideoRecord(s) for an alert visible/hidden when an
    operator approves or rejects it (clips are matched to alerts by external_id,
    since there is no FK between them).
    """
    if not isinstance(payload, dict):
        return []
    ids: List[str] = []
    clips = payload.get("clip")
    if isinstance(clips, dict):
        ext = str(clips.get("external_id") or "").strip()
        if ext:
            ids.append(ext)
    return list(dict.fromkeys(ids))


class EventClipService:
    # Clip layout: PRE_EVENT_S before the event + POST_EVENT_S after = CLIP_DURATION_S total.
    # Default: 1 min 30 sec of pre-roll + 30 sec of post-roll = 2 min total.
    PRE_EVENT_S = 90
    POST_EVENT_S = 30
    CLIP_DURATION_S = PRE_EVENT_S + POST_EVENT_S
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

        capture_enabled_raw = os.getenv("VIDEO_CLIP_CAPTURE_ENABLED", "").strip().lower()
        capture_enabled = (
            capture_enabled_raw not in {"0", "false", "no", "off"}
            if capture_enabled_raw
            else True
        )
        self.storage_enabled = bool(self.connection_string)
        self.enabled = bool(capture_enabled and self.playback_base_url)

        # Env overrides for the class defaults. Pre/post-roll are the source of
        # truth for the clip layout; the total duration is always derived from
        # them so the event stays anchored at the PRE_EVENT_S mark.
        self.PRE_EVENT_S = env_int("VIDEO_CLIP_PRE_EVENT_S", self.PRE_EVENT_S)
        self.POST_EVENT_S = env_int("VIDEO_CLIP_POST_EVENT_S", self.POST_EVENT_S)
        self.CLIP_DURATION_S = self.PRE_EVENT_S + self.POST_EVENT_S
        self.MINIMUM_DURATION_S = env_int("VIDEO_CLIP_MIN_DURATION_S", self.MINIMUM_DURATION_S)
        self.SAS_TTL_HOURS = env_int("VIDEO_CLIP_SAS_TTL_HOURS", self.SAS_TTL_HOURS)
        self.HTTP_TIMEOUT_S = env_float("VIDEO_CLIP_HTTP_TIMEOUT_S", self.HTTP_TIMEOUT_S)

        raw_fmt = (os.getenv("VIDEO_CLIP_DOWNLOAD_FORMAT") or "").strip().lower()
        if raw_fmt:
            self.DOWNLOAD_FORMAT = raw_fmt

        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(self.HTTP_TIMEOUT_S, connect=min(10.0, self.HTTP_TIMEOUT_S))
        )
        self._blob_service: Optional[BlobServiceClient] = None
        self._session_factory: Optional[SessionFactory] = None
        self._camera_locks: Dict[str, asyncio.Lock] = {}
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
            event_time = now
        else:
            try:
                event_time = datetime.fromtimestamp(float(event_ts_ms) / 1000.0, tz=timezone.utc)
            except (TypeError, ValueError, OSError, OverflowError):
                event_time = now

        # Anchor the window on the event itself: PRE_EVENT_S of context before the
        # trigger frame, then POST_EVENT_S after so viewers see what happened next.
        start_time = event_time - timedelta(seconds=self.PRE_EVENT_S)
        end_time = event_time + timedelta(seconds=self.POST_EVENT_S)

        # If the post-event tail hasn't been recorded yet, clamp to "now" so we
        # still return a valid window; the caller waits before invoking, so this
        # path should be rare.
        if end_time > now:
            end_time = now
            if start_time > end_time:
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

    def _build_playback_clip_url(
        self,
        *,
        path: str,
        start_time: datetime,
        duration_s: int,
    ) -> str:
        params = urlencode(
            {
                "path": path,
                "start": self._iso_utc(start_time),
                "duration": f"{int(duration_s)}s",
                "format": self.DOWNLOAD_FORMAT,
            }
        )
        return f"{self.playback_base_url}/get?{params}"

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
        requires_approval: bool = False,
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
                visible=not requires_approval,
                approval_status="pending" if requires_approval else "approved",
            )
            db.add(row)
            await db.commit()
            await db.refresh(row)

    async def set_clips_approval(self, *, external_ids: List[str], approved: bool) -> int:
        """Flip stored clips' operator-approval state, matched by external_id.

        Called when an operator approves/rejects an alert: approval makes the
        captured playback visible to the end user; rejection keeps it hidden.
        Returns the number of rows updated.
        """
        if self._session_factory is None:
            return 0

        clean_ids = [str(e).strip() for e in (external_ids or []) if str(e or "").strip()]
        if not clean_ids:
            return 0

        async with self._session_factory() as db:
            result = await db.execute(
                update(VideoRecord)
                .where(VideoRecord.external_id.in_(clean_ids))
                .values(
                    visible=bool(approved),
                    approval_status="approved" if approved else "rejected",
                )
                .execution_options(synchronize_session=False)
            )
            await db.commit()
            return int(result.rowcount or 0)

    async def update_overlay_payload(
        self,
        *,
        camera_uuid: str,
        external_id: str,
        overlay_payload: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if self._session_factory is None or not overlay_payload:
            return overlay_payload

        try:
            camera_uuid_obj = uuid.UUID(str(camera_uuid))
        except Exception:
            return overlay_payload

        async with self._session_factory() as db:
            row = (
                await db.execute(
                    select(VideoRecord)
                    .where(
                        VideoRecord.camera_uuid == camera_uuid_obj,
                        VideoRecord.external_id == str(external_id),
                    )
                    .order_by(VideoRecord.id.desc())
                )
            ).scalars().first()

            if row is None:
                return overlay_payload

            merged = _merge_overlay_payloads(
                getattr(row, "overlay_payload", None),
                overlay_payload,
                default_camera_uuid=str(camera_uuid),
            )
            row.overlay_payload = merged
            await db.commit()
            return merged
              
    async def capture_pre_event_clip(
        self,
        *,
        camera_uuid: str,
        ctx: CameraContext,
        event_ts_ms: Optional[int] = None,
        trigger: Optional[str] = None,
        overlay_payload: Optional[Dict[str, Any]] = None,
        requires_approval: bool = False,
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


            # Wait long enough for MediaMTX to flush the POST_EVENT_S tail of the
            # recording (segments are written on a fixed cadence — see
            # recordSegmentDuration in main-prod.bicep). The +2s slack covers the
            # segment-boundary rounding so /list reports the full tail before we
            # call /get. This wait now runs inside the background finalize task,
            # so it does not block notification persistence / web / email.
            #
            # IMPORTANT: the wait MUST happen before _get_capture_window(), because
            # that method clamps end_time to "now". The capture is kicked off right
            # after the event, so if we computed the window first, end_time would be
            # clamped from event+POST_EVENT_S back to ~event (no post-roll), leaving
            # the event jammed against the very end of the clip. Sleeping first lets
            # "now" advance past event+POST_EVENT_S so the full tail survives.
            if event_ts_ms is not None:
                event_time = datetime.fromtimestamp(float(event_ts_ms) / 1000.0, tz=timezone.utc)
                tail_ready_at = event_time + timedelta(seconds=self.POST_EVENT_S + 2)
                wait_s = (tail_ready_at - datetime.now(timezone.utc)).total_seconds()
                if wait_s > 0:
                    await asyncio.sleep(wait_s)

            start_time, end_time = self._get_capture_window(event_ts_ms)
            external_id = f"{camera_key}-{int(end_time.timestamp())}-{uuid.uuid4().hex[:10]}"

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

                if not self.storage_enabled:
                    recording_url = self._build_playback_clip_url(
                        path=path,
                        start_time=clip_start,
                        duration_s=duration_s,
                    )
                    try:
                        await self._save_video_record(
                            camera_uuid=camera_key,
                            external_id=external_id,
                            start_time=clip_start,
                            end_time=clip_end,
                            duration_s=duration_s,
                            status="completed",
                            storage_key="",
                            recording_url=recording_url,
                            overlay_payload=overlay_payload,
                            requires_approval=requires_approval,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to persist playback-backed video_record camera=%s external_id=%s",
                            camera_key,
                            external_id,
                        )

                    result = ClipCaptureResult(
                        external_id=external_id,
                        storage_key="",
                        recording_url=recording_url,
                        status="completed",
                        start_time=clip_start,
                        end_time=clip_end,
                        duration=duration_s,
                        path=path,
                    )
                    return result.to_payload()

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
                try:
                    recording_url = await self._upload_blob(
                        blob_name=storage_key,
                        payload=payload,
                    )
                except Exception:
                    logger.warning(
                        "Blob upload failed for clip camera=%s path=%s external_id=%s; falling back to direct playback URL",
                        camera_key,
                        path,
                        external_id,
                        exc_info=True,
                    )
                    storage_key = ""
                    recording_url = self._build_playback_clip_url(
                        path=path,
                        start_time=clip_start,
                        duration_s=duration_s,
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
                        requires_approval=requires_approval,
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
