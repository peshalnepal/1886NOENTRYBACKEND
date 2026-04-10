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
from sqlalchemy import select
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


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_positive_int(value: Any) -> Optional[int]:
    parsed = _coerce_int(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _normalize_overlay_box(raw_box: Any) -> Optional[Dict[str, int]]:
    if isinstance(raw_box, dict):
        keys = ("x1", "y1", "x2", "y2")
        if not all(key in raw_box for key in keys):
            return None
        values = tuple(_coerce_int(raw_box.get(key)) for key in keys)
    elif isinstance(raw_box, (list, tuple)) and len(raw_box) >= 4:
        values = tuple(_coerce_int(raw_box[idx]) for idx in range(4))
    else:
        return None

    if any(value is None for value in values):
        return None

    x1, y1, x2, y2 = values
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _normalize_box_norm(raw: Any) -> Optional[Dict[str, float]]:
    if not isinstance(raw, dict):
        return None
    try:
        x = float(raw["x"])
        y = float(raw["y"])
        w = float(raw["w"])
        h = float(raw["h"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _normalize_overlay_detection(raw_detection: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_detection, dict):
        return None

    box = _normalize_overlay_box(raw_detection.get("box") or raw_detection.get("bbox"))
    if box is None:
        return None

    try:
        conf = float(raw_detection.get("conf", 0.0) or 0.0)
    except (TypeError, ValueError):
        conf = 0.0

    result: Dict[str, Any] = {
        "cls_name": str(raw_detection.get("cls_name") or raw_detection.get("class") or "obj"),
        "conf": conf,
        "box": box,
    }
    box_norm = _normalize_box_norm(raw_detection.get("box_norm"))
    if box_norm is not None:
        result["box_norm"] = box_norm
    return result


def _normalize_overlay_frame(raw_frame: Any, *, default_camera_uuid: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_frame, dict):
        return None

    frame_ts_ms = _coerce_int(raw_frame.get("frame_ts_ms"))
    frame_seq = _coerce_int(raw_frame.get("frame_seq"))
    if frame_ts_ms is None or frame_seq is None:
        return None

    detections: List[Dict[str, Any]] = []
    seen = set()
    for raw_detection in list(raw_frame.get("detections") or []):
        normalized = _normalize_overlay_detection(raw_detection)
        if normalized is None:
            continue
        box = normalized["box"]
        key = (
            normalized["cls_name"],
            normalized["conf"],
            box["x1"],
            box["y1"],
            box["x2"],
            box["y2"],
        )
        if key in seen:
            continue
        seen.add(key)
        detections.append(normalized)

    if not detections:
        return None

    frame: Dict[str, Any] = {
        "frame_ts_ms": int(frame_ts_ms),
        "frame_seq": int(frame_seq),
        "detections": detections,
    }
    camera_uuid = str(raw_frame.get("camera_uuid") or default_camera_uuid or "").strip()
    if camera_uuid:
        frame["camera_uuid"] = camera_uuid
    frame_w = _coerce_positive_int(raw_frame.get("frame_w"))
    frame_h = _coerce_positive_int(raw_frame.get("frame_h"))
    if frame_w is not None:
        frame["frame_w"] = frame_w
    if frame_h is not None:
        frame["frame_h"] = frame_h
    return frame


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
        self.storage_enabled = bool(self.connection_string)
        self.enabled = bool(capture_enabled and self.playback_base_url)

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
                    self._recent_by_camera[camera_key] = (time.monotonic(), result)
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
