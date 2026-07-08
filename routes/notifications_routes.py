import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.responses import StreamingResponse

from application.services.notification import WebNotificationHub
from application.services.notification.types import NotificationMessage
from application.repositories.notification_repository import NotificationRepository
from application.repositories.site_repository import SiteRepository
from application.repositories._helpers import as_uuid as _as_uuid
from application.services.authz_service import AuthzService
from application.services.user_snapshot_cache import (
    CachedUserSnapshot,
    UserSnapshotCache,
    UserSnapshotLookupError,
)
from core.database_orm import Notification, User
from core.schemas import (
    ChartPoint,
    ClearNotificationsRequest,
    DeleteNotificationsRequest,
    DetectionsOverTimeOut,
    NoteAttributes,
    NotificationOut,
)
from core.security.tokens import decode_access_token
from dependencies import (
    get_async_db,
    get_current_user,
    get_notification_hub,
    get_notification_service,
    get_session_factory,
    get_user_snapshot_cache,
    RequirePermission,
    OrgContext,
)
from core.security.roles import Permission
from application.services.notification import NotificationService

router = APIRouter(prefix="/notifications")
logger = logging.getLogger(__name__)

notif_repo = NotificationRepository()
site_repo = SiteRepository()
async def _notif_site_scope(db: AsyncSession, ctx: OrgContext) -> List[uuid.UUID]:
    """Site UUIDs whose notifications the caller may read: all org sites for
    admins/operators, only granted sites for plain members."""
    org_sites = await site_repo.list_site_uuids(db, org_id=ctx.org_id)
    accessible = await AuthzService.accessible_site_uuids(
        db, user=ctx.user, org_id=ctx.org_id, role=ctx.role
    )
    if accessible is None:
        return org_sites
    return [s for s in org_sites if s in accessible]


async def _resolve_target_sites(
    db: AsyncSession, ctx: OrgContext, site_uuid: Optional[str]
) -> Tuple[Optional[uuid.UUID], List[uuid.UUID]]:
    """Parse the requested site filter and intersect it with the caller's
    readable sites. Returns ``(parsed_uuid, target_sites)``."""
    su = _as_uuid(site_uuid)
    target_sites = await _notif_site_scope(db, ctx)
    if su is not None:
        target_sites = [su] if su in target_sites else []
    return su, target_sites


# -------------------------------------------------------------------
# stream helpers
# -------------------------------------------------------------------
def _sse(data: str, event: Optional[str] = None) -> str:
    if event:
        return f"event: {event}\ndata: {data}\n\n"
    return f"data: {data}\n\n"


def _extract_bearer_token(auth_header: str, access_token: Optional[str]) -> str:
    token = ""

    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()

    if not token:
        token = str(access_token or "").strip()

    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    return token


async def _resolve_stream_user(
    *,
    auth_header: str,
    access_token: Optional[str],
    cache: UserSnapshotCache,
    session_factory: async_sessionmaker[AsyncSession],
) -> CachedUserSnapshot:
    token = _extract_bearer_token(auth_header, access_token)

    try:
        payload = decode_access_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    raw_user_id = payload.get("user_id") or payload.get("sub")
    try:
        user_id = int(raw_user_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token payload")

    try:
        user = await cache.get(session_factory=session_factory, user_id=user_id)
    except UserSnapshotLookupError:
        raise HTTPException(status_code=503, detail="Database not available")

    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return user


# -------------------------------------------------------------------
# shared helpers
# -------------------------------------------------------------------


def _validate_pagination(limit: int, offset: int) -> Tuple[int, int]:
    if limit <= 0:
        raise HTTPException(status_code=422, detail="limit must be positive")
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must be non-negative")
    return min(int(limit), 500), int(offset)


def _payload_msg(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    msg = payload.get("msg")
    if isinstance(msg, dict):
        return msg
    return payload


def _payload_notes(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw = payload.get("notes")
    if not isinstance(raw, list):
        return []
    return [n for n in raw if isinstance(n, dict)]


def _to_out(n: Notification) -> NotificationOut:
    msg = _payload_msg(n.payload)
    extra = n.payload.get("extra") if isinstance(n.payload, dict) else None

    image_url = ""
    image_storage_key = ""
    clip_url = ""
    clip_status = ""

    if isinstance(extra, dict):
        image_url = str(extra.get("image_url") or "").strip()
        image_storage_key = str(extra.get("image_storage_key") or "").strip()

    if not image_url:
        image_url = str(msg.get("image_url") or "").strip()
    if not image_storage_key:
        image_storage_key = str(msg.get("image_storage_key") or "").strip()

    clip_url = str(msg.get("clip_url") or "").strip()
    clip_status = str(msg.get("clip_status") or "").strip()

    if isinstance(extra, dict) and not clip_url:
        clip_payload = extra.get("clip")
        if isinstance(clip_payload, dict):
            clip_url = str(clip_payload.get("recording_url") or "").strip()
            clip_status = str(clip_payload.get("status") or "").strip()

    return NotificationOut(
        id=int(n.id),
        user_id=int(n.user_id),
        site_uuid=str(n.site_uuid),
        camera_uuid=str(n.camera_uuid) if n.camera_uuid else None,
        site_name=str(msg.get("site_name") or "") or None,
        camera_name=str(msg.get("camera_name") or "") or None,
        device_uuid=str(n.device_uuid) if n.device_uuid else None,
        event_type=str(n.event_type),
        title=n.title,
        message=n.message,
        payload=n.payload,
        image_url=image_url or None,
        image_storage_key=image_storage_key or None,
        clip_url=clip_url or None,
        clip_status=clip_status or None,
        notes=_payload_notes(n.payload),
        detected_at=n.detected_at,
        created_at=n.created_at,
        read_at=n.read_at,
        sent_at=n.sent_at,
        status=str(n.status),
    )


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


_OBJECT_CLASS_ALIASES: Dict[str, str] = {
    "person": "person",
    "people": "person",
    "persons": "person",
    "car": "car",
    "cars": "car",
    "truck": "truck",
    "trucks": "truck",
    "motorcycle": "motorcycle",
    "motorcycles": "motorcycle",
    "motor cycle": "motorcycle",
    "motor bike": "motorcycle",
    "motorbike": "motorcycle",
    "motor-bike": "motorcycle",
}


def _canonical_object_class(raw: Any) -> Optional[str]:
    if raw is None:
        return None

    key = str(raw).strip().lower()
    if not key or key == "all":
        return None

    key = key.replace("_", " ").replace("-", " ")
    key = " ".join(key.split())

    normalized = _OBJECT_CLASS_ALIASES.get(key)
    if normalized:
        return normalized

    if key.endswith("s"):
        normalized = _OBJECT_CLASS_ALIASES.get(key[:-1])
        if normalized:
            return normalized

    return None


def _normalize_object_class(raw: Optional[str]) -> Optional[str]:
    normalized = _canonical_object_class(raw)
    if normalized:
        return normalized
    if raw is None or str(raw).strip().lower() in {"", "all"}:
        return None

    raise HTTPException(
        status_code=422,
        detail="Invalid object_class. Use one of: person, car, truck, motorcycle",
    )


def _extract_object_classes(event_type: Any, title: Any, message: Any, payload: Any) -> Set[str]:
    msg = _payload_msg(payload)
    classes: Set[str] = set()

    for source in (
        msg.get("cls_names"),
        payload.get("cls_names") if isinstance(payload, dict) else None,
    ):
        if isinstance(source, (list, tuple)):
            for item in source:
                normalized = _canonical_object_class(item)
                if normalized:
                    classes.add(normalized)

    for source in (
        msg.get("cls_name"),
        payload.get("cls_name") if isinstance(payload, dict) else None,
    ):
        normalized = _canonical_object_class(source)
        if normalized:
            classes.add(normalized)

    hint = " ".join(
        [
            str(event_type or ""),
            str(title or ""),
            str(message or ""),
        ]
    ).lower()

    if re.search(r"\bperson\b|\bpeople\b", hint):
        classes.add("person")
    if re.search(r"\bcar\b|\bcars\b", hint):
        classes.add("car")
    if re.search(r"\btruck\b|\btrucks\b", hint):
        classes.add("truck")
    if re.search(r"\bmotor[\s-]?cycle\b|\bmotor bike\b|\bmotorbike\b", hint):
        classes.add("motorcycle")

    return classes


def _is_roi_notification(event_type: Any, title: Any, message: Any, payload: Any) -> bool:
    hint = " ".join([str(event_type or ""), str(title or ""), str(message or "")]).lower()
    if "roi" in hint:
        return True

    msg = _payload_msg(payload)
    if msg.get("roi_id") not in (None, "", 0):
        return True

    if isinstance(payload, dict) and payload.get("roi_id") not in (None, "", 0):
        return True

    return False


# -------------------------------------------------------------------
# camera mode cache
# -------------------------------------------------------------------
from typing import NamedTuple


class _CameraMode(NamedTuple):
    is_enabled: bool
    detection_enabled: bool
    notification_enabled: bool
    use_site_schedule: bool
    roi: Any
    site_config: Any
    site_settings_id: Any
    timezone: str


_camera_mode_cache: Dict[uuid.UUID, _CameraMode] = {}
_inflight: Dict[uuid.UUID, "asyncio.Future[_CameraMode]"] = {}


async def invalidate_camera_mode_cache(camera_uuid: Optional[uuid.UUID] = None) -> None:
    """Remove one or all camera mode cache entries.

    Does NOT cancel in-flight DB lookups so waiting callers still receive
    the (stale) result they were waiting for — the result simply won't be
    written back to the cache because the inflight slot has been cleared.
    """
    if camera_uuid is None:
        _camera_mode_cache.clear()
        _inflight.clear()
    else:
        _camera_mode_cache.pop(camera_uuid, None)
        _inflight.pop(camera_uuid, None)


async def _get_camera_mode_cached(
    request: Any, *, cam_uuid_obj: uuid.UUID
) -> _CameraMode:
    """Fetch camera+site mode from cache, coalescing concurrent requests."""
    if cam_uuid_obj in _camera_mode_cache:
        return _camera_mode_cache[cam_uuid_obj]

    existing = _inflight.get(cam_uuid_obj)
    if existing is not None:
        return await existing

    loop = asyncio.get_running_loop()
    fut: asyncio.Future[_CameraMode] = loop.create_future()
    _inflight[cam_uuid_obj] = fut

    try:
        session_factory = request.app.state.session_factory
        async with session_factory() as db:
            row = await notif_repo.get_camera_mode_row(db, camera_uuid=cam_uuid_obj)

        mode = (
            _CameraMode(*row)
            if row is not None
            else _CameraMode(False, False, False, False, None, None, None, "UTC")
        )

        if _inflight.get(cam_uuid_obj) is fut:
            _camera_mode_cache[cam_uuid_obj] = mode
            _inflight.pop(cam_uuid_obj, None)

        fut.set_result(mode)
        return mode

    except Exception as exc:
        if _inflight.get(cam_uuid_obj) is fut:
            _inflight.pop(cam_uuid_obj, None)
        if not fut.done():
            fut.set_exception(exc)
        raise


# -------------------------------------------------------------------
# routes
# -------------------------------------------------------------------
@router.get("/stream")
async def notifications_stream(
    request: Request,
    access_token: Optional[str] = None,
    hub: WebNotificationHub = Depends(get_notification_hub),
    user_snapshot_cache: UserSnapshotCache = Depends(get_user_snapshot_cache),
    session_factory: async_sessionmaker[AsyncSession] = Depends(get_session_factory),
):
    """
    User-scoped notifications SSE stream.
    Authentication can be provided via Authorization header or access_token query param.
    """
    user = await _resolve_stream_user(
        auth_header=request.headers.get("authorization", ""),
        access_token=access_token,
        cache=user_snapshot_cache,
        session_factory=session_factory,
    )
    user_id = int(user.id)
    q = await hub.subscribe(user_id=user_id)

    async def gen():
        last_ping = time.monotonic()
        try:
            while True:
                if await request.is_disconnected():
                    break

                now = time.monotonic()
                if (now - last_ping) > 20.0:
                    last_ping = now
                    yield _sse(json.dumps({"ts": int(time.time() * 1000)}), event="ping")

                try:
                    msg = await asyncio.wait_for(q.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                yield _sse(msg.model_dump_json(), event="notification")
        finally:
            await hub.unsubscribe(user_id=user_id, q=q)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


@router.get("", response_model=List[NotificationOut])
async def list_notifications(
    site_uuid: Optional[str] = None,
    camera_uuid: Optional[str] = None,
    unread_only: bool = False,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
):
    limit, offset = _validate_pagination(limit, offset)
    cu = _as_uuid(camera_uuid)
    _, target_sites = await _resolve_target_sites(db, ctx, site_uuid)
    rows = await notif_repo.list_notifications(
        db,
        site_uuids=target_sites,
        camera_uuid=cu,
        only_visible=True,
        only_unread=True if unread_only else None,
        limit=limit,
        offset=offset,
        order_desc=True,
    )
    return [_to_out(r) for r in rows]

@router.get("/pending", response_model=List[NotificationOut])
async def list_pending_notifications(
    site_uuid: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ALERTS_APPROVE)),
):
    """Operator queue: alerts awaiting approval for the operator's org."""
    limit, offset = _validate_pagination(limit, offset)
    _, target_sites = await _resolve_target_sites(db, ctx, site_uuid)

    rows = await notif_repo.list_notifications(
        db,
        site_uuids=target_sites,
        approval_status="pending",
        limit=limit,
        offset=offset,
        order_desc=True,
    )
    return [_to_out(r) for r in rows]


class ApprovalRequest(BaseModel):
    notification_ids: List[int] = Field(default_factory=list)
    # When true, an approved alert also emails the site's configured recipients
    # (the operator's "important" opt-in). Ignored on reject.
    email: bool = False


async def _publish_approved_to_owner(
    *, hub: WebNotificationHub, db: AsyncSession, ids: List[int], target_sites: List[uuid.UUID]
) -> None:
    """Push freshly-approved alerts to the end user's realtime stream.

    The held alert was routed to operators only; on approval the owner
    (``Notification.user_id``) must finally receive it live. We rebuild the
    message from the persisted payload so the user sees the exact alert.
    """
    rows = await notif_repo.list_notifications(
        db, ids=ids, site_uuids=target_sites, approval_status="approved"
    )
    for r in rows:
        payload = r.payload if isinstance(r.payload, dict) else {}
        msg_payload = payload.get("msg")
        if not isinstance(msg_payload, dict):
            continue
        try:
            msg = NotificationMessage.model_validate(
                {**msg_payload, "db_id": int(r.id), "approval_status": "approved"}
            )
        except Exception:
            logger.warning("Could not rebuild approved notification for publish id=%s", r.id, exc_info=True)
            continue
        await hub.publish(msg)


# Strong refs to fire-and-forget urgent-report tasks (asyncio keeps only weak
# references to running tasks).
_urgent_report_tasks: set = set()


def _queue_urgent_report(
    *,
    ctx: OrgContext,
    ids: List[int],
    session_factory,
) -> None:
    """Archive an urgent report for alerts approved with the email opt-in.

    The operator chose "approve + email", meaning the alerts went to users
    immediately — that decision is captured as a downloadable urgent report in
    the org archive. Fire-and-forget so approval never blocks on PDF work.
    """
    if ctx.org_id is None:
        return  # platform-admin super context has no single org to file under

    from application.services.report import PdfReportGenerator

    generator = PdfReportGenerator(
        session_factory=session_factory,
        email=None,  # the alert email already went out; the report is archive-only
    )
    task = asyncio.create_task(
        generator.build_report(
            org_id=int(ctx.org_id),
            report_type="urgent",
            notification_ids=[int(i) for i in ids],
            persist=True,
            generated_by=int(ctx.user.id),
            generated_by_email=str(getattr(ctx.user, "email", "") or "") or None,
        ),
        name=f"urgent_report:{ctx.org_id}",
    )
    _urgent_report_tasks.add(task)
    task.add_done_callback(_urgent_report_tasks.discard)


async def _decide_notifications(
    *,
    db: AsyncSession,
    ctx: OrgContext,
    ids: List[int],
    approve: bool,
    hub: Optional[WebNotificationHub] = None,
    notification_service: Optional[NotificationService] = None,
    email_owner: bool = False,
    session_factory=None,
) -> int:
    # Authorization is enforced by RequirePermission(ALERTS_APPROVE) on the
    # calling routes; this helper only performs the state change.
    if not ids:
        raise HTTPException(status_code=422, detail="notification_ids required")
    target_sites = await _notif_site_scope(db, ctx)
    affected = await notif_repo.set_approval(
        db,
        ids=ids,
        approval_status="approved" if approve else "rejected",
        visible=approve,
        approved_by=int(ctx.user.id),
        site_uuids=target_sites,
    )
    await db.commit()
    if not affected:
        raise HTTPException(status_code=404, detail="No matching pending notifications")
    # On approval the alert leaves the operator-only feed and goes live to the
    # end user. Rejection stays hidden, so nothing is published.
    if approve and hub is not None:
        try:
            await _publish_approved_to_owner(hub=hub, db=db, ids=ids, target_sites=target_sites)
        except Exception:
            logger.exception("Failed to publish approved notifications to owner ids=%s", ids)
    if notification_service is not None:
        try:
            await notification_service.set_clips_approval_for_notifications(
                notification_ids=ids, site_uuids=target_sites, approved=approve,
            )
            # On rejection, also delete the image + clip blobs linked to the
            # alert (the hidden notification/clip rows are reaped by retention).
            if not approve:
                await notification_service.purge_alert_media_for_notifications(
                    notification_ids=ids, site_uuids=target_sites,
                )
        except Exception:
            logger.exception("Failed to reconcile clip approval for notifications ids=%s", ids)
    # When the operator marks the alert important, also email the site's
    # configured recipients. Fire-and-forget so the request doesn't block on SMTP.
    if approve and email_owner and notification_service is not None:
        try:
            notification_service.queue_approved_emails(notification_ids=ids, site_uuids=target_sites)
        except Exception:
            logger.exception("Failed to queue approval emails ids=%s", ids)
        # Approve-with-email = urgent: archive a downloadable urgent report for
        # exactly these alerts alongside the immediate email.
        if session_factory is not None:
            try:
                _queue_urgent_report(ctx=ctx, ids=ids, session_factory=session_factory)
            except Exception:
                logger.exception("Failed to queue urgent report ids=%s", ids)
    return affected


@router.post("/{notification_id}/approve")
async def approve_notification(
    notification_id: int,
    email: bool = False,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ALERTS_APPROVE)),
    hub: WebNotificationHub = Depends(get_notification_hub),
    notification_service: NotificationService = Depends(get_notification_service),
    session_factory=Depends(get_session_factory),
):
    """Operator approves a held alert; it becomes visible to admins/members.

    Pass `?email=true` to also email the site's configured recipients (the
    operator's "important" opt-in). Emailed approvals are archived as an
    urgent report; plain approvals roll into the daily general report.
    """
    affected = await _decide_notifications(
        db=db, ctx=ctx, ids=[notification_id], approve=True, hub=hub,
        notification_service=notification_service, email_owner=email,
        session_factory=session_factory,
    )
    return {"approved": affected}


@router.post("/{notification_id}/reject")
async def reject_notification(
    notification_id: int,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ALERTS_APPROVE)),
    notification_service: NotificationService = Depends(get_notification_service),
):
    """Operator rejects a held alert; it stays hidden from admins/members."""
    affected = await _decide_notifications(
        db=db, ctx=ctx, ids=[notification_id], approve=False,
        notification_service=notification_service,
    )
    return {"rejected": affected}


@router.post("/approve")
async def approve_notifications_bulk(
    payload: ApprovalRequest,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ALERTS_APPROVE)),
    hub: WebNotificationHub = Depends(get_notification_hub),
    notification_service: NotificationService = Depends(get_notification_service),
    session_factory=Depends(get_session_factory),
):
    affected = await _decide_notifications(
        db=db, ctx=ctx, ids=payload.notification_ids, approve=True, hub=hub,
        notification_service=notification_service, email_owner=payload.email,
        session_factory=session_factory,
    )
    return {"approved": affected}


@router.post("/reject")
async def reject_notifications_bulk(
    payload: ApprovalRequest,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ALERTS_APPROVE)),
    notification_service: NotificationService = Depends(get_notification_service),
):
    affected = await _decide_notifications(
        db=db, ctx=ctx, ids=payload.notification_ids, approve=False,
        notification_service=notification_service,
    )
    return {"rejected": affected}


class AddNoteRequest(BaseModel):
    # All parts optional; at least one must be present (see validator). A note
    # carries free-text `note`, the operator `action` taken, and structured
    # class-aware `attributes` (vehicle model/color/direction or person
    # gender/clothing/direction) for the report's Notes column bullets.
    note: Optional[str] = Field(default=None, max_length=2000)
    action: Optional[str] = Field(default=None, max_length=2000)
    attributes: Optional[NoteAttributes] = None

    @model_validator(mode="after")
    def _require_some_content(self) -> "AddNoteRequest":
        has_attrs = self.attributes is not None and any(
            str(v or "").strip() for v in self.attributes.model_dump().values()
        )
        if not (self.note or "").strip() and not (self.action or "").strip() and not has_attrs:
            raise ValueError("Provide at least one of note, action, or attributes.")
        return self


@router.post("/{notification_id}/note", response_model=NotificationOut)
async def add_notification_note(
    notification_id: int,
    payload: AddNoteRequest,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ALERTS_APPROVE)),
):
    """Attach an operator note to a notification.

    Notes accumulate on the notification so they can be rolled up into the
    organization's report. Scoped to the operator's readable sites.
    """
    target_sites = await _notif_site_scope(db, ctx)
    author_name = getattr(ctx.user, "user_name", None) or getattr(ctx.user, "email", None)
    row = await notif_repo.append_note(
        db,
        notification_id=notification_id,
        text=payload.note,
        action=payload.action,
        attributes=payload.attributes.model_dump(exclude_none=True) if payload.attributes else None,
        author_id=int(ctx.user.id),
        author_name=author_name,
        site_uuids=target_sites,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Notification not found")
    await db.commit()
    return _to_out(row)


@router.post("/delete")
async def delete_notifications_post(
    payload: DeleteNotificationsRequest,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_SETTINGS)),
    notification_service: NotificationService = Depends(get_notification_service),
):
    return await notification_service.handle_deletion_event(
        user_id=int(ctx.user.id),
        notification_ids=payload.notification_ids,
        site_uuid=payload.site_uuid,
        camera_uuid=payload.camera_uuid,
    )

@router.delete("")
async def delete_notifications(
    payload: DeleteNotificationsRequest,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_SETTINGS)),
    notification_service: NotificationService = Depends(get_notification_service),
):
    return await notification_service.handle_deletion_event(
        user_id=int(ctx.user.id),
        notification_ids=payload.notification_ids,
        site_uuid=payload.site_uuid,
        camera_uuid=payload.camera_uuid,
    )
    
@router.get("/detections-over-time", response_model=DetectionsOverTimeOut)
async def detections_over_time(
    site_uuid: Optional[str] = None,
    hours: int = 24,
    object_class: Optional[str] = None,
    roi_only: bool = False,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
):
    class_filter = _normalize_object_class(object_class)
    su, target_sites = await _resolve_target_sites(db, ctx, site_uuid)

    hours_i = max(1, min(int(hours), 24 * 7))
    bucket_minutes = 60 if hours_i <= 24 else 24 * 60

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours_i)
    bucket_ms = bucket_minutes * 60 * 1000
    start_ms = int(start.timestamp() * 1000)
    aligned_start_ms = start_ms - (start_ms % bucket_ms)
    now_ms = int(now.timestamp() * 1000)

    needs_payload_filter = bool(class_filter or roi_only)

    counts: Dict[int, int] = await notif_repo.aggregate_detections_over_time(
        db,
        site_uuids=target_sites,
        start=start,
        bucket_minutes=bucket_minutes,
        bucket_ms=bucket_ms,
        site_uuid=su,
        needs_payload_filter=needs_payload_filter,
        roi_only=bool(roi_only),
        class_filter=class_filter,
        as_utc_fn=_as_utc,
        is_roi_fn=_is_roi_notification,
        extract_classes_fn=_extract_object_classes,
    )

    points: List[ChartPoint] = []
    cursor = aligned_start_ms
    while cursor <= now_ms:
        points.append(
            ChartPoint(
                bucket_start=datetime.fromtimestamp(cursor / 1000.0, tz=timezone.utc),
                count=int(counts.get(cursor, 0)),
            )
        )
        cursor += bucket_ms

    total = sum(p.count for p in points)

    return DetectionsOverTimeOut(
        user_id=int(ctx.user.id),
        site_uuid=str(su) if su else None,
        hours=hours_i,
        object_class=class_filter,
        roi_only=bool(roi_only),
        bucket_minutes=bucket_minutes,
        **{"from": datetime.fromtimestamp(aligned_start_ms / 1000.0, tz=timezone.utc)},
        to=now,
        total=int(total),
        points=points,
    )


@router.get("/unread-count")
async def unread_count(
    site_uuid: Optional[str] = None,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
):
    su, target_sites = await _resolve_target_sites(db, ctx, site_uuid)
    count = await notif_repo.count_notifications(
        db,
        site_uuids=target_sites,
        only_unread=True,
        only_visible=True,
    )

    return {
        "user_id": int(ctx.user.id),
        "site_uuid": str(su) if su else None,
        "unread": int(count),
    }
