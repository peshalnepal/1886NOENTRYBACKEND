import uuid
import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field, PositiveInt
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from application.services.clip_storage import EventClipService
from core.database_orm import Camera, Site, VideoRecord, User
from dependencies import get_async_db, get_current_user


router = APIRouter(prefix="/clips", tags=["clips"])
logger = logging.getLogger(__name__)


class ClipOut(BaseModel):
    id: int
    camera_uuid: str
    camera_name: Optional[str] = None
    camera_code: Optional[str] = None
    site_uuid: Optional[str] = None
    site_name: Optional[str] = None
    site_code: Optional[str] = None
    external_id: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[int] = None
    status: str
    recording_url: Optional[str] = None
    storage_key: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime


class BulkClipDeleteRequest(BaseModel):
    clip_ids: List[PositiveInt] = Field(..., min_length=1)


class BulkClipDeleteResponse(BaseModel):
    requested: int
    deleted: int
    deleted_ids: List[int] = Field(default_factory=list)


def _normalize_clip_ids(raw_ids: List[int]) -> List[int]:
    ordered_ids: List[int] = []
    seen: set[int] = set()

    for raw_id in raw_ids:
        clip_id = int(raw_id)
        if clip_id <= 0 or clip_id in seen:
            continue
        seen.add(clip_id)
        ordered_ids.append(clip_id)

    return ordered_ids


async def _fetch_owned_clips(
    *,
    db: AsyncSession,
    user_id: int,
    clip_ids: List[int],
) -> List[VideoRecord]:
    ordered_ids = _normalize_clip_ids(clip_ids)
    if not ordered_ids:
        return []

    rows = (
        await db.execute(
            select(VideoRecord)
            .join(Camera, Camera.camera_uuid == VideoRecord.camera_uuid)
            .where(
                VideoRecord.id.in_(ordered_ids),
                Camera.user_id == int(user_id),
            )
        )
    ).scalars().all()

    clips_by_id = {int(clip.id): clip for clip in rows}
    return [clips_by_id[clip_id] for clip_id in ordered_ids if clip_id in clips_by_id]


async def _delete_clip_records(
    *,
    db: AsyncSession,
    clips: List[VideoRecord],
) -> int:
    if not clips:
        return 0

    clip_service = EventClipService()
    try:
        for clip in clips:
            storage_key = str(clip.storage_key or "").strip()
            if storage_key:
                try:
                    await clip_service.delete_blob(blob_name=storage_key)
                except Exception:
                    logger.warning(
                        "Failed deleting clip blob %s; removing DB record anyway",
                        storage_key,
                        exc_info=True,
                    )

            await db.delete(clip)

        await db.commit()
    finally:
        await clip_service.close()

    return len(clips)


@router.get("", response_model=List[ClipOut])
async def list_clips(
    site_uuid: Optional[uuid.UUID] = None,
    camera_uuid: Optional[uuid.UUID] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_async_db),
    user: User = Depends(get_current_user),
):
    if limit <= 0:
        raise HTTPException(status_code=422, detail="limit must be positive")
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must be non-negative")

    stmt = (
        select(
            VideoRecord,
            Camera.name,
            Camera.camera_code,
            Site.site_uuid,
            Site.name,
            Site.site_code,
        )
        .join(Camera, Camera.camera_uuid == VideoRecord.camera_uuid)
        .join(Site, Site.site_uuid == Camera.site_uuid)
        .where(Camera.user_id == int(user.id))
        .order_by(desc(VideoRecord.created_at), desc(VideoRecord.id))
        .offset(int(offset))
        .limit(min(int(limit), 200))
    )

    if site_uuid is not None:
        stmt = stmt.where(Camera.site_uuid == site_uuid)

    if camera_uuid is not None:
        stmt = stmt.where(VideoRecord.camera_uuid == camera_uuid)

    if status:
        stmt = stmt.where(VideoRecord.status == str(status).strip())

    rows = (await db.execute(stmt)).all()
    return [
        ClipOut(
            id=int(record.id),
            camera_uuid=str(record.camera_uuid),
            camera_name=camera_name,
            camera_code=camera_code,
            site_uuid=str(record_site_uuid) if record_site_uuid is not None else None,
            site_name=site_name,
            site_code=site_code,
            external_id=str(record.external_id),
            start_time=record.start_time,
            end_time=record.end_time,
            duration=int(record.duration) if record.duration is not None else None,
            status=str(record.status),
            recording_url=record.recording_url,
            storage_key=record.storage_key,
            error=record.error,
            created_at=record.created_at,
        )
        for record, camera_name, camera_code, record_site_uuid, site_name, site_code in rows
    ]


@router.delete("", response_model=BulkClipDeleteResponse)
async def delete_clips(
    payload: BulkClipDeleteRequest,
    db: AsyncSession = Depends(get_async_db),
    user: User = Depends(get_current_user),
):
    clip_ids = _normalize_clip_ids([int(value) for value in payload.clip_ids])
    if not clip_ids:
        raise HTTPException(status_code=422, detail="Select at least one clip to delete")

    clips = await _fetch_owned_clips(db=db, user_id=int(user.id), clip_ids=clip_ids)
    deleted = await _delete_clip_records(db=db, clips=clips)
    return BulkClipDeleteResponse(
        requested=len(clip_ids),
        deleted=deleted,
        deleted_ids=[int(clip.id) for clip in clips],
    )


@router.delete("/{clip_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_clip(
    clip_id: int,
    db: AsyncSession = Depends(get_async_db),
    user: User = Depends(get_current_user),
):
    clips = await _fetch_owned_clips(db=db, user_id=int(user.id), clip_ids=[clip_id])
    if not clips:
        raise HTTPException(status_code=404, detail="Clip not found")

    await _delete_clip_records(db=db, clips=clips)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
