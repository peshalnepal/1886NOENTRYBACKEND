import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from application.services.clip_storage import EventClipService
from core.database_orm import Camera, Site, VideoRecord, User
from dependencies import get_async_db, get_current_user


router = APIRouter(prefix="/clips", tags=["clips"])


class ClipOut(BaseModel):
    id: int
    camera_uuid: str
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


@router.get("", response_model=List[ClipOut])
async def list_clips(
    site_uuid: Optional[uuid.UUID] = None,
    camera_uuid: Optional[uuid.UUID] = None,
    status: Optional[str] = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_async_db),
    user: User = Depends(get_current_user),
):
    if limit <= 0:
        raise HTTPException(status_code=422, detail="limit must be positive")

    stmt = (
        select(
            VideoRecord,
            Camera.camera_code,
            Site.site_uuid,
            Site.name,
            Site.site_code,
        )
        .join(Camera, Camera.camera_uuid == VideoRecord.camera_uuid)
        .join(Site, Site.site_uuid == Camera.site_uuid)
        .where(Camera.user_id == int(user.id))
        .order_by(desc(VideoRecord.created_at), desc(VideoRecord.id))
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
        for record, camera_code, record_site_uuid, site_name, site_code in rows
    ]


@router.delete("/{clip_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_clip(
    clip_id: int,
    db: AsyncSession = Depends(get_async_db),
    user: User = Depends(get_current_user),
):
    clip = (
        await db.execute(
            select(VideoRecord)
            .join(Camera, Camera.camera_uuid == VideoRecord.camera_uuid)
            .where(
                VideoRecord.id == int(clip_id),
                Camera.user_id == int(user.id),
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if clip is None:
        raise HTTPException(status_code=404, detail="Clip not found")

    storage_key = str(clip.storage_key or "").strip()
    if storage_key:
        clip_service = EventClipService()
        try:
            await clip_service.delete_blob(blob_name=storage_key)
        finally:
            await clip_service.close()

    await db.delete(clip)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
