import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database_orm import Camera, VideoRecord, User
from dependencies import get_async_db, get_current_user


router = APIRouter(prefix="/clips", tags=["clips"])


class ClipOut(BaseModel):
    id: int
    camera_uuid: str
    camera_code: Optional[str] = None
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
    camera_uuid: Optional[uuid.UUID] = None,
    status: Optional[str] = None,
    limit: int = 100,
    db: AsyncSession = Depends(get_async_db),
    user: User = Depends(get_current_user),
):
    if limit <= 0:
        raise HTTPException(status_code=422, detail="limit must be positive")

    stmt = (
        select(VideoRecord, Camera.camera_code)
        .join(Camera, Camera.camera_uuid == VideoRecord.camera_uuid)
        .where(Camera.user_id == int(user.id))
        .order_by(desc(VideoRecord.created_at), desc(VideoRecord.id))
        .limit(min(int(limit), 200))
    )

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
        for record, camera_code in rows
    ]
