# application/repositories/pipeline_repository.py

import uuid
from typing import List, Optional, Sequence

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import IntegrityError

# Adjust this import to wherever your ORM models live.
# You said you keep them all in database.py and import via core.database in other files.
from core.database_orm import Camera, ChannelConfiguration, Pipeline, PipelineCamera, CameraDevice, Device

from dto import ChannelConfig


class PipelineRepository:
    """
    DB access for Pipeline <-> Cameras <-> ChannelConfiguration.

    Notes:
    - No lazy loading: queries use selectinload explicitly.
    - No commits: caller (Manager/service) controls transaction boundaries.
    """

    async def pipeline_exists(self, db: AsyncSession, pipeline_id: uuid.UUID) -> bool:
        stmt = select(Pipeline.id).where(Pipeline.id == pipeline_id)
        row = (await db.execute(stmt)).scalar_one_or_none()
        return row is not None
    

    async def upsert_pipeline(
        self,
        db: AsyncSession,
        user_id: int,
        pipeline_id: Optional[uuid.UUID] = None,
        *,
        name: str = "default",
        is_active: bool = True,
    ) -> Pipeline:
        """
        Upsert semantics (stable per-user default):
        - If pipeline_id is None:
            -> return existing pipeline for (user_id, name) if present (and update is_active)
            -> otherwise create a new pipeline with (user_id, name)
        - If pipeline_id is provided:
            -> update if exists
            -> else create with that exact id (and attach to user_id)
        Always flush so pipeline.id is available to caller.
        """
        if user_id is None:
            raise ValueError("user_id is required for upsert_pipeline()")

        if pipeline_id is None:
            q = (
                select(Pipeline)
                .where(Pipeline.user_id == user_id, Pipeline.name == name)
                .order_by(Pipeline.created_at.asc())
            )
            pipeline = (await db.execute(q)).scalars().first()

            if pipeline is not None:
                pipeline.is_active = bool(is_active)
                pipeline.name = name
                await db.flush()
                return pipeline

            pipeline = Pipeline(user_id=user_id, name=name, is_active=bool(is_active))
            db.add(pipeline)

            # If you have (user_id, name) UNIQUE constraint, this handles race conditions.
            try:
                await db.flush()
                return pipeline
            except IntegrityError:
                # Another concurrent request created it first.
                await db.rollback()
                pipeline = (await db.execute(q)).scalars().first()
                if pipeline is None:
                    raise
                pipeline.is_active = bool(is_active)
                await db.flush()
                return pipeline

        stmt = select(Pipeline).where(Pipeline.id == pipeline_id)
        pipeline = (await db.execute(stmt)).scalar_one_or_none()

        if pipeline is None:
            pipeline = Pipeline(
                id=pipeline_id,
                user_id=user_id,
                name=name,
                is_active=bool(is_active),
            )
            db.add(pipeline)
            await db.flush()
            return pipeline

        # Safety: prevent cross-user accidental reuse
        if pipeline.user_id is not None and int(pipeline.user_id) != int(user_id):
            raise ValueError("pipeline_id belongs to a different user")

        pipeline.user_id = user_id
        pipeline.name = name or pipeline.name
        pipeline.is_active = bool(is_active)
        await db.flush()
        return pipeline

    async def get_full_pipeline(self, db: AsyncSession, pipeline_id: uuid.UUID) -> Optional[Pipeline]:
        """
        Returns a Pipeline with cameras and each camera's channel_configuration loaded.

        Also attaches a computed attribute:
          pipeline.channel_configurations = [ChannelConfiguration, ...]
        so your Manager._load_channel_configs() works unchanged.
        """
        stmt = (
            select(Pipeline)
            .where(Pipeline.id == pipeline_id)
            .options(
                selectinload(Pipeline.cameras).selectinload(Camera.channel_configuration),
                selectinload(Pipeline.cameras).selectinload(Camera.devices),
            )
        )

        pipeline = (await db.execute(stmt)).scalar_one_or_none()
        if pipeline is None:
            return None

        # Attach computed list so Manager can iterate `pipeline.channel_configurations`
        # and access `.configuration` on each entry.
        channel_cfgs: List[ChannelConfiguration] = []
        for cam in (pipeline.cameras or []):
            if cam.channel_configuration is not None:
                channel_cfgs.append(cam.channel_configuration)

        setattr(pipeline, "channel_configurations", channel_cfgs)
        return pipeline
