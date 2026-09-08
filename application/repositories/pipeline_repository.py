"""Pipeline <-> Camera <-> ChannelConfiguration persistence.

Relations are always eager-loaded with `selectinload` (no lazy loading under
async). Never commits: the caller owns the transaction.
"""

import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.database_orm import Camera, Pipeline

# Cameras plus the two relations every pipeline consumer needs.
_PIPELINE_LOAD_OPTIONS = (
    selectinload(Pipeline.cameras).selectinload(Camera.channel_configuration),
    selectinload(Pipeline.cameras).selectinload(Camera.device),
)


class PipelineRepository:
    async def pipeline_exists(self, db: AsyncSession, pipeline_id: uuid.UUID) -> bool:
        row = (
            await db.execute(select(Pipeline.id).where(Pipeline.id == pipeline_id))
        ).scalar_one_or_none()
        return row is not None

    def _attach_channel_configs(self, pipeline: Optional[Pipeline]) -> Optional[Pipeline]:
        """Expose the pipeline's channel configs as a flat `channel_configurations`
        attribute, which callers read instead of walking every camera."""
        if pipeline is None:
            return None

        pipeline.channel_configurations = [
            cam.channel_configuration
            for cam in (pipeline.cameras or [])
            if cam.channel_configuration is not None
        ]
        return pipeline

    async def upsert_pipeline(
        self,
        db: AsyncSession,
        user_id: int,
        pipeline_id: Optional[uuid.UUID] = None,
        *,
        name: str = "default",
        is_active: bool = True,
    ) -> Pipeline:
        if user_id is None:
            raise ValueError("user_id is required for upsert_pipeline()")

        if pipeline_id:
            pipeline = await db.get(Pipeline, pipeline_id)
            if pipeline and pipeline.user_id is not None and int(pipeline.user_id) != int(user_id):
                raise ValueError("pipeline_id belongs to a different user")
            if not pipeline:
                pipeline = Pipeline(id=pipeline_id, user_id=user_id)
                db.add(pipeline)
        else:
            by_name = select(Pipeline).where(
                Pipeline.user_id == user_id, Pipeline.name == name
            )
            pipeline = (await db.execute(by_name)).scalars().first()
            if not pipeline:
                pipeline = Pipeline(user_id=user_id, name=name)
                db.add(pipeline)
                try:
                    await db.flush()
                except IntegrityError:
                    # A concurrent caller inserted the same (user, name) first.
                    await db.rollback()
                    pipeline = (await db.execute(by_name)).scalars().first()
                    if not pipeline:
                        raise

        pipeline.user_id = user_id
        pipeline.name = name
        pipeline.is_active = bool(is_active)

        await db.flush()
        return pipeline

    async def get_full_pipeline(
        self, db: AsyncSession, pipeline_id: uuid.UUID
    ) -> Optional[Pipeline]:
        stmt = (
            select(Pipeline)
            .where(Pipeline.id == pipeline_id)
            .options(*_PIPELINE_LOAD_OPTIONS)
        )
        return self._attach_channel_configs((await db.execute(stmt)).scalar_one_or_none())

    async def get_pipeline_by_userid(
        self, db: AsyncSession, user_id: int
    ) -> Optional[Pipeline]:
        stmt = (
            select(Pipeline)
            .where(Pipeline.user_id == int(user_id), Pipeline.name == "default")
            .options(*_PIPELINE_LOAD_OPTIONS)
            .order_by(Pipeline.created_at.asc())
        )
        return self._attach_channel_configs((await db.execute(stmt)).scalars().first())