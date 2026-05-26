# application/repositories/pipeline_repository.py

import uuid
from typing import List, Optional, Sequence

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import IntegrityError

# Adjust this import to wherever your ORM models live.
# You said you keep them all in database.py and import via core.database in other files.
from core.database_orm import Camera, ChannelConfiguration, Pipeline, PipelineCamera, Device


class PipelineRepository:
    """
    DB access for Pipeline <-> Cameras <-> ChannelConfiguration.

    Notes:
    - No lazy loading: queries use selectinload explicitly.
    - No commits: caller (Manager/service) controls transaction boundaries.
    """

    async def pipeline_exists(self, db: AsyncSession, pipeline_id: uuid.UUID) -> bool:
        stmt = select(Pipeline).where(Pipeline.id == pipeline_id)
        row = (await db.execute(stmt)).scalars().first()
        return row is not None
    
    def _attach_channel_configs(self, pipeline: Optional[Pipeline]) -> Optional[Pipeline]:
        """Helper to attach channel configs and avoid repeating the loop 3 times."""
        if pipeline is None:
            return None
        
        channel_cfgs = [
            cam.channel_configuration 
            for cam in (pipeline.cameras or []) 
            if cam.channel_configuration is not None
        ]
        setattr(pipeline, "channel_configurations", channel_cfgs)
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

        # 1. Fetch or initialize the pipeline
        if pipeline_id:
            pipeline = await db.get(Pipeline, pipeline_id)
            if pipeline and pipeline.user_id is not None and int(pipeline.user_id) != int(user_id):
                raise ValueError("pipeline_id belongs to a different user")
            if not pipeline:
                pipeline = Pipeline(id=pipeline_id, user_id=user_id)
                db.add(pipeline)
        else:
            q = select(Pipeline).where(Pipeline.user_id == user_id, Pipeline.name == name)
            pipeline = (await db.execute(q)).scalars().first()
            if not pipeline:
                pipeline = Pipeline(user_id=user_id, name=name)
                db.add(pipeline)
                try:
                    await db.flush()  # Catch race condition early
                except IntegrityError:
                    await db.rollback()
                    pipeline = (await db.execute(q)).scalars().first()
                    if not pipeline: raise

        # 2. Apply common updates
        pipeline.user_id = user_id
        pipeline.name = name
        pipeline.is_active = bool(is_active)

        # Flush only — the caller (service/route) owns the transaction.
        await db.flush()

        return pipeline

    async def get_full_pipeline(self, db: AsyncSession, pipeline_id: uuid.UUID) -> Optional[Pipeline]:
        stmt = select(Pipeline).where(Pipeline.id == pipeline_id).options(
            selectinload(Pipeline.cameras).selectinload(Camera.channel_configuration),
            selectinload(Pipeline.cameras).selectinload(Camera.device),
        )
        return self._attach_channel_configs((await db.execute(stmt)).scalar_one_or_none())
    
    async def get_pipeline_by_userid(self, db: AsyncSession, user_id: int) -> Optional[Pipeline]:
        stmt = select(Pipeline).where(Pipeline.user_id == int(user_id), Pipeline.name == "default").options(
            selectinload(Pipeline.cameras).selectinload(Camera.channel_configuration),
            selectinload(Pipeline.cameras).selectinload(Camera.device),
        ).order_by(Pipeline.created_at.asc())
        return self._attach_channel_configs((await db.execute(stmt)).scalars().first())
    
    async def get_default_pipeline_for_user(self, db: AsyncSession, user_id: int) -> Optional[Pipeline]:
        return await self.get_pipeline_by_userid(db, user_id)