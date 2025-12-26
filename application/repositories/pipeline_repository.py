# application/repositories/pipeline_repository.py

import uuid
from typing import List, Optional, Sequence

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

# Adjust this import to wherever your ORM models live.
# You said you keep them all in database.py and import via core.database in other files.
from core.database_orm import Camera, ChannelConfiguration, Pipeline, PipelineCamera

from domain.channel import ChannelConfig


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
        pipeline_id: Optional[uuid.UUID] = None,
        *,
        name: str = "default",
        is_active: bool = True,
    ) -> Pipeline:
        """
        If pipeline_id is None -> create new pipeline and let ORM/DB generate UUID.
        If pipeline_id provided:
        - update if exists
        - else create with that exact id
        Always flush so pipeline.id is available to caller.
        """
        if pipeline_id is None:
            # IMPORTANT: don't set id at all, let default generate it
            pipeline = Pipeline(name=name, is_active=is_active)
            db.add(pipeline)
            await db.flush()          # INSERT happens here
            return pipeline           # pipeline.id is now populated

        # pipeline_id provided: try fetch
        stmt = select(Pipeline).where(Pipeline.id == pipeline_id)
        pipeline = (await db.execute(stmt)).scalar_one_or_none()

        if pipeline is None:
            # create with explicit id
            pipeline = Pipeline(id=pipeline_id, name=name, is_active=is_active)
            db.add(pipeline)
            await db.flush()
            return pipeline

        # update
        pipeline.name = name or pipeline.name
        pipeline.is_active = is_active
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
                selectinload(Pipeline.cameras).selectinload(Camera.channel_configuration)
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
