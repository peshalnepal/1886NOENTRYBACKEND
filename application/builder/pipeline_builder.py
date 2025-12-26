import logging
from application.channels.channel import VideoChannel
from domain.template import Template as DomainTemplate
from domain.model import ModelPipeline
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class PipelineBuilder:

    def __init__(self, db_session: AsyncSession):
        """
        Initializes the PipelineBuilder.

        Args:
            db_session: An active asynchronous SQLAlchemy session.
        """
        self.db_session = db_session

    async def create(self, template: DomainTemplate) -> ModelPipeline:
        channels = []
        for cfg in (template.configs or []):
            channels.append(VideoChannel(config=cfg))

        return ModelPipeline(
            pipeline_id=template.id,
            model=None,
            channels=channels,
        )

