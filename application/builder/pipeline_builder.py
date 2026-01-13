import logging
from application.channels.channel import VideoChannel
from domain.template import Template as DomainTemplate
from domain.model_pipeline import ModelPipeline
from sqlalchemy.ext.asyncio import AsyncSession
from application.models.yolo_model import YoloMultiTaskModel
from application.models.yolo_config import YoloModelConfig

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
        channels = [VideoChannel(config=cfg) for cfg in (template.configs or [])]

        model = None

        cfg = getattr(template, "model_cfg", None)
        if cfg is not None:
            if isinstance(cfg, dict):
                cfg = YoloModelConfig.model_validate(cfg)
            model = YoloMultiTaskModel(cfg)

        return ModelPipeline(
            pipeline_id=template.id,
            model=model,
            channels=channels,
        )


