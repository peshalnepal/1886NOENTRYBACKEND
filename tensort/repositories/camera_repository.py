# repositories/camera_repository.py  (Python 3.6)
"""
Persistence for camera configurations.

All SQLite access for `camera_configs` goes through this class. The repository
owns its own sessions (the edge service has no request-scoped session to
inherit) and commits, which is what the previous inline `PipelineRuntime`
methods did.

Async methods are the ones the pipeline loop calls; `list_all()` is sync
because it runs during construction, before the async loop is servicing work.
"""

import logging
from typing import Any, Dict, List

from sqlalchemy import select

try:
    # Script mode (python main.py from Backend/tensort)
    from database import db_manager, AsyncSessionLocal
    from database_orm import CameraConfig
except Exception:
    # Package mode (python -m Backend.tensort.main)
    from ..database import db_manager, AsyncSessionLocal
    from ..database_orm import CameraConfig

logger = logging.getLogger(__name__)


class CameraRepository(object):
    """Stateless repository over the `camera_configs` table."""

    async def save(self, camera_uuid: str, cfg_data: Dict[str, Any]) -> None:
        """
        Upsert a camera configuration.

        Swallows exceptions (logged) so a persistence failure never takes down
        the caller's pipeline mutation — same behaviour as the original
        `_save_camera_to_db_async`.
        """
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(CameraConfig).filter_by(camera_uuid=camera_uuid)
                )
                existing = result.scalar_one_or_none()

                if existing:
                    existing.source_url = cfg_data.get("source_url", existing.source_url)
                    existing.config_json = cfg_data
                else:
                    cam_config = CameraConfig(
                        channel_id=cfg_data.get("channel_id", camera_uuid),
                        camera_uuid=camera_uuid,
                        user_id=cfg_data.get("user_id", 1),
                        source_url=cfg_data["source_url"],
                        config_json=cfg_data,
                    )
                    session.add(cam_config)
                await session.commit()
        except Exception as e:
            logger.exception("Failed to save camera %s to database: %s", camera_uuid, e)

    async def delete(self, camera_uuid: str) -> None:
        """Delete a camera configuration if present."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(CameraConfig).filter_by(camera_uuid=camera_uuid)
                )
                camera = result.scalar_one_or_none()
                if camera:
                    await session.delete(camera)
                    await session.commit()
        except Exception as e:
            logger.exception("Failed to delete camera %s from database: %s", camera_uuid, e)

    def list_all(self) -> List[Dict[str, Any]]:
        """
        Read every stored camera as plain dicts, for startup restore.

        Returns detached data rather than ORM rows so the session can be closed
        here and the caller never touches a lazily-loaded attribute.
        """
        session = db_manager.get_session()
        try:
            rows = session.query(CameraConfig).all()
            return [
                {
                    "camera_uuid": row.camera_uuid,
                    "config_json": dict(row.config_json or {}),
                }
                for row in rows
            ]
        finally:
            session.close()

    def initialize_tables(self) -> bool:
        """Create/verify tables. Delegates to the shared DatabaseManager."""
        return db_manager.initialize_tables()
