"""Manager cleanup paths (user, device, site).

Extracted from the former monolithic application/services/manager.py.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Set

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.database_orm import Site
from application.services.pipeline import ModelPipeline
from application.services.manager.controllers._state import ManagerState

logger = logging.getLogger(__name__)


class CleanupController:
    def __init__(self, state: ManagerState):
        self._state = state

    async def cleanup_user_resources(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        camera_uuids: Optional[List[uuid.UUID]] = None,
    ) -> Dict[str, Any]:
        """
        Clean up external/runtime resources for a user before deleting the DB user row.
        This does not delete DB rows directly; caller should delete the User after this succeeds.

        `camera_uuids` restricts the edge/WebRTC teardown to a specific set —
        the cameras of organizations actually being dissolved. Cameras are
        org-owned, so tearing down by `user_id` would black out live feeds that
        the departing member merely created and the rest of the org still
        depends on. Pass an empty list to skip camera teardown entirely; omit
        the argument for the legacy user-scoped behaviour.
        """
        uid = int(user_id)
        errors: List[str] = []
        edge_deleted: List[str] = []
        streams_deleted: List[str] = []

        if camera_uuids is None:
            user_cameras = await self._state.channel_repo.list_cameras(
                db, user_id=uid, include_device=True
            )
        elif camera_uuids:
            user_cameras = await self._state.channel_repo.list_cameras(
                db, camera_uuids=list(camera_uuids), include_device=True
            )
        else:
            user_cameras = []

        stream_seen: Set[str] = set()

        # Best-effort external cleanup while camera/device metadata still exists.
        for cam in user_cameras:
            cam_uuid_str = str(cam.camera_uuid)
            for dev in ([cam.device] if getattr(cam, "device", None) else []):
                dev_url = str(getattr(dev, "device_url", "") or "").strip()
                if not dev_url:
                    continue
                try:
                    await self._state.edge.delete_camera(device_url=dev_url, camera_uuid=cam_uuid_str)
                    edge_deleted.append(cam_uuid_str)
                except Exception as e:
                    logger.warning(
                        "Failed deleting edge camera during user cleanup user=%s camera=%s device=%s",
                        uid,
                        cam_uuid_str,
                        getattr(dev, "device_uuid", None),
                        exc_info=True,
                    )
                    errors.append(f"edge:{cam_uuid_str}:{e}")

            stream_key = str(getattr(cam, "camera_code", "") or "").strip()
            if stream_key and stream_key not in stream_seen:
                stream_seen.add(stream_key)
                try:
                    await self._state.webrtc.delete_stream(stream_key=stream_key)
                    streams_deleted.append(stream_key)
                except Exception as e:
                    logger.warning(
                        "Failed deleting WebRTC stream during user cleanup user=%s stream=%s",
                        uid,
                        stream_key,
                        exc_info=True,
                    )
                    errors.append(f"webrtc:{stream_key}:{e}")

        pipeline_to_shutdown: Optional[ModelPipeline] = None
        user_lock = self._state.get_user_lock(uid)
        async with user_lock:
            pipeline_to_shutdown = self._state.pipelines_by_user.pop(uid, None)
            self._state.pipeline_id_by_user.pop(uid, None)

        if pipeline_to_shutdown is not None:
            try:
                await pipeline_to_shutdown.shutdown()
            except Exception as e:
                logger.warning("Failed shutting down in-memory pipeline for deleted user=%s", uid, exc_info=True)
                errors.append(f"pipeline_shutdown:{e}")

        return {
            "user_id": uid,
            "camera_count": len(user_cameras),
            "edge_deleted_count": len(edge_deleted),
            "stream_deleted_count": len(streams_deleted),
            "errors": errors,
        }

    async def cleanup_device_resources(self, db: AsyncSession, *, device_uuid: uuid.UUID,active: Optional[ModelPipeline]) -> None:
        """
        Called when a Device is about to be deleted.
        Finds all cameras on this device and sends DELETE to the edge service.
        """
        try:
            dev = await self._state.device_repo.get_device(db, device_uuid=device_uuid)
            if not dev or not getattr(dev, "device_url", None):
                return

            cameras_on_device = await self._state.channel_repo.list_cameras(
                db, device_uuid=device_uuid
            )

            for cam in cameras_on_device:
                try:
                    await self._state.edge.delete_camera(device_url=dev.device_url, camera_uuid=str(cam.camera_uuid))
                    if cam.camera_code:
                        await self._state.webrtc.delete_stream(stream_key=str(cam.camera_code))
                    await self._state.channel_repo.delete_camera(db, camera_uuid=cam.camera_uuid)
                    if active:
                        try:
                            await active.remove_channel(cam.camera_uuid)
                        except Exception:
                            logger.warning("Failed removing channel from cached ModelPipeline", exc_info=True)

                except Exception:
                    logger.warning("Failed cleanup camera %s on device %s", cam.camera_uuid, dev.device_uuid, exc_info=True)
        except Exception:
            logger.exception("Error during device cleanup for %s", device_uuid)
            
    async def _get_site(self, db: AsyncSession, user_id: int, site_uuid: uuid.UUID) -> Site:
        site = await self._state.site_repo.get_site(
            db, site_uuid=site_uuid, user_id=user_id, raise_if_missing=False
        )
        if not site:
            raise HTTPException(status_code=404, detail="Site not found")
        return site
    
    async def cleanup_site_resources(self, db: AsyncSession, *,user_id: int, site_uuid: uuid.UUID,active: Optional[ModelPipeline]) -> None:
        """
        Called when a Site is about to be deleted.
        Finds all cameras on this site and sends DELETE to the edge service.
        """
        try:
            site = await self._get_site(db,user_id=user_id, site_uuid=site_uuid)
            if not site.site_uuid:
                return
            cameras_on_site = await self._state.channel_repo.list_cameras(
                db, site_uuid=site_uuid, include_device=True
            )

            for cam in cameras_on_site:
                # Isolated per camera, like cleanup_device_resources: one
                # unreachable edge device must not abort the whole site
                # teardown and strand the remaining cameras.
                try:
                    for dev in ([cam.device] if getattr(cam, "device", None) else []):
                        dev_url = str(getattr(dev, "device_url", "") or "").strip()
                        if not dev_url:
                            continue
                        await self._state.edge.delete_camera(device_url=dev_url, camera_uuid=str(cam.camera_uuid))
                    if cam.camera_code:
                        await self._state.webrtc.delete_stream(stream_key=str(cam.camera_code))
                    await self._state.channel_repo.delete_camera(db, camera_uuid=cam.camera_uuid)
                    if active:
                        try:
                            await active.remove_channel(cam.camera_uuid)
                        except Exception:
                            logger.warning("Failed removing channel from cached ModelPipeline", exc_info=True)

                except Exception:
                    logger.warning(
                        "Failed cleanup camera %s on site %s",
                        cam.camera_uuid,
                        site_uuid,
                        exc_info=True,
                    )

        except Exception:
            logger.exception("Error during site cleanup for %s", site_uuid)
