"""Manager channel CRUD controllers.

Extracted from the former monolithic application/services/manager.py.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from application.dtos import CameraUpsertDTO
from application.channels.channel import VideoChannel
from domain.events import VideoChannelEvent
from application.services.pipeline import ModelPipeline

from application.services.manager.helpers import (
    _only_jetson_config,
    build_video_channel_config,
)
from application.services.manager.types import CameraOut
from application.services.manager.controllers._state import ManagerState
from application.services.manager.controllers.schedule import ScheduleResolver

logger = logging.getLogger(__name__)


class ChannelController:
    def __init__(self, state: ManagerState, schedule_resolver: ScheduleResolver):
        self._state = state
        self._schedule_resolver = schedule_resolver

    def _patch_to_dict(self, obj: Any) -> Dict[str, Any]:
        nullable_keep = {"roi"}
        if obj is None:
            out: Dict[str, Any] = {}
        elif hasattr(obj, "model_dump"):
            out = obj.model_dump(exclude_unset=True)
            fields_set = getattr(obj, "model_fields_set", None)
            if fields_set is None:
                fields_set = getattr(obj, "__fields_set__", None)
            if fields_set is not None and "roi" in fields_set and "roi" not in out:
                out["roi"] = None
            out = {k: v for k, v in out.items() if v is not None or k in nullable_keep}
        elif isinstance(obj, dict):
            out = {k: v for k, v in obj.items() if v is not None or k in nullable_keep}
        else:
            out = {}

        if "enabled" not in out and "is_enabled" in out:
            out["enabled"] = out["is_enabled"]
        if "detection_enabled" not in out and "is_detection_enabled" in out:
            out["detection_enabled"] = out["is_detection_enabled"]
        if "notification_enabled" not in out and "is_notification_enabled" in out:
            out["notification_enabled"] = out["is_notification_enabled"]

        return out

    def _as_uuid(self, v: Any, name: str) -> uuid.UUID:
        if isinstance(v, uuid.UUID):
            return v
        try:
            return uuid.UUID(str(v))
        except Exception as e:
            raise ValueError(f"Invalid {name}: {v}") from e

    async def _bg_edge_upsert(self, *, device_url: str, payload: dict) -> None:
        cam_uuid = payload.get("camera_uuid", "unknown")
        coro = self._state.edge.upsert_camera(device_url=device_url, payload=payload)
        await self._safe_bg_call(coro, "Background edge upsert succeeded camera_uuid=", "Background edge upsert failed camera_uuid=", cam_uuid)

    async def _bg_edge_delete(self, *, device_url: str, camera_uuid: str) -> None:
        coro = self._state.edge.delete_camera(device_url=device_url, camera_uuid=camera_uuid)
        await self._safe_bg_call(coro, "Background edge delete succeeded camera_uuid=", "Background edge delete failed camera_uuid=", camera_uuid)

    async def _bg_edge_patch(self, *, device_url: str, camera_uuid: str, patch: dict) -> None:
        coro = self._state.edge.patch_camera(device_url=device_url, camera_uuid=camera_uuid, patch=patch)
        await self._safe_bg_call(coro, "Background edge patch succeeded camera_uuid=", "Background edge patch failed camera_uuid=", camera_uuid)

    async def _bg_webrtc_update_stream(self, *, stream_key: str, source_url: str) -> None:
        coro = self._state.webrtc.update_stream(stream_key=stream_key, source_url=source_url)
        await self._safe_bg_call(coro, "Background WebRTC update succeeded stream_key=", "Background WebRTC update failed stream_key=", stream_key)

    async def _safe_bg_call(self, coro: asyncio.coroutine, success_msg: str, error_msg: str, identifier: str) -> None:
        try:
            await coro
            logger.debug(f"{success_msg} %s", identifier)
        except Exception:
            logger.warning(f"{error_msg} %s — run Sync to fix", identifier, exc_info=True)


    async def add_channel(
        self,
        db: AsyncSession,
        *,
        pid: uuid.UUID,
        ev: VideoChannelEvent,
        user_id: int,
        camera_code_prefix: str,
        model_pipeline: ModelPipeline,
    ) -> Tuple[List[CameraOut], List[Dict[str, Any]]]:

        patch = self._patch_to_dict(getattr(ev, "configs", None))

        source_url = patch.get("source_url")
        if not source_url:
            raise ValueError("Create_Channel requires source_url")

        site_uuid = patch.get("site_uuid")
        if not site_uuid:
            raise ValueError("Create_Channel requires site_uuid")
        site_uuid = self._as_uuid(site_uuid, "site_uuid")

        cam_uuid = patch.get("camera_uuid") or uuid.uuid4()
        cam_uuid = self._as_uuid(cam_uuid, "camera_uuid")
        patch["camera_uuid"] = cam_uuid
        patch["channel_id"] = cam_uuid
        camera_code = f"{camera_code_prefix}-{cam_uuid.hex[:8]}"
        
        # 1. Validate ownership and find device FIRST (before creating stream)
        site = await self._state.site_repo.get_site(
            db, site_uuid=site_uuid, user_id=int(user_id), raise_if_missing=False
        )
        if site is None:
            raise ValueError(f"Site not found for user: {site_uuid}")

        raw_device_uuid = patch.get("device_uuid")
        requested_device_uuid = (
            self._as_uuid(raw_device_uuid, "device_uuid")
            if raw_device_uuid is not None else None
        )
        
        site_devices = await self._state.device_repo.list_devices(db, site_uuid=site_uuid, user_id=user_id)

        if requested_device_uuid is not None:
            dev = await self._state.device_repo.get_device(db, device_uuid=requested_device_uuid, user_id=user_id)
            if not dev:
                raise ValueError(f"Device not found: {requested_device_uuid}")
            if site_devices and dev.device_uuid not in [d.device_uuid for d in site_devices]:
                raise ValueError(f"Device {requested_device_uuid} is not linked to site {site_uuid}.")
        else:
            if not site_devices:
                raise ValueError(f"Site {site_uuid} has no linked device. Link a device to the site first.")
            enabled_devices = [d for d in site_devices if bool(getattr(d, "is_enabled", True))]
            candidates = enabled_devices or site_devices
            if len(candidates) == 1:
                dev = candidates[0]
            else:
                raise ValueError(f"Site {site_uuid} has multiple linked devices. Specify device_uuid explicitly.")

        device_uuid = dev.device_uuid
        patch["device_uuid"] = device_uuid

        # 2. Create WebRTC stream NOW
        webrtc_url = await self._state.webrtc.ensure_stream(stream_key=str(camera_code), source_url=str(source_url))

        # 3. Upsert to DB, wrapped in try/except to rollback stream on failure
        try:
            cam, cfg_json, tz = await self._state.channel_repo.upsert_camera_from_channel_config(
                db,
                dto=CameraUpsertDTO(
                    pipeline_id=pid,
                    channel_config=patch,
                    user_id=user_id,
                    cam_uuid=cam_uuid,
                    camera_code=camera_code,
                    site_uuid=site_uuid,
                    webrtc_url=webrtc_url,
                    device_uuid=device_uuid,
                ),
            )
        except Exception:
            try:
                await self._state.webrtc.delete_stream(stream_key=str(camera_code))
            except Exception:
                logger.warning("Failed to rollback WebRTC stream after DB upsert failure", exc_info=True)
            raise
            
        det_enabled = bool(getattr(cam, "is_detection_enabled", True))
        schedule_state = await self._schedule_resolver.resolve_runtime_schedule(
            db,
            cam=cam,
            cfg_json=cfg_json,
            cfg_timezone=tz,
        )

        edge_payload = {
            **_only_jetson_config(patch),
            "enabled":bool(det_enabled),
            "detection_enabled": det_enabled,
            "notification_enabled": bool(getattr(cam, "is_notification_enabled", True)),
            "camera_uuid": str(cam.camera_uuid),
            "source_url": cam.source_url,
        }
        
        if det_enabled:
            self._state.spawn_bg(
                self._bg_edge_upsert(device_url=dev.device_url, payload=edge_payload),
                name=f"edge_upsert_{cam.camera_uuid}"
            )
        else:
            self._state.spawn_bg(
                self._bg_edge_delete(device_url=dev.device_url, camera_uuid=str(cam.camera_uuid)),
                name=f"edge_delete_{cam.camera_uuid}"
            )


        vcc = build_video_channel_config(
            cam,
            device_uuid=device_uuid,
            device_url=dev.device_url,
            schedule_state=schedule_state,
            cfg_json=cfg_json or {},
            default_request_timeout_s=self._state.default_request_timeout_s,
        )
        await model_pipeline.add_channel(VideoChannel(config=vcc))

        cameras_out = [
            CameraOut(
                camera_uuid=cam.camera_uuid,
                camera_code=getattr(cam, "camera_code", None),
                name=getattr(cam, "name", None),
                location=getattr(cam, "location", None),
                site_uuid=cam.site_uuid,
                source_url=cam.source_url,
                webrtc_url=cam.webrtc_url,
                enabled=bool(cam.is_enabled),
                detection_enabled=bool(cam.is_detection_enabled),
                notification_enabled=bool(cam.is_notification_enabled),
                device_uuid=device_uuid,
                device_url=dev.device_url,
                sample_fps=float(patch.get("sample_fps", 5.0)),
                decode_backend=str(patch.get("decode_backend", "gstreamer")),
                resize=patch.get("resize"),
                emit_format=str(patch.get("emit_format", "raw")),
                jpeg_quality=int(patch.get("jpeg_quality", 80)),
                roi=cam.roi,
                configuration=cfg_json or {},
                timezone=tz,
                notification_trigger_mode=str(getattr(cam, "notification_trigger_mode", "inherit") or "inherit"),
                camera_playback_enabled=str(getattr(cam, "camera_playback_enabled", "inherit") or "inherit"),
                use_site_schedule=schedule_state.get("use_site_schedule"),
                created_at=getattr(cam, "created_at", None),
                updated_at=getattr(cam, "updated_at", None),
            )
        ]

        events_out = [
            {
                "event_type": "Create_Channel",
                "camera_uuid": str(cam.camera_uuid),
                "site_uuid": str(cam.site_uuid),
                "device_uuid": str(device_uuid),
                "source_url": cam.source_url,
                "webrtc_url": cam.webrtc_url,
            }
        ]
        return cameras_out, events_out


    async def edit_channel(
        self,
        db: AsyncSession,
        *,
        pid: uuid.UUID,
        ev: VideoChannelEvent,
        user_id: int,
        model_pipeline: ModelPipeline,
    ) -> Tuple[List[CameraOut], List[Dict[str, Any]]]:

        cam_uuid = getattr(ev, "channel_id", None) or getattr(ev, "camera_uuid", None)
        if cam_uuid is None:
            raise ValueError("Edit_Channel missing channel_id (camera_uuid).")
        cam_uuid = self._as_uuid(cam_uuid, "camera_uuid")

        full = await self._state.channel_repo.get_camera_full(db, camera_uuid=cam_uuid)
        if not full:
            raise ValueError(f"Camera not found: {cam_uuid}")

        cam_db, chan_cfg_db, existing_pid = full
        if int(getattr(cam_db, "user_id", -1)) != int(user_id):
            raise ValueError(f"Camera does not belong to user: {cam_uuid}")
        if existing_pid is not None and existing_pid != pid:
            raise ValueError("Camera does not belong to provided pipeline_id")

        old_rtsp = cam_db.source_url
        old_webrtc = cam_db.webrtc_url
        patch = self._patch_to_dict(getattr(ev, "configs", None))
        patch.pop("webrtc_url", None)

        old_devices = await self._state.device_repo.list_devices(db, camera_uuid=cam_uuid)
        if len(old_devices) > 1:
            logger.warning(
                "Camera %s has %s linked devices during edit; using most recent device %s",
                cam_uuid,
                len(old_devices),
                getattr(old_devices[0], "device_uuid", None),
            )
        old_dev = old_devices[0] if old_devices else None
        if old_dev is not None and not getattr(old_dev, "device_url", None):
            raise ValueError(f"Assigned device has no device_url for camera {cam_uuid}")

        # Determine new device: requested > preserve old > auto-pick
        requested_device_uuid = patch.get("device_uuid")
        if requested_device_uuid is not None:
            requested_device_uuid = self._as_uuid(requested_device_uuid, "device_uuid")
            site_devices = await self._state.device_repo.list_devices(db, site_uuid=cam_db.site_uuid, user_id=user_id)
            new_dev = await self._state.device_repo.get_device(db, device_uuid=requested_device_uuid, user_id=user_id)
            if not new_dev:
                raise ValueError(f"Device not found: {requested_device_uuid}")
            if site_devices and new_dev.device_uuid not in [d.device_uuid for d in site_devices]:
                raise ValueError(f"Device {requested_device_uuid} is not linked to site {cam_db.site_uuid}.")
        elif old_dev is not None:
            # PRESERVE: Keep existing device if not changing
            new_dev = old_dev
        else:
            # AUTO-PICK: Try if exactly 1 device in site
            site_devices = await self._state.device_repo.list_devices(db, site_uuid=cam_db.site_uuid, user_id=user_id)
            if len(site_devices) == 1:
                new_dev = site_devices[0]
                logger.info("Auto-linked camera %s to device %s", cam_uuid, new_dev.device_uuid)
            else:
                raise ValueError("Camera has no device assigned. Link a Device to this site or assign explicitly.")
                
        new_device_uuid = new_dev.device_uuid
        patch["device_uuid"] = new_device_uuid
        # merge config json
        merged_cfg: Dict[str, Any] = {}
        if chan_cfg_db and getattr(chan_cfg_db, "configuration", None):
            merged_cfg.update(chan_cfg_db.configuration or {})
        merged_cfg.update(patch)

        cam2, cfg_json, tz = await self._state.channel_repo.upsert_camera_from_channel_config(
            db,
            dto=CameraUpsertDTO(
                pipeline_id=pid,
                channel_config={
                    **merged_cfg,
                    "camera_uuid": cam_uuid,
                    "webrtc_url": old_webrtc,
                    "source_url": merged_cfg.get("source_url", old_rtsp),
                    "enabled": merged_cfg.get("enabled", cam_db.is_enabled),
                    "detection_enabled": merged_cfg.get("detection_enabled", cam_db.is_detection_enabled),
                    "notification_enabled": merged_cfg.get("notification_enabled", cam_db.is_notification_enabled),
                },
                site_uuid=cam_db.site_uuid,
                webrtc_url=old_webrtc,
                device_uuid=new_device_uuid,
            ),
        )

        stale_old_devices = [
            dev for dev in old_devices
            if getattr(dev, "device_uuid", None) != new_device_uuid and getattr(dev, "device_url", None)]
        for dev in stale_old_devices:
            self._state.spawn_bg(
                self._bg_edge_delete(device_url=dev.device_url, camera_uuid=str(cam_uuid)),
                name=f"edge_delete_stale_{cam_uuid}_{dev.device_uuid}"
            )

        device_changed = old_dev is None or old_dev.device_uuid != new_device_uuid
        if len(old_devices) != 1 or device_changed:
            await self._state.channel_repo.set_camera_device(db, camera_uuid=cam_uuid, device_uuid=new_device_uuid)
        if cam2.source_url != old_rtsp and cam2.camera_code:
            self._state.spawn_bg(
                self._bg_webrtc_update_stream(stream_key=str(cam2.camera_code), source_url=cam2.source_url),
                name=f"webrtc_update_{cam_uuid}"
            )

        det_enabled = bool(cam2.is_detection_enabled)
        schedule_state = await self._schedule_resolver.resolve_runtime_schedule(
            db,
            cam=cam2,
            cfg_json=cfg_json,
            cfg_timezone=tz,
        )
        
        if det_enabled:
            if device_changed:
                edge_payload = {
                    **_only_jetson_config(merged_cfg),
                    "enabled": bool(det_enabled),
                    "detection_enabled": det_enabled,
                    "notification_enabled": bool(cam2.is_notification_enabled),
                    "camera_uuid": str(cam_uuid),
                    "source_url": cam2.source_url,
                }
                self._state.spawn_bg(
                    self._bg_edge_upsert(device_url=new_dev.device_url, payload=edge_payload),
                    name=f"edge_upsert_{cam_uuid}"
                )
            else:
                edge_patch = _only_jetson_config(patch)
                edge_patch.setdefault("source_url", cam2.source_url)
                edge_patch["enabled"] =bool(det_enabled)
                edge_patch["detection_enabled"] = det_enabled
                edge_patch["notification_enabled"] = bool(cam2.is_notification_enabled)
                self._state.spawn_bg(
                    self._bg_edge_patch(device_url=new_dev.device_url, camera_uuid=str(cam_uuid), patch=edge_patch),
                    name=f"edge_patch_{cam_uuid}"
                )
        else:
            self._state.spawn_bg(
                self._bg_edge_delete(device_url=new_dev.device_url, camera_uuid=str(cam_uuid)),
                name=f"edge_delete_{cam_uuid}"
            )

        vcc = build_video_channel_config(
            cam2,
            device_uuid=new_device_uuid,
            device_url=new_dev.device_url,
            schedule_state=schedule_state,
            cfg_json=merged_cfg,
            include_capture_fields=False,
        )
        await model_pipeline.edit_channel(VideoChannel(config=vcc))

        cameras_out = [
            CameraOut(
                camera_uuid=cam2.camera_uuid,
                camera_code=getattr(cam2, "camera_code", None),
                name=getattr(cam2, "name", None),
                location=getattr(cam2, "location", None),
                site_uuid=cam2.site_uuid,
                source_url=cam2.source_url,
                webrtc_url=cam2.webrtc_url,
                enabled=bool(cam2.is_enabled),
                detection_enabled=bool(cam2.is_detection_enabled),
                notification_enabled=bool(cam2.is_notification_enabled),
                device_uuid=new_device_uuid,
                device_url=new_dev.device_url,
                sample_fps=float(merged_cfg.get("sample_fps", 5.0)),
                decode_backend=str(merged_cfg.get("decode_backend", "gstreamer")),
                resize=merged_cfg.get("resize"),
                emit_format=str(merged_cfg.get("emit_format", "raw")),
                jpeg_quality=int(merged_cfg.get("jpeg_quality", 80)),
                roi=cam2.roi,
                configuration=cfg_json or {},
                timezone=tz,
                notification_trigger_mode=str(getattr(cam2, "notification_trigger_mode", "inherit") or "inherit"),
                camera_playback_enabled=str(getattr(cam2, "camera_playback_enabled", "inherit") or "inherit"),
                use_site_schedule=schedule_state.get("use_site_schedule"),
                created_at=getattr(cam2, "created_at", None),
                updated_at=getattr(cam2, "updated_at", None),
            )
        ]

        events_out = [{
            "event_type": "Edit_Channel",
            "camera_uuid": str(cam2.camera_uuid),
            "source_url": cam2.source_url,
            "webrtc_url": cam2.webrtc_url,
            "device_uuid": str(new_device_uuid),
            "patch": patch,
        }]

        return cameras_out, events_out

    async def remove_channel(
        self,
        db: AsyncSession,
        *,
        pid: uuid.UUID,
        ev: VideoChannelEvent,
        user_id: int,
        model_pipeline: ModelPipeline,
    ) -> Tuple[List[CameraOut], List[Dict[str, Any]]]:

        cam_uuid = getattr(ev, "channel_id", None) or getattr(ev, "camera_uuid", None)
        if cam_uuid is None:
            raise ValueError("Remove_Channel missing channel_id (camera_uuid).")
        cam_uuid = self._as_uuid(cam_uuid, "camera_uuid")

        full = await self._state.channel_repo.get_camera_full(db, camera_uuid=cam_uuid)
        if full:
            cam_db, _cfg, existing_pid = full
            if int(getattr(cam_db, "user_id", -1)) != int(user_id):
                raise ValueError(f"Camera does not belong to user: {cam_uuid}")
            if existing_pid is not None and existing_pid != pid:
                raise ValueError("Camera does not belong to provided pipeline_id")

            try:
                devices = await self._state.device_repo.list_devices(db, camera_uuid=cam_uuid)
                seen_urls: Set[str] = set()
                for dev in devices:
                    dev_url = str(getattr(dev, "device_url", "") or "").strip()
                    if not dev_url or dev_url in seen_urls:
                        continue
                    seen_urls.add(dev_url)
                    await self._state.edge.delete_camera(device_url=dev_url, camera_uuid=str(cam_uuid))
            except Exception:
                logger.warning("Edge delete failed during camera removal", exc_info=True)
            try:
                if cam_db.camera_code:
                    await self._state.webrtc.delete_stream(stream_key=str(cam_db.camera_code))
            except Exception:
                logger.warning("WebRTC delete failed during camera removal", exc_info=True)

            await self._state.channel_repo.delete_camera(db, camera_uuid=cam_db.camera_uuid)

        if model_pipeline:
            try:
                await model_pipeline.remove_channel(cam_uuid)
            except Exception:
                logger.warning("Failed removing channel from cached ModelPipeline", exc_info=True)

        events_out = [{"event_type": "Remove_Channel", "camera_uuid": str(cam_uuid)}]
        return [], events_out
