"""Device (Jetson edge box) CRUD and edge reconcile."""

import asyncio
import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from application.dtos import DeviceCreateDTO, DeviceUpdateDTO
from application.repositories.channel_repository import ChannelRepository
from application.repositories.device_repository import DeviceRepository
from application.services.manager import EdgeDeviceUnavailableError, Manager
from core.coercions import gen_code, is_blank
from core.database_orm import Device
from application.services.inventory_service import InventoryService
from core.schemas import (
    DeviceCameraOut,
    DeviceCamerasOut,
    DeviceCreate,
    DeviceOut,
    DeviceUpdate,
    EdgeReconcileOut,
    InventoryCameraOut,
    InventoryRefreshOut,
)
from core.security.roles import Permission
from dependencies import (
    get_async_db,
    get_manager,
    get_manager_optional,
    OrgContext,
    RequirePermission,
)
from routes._errors import DEVICE_NOT_FOUND

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/devices", tags=["devices"])

device_repo = DeviceRepository()


async def _get_device_or_404(db: AsyncSession, org_id: int, device_uuid: uuid.UUID) -> Device:
    device = await device_repo.get_device(db, device_uuid=device_uuid, org_id=org_id)
    if not device:
        raise HTTPException(status_code=404, detail=DEVICE_NOT_FOUND)
    return device


def _device_owner_id(device: Device, ctx: OrgContext) -> int:
    """Runtime pipelines key off a user id; prefer the device creator,
    falling back to the acting admin."""
    return int(device.user_id) if device.user_id is not None else int(ctx.user.id)


def _camera_owner_id_for_device(camera, device_owner_id: int) -> int:
    """Whose runtime pipeline holds this camera.

    A camera on a shared device may have been created by a different member, so
    its own `user_id` is the pipeline key when it still has one; the device
    owner is the fallback after a member deletion SET NULL it.
    """
    return (
        int(camera.user_id)
        if getattr(camera, "user_id", None) is not None
        else int(device_owner_id)
    )


async def _cleanup_device_runtime(
    manager: Manager,
    db: AsyncSession,
    *,
    device_uuid: uuid.UUID,
    owner_id: int,
) -> None:
    """Best-effort runtime teardown with bounded manager calls."""
    try:
        active_pipeline = manager.get_loaded_pipeline(user_id=owner_id)
    except Exception:
        logger.warning(
            "Could not inspect loaded pipeline for device=%s; "
            "channel eviction limited to the DB snapshot",
            device_uuid,
            exc_info=True,
        )
        active_pipeline = None

    try:
        await asyncio.wait_for(
            manager.cleanup_device_resources(
                db,
                device_uuid=device_uuid,
                active=active_pipeline,
            ),
            timeout=10.0,
        )
    except Exception:
        logger.warning(
            "Best-effort device cleanup failed for %s; proceeding with DB delete.",
            device_uuid,
            exc_info=True,
        )


@router.get("", response_model=List[DeviceOut])
async def list_devices(
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
):
    return await device_repo.list_devices(db, org_id=ctx.org_id, order_by_recent=True)


@router.post("", response_model=DeviceOut, status_code=status.HTTP_201_CREATED)
async def create_device(
    payload: DeviceCreate,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_DEVICES)),
):
    device_code = payload.device_code
    if is_blank(device_code):
        device_code = gen_code("dev")

    target_org_id = ctx.org_id if ctx.org_id is not None else payload.org_id
    if target_org_id is None:
        raise HTTPException(
            status_code=422,
            detail="org_id is required when creating a device as a platform admin.",
        )

    device = await device_repo.create_device(
        db,
        dto=DeviceCreateDTO(
            org_id=int(target_org_id),
            user_id=ctx.user.id,
            created_by=ctx.user.id,
            device_url=payload.device_url,
            name=payload.name,
            device_code=device_code,
            is_enabled=payload.is_enabled,
        ),
    )
    await db.commit()
    await db.refresh(device)
    return device


@router.get("/{device_uuid}", response_model=DeviceOut)
async def get_device(
    device_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
):
    return await _get_device_or_404(db, ctx.org_id, device_uuid)


@router.patch("/{device_uuid}", response_model=DeviceOut)
async def update_device(
    device_uuid: uuid.UUID,
    payload: DeviceUpdate,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_DEVICES)),
):
    device = await _get_device_or_404(db, ctx.org_id, device_uuid)

    data = payload.model_dump(exclude_unset=True)
    # An explicitly blanked device_code means "regenerate it".
    if "device_code" in data and is_blank(data.get("device_code")):
        data["device_code"] = gen_code("dev")

    update_fields = {k: v for k, v in data.items() if v is not None}
    if update_fields:
        await device_repo.update_device(
            db,
            device_uuid=device.device_uuid,
            dto=DeviceUpdateDTO(**update_fields),
        )
    await db.commit()
    await db.refresh(device)
    return device


@router.delete("/{device_uuid}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_device(
    device_uuid: uuid.UUID,
    cascade: bool = False,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_DEVICES)),
    manager: Optional[Manager] = Depends(get_manager_optional),
):
    """Delete a device, and optionally the cameras that stream through it.

    `camera.device_uuid` is ON DELETE SET NULL, so dropping a device that still
    has cameras would leave them behind pointing at nothing: still listed, still
    flagged enabled, but with no edge box left to stream them. That is never
    what the caller wants, so by default this refuses — mirroring the same guard
    on unlinking a device from a site.

    Pass `cascade=true` to delete those cameras properly first. Each one goes
    through the full camera teardown (edge, WebRTC, pipeline, notifications,
    clips and blobs) rather than being dropped by a bare SQL delete, so nothing
    is stranded on the edge or in storage.
    """
    from routes.camera_routes import perform_camera_deletion

    device = await _get_device_or_404(db, ctx.org_id, device_uuid)
    owner_id = _device_owner_id(device, ctx)

    channel_repo = ChannelRepository()
    attached = await channel_repo.list_cameras(
        db, device_uuid=device.device_uuid, include_device=True
    )

    if attached and not cascade:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot delete this device while {len(attached)} camera(s) are "
                f"assigned to it. Move or delete those cameras first, or retry "
                f"with cascade=true to delete them along with the device."
            ),
        )

    for cam in attached:
        cam_uuid = cam.camera_uuid
        try:
            await perform_camera_deletion(
                camera_uuid=cam_uuid,
                cam=cam,
                owner_id=_camera_owner_id_for_device(cam, owner_id),
                db=db,
                manager=manager,
            )
        except Exception:
            logger.exception(
                "[Device Delete] Camera teardown failed cam=%s device=%s; continuing",
                cam_uuid,
                device_uuid,
            )

    # Best-effort cleanup — DB delete must succeed even if manager/edge is down.
    if manager is not None:
        await _cleanup_device_runtime(
            manager,
            db,
            device_uuid=device_uuid,
            owner_id=owner_id,
        )

    await device_repo.delete_device(db, device_uuid=device.device_uuid)
    await db.commit()
    return None

@router.get("/{device_uuid}/cameras", response_model=DeviceCamerasOut)
async def list_device_cameras(
    device_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
    manager: Optional[Manager] = Depends(get_manager_optional),
):
    """The cameras the device currently has loaded, read from the device itself.

    A live read with no cloud copy: an unreachable device reports `reachable`
    false rather than an error, so the caller can tell "no cameras" apart from
    "could not ask".
    """
    device = await _get_device_or_404(db, ctx.org_id, device_uuid)
    if manager is None:
        return DeviceCamerasOut(device_uuid=device.device_uuid, reachable=False)

    try:
        loaded = await manager.edge.list_cameras(device_url=device.device_url)
    except Exception:
        logger.warning(
            "Could not list cameras on device=%s", device.device_uuid, exc_info=True
        )
        return DeviceCamerasOut(device_uuid=device.device_uuid, reachable=False)

    known = {
        str(cam.camera_uuid): cam
        for cam in await ChannelRepository().list_cameras(
            db, device_uuid=device.device_uuid
        )
    }
    cameras = [
        DeviceCameraOut(
            camera_uuid=uuid.UUID(camera_uuid),
            name=getattr(known.get(camera_uuid), "name", None),
            site_uuid=getattr(known.get(camera_uuid), "site_uuid", None),
            known_to_cloud=camera_uuid in known,
        )
        for camera_uuid in sorted(loaded)
    ]
    return DeviceCamerasOut(
        device_uuid=device.device_uuid, reachable=True, cameras=cameras
    )


@router.post("/{device_uuid}/inventory/refresh", response_model=InventoryRefreshOut)
async def refresh_device_inventory(
    device_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_DEVICES)),
    manager: Optional[Manager] = Depends(get_manager_optional),
):
    """Ask the device what cameras it can see, and store the answer."""
    device = await _get_device_or_404(db, ctx.org_id, device_uuid)

    summary = await InventoryService(manager=manager).refresh_from_device(
        db, device_uuid=device.device_uuid, device_url=device.device_url
    )
    if not summary["fetched"]:
        await db.rollback()
        return InventoryRefreshOut(
            fetched=False,
            detail="Device did not answer; stored inventory is unchanged.",
        )

    await db.commit()
    return InventoryRefreshOut(**summary)


@router.get("/{device_uuid}/inventory", response_model=List[InventoryCameraOut])
async def list_device_inventory(
    device_uuid: uuid.UUID,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_READ)),
):
    """Every camera this device has reported, added or not."""
    device = await _get_device_or_404(db, ctx.org_id, device_uuid)

    rows = await InventoryService().describe_for_device(
        db, device_uuid=device.device_uuid
    )
    return [InventoryCameraOut(**row) for row in rows]


@router.post("/{device_uuid}/edge/reconcile", response_model=EdgeReconcileOut)
async def reconcile_edge_cameras(
    device_uuid: uuid.UUID,
    dry_run: bool = False,
    delete_unknown: bool = True,
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_DEVICES)),
    manager: Manager = Depends(get_manager),
):
    device = await _get_device_or_404(db, ctx.org_id, device_uuid)
    owner_id = _device_owner_id(device, ctx)

    try:
        result = await manager.reconcile_device_edge_simple(
            device_uuid=device_uuid,
            user_id=owner_id,
            dry_run=dry_run,
            delete_unknown=delete_unknown,
        )
    except EdgeDeviceUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return EdgeReconcileOut(
        device_uuid=device.device_uuid,
        device_url=device.device_url,
        **result,
    )
