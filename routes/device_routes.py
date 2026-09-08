"""Device (Jetson edge box) CRUD and edge reconcile."""

import asyncio
import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from application.dtos import DeviceCreateDTO, DeviceUpdateDTO
from application.repositories.device_repository import DeviceRepository
from application.services.manager import EdgeDeviceUnavailableError, Manager
from core.coercions import gen_code, is_blank
from core.database_orm import Device
from core.schemas import DeviceCreate, DeviceOut, DeviceUpdate, EdgeReconcileOut
from core.security.roles import Permission
from dependencies import (
    get_async_db,
    get_manager,
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


async def _cleanup_device_runtime(
    manager: Manager,
    db: AsyncSession,
    *,
    device_uuid: uuid.UUID,
    owner_id: int,
) -> None:
    """Best-effort runtime teardown with bounded manager calls."""
    try:
        active_pipeline = await asyncio.wait_for(
            manager.get_activepipeline(user_id=owner_id),
            timeout=5.0,
        )
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
    db: AsyncSession = Depends(get_async_db),
    ctx: OrgContext = Depends(RequirePermission(Permission.ORG_MANAGE_DEVICES)),
    manager: Optional[Manager] = Depends(get_manager),
):
    device = await _get_device_or_404(db, ctx.org_id, device_uuid)
    owner_id = _device_owner_id(device, ctx)

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
