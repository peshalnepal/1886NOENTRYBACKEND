"""Canonical HTTP error responses shared across routers.

Keeps the wire-visible `detail` strings in one place so they stay consistent.
"""

from __future__ import annotations

from fastapi import HTTPException

CAMERA_NOT_FOUND = "Camera not found"
DEVICE_NOT_FOUND = "Device not found"
ORGANIZATION_NOT_FOUND = "Organization not found"
SITE_NOT_FOUND = "Site not found"
USER_NOT_FOUND = "User not found"


def not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=404, detail=detail)


def camera_not_found() -> HTTPException:
    return not_found(CAMERA_NOT_FOUND)


def site_not_found() -> HTTPException:
    return not_found(SITE_NOT_FOUND)


def device_not_found() -> HTTPException:
    return not_found(DEVICE_NOT_FOUND)
