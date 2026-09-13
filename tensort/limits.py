"""Capacity policy shared by camera admission and the inference pipeline."""

import os

SUPPORTED_MAX_CAMERAS = 8


class CameraCapacityError(ValueError):
    """Adding an enabled camera would exceed this device's configured limit."""


def camera_limit():
    """Allow a smaller site limit, with a hard ceiling of eight cameras."""
    try:
        configured = int(os.getenv("MAX_CAMERAS", str(SUPPORTED_MAX_CAMERAS)))
    except ValueError:
        configured = SUPPORTED_MAX_CAMERAS
    return max(1, min(configured, SUPPORTED_MAX_CAMERAS))
