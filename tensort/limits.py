"""Capacity policy shared by camera admission and the inference pipeline."""

import os

DEFAULT_MAX_CAMERAS = 8


class CameraCapacityError(ValueError):
    """Adding an enabled camera would exceed this device's configured limit."""


def camera_limit():
    """Read the admission limit from configuration, without a hidden ceiling."""
    try:
        configured = int(os.getenv("MAX_CAMERAS", str(DEFAULT_MAX_CAMERAS)))
    except ValueError:
        configured = DEFAULT_MAX_CAMERAS
    return max(1, configured)
