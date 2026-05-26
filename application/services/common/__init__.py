"""Shared building blocks used by more than one application service.

These were previously copy-pasted into both ``ModelPipeline`` and
``NotificationService``. They are kept here so there is exactly one
implementation of each.
"""

from application.services.common.background import BackgroundTasks, fire_and_forget
from application.services.common.camera_context import CameraContextResolver
from application.services.common.site_settings_cache import SiteTriggerModeResolver

__all__ = [
    "BackgroundTasks",
    "CameraContextResolver",
    "SiteTriggerModeResolver",
    "fire_and_forget",
]
