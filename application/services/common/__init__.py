"""Shared building blocks used by more than one application service.

``ModelPipeline`` and ``NotificationService`` both need these, so there is
exactly one implementation of each here.
"""

from application.services.common.background import BackgroundTasks, fire_and_forget
from application.services.common.camera_context import CameraContextResolver
from application.services.common.site_settings_cache import (
    SiteArmStateResolver,
    SiteTriggerModeResolver,
)

__all__ = [
    "BackgroundTasks",
    "CameraContextResolver",
    "SiteArmStateResolver",
    "SiteTriggerModeResolver",
    "fire_and_forget",
]
