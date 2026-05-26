"""Notification package.

Re-exports the public surface that used to be importable from the former
``application.services.notification`` module. Existing imports such as::

    from application.services.notification import NotificationService

continue to work unchanged.
"""

from application.services.notification.email_notifier import EmailNotifier
from application.services.notification.hub import WebNotificationHub
from application.services.notification.service import NotificationService
from application.services.notification.flusher import NotificationFlusher
from application.services.notification.clip_manager import ClipManager
from application.services.notification.deleter import NotificationDeleter
from application.services.notification.types import (
    BufferedNotification,
    CameraMode,
    EmailConfig,
    NotificationMessage,
    SitePrerecordPlan,
)

__all__ = [
    "BufferedNotification",
    "CameraMode",
    "EmailConfig",
    "EmailNotifier",
    "NotificationMessage",
    "NotificationService",
    "NotificationFlusher",
    "ClipManager",
    "NotificationDeleter",
    "SitePrerecordPlan",
    "WebNotificationHub",
]
