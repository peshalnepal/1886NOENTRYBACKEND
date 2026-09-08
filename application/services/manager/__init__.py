"""Manager package: pipeline + edge-device orchestration.

Re-exports the public surface (`Manager` and its DTOs/exceptions) so callers
import from the package rather than from individual controller modules.
"""

from application.services.manager.service import Manager
from application.services.manager.types import (
    CameraOut,
    EdgeDeviceUnavailableError,
    PipelineUpdateResult,
)

__all__ = [
    "CameraOut",
    "EdgeDeviceUnavailableError",
    "Manager",
    "PipelineUpdateResult",
]
