"""Manager package.

Re-exports the public surface that used to be importable from the former
``application.services.manager`` module. Existing imports such as::

    from application.services.manager import Manager, EdgeDeviceUnavailableError

continue to work unchanged.
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
