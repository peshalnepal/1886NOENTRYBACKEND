# repositories/ - persistence layer for the Jetson edge service.
#
# Everything that touches the SQLite session lives here; PipelineRuntime and the
# route modules go through CameraRepository instead of building queries inline.

try:
    # Script mode (python main.py from Backend/tensort)
    from repositories.camera_repository import CameraRepository
    from repositories.discovery_repository import DiscoveryRepository
except Exception:
    # Package mode (python -m Backend.tensort.main)
    from .camera_repository import CameraRepository
    from .discovery_repository import DiscoveryRepository

__all__ = ["CameraRepository", "DiscoveryRepository"]
