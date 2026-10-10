"""MiniPC discovery, relay and Jetson API gateway.

Run from this folder: python main.py   (or from its parent: python -m NVR.main)
"""

if not __package__:
    import importlib.util
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location(
        "NVR", _root / "__init__.py", submodule_search_locations=[str(_root)])
    sys.modules["NVR"] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules["NVR"])
    __package__ = "NVR"

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from .application.services.adoption_service import AdoptionService
from .application.services.discovery_adapter import DiscoveryAdapter
from .application.services.frame_verifier import FrameVerifier
from .application.services.sweep_service import SweepService
from .config import Settings
from .core.database import Database
from .jetson import JetsonClient
from .relay import MediaRelay
from .routes import router
from .recordings import RecordingService, router as recordings_router

logger = logging.getLogger("nvr")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        database = Database(settings.minipc_db_path)
        await database.init()

        adapter = DiscoveryAdapter(settings)
        verifier = FrameVerifier(settings.ffmpeg_bin, timeout_s=settings.verify_timeout_s,
                                 max_concurrency=settings.verify_concurrency,
                                 per_host=settings.verify_per_host)
        if not verifier.available():
            logger.error("%s not found: no camera can be verified or adopted "
                         "(install ffmpeg or set FFMPEG_BIN)", settings.ffmpeg_bin)
        jetson = JetsonClient(settings)
        relay = MediaRelay(settings, database.session_factory, adapter)
        recordings = RecordingService(settings, database.session_factory, relay)
        await recordings.start()
        app.state.recordings = recordings
        adoption = AdoptionService(database.session_factory, jetson, relay)
        sweep = SweepService(settings, database.session_factory, adapter, verifier, relay, adoption)
        app.state.jetson, app.state.relay, app.state.sweep = jetson, relay, sweep
        sweep.start()
        try:
            yield
        finally:
            await sweep.stop()
            await recordings.close()
            await jetson.close()
            await relay.close()
            adapter.close()
            await database.dispose()

    app = FastAPI(title="NOENTRY NVR gateway", lifespan=lifespan)
    # Register local endpoints before the proxy catch-all in both namespaces.
    app.include_router(recordings_router, prefix="/api")
    app.include_router(recordings_router)
    app.include_router(router, prefix="/api")
    app.include_router(router)
    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    uvicorn.run(create_app(settings), host=settings.minipc_bind_host, port=settings.minipc_port)


if __name__ == "__main__":
    main()
