"""Register or repoint verified cameras on Jetson, pointing it at the relay path."""

from uuid import uuid4

from sqlalchemy import select

from ...core.database_orm import Camera
from ...jetson import JetsonError
from .discovery_adapter import redact


class AdoptionService:
    def __init__(self, session_factory, jetson, relay):
        self._sessions = session_factory
        self._jetson = jetson
        self._relay = relay

    async def reconcile(self, identities):
        """POST missing channels, PATCH moved ones. Returns safe error strings.

        HTTP 409 becomes capacity_pending and retries next sweep. Existing channel
        settings on Jetson are never touched.
        """
        # Snapshot once; never keep a SQLite transaction open across HTTP calls.
        async with self._sessions() as db:
            cameras = list((await db.execute(select(Camera).where(Camera.identity.in_(identities)))).scalars())
        if not cameras:
            return []
        try:
            remote = await self._jetson.cameras()
        except JetsonError as exc:
            for camera in cameras:
                await self._save(camera.id, camera.camera_uuid, "error", str(exc))
            return [str(exc)]

        errors = []
        for camera in cameras:
            if self._relay.states.get(camera.identity) != "configured":
                await self._save(camera.id, camera.camera_uuid, "error", "MediaMTX relay is not configured")
                continue
            source = self._relay.jetson_url(camera.identity)
            # Jetson reads the relay URL. Keep raw URL matching for pre-relay cameras.
            matches = [r for r in remote if (
                r.get("camera_uuid") == camera.camera_uuid and camera.camera_uuid
            ) or (r.get("config") or {}).get("discovery_identity") == camera.identity
                or redact(r.get("source_url") or "") == redact(source)
                or redact(r.get("source_url") or "") == camera.source_url]
            if len(matches) > 1:
                error = "Ambiguous Jetson camera match for {}".format(camera.identity)
                await self._save(camera.id, camera.camera_uuid, "error", error)
                errors.append(error)
                continue
            existing = matches[0] if matches else None
            camera_uuid = (existing or {}).get("camera_uuid") or camera.camera_uuid or str(uuid4())
            # Persist the UUID BEFORE sending: retries after a lost response use it again.
            await self._save(camera.id, camera_uuid, "pending", None)
            try:
                if existing is None:
                    await self._jetson.request("POST", "/cameras", json={
                        "source_url": source,
                        "config": {"camera_uuid": camera_uuid,
                                   "discovery_identity": camera.identity,
                                   "detection_enabled": camera.detection_wanted},
                    })
                elif existing.get("source_url") != source:
                    await self._jetson.request("PATCH", "/cameras/" + camera_uuid, json={"source_url": source})
                await self._save(camera.id, camera_uuid, "pushed", None)
            except JetsonError as exc:
                state = "capacity_pending" if exc.status_code == 409 else "error"
                await self._save(camera.id, camera_uuid, state, str(exc))
                errors.append("{}: {}".format(camera.identity, exc))
        return errors

    async def _save(self, camera_id, camera_uuid, state, error):
        async with self._sessions() as db, db.begin():
            camera = await db.get(Camera, camera_id)
            if camera is not None:
                camera.camera_uuid = camera_uuid
                camera.jetson_state = state
                camera.jetson_error = error
