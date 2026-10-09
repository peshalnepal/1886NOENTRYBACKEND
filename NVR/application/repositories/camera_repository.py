"""Camera persistence. Never commits; the caller owns the transaction."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..dtos import CandidateDTO, SourceDTO
from ...core.database_orm import Camera

_METADATA_FIELDS = ("serial_number", "mac_address", "model", "firmware", "device_name")


class CameraMatchConflict(ValueError):
    """Discovery cannot safely choose an existing camera for a candidate."""


class CameraRepository:
    async def find_for_candidate(self, db: AsyncSession, candidate: CandidateDTO) -> Optional[Camera]:
        """Match identifiers from every fresh source against the camera inventory.

        Discovery normalizes serials and MACs before building the candidate;
        save_verified stores those same normalized values. Device identifiers
        take precedence over IP-based identities. Never pick the first of several
        matches, and never overwrite conflicting device metadata on an IP match.
        """
        serials = {source.serial_number for source in candidate.sources if source.serial_number}
        macs = {source.mac_address for source in candidate.sources if source.mac_address}
        keys = set(candidate.source_identities) | {candidate.dedupe_key}
        keys.update("serial:" + serial for serial in serials)
        keys.update("mac:" + mac for mac in macs)
        device_keys = {key for key in keys if key.startswith(("serial:", "mac:"))}

        matches = list((await db.execute(select(Camera).where(or_(
            Camera.serial_number.in_(serials), Camera.mac_address.in_(macs),
            Camera.identity.in_(keys), Camera.dedupe_key.in_(keys),
        )).order_by(Camera.id))).scalars())
        device_matches = [camera for camera in matches if (
            camera.serial_number in serials or camera.mac_address in macs
            or camera.identity in device_keys or camera.dedupe_key in device_keys
        )]
        matches = device_matches or matches
        if len(matches) > 1:
            raise CameraMatchConflict("Ambiguous camera match for {}: camera rows {}".format(
                candidate.dedupe_key, ", ".join(str(camera.id) for camera in matches)))
        if not matches:
            return None
        camera = matches[0]
        if not device_matches and (
            (serials and camera.serial_number and camera.serial_number not in serials)
            or (macs and camera.mac_address and camera.mac_address not in macs)
        ):
            raise CameraMatchConflict("Conflicting device identifiers for IP-based match {}: camera row {}".format(
                candidate.dedupe_key, camera.id))
        return camera

    async def save_verified(
        self, db: AsyncSession, camera: Optional[Camera], candidate: CandidateDTO, winner: SourceDTO,
        nvr_id: Optional[int], now: datetime,
    ) -> tuple[Camera, Optional[Dict[str, Any]]]:
        """Save the last verified route and sighting; return the camera and route change.

        Discovery observations may already point elsewhere. Only a decoded frame
        can replace this snapshot, which is what the relay and adoption consume.
        """
        if camera is None:
            camera = Camera(identity=winner.source_identity, first_seen_at=now, created_at=now)
            db.add(camera)
        previous_url, previous_kind, previous_host = camera.source_url, camera.source_kind, camera.ip_address
        camera.dedupe_key = candidate.dedupe_key
        camera.source_kind = winner.kind
        camera.nvr_id = nvr_id if winner.kind == "nvr" else None
        camera.channel_no = winner.channel_no
        camera.ip_address = winner.host
        camera.rtsp_port = winner.rtsp_port
        camera.source_url = winner.source_url

        # Winner first, then alternate routes; never blank a known value.
        ordered = [winner] + [s for s in candidate.sources if s.source_identity != winner.source_identity]
        for field in _METADATA_FIELDS:
            value = next((getattr(s, field) for s in ordered if getattr(s, field)), None)
            if value:
                setattr(camera, field, value)

        camera.is_present = True
        if camera.first_frame_at is None:
            camera.first_frame_at = now
        camera.consecutive_misses = 0
        camera.alerted = False
        camera.missing_since = None
        camera.last_seen_at = now
        camera.updated_at = now
        await db.flush()

        if previous_url is None or previous_url == camera.source_url:
            return camera, None
        return camera, {
            "identity": camera.identity,
            "camera_uuid": camera.camera_uuid,
            "ip_address": camera.ip_address,
            "source_url": camera.source_url,
            "previous_source_url": previous_url,
            "source_kind": camera.source_kind,
            "channel_no": camera.channel_no,
            "reason": ("ip_change" if previous_kind == camera.source_kind and previous_host != camera.ip_address
                       else "source_switch"),
        }

    async def mark_missing(
        self, db: AsyncSession, present_camera_ids: set[int], miss_threshold: int, now: datetime,
    ) -> List[Dict[str, Any]]:
        """Age every camera not verified this sweep; return those newly crossing the
        threshold. A grace window of `miss_threshold` sweeps, then ONE alert that
        re-arms on recovery."""
        newly_missing = []
        for camera in (await db.execute(select(Camera))).scalars():
            if camera.id in present_camera_ids:
                continue
            camera.consecutive_misses += 1
            camera.updated_at = now
            if camera.missing_since is None:
                camera.missing_since = now
            if camera.consecutive_misses < miss_threshold:
                continue
            camera.is_present = False
            if not camera.alerted:
                camera.alerted = True
                newly_missing.append(camera.id)
        await db.flush()
        if not newly_missing:
            return []
        return [camera.to_dict() for camera in (await db.execute(
            select(Camera).where(Camera.id.in_(newly_missing)).order_by(Camera.id)
            .execution_options(populate_existing=True)
        )).scalars()]

    async def list_roster(self, db: AsyncSession) -> List[Camera]:
        """Present cameras first, then latest sighting."""
        return list((await db.execute(
            select(Camera)
            .order_by(Camera.is_present.desc(), Camera.last_seen_at.desc(), Camera.id)
            .execution_options(populate_existing=True)
        )).scalars())

    async def forget(self, db: AsyncSession, identity: str) -> bool:
        """Delete one camera. Rediscovery will add it back."""
        camera = (await db.execute(select(Camera).where(Camera.identity == identity))).scalar_one_or_none()
        if camera is None:
            return False
        await db.delete(camera)
        await db.flush()
        return True
