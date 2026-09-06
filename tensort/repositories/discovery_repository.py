# repositories/discovery_repository.py  (Python 3.6)
"""
Persistence for the discovered-camera roster.

Deliberately synchronous, unlike CameraRepository's async write paths. The
discovery scanner runs in its own worker thread (it does blocking socket I/O
that must never touch the pipeline's event loop), so it has no loop to await
on. The sync SQLAlchemy session is the correct tool there.

Every method owns its session and commits, because there is no request-scoped
transaction on the edge for a caller to own.
"""

import logging
from datetime import datetime

try:
    # Script mode (python main.py from Backend/tensort)
    from database import db_manager
    from database_orm import DiscoveredCamera
except Exception:
    # Package mode (python -m Backend.tensort.main)
    from ..database import db_manager
    from ..database_orm import DiscoveredCamera

logger = logging.getLogger(__name__)


class DiscoveryRepository(object):
    """Stateless repository over the `discovered_cameras` table."""

    def list_all(self):
        """Every roster row as a plain dict, newest-seen first."""
        session = db_manager.get_session()
        try:
            rows = (
                session.query(DiscoveredCamera)
                .order_by(DiscoveredCamera.is_present.desc(), DiscoveredCamera.last_seen_at.desc())
                .all()
            )
            return [row.to_dict() for row in rows]
        except Exception:
            logger.exception("Failed to list discovered cameras")
            return []
        finally:
            session.close()

    def get_by_identity(self, identity):
        session = db_manager.get_session()
        try:
            row = (
                session.query(DiscoveredCamera)
                .filter(DiscoveredCamera.identity == identity)
                .one_or_none()
            )
            return row.to_dict() if row else None
        finally:
            session.close()

    def mark_seen(self, device, source_url=None):
        """Record that `device` (a discovery.HikDevice) is present right now.

        Returns a dict describing the transition:
            {"identity", "is_new", "recovered", "ip_changed", "row"}

        `is_new` means this physical camera has never been seen before — that
        is what triggers auto-provisioning. `recovered` means it was previously
        absent and has come back, which clears any outstanding missing alert.
        """
        now = datetime.utcnow()
        identity = device.identity()

        session = db_manager.get_session()
        try:
            row = (
                session.query(DiscoveredCamera)
                .filter(DiscoveredCamera.identity == identity)
                .one_or_none()
            )

            is_new = row is None
            recovered = False
            ip_changed = False

            if row is None:
                row = DiscoveredCamera(
                    identity=identity,
                    first_seen_at=now,
                    created_at=now,
                )
                session.add(row)
            else:
                recovered = not bool(row.is_present)
                ip_changed = bool(row.ip_address) and row.ip_address != device.ip

            row.ip_address = device.ip
            row.mac_address = device.mac
            row.serial_number = device.serial_number
            row.model = device.model
            row.firmware = device.firmware
            row.device_name = device.device_name
            if source_url:
                row.source_url = source_url

            row.is_present = True
            row.consecutive_misses = 0
            row.alerted = False
            row.missing_since = None
            row.last_seen_at = now
            row.updated_at = now

            session.commit()
            return {
                "identity": identity,
                "is_new": is_new,
                "recovered": recovered,
                "ip_changed": ip_changed,
                "row": row.to_dict(),
            }
        except Exception:
            session.rollback()
            logger.exception("Failed to mark discovered camera %s as seen", identity)
            return {
                "identity": identity,
                "is_new": False,
                "recovered": False,
                "ip_changed": False,
                "row": None,
            }
        finally:
            session.close()

    def mark_missing(self, present_identities, miss_threshold=2):
        """Age out every roster row not in `present_identities`.

        Returns the rows that crossed the alert threshold on THIS sweep — i.e.
        newly-missing cameras. Rows already alerted are not returned again, so
        a camera that has been unplugged for a week does not re-alert every
        minute; it stays in the roster as absent and shows up in the report.
        """
        now = datetime.utcnow()
        present = set(present_identities or ())
        newly_missing = []

        session = db_manager.get_session()
        try:
            rows = session.query(DiscoveredCamera).all()
            for row in rows:
                if row.identity in present:
                    continue

                row.consecutive_misses = int(row.consecutive_misses or 0) + 1
                row.updated_at = now

                if row.missing_since is None:
                    row.missing_since = now

                if row.consecutive_misses < miss_threshold:
                    # Still inside the grace window — a rebooting camera or one
                    # dropped probe must not raise an alert.
                    continue

                row.is_present = False
                if not row.alerted:
                    row.alerted = True
                    newly_missing.append(row.to_dict())

            session.commit()
            return newly_missing
        except Exception:
            session.rollback()
            logger.exception("Failed to age out missing discovered cameras")
            return []
        finally:
            session.close()

    def attach_camera_uuid(self, identity, camera_uuid, source_url=None):
        """Link a roster row to the camera_configs row it was provisioned into."""
        session = db_manager.get_session()
        try:
            row = (
                session.query(DiscoveredCamera)
                .filter(DiscoveredCamera.identity == identity)
                .one_or_none()
            )
            if row is None:
                return False
            row.camera_uuid = str(camera_uuid) if camera_uuid else None
            if source_url:
                row.source_url = source_url
            row.updated_at = datetime.utcnow()
            session.commit()
            return True
        except Exception:
            session.rollback()
            logger.exception("Failed to attach camera_uuid to %s", identity)
            return False
        finally:
            session.close()

    def forget(self, identity):
        """Permanently drop a roster row (decommissioned camera)."""
        session = db_manager.get_session()
        try:
            row = (
                session.query(DiscoveredCamera)
                .filter(DiscoveredCamera.identity == identity)
                .one_or_none()
            )
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True
        except Exception:
            session.rollback()
            logger.exception("Failed to forget discovered camera %s", identity)
            return False
        finally:
            session.close()
