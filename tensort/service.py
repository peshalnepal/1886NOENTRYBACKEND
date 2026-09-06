# service.py  (Python 3.6)
"""
Scheduled Hikvision camera discovery for the Jetson edge.

The Jetson is the source of truth for which cameras exist. This service is what
makes that true: every minute it sweeps the network, adopts cameras it has
never seen before into the detection pipeline, and flags cameras that have
stopped answering so the frontend can raise an alert.

Threading model
---------------
Discovery does blocking socket work (UDP multicast, then up to 254 HTTP probes)
that takes seconds. Running that on the pipeline's asyncio loop would stall
every camera's decode task, so the scheduler owns a dedicated daemon thread and
reaches the pipeline only through `PipelineRuntime`'s existing thread-safe
wrappers — the same ones the Flask request threads use.

Sweep lifecycle
---------------
1. `scan_network()` returns the confirmed Hikvision cameras.
2. Each is marked present in the roster. A camera with no roster row is NEW.
3. New cameras are auto-provisioned: a UUID is minted, an RTSP URL is built
   from the shared credentials, and `runtime.add_camera()` starts a channel.
4. Roster rows absent from the sweep age towards missing; once past the miss
   threshold they raise a one-shot alert.
5. The resulting report is broadcast on the SSE stream and cached so the
   cloud's sync/reconcile call can pull it.

Why UUIDs are minted here
-------------------------
`PipelineRuntime.add_camera` refuses to generate IDs — the cloud normally owns
them. But an auto-discovered camera has no cloud row yet, and the cloud's
`EdgeInferenceClient.list_cameras` discards any entry whose camera_uuid is not
a parseable UUID. So the edge mints a real uuid4 and the cloud adopts it on the
next sync. The `discovered: True` marker in the config is what tells the cloud
(and a human reading the DB) that this row originated at the edge.
"""

import logging
import threading
import time
import uuid as uuid_mod
from datetime import datetime

try:
    # Script mode (python main.py from Backend/tensort)
    from discovery import DiscoveryConfig, scan_network
    from env_utils import env_bool, env_int
    from repositories import DiscoveryRepository
except ModuleNotFoundError:
    # Package mode (python -m Backend.tensort.main). Only a genuinely absent
    # top-level module falls through to here — an ImportError raised *inside* a
    # module that does exist (a missing DB driver, say) propagates instead of
    # being masked by a relative import that cannot work in script mode.
    from .discovery import DiscoveryConfig, scan_network
    from .env_utils import env_bool, env_int
    from .repositories import DiscoveryRepository

logger = logging.getLogger("jetson-discovery-service")


def _utc_now_iso():
    return datetime.utcnow().isoformat() + "Z"


class DiscoveryService(object):
    """Runs a discovery sweep on a fixed interval and keeps the last report.

    Public surface used by the routes:
        start() / stop()
        run_once(force=False) -> report dict   (also what /sync triggers)
        last_report()         -> report dict or None
        roster()              -> list of roster dicts
        forget(identity)      -> bool
        status()              -> scheduler state dict
    """

    def __init__(self, runtime, repository=None, config=None):
        self._runtime = runtime
        self._repo = repository or DiscoveryRepository()
        self._cfg = config or DiscoveryConfig()

        self.interval_s = env_int("DISCOVERY_INTERVAL_S", 60, minimum=10)
        self.miss_threshold = env_int("DISCOVERY_MISS_THRESHOLD", 2, minimum=1)
        self.auto_add = env_bool("DISCOVERY_AUTO_ADD", True)

        self._thread = None
        self._stop_evt = threading.Event()

        # `_sweep_lock` serialises sweeps. A /sync-triggered sweep and the
        # scheduled one must never interleave — two concurrent sweeps would
        # both see a camera as "new" and provision it twice.
        self._sweep_lock = threading.Lock()

        self._report_lock = threading.Lock()
        self._last_report = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        """Start the background sweep loop. Idempotent."""
        if not self._cfg.enabled:
            logger.info("Camera discovery is disabled (DISCOVERY_ENABLED=false)")
            return False
        if self._thread is not None and self._thread.is_alive():
            return True

        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._loop, name="discovery-scheduler", daemon=True
        )
        self._thread.start()
        logger.info(
            "Camera discovery scheduler started (every %ds, miss threshold %d, auto_add=%s)",
            self.interval_s, self.miss_threshold, self.auto_add,
        )
        return True

    def stop(self, timeout_s=5.0):
        self._stop_evt.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_s)

    def _loop(self):
        # The first sweep happens immediately so a freshly-booted Jetson has
        # its cameras up without waiting a full interval.
        while not self._stop_evt.is_set():
            try:
                self.run_once()
            except Exception:
                logger.exception("Discovery sweep failed; will retry next interval")
            # wait() returns early when stop() is called, so shutdown is prompt.
            self._stop_evt.wait(self.interval_s)

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    def run_once(self, force=False):
        """Run one sweep and return its report.

        `force` is accepted for symmetry with the route; a sweep already in
        flight is never duplicated, the caller just waits for it and gets the
        report it produced.
        """
        if not self._cfg.enabled and not force:
            return self._empty_report(reason="discovery_disabled")

        if not self._sweep_lock.acquire(False):
            # Another sweep is running. Block until it finishes, then hand back
            # its result rather than starting a redundant scan.
            with self._sweep_lock:
                report = self.last_report()
                if report is None:
                    return self._empty_report(reason="no_report_yet")
                report["coalesced"] = True
                return report

        try:
            return self._sweep()
        finally:
            self._sweep_lock.release()

    def _sweep(self):
        started = time.time()
        devices = scan_network(self._cfg)

        present_identities = []
        new_cameras = []
        recovered = []
        ip_changes = []
        errors = []

        for dev in devices:
            identity = dev.identity()
            present_identities.append(identity)
            source_url = self._cfg.rtsp_url_for(dev.ip)

            state = self._repo.mark_seen(dev, source_url=source_url)

            if state.get("ip_changed"):
                ip_changes.append({
                    "identity": identity,
                    "ip_address": dev.ip,
                    "source_url": source_url,
                })

            if state.get("recovered"):
                recovered.append(dev.to_dict())

            if not state.get("is_new"):
                continue

            if not self.auto_add:
                new_cameras.append(self._camera_entry(dev, identity, source_url, adopted=False))
                continue

            try:
                new_cameras.append(self._adopt(dev, identity, source_url))
            except Exception as e:
                logger.exception("Failed to adopt discovered camera %s", identity)
                errors.append("Failed to add {}: {}".format(identity, e))

        for change in ip_changes:
            try:
                self._repoint(change)
            except Exception as e:
                logger.exception("Failed to repoint camera after IP change: %s", change["identity"])
                errors.append("Failed to repoint {}: {}".format(change["identity"], e))

        missing = self._repo.mark_missing(present_identities, miss_threshold=self.miss_threshold)

        report = {
            "type": "discovery_report",
            "generated_at": _utc_now_iso(),
            "duration_s": round(time.time() - started, 2),
            "scanned": len(devices),
            "present": [d.to_dict() for d in devices],
            "new_cameras": new_cameras,
            "missing_cameras": missing,
            "recovered_cameras": recovered,
            "ip_changes": ip_changes,
            "roster": self._repo.list_all(),
            "errors": errors,
        }

        with self._report_lock:
            self._last_report = report

        if new_cameras or missing or recovered or ip_changes:
            logger.info(
                "Discovery: %d present, %d new, %d missing, %d recovered, %d ip change(s)",
                len(devices), len(new_cameras), len(missing), len(recovered), len(ip_changes),
            )
            self._broadcast(report)
        else:
            logger.debug("Discovery: %d present, no changes", len(devices))

        return report

    # ------------------------------------------------------------------
    # Pipeline adoption
    # ------------------------------------------------------------------
    @staticmethod
    def _camera_entry(dev, identity, source_url, adopted, camera_uuid=None, **extra):
        """The per-camera shape reported under `new_cameras`."""
        entry = {
            "identity": identity,
            "ip_address": dev.ip,
            "source_url": source_url,
            "camera_uuid": camera_uuid,
            "adopted": adopted,
            "model": dev.model,
            "serial_number": dev.serial_number,
        }
        entry.update(extra)
        return entry

    @staticmethod
    def _host_of(url):
        """Hostname of a stream URL, ignoring credentials, port and path."""
        if not url or not isinstance(url, str):
            return None
        authority = url.split("://", 1)[-1].split("/", 1)[0]
        if "@" in authority:
            authority = authority.rsplit("@", 1)[1]
        return authority.split(":", 1)[0].strip().lower() or None

    def _existing_camera_uuid_for(self, dev, identity, source_url):
        """Find an already-provisioned camera for this physical device.

        Guards the case where the roster says "new" but the pipeline already
        has the camera — e.g. the roster table was recreated while
        camera_configs survived, or the camera was pushed down by the cloud
        before discovery ever ran. Adopting again would give one physical
        camera two channels, both decoding the same stream.

        Matched on discovery identity first, then on the host in source_url,
        which is what catches a cloud-provisioned camera that has no discovery
        provenance at all.
        """
        try:
            cameras = self._runtime.list_cameras()
        except Exception:
            return None

        host = self._host_of(source_url)

        for cam in cameras or []:
            cfg = cam.get("config") or {}
            if cfg.get("discovery_identity") == identity:
                return cam.get("camera_uuid")
            if dev.serial_number and cfg.get("discovery_serial") == dev.serial_number:
                return cam.get("camera_uuid")
            if host and self._host_of(cam.get("source_url")) == host:
                return cam.get("camera_uuid")
        return None

    def _adopt(self, dev, identity, source_url):
        """Provision a newly-discovered camera into the running pipeline."""
        existing = self._existing_camera_uuid_for(dev, identity, source_url)
        if existing:
            self._repo.attach_camera_uuid(identity, existing, source_url=source_url)
            logger.info(
                "Discovered camera %s is already provisioned as %s; linked instead of re-adding",
                dev.ip, existing,
            )
            return self._camera_entry(
                dev, identity, source_url,
                adopted=False, camera_uuid=existing, already_present=True,
            )

        camera_uuid = str(uuid_mod.uuid4())
        result = self._runtime.add_camera(source_url, {
            "camera_uuid": camera_uuid,
            "channel_id": camera_uuid,
            "enabled": True,
            "detection_enabled": True,
            "notification_enabled": True,
            "discovered": True,
            "discovery_identity": identity,
            "discovery_ip": dev.ip,
            "discovery_model": dev.model,
            "discovery_serial": dev.serial_number,
            "discovery_name": dev.device_name,
        })
        self._repo.attach_camera_uuid(identity, camera_uuid, source_url=source_url)

        logger.info(
            "Adopted new Hikvision camera %s (%s) as %s",
            dev.ip, dev.model or "unknown model", camera_uuid,
        )

        return self._camera_entry(
            dev, identity, source_url,
            adopted=True, camera_uuid=camera_uuid,
            config=result.get("config") if isinstance(result, dict) else None,
        )

    def _repoint(self, change):
        """Update an adopted camera's source_url after its IP moved."""
        row = self._repo.get_by_identity(change["identity"])
        if not row or not row.get("camera_uuid"):
            return
        self._runtime.patch_camera(row["camera_uuid"], {
            "source_url": change["source_url"],
            "discovery_ip": change["ip_address"],
        })
        logger.info(
            "Repointed camera %s to %s after IP change",
            row["camera_uuid"], change["ip_address"],
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def _broadcast(self, report):
        """Push the report onto the pipeline's SSE broadcaster.

        The message carries no `camera_uuid`, so per-camera SSE subscribers
        correctly ignore it — this is a fleet-level event and only the
        all-cameras stream should receive it.

        The payload is trimmed to the deltas: the full roster can be hundreds
        of rows and the broadcaster's queues are bounded at 200 messages.
        """
        pipeline = getattr(self._runtime, "pipeline", None)
        loop = getattr(self._runtime, "loop", None)
        if pipeline is None or loop is None:
            return

        event = {k: report[k] for k in (
            "type", "generated_at", "scanned", "new_cameras",
            "missing_cameras", "recovered_cameras", "ip_changes", "errors",
        )}

        try:
            # broadcast() is synchronous (put_nowait only) but touches the
            # subscriber list owned by the pipeline loop, so hop threads.
            loop.call_soon_threadsafe(pipeline.broadcaster.broadcast, event)
        except Exception:
            logger.debug("Failed to broadcast discovery report", exc_info=True)

    def last_report(self):
        with self._report_lock:
            return dict(self._last_report) if self._last_report else None

    def roster(self):
        return self._repo.list_all()

    def forget(self, identity):
        return self._repo.forget(identity)

    def status(self):
        """Scheduler state, for /health and the discovery status route."""
        report = self.last_report()
        roster = report["roster"] if report else []
        return {
            "enabled": bool(self._cfg.enabled),
            "running": self._thread is not None and self._thread.is_alive(),
            "interval_s": self.interval_s,
            "miss_threshold": self.miss_threshold,
            "auto_add": self.auto_add,
            "last_sweep_at": report["generated_at"] if report else None,
            "last_sweep_duration_s": report["duration_s"] if report else None,
            "present_count": len(report["present"]) if report else 0,
            "missing_count": sum(1 for r in roster if not r.get("is_present")),
        }

    def _empty_report(self, reason=""):
        return {
            "type": "discovery_report",
            "generated_at": _utc_now_iso(),
            "duration_s": 0.0,
            "scanned": 0,
            "present": [],
            "new_cameras": [],
            "missing_cameras": [],
            "recovered_cameras": [],
            "ip_changes": [],
            "roster": self._repo.list_all(),
            "errors": [],
            "reason": reason,
        }
