"""Discover working cameras, preserve their identities, and schedule adoption.

Network probes and video verification run on a discovery thread. Runtime methods
bridge to the asyncio pipeline safely. Read `_sweep()` for the complete flow:
scan candidates, verify frames, adopt/update cameras, then report missing cameras.
Discovery mints UUIDs only for cameras that the cloud has not provisioned yet.
"""

import logging
import threading
import time
import uuid as uuid_mod
from datetime import datetime
from urllib.parse import urlsplit

if __package__:
    from .discovery import DiscoveryConfig, scan_network
    from .env_utils import env_bool, env_int
    from .limits import CameraCapacityError, camera_limit
    from .repositories import DiscoveryRepository
else:
    from discovery import DiscoveryConfig, scan_network
    from env_utils import env_bool, env_int
    from limits import CameraCapacityError, camera_limit
    from repositories import DiscoveryRepository

logger = logging.getLogger("jetson-discovery-service")


def _utc_now_iso():
    return datetime.utcnow().isoformat() + "Z"


class DiscoveryService(object):
    """Runs a discovery sweep on a fixed interval and keeps the last report.

    Public surface used by the routes:
        start() / stop()
        run_once(force=False) -> report dict
        request_scan()        -> schedule a nonblocking refresh for /sync
        last_report()         -> report dict or None
        roster()              -> list of roster dicts
        forget(identity)      -> bool
        status()              -> scheduler state dict
    """

    def __init__(self, runtime, repository=None, config=None):
        self._runtime = runtime
        self._repo = repository or DiscoveryRepository()
        self._cfg = config or DiscoveryConfig()

        self.interval_s = env_int("DISCOVERY_INTERVAL_S", 150, minimum=60)
        self.miss_threshold = env_int("DISCOVERY_MISS_THRESHOLD", 2, minimum=1)
        self.auto_add = env_bool("DISCOVERY_AUTO_ADD", True)

        self._last_sweep_finished_at = None

        self._thread = None
        self._stop_evt = threading.Event()

        # `_sweep_lock` serialises sweeps. A /sync-triggered sweep and the
        # scheduled one must never interleave — two concurrent sweeps would
        # both see a camera as "new" and provision it twice.
        self._sweep_lock = threading.Lock()

        self._report_lock = threading.Lock()
        self._last_report = None
        self._selected_identities = set()
        self._request_lock = threading.Lock()
        self._requested_scan_thread = None

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
        if self._requested_scan_thread is not None and self._requested_scan_thread.is_alive():
            self._requested_scan_thread.join(timeout=timeout_s)

    def request_scan(self):
        """Schedule one refresh without holding an HTTP request open for video.

        Frame verification across staggered WAN cameras can exceed the cloud's
        request timeout. Repeated sync requests share the in-flight scan.
        """
        if not self._cfg.enabled or self._stop_evt.is_set():
            return False
        with self._request_lock:
            if self._sweep_lock.locked() or (self._requested_scan_thread is not None and
                                           self._requested_scan_thread.is_alive()):
                return True

            if self._within_quiet_period():
                return True

            def refresh():
                try:
                    self.run_once()
                except Exception:
                    logger.exception("Requested discovery sweep failed; retaining last report")

            self._requested_scan_thread = threading.Thread(target=refresh, name="discovery-refresh", daemon=True)
            self._requested_scan_thread.start()
            return True

    def _within_quiet_period(self):
        """True while `interval_s` has not elapsed since the last sweep finished."""
        if self._last_sweep_finished_at is None:
            return False
        return (time.time() - self._last_sweep_finished_at) < self.interval_s

    def _loop(self):
        # The first sweep happens immediately so a freshly-booted Jetson has
        # its cameras up without waiting a full interval.
        while not self._stop_evt.is_set():
            try:
                self.run_once()
            except Exception:
                logger.exception("Discovery sweep failed; will retry next interval")
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
        # Select before opening streams. Offline selected cameras retain their
        # slot, so reconnects do not churn identities or displace other sources.
        devices = devices[:camera_limit()]
        previous_selection = self._selected_identities
        selected_identities = {dev.identity() for dev in devices}
        self._selected_identities = selected_identities
        if self.auto_add and hasattr(self._runtime, "select_discovery_sources"):
            self._runtime.select_discovery_sources([self._cfg.rtsp_url_for(dev) for dev in devices])

        report = self._empty_report()
        report.pop("reason")
        report["scanned"] = len(devices)
        source_updates = []
        for dev in devices:
            if self._stop_evt.is_set():
                return self.last_report() or self._empty_report(reason="stopped")
            self._inspect_candidate(dev, report, source_updates)

        for change in source_updates:
            try:
                self._repoint(change)
                report["ip_changes"].append(change)
            except Exception as error:
                logger.exception("Failed to update discovered camera source: %s", change["identity"])
                report["errors"].append("Failed to repoint {}: {}".format(change["identity"], error))

        if self._stop_evt.is_set():
            # A partial sweep must not age out cameras that were not tested.
            return self.last_report() or self._empty_report(reason="stopped")
        present_identities = [camera["identity"] for camera in report["present"]]
        # Excluded rows are historical, not newly missing selected cameras.
        excluded = [row["identity"] for row in self._repo.list_all()
                    if row["identity"] not in (selected_identities | previous_selection)]
        report["missing_cameras"] = self._repo.mark_missing(
            present_identities + excluded, miss_threshold=self.miss_threshold)
        report["roster"] = [row for row in self._repo.list_all()
                            if row["identity"] in selected_identities]
        report["generated_at"] = _utc_now_iso()
        report["duration_s"] = round(time.time() - started, 2)
        # Stamped on completion, not on start: a sweep that overruns the quiet
        # period should still leave a full gap behind it before the next one.
        self._last_sweep_finished_at = time.time()
        self._publish_report(report)
        return report

    def _inspect_candidate(self, dev, report, source_updates):
        """Verify a candidate, record presence, and adopt it when permitted."""
        identity = dev.identity()
        source_url = self._cfg.rtsp_url_for(dev)
        if not self._runtime.verify_source(source_url, self._stop_evt):
            report["unverified_candidates"].append(identity)
            return
        report["present"].append(dev.to_dict())
        state = self._repo.mark_seen(dev, source_url=source_url)

        existing = self._existing_camera_uuid_for(dev, identity, source_url)
        current = next((cam for cam in self._runtime.list_cameras()
                        if cam.get("camera_uuid") == existing), None)
        if current is not None and current.get("source_url") != source_url:
            source_updates.append({
                "identity": identity,
                "ip_address": dev.ip,
                "source_url": source_url,
                "camera_uuid": existing,
            })

        if state.get("recovered"):
            report["recovered_cameras"].append(dev.to_dict())

        # Retry a previously seen camera whose admission failed (for
        # example because all eight slots were occupied at the last scan).
        row = state.get("row")
        needs_adoption = state.get("is_new") or (row is not None and not row.get("camera_uuid"))
        if not needs_adoption:
            return

        if not self.auto_add:
            report["new_cameras"].append(
                self._camera_entry(dev, identity, source_url, adopted=False)
            )
            return

        try:
            report["new_cameras"].append(self._adopt(dev, identity, source_url))
        except CameraCapacityError:
            report["capacity_rejected"].append(identity)
            report["errors"].append("At capacity, not added: {}".format(identity))
        except Exception as e:
            logger.exception("Failed to adopt discovered camera %s", identity)
            report["errors"].append("Failed to add {}: {}".format(identity, e))

    def _publish_report(self, report):
        """Cache the completed sweep and broadcast inventory changes."""
        with self._report_lock:
            self._last_report = report
        if report["capacity_rejected"]:
            logger.warning("Camera capacity reached; not added: %s",
                           ", ".join(report["capacity_rejected"]))
        changes = ("new_cameras", "missing_cameras", "recovered_cameras", "ip_changes")
        if any(report[key] for key in changes):
            logger.info("Discovery: %d present, %d new, %d missing, %d recovered, %d source changes",
                        len(report["present"]), *(len(report[key]) for key in changes))
            self._broadcast(report)

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
    def _stream_endpoint(url):
        """Compare streams without credentials; retain the NVR channel path."""
        try:
            parsed = urlsplit(url or "")
            return (parsed.scheme, parsed.hostname, parsed.port or 554,
                    parsed.path, parsed.query)
        except ValueError:
            return None

    def _existing_camera_uuid_for(self, dev, identity, source_url):
        """Prefer saved identity over endpoint, then direct-camera host fallback."""
        cameras = self._runtime.list_cameras() or []
        row = self._repo.get_by_identity(identity)
        linked_uuid = row.get("camera_uuid") if row else None
        is_nvr = bool(dev.is_nvr)

        # Check stronger matches across the whole list before weaker fallbacks.
        for camera in cameras:
            if linked_uuid and camera.get("camera_uuid") == linked_uuid:
                return linked_uuid
        for camera in cameras:
            config = camera.get("config") or {}
            if config.get("discovery_identity") == identity:
                return camera.get("camera_uuid")
            same_serial = dev.serial_number and config.get("discovery_serial") == dev.serial_number
            same_channel = not is_nvr or config.get("discovery_channel") == dev.channel
            if same_serial and same_channel:
                return camera.get("camera_uuid")

        target = self._stream_endpoint(source_url)
        for camera in cameras:
            if target is not None and self._stream_endpoint(camera.get("source_url")) == target:
                return camera.get("camera_uuid")

        # NVR channels share a host, so host-only matching would merge cameras.
        if not is_nvr and target is not None and target[1]:
            for camera in cameras:
                saved = self._stream_endpoint(camera.get("source_url"))
                if saved is not None and saved[1] == target[1]:
                    return camera.get("camera_uuid")
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
            # Channel + nvr flag are what disambiguate two cameras that share a
            # host (and sometimes a serial) behind the same NVR.
            "discovery_channel": getattr(dev, "channel", 1),
            "discovery_is_nvr": bool(getattr(dev, "is_nvr", False)),
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
        """Apply a verified source change; a failed update retries next sweep."""
        self._runtime.patch_camera(change["camera_uuid"], {
            "source_url": change["source_url"],
            "discovery_ip": change["ip_address"],
        })
        logger.info(
            "Updated camera %s source at %s",
            change["camera_uuid"], change["ip_address"],
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

    def current_report(self):
        """Expose verified selected rows during a long sweep for cloud adoption.

        Partial reports must not be used to infer that absent cameras are offline.
        """
        report = self.last_report()
        if not self._sweep_lock.locked() and report is not None:
            return report
        report = self._empty_report("discovery_pending")
        report["partial"] = True
        active = {camera["camera_uuid"] for camera in self._runtime.list_cameras()}
        report["roster"] = [row for row in self._repo.list_all()
                            if row["identity"] in self._selected_identities
                            and row.get("camera_uuid") in active]
        return report

    def roster(self):
        return [row for row in self._repo.list_all()
                if row["identity"] in self._selected_identities]

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
            "in_quiet_period": self._within_quiet_period(),
            "scan_in_progress": self._sweep_lock.locked(),
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
            "unverified_candidates": [],
            "capacity_rejected": [],
            "present": [],
            "new_cameras": [],
            "missing_cameras": [],
            "recovered_cameras": [],
            "ip_changes": [],
            "roster": [],
            "errors": [],
            "reason": reason,
        }
