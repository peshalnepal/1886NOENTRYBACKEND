"""Scan, verify, persist, then provision the relay and reconcile Jetson channels.

Sweeps are serialized and shared by concurrent callers. Missing cameras are retained.
"""

import asyncio
import contextlib
import logging
import time
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..dtos import CandidateDTO, SourceDTO
from ..repositories import CameraMatchConflict, CameraRepository, NvrRepository
from .discovery_adapter import DiscoveryAdapter
from ...config import Settings
from ...core.database_orm import iso, utc_now

logger = logging.getLogger(__name__)

# Changes worth logging after a sweep.
CHANGE_KEYS = ("new_cameras", "missing_cameras", "recovered_cameras", "ip_changes")
# Report lists whose rows carry a camera source_url; the relay rewrites them.
EXPORTED_KEYS = ("roster", "present", "new_cameras", "recovered_cameras",
                 "missing_cameras", "ip_changes", "capacity_rejected")


def empty_report(nvr_mode: str, reason: str = "") -> Dict[str, Any]:
    report = {key: [] for key in EXPORTED_KEYS + ("unverified_candidates", "errors")}
    report.update(type="discovery_report", generated_at=iso(utc_now()), duration_s=0.0,
                  scanned=0, nvr_mode=nvr_mode, nvr=None)
    if reason:
        report["reason"] = reason
    return report


class SweepService:
    def __init__(self, settings: Settings, session_factory: async_sessionmaker,
                 adapter: DiscoveryAdapter, verifier, relay, adoption):
        self._settings = settings
        self._session_factory = session_factory
        self._adapter = adapter
        self._verifier = verifier
        self._relay = relay
        self._adoption = adoption
        self._cameras = CameraRepository()
        self._nvrs = NvrRepository()

        self._inflight: Optional[asyncio.Task] = None
        self._loop_task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._last_report: Optional[Dict[str, Any]] = None
        self._last_finished_at: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Start the background scheduler; False when DISCOVERY_ENABLED=false."""
        if not self._settings.discovery_enabled:
            logger.info("Discovery is disabled (DISCOVERY_ENABLED=false)")
            return False
        if self._loop_task is None or self._loop_task.done():
            self._stop.clear()
            self._loop_task = asyncio.create_task(self._loop(), name="nvr-sweep-scheduler")
            logger.info("Sweep scheduler started (every %ds, miss threshold %d, NVR mode %s)",
                        self._settings.discovery_interval_s, self._settings.discovery_miss_threshold,
                        self._settings.nvr_mode)
        return True

    async def stop(self) -> None:
        self._stop.set()
        for task in (self._loop_task, self._inflight):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    async def _loop(self) -> None:
        # First sweep immediately, so a freshly booted MiniPC has a roster.
        while not self._stop.is_set():
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Discovery sweep failed; retrying next interval")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=self._settings.discovery_interval_s)

    def request_scan(self) -> bool:
        """Schedule a sweep without waiting (for /sync). Shares one in flight; honours
        the quiet period. False only when discovery is disabled or stopping."""
        if not self._settings.discovery_enabled or self._stop.is_set():
            return False
        if not self.scan_in_progress() and not self._within_quiet_period():
            self._inflight = asyncio.create_task(self._sweep(), name="nvr-sweep")
            self._inflight.add_done_callback(_log_task_failure)
        return True

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    async def run_once(self, force: bool = False) -> Dict[str, Any]:
        """Await a complete sweep, sharing one already running. `force` ignores
        DISCOVERY_ENABLED (an explicit POST /discovery/scan)."""
        if not self._settings.discovery_enabled and not force:
            return empty_report(self._settings.nvr_mode, "discovery_disabled")
        if self.scan_in_progress():
            # shield: a caller that disconnects must not cancel the shared sweep.
            report = dict(await asyncio.shield(self._inflight))
            report["coalesced"] = True
            return report
        self._inflight = asyncio.create_task(self._sweep(), name="nvr-sweep")
        return await asyncio.shield(self._inflight)

    async def _sweep(self) -> Dict[str, Any]:
        started = time.monotonic()
        scan = await self._adapter.scan()
        # scan={"nvr": {
        #     "host": "192.168.1.10",
        #     "rtsp_port": 554,
        #     "http_port": 80,
        #     "origin": "static",
        #     "channels_spec": "1-2",
        #     "responded": true
        #     },
        #     "candidates": [
        #     {
        #         "sources": [
        #         {
        #             "source_identity": "serial:CAM001#1",
        #             "kind": "nvr",
        #             "host": "192.168.1.10",
        #             "rtsp_port": 554,
        #             "channel_no": 1,
        #             "source_url": "rtsp://192.168.1.10:554/Streaming/Channels/102",
        #             "serial_number": "CAM001",
        #             "mac_address": null,
        #             "model": "DS-2CD2143G0-I",
        #             "firmware": null,
        #             "device_name": "Front Door"
        #         }
        #         ]
        #     },
        #     {
        #         "sources": [
        #         {
        #             "source_identity": "ip:192.168.1.10#2",
        #             "kind": "nvr",
        #             "host": "192.168.1.10",
        #             "rtsp_port": 554,
        #             "channel_no": 2,
        #             "source_url": "rtsp://192.168.1.10:554/Streaming/Channels/202",
        #             "serial_number": null,
        #             "mac_address": null,
        #             "model": "NVR-Static-Channel",
        #             "firmware": null,
        #             "device_name": "192.168.1.10 ch2"
        #         }
        #         ]
        #     },
        #     {
        #         "sources": [
        #         {
        #             "source_identity": "serial:CAM003",
        #             "kind": "direct",
        #             "host": "192.168.1.30",
        #             "rtsp_port": 554,
        #             "channel_no": null,
        #             "source_url": "rtsp://192.168.1.30:554/Streaming/Channels/102",
        #             "serial_number": "CAM003",
        #             "mac_address": null,
        #             "model": "DS-2CD2143G0-I",
        #             "firmware": "V5.7",
        #             "device_name": "Back Door"
        #         }
        #         ]
        #     }
        #     ],
        #     "errors": []}

        winners = await asyncio.gather(*(self._verify(candidate) for candidate in scan.candidates))
        # winners=  =[
        #     {
        #     "source_identity": "serial:CAM001#1",
        #     "kind": "nvr",
        #     "host": "192.168.1.10",
        #     "rtsp_port": 554,
        #     "channel_no": 1,
        #     "source_url": "rtsp://192.168.1.10:554/Streaming/Channels/102",
        #     "serial_number": "CAM001",
        #     "mac_address": null,
        #     "model": "DS-2CD2143G0-I",
        #     "firmware": null,
        #     "device_name": "Front Door"
        #     },
        #     {
        #     "source_identity": "ip:192.168.1.10#2",
        #     "kind": "nvr",
        #     "host": "192.168.1.10",
        #     "rtsp_port": 554,
        #     "channel_no": 2,
        #     "source_url": "rtsp://192.168.1.10:554/Streaming/Channels/202",
        #     "serial_number": null,
        #     "mac_address": null,
        #     "model": "NVR-Static-Channel",
        #     "firmware": null,
        #     "device_name": "192.168.1.10 ch2"
        #     },
        #     {
        #     "source_identity": "serial:CAM003",
        #     "kind": "direct",
        #     "host": "192.168.1.30",
        #     "rtsp_port": 554,
        #     "channel_no": null,
        #     "source_url": "rtsp://192.168.1.30:554/Streaming/Channels/102",
        #     "serial_number": "CAM003",
        #     "mac_address": null,
        #     "model": "DS-2CD2143G0-I",
        #     "firmware": "V5.7",
        #     "device_name": "Back Door"
        #     }
        # ]

        report = empty_report(self._settings.nvr_mode)
        report["scanned"] = len(scan.candidates)
        report["errors"].extend(scan.errors)
        now = utc_now()
        async with self._session_factory() as db:
            async with db.begin():
                nvr_id = await self._sweep_nvr(db, scan.nvr, winners, report, now)
                present_ids = set()
                for candidate, winner in zip(scan.candidates, winners):
                    camera_id = await self._record(db, candidate, winner, nvr_id, report, now)
                    if camera_id is not None:
                        present_ids.add(camera_id)
                report["missing_cameras"] = await self._cameras.mark_missing(
                    db, present_ids, self._settings.discovery_miss_threshold, now)

        # Relay first: adoption only points Jetson at a configured relay path.
        present = [entry["identity"] for entry in report["present"]]
        # this report present will look something like ZipFile The class for reading and writing ZIP files.  See section 
        #  {
        #     "identity": camera.identity,# this is source identity made by serial number with ip
        #     "ip": camera.ip_address, 
        #     "ip_address": camera.ip_address,
        #     "serial_number": camera.serial_number,
        #     "model": camera.model,
        #     "firmware": camera.firmware,
        #     "device_name": camera.device_name,
        #     "mac": camera.mac_address,
        #     "channel": camera.channel_no or 1,
        #     "rtsp_port": camera.rtsp_port,
        #     "is_nvr": camera.source_kind == "nvr",
        #     "source_kind": camera.source_kind,
        #     "nvr_id": camera.nvr_id,
        #     "channel_no": camera.channel_no,
        #     "source_url": camera.source_url,
        # }
        report["errors"].extend(await self._relay.reconcile(present))
        report["errors"].extend(await self._adoption.reconcile(present))

        report["roster"] = await self._roster_rows()
        by_identity = {row["identity"]: row for row in report["roster"]}
        for entry in report["new_cameras"]:
            row = by_identity[entry["identity"]]
            entry.update(camera_uuid=row["camera_uuid"], adopted=row["jetson_state"] == "pushed")
        report["capacity_rejected"] = [row for row in report["roster"]
                                       if row["jetson_state"] == "capacity_pending"]
        for key in EXPORTED_KEYS:
            report[key] = [self._relay.export(row) for row in report[key]]

        report["generated_at"] = iso(utc_now())
        report["duration_s"] = round(time.monotonic() - started, 2)
        self._publish(report)
        return report

    async def _verify(self, candidate: CandidateDTO) -> Optional[SourceDTO]:
        """Try routes best-first; stop at the first decoded frame."""
        for source in candidate.sources:
            result = await self._verifier.verify(self._adapter.render(source.kind, source.source_url))
            if result.ok:
                return source
        return None

    async def _sweep_nvr(self, db: AsyncSession, nvr, winners, report, now) -> Optional[int]:
        """Update the NVR row's presence: an ISAPI answer or any decoded channel counts."""
        if nvr is None:
            if await self._nvrs.clear(db):
                logger.warning("NVR_HOST is empty: removed the stored physical NVR row")
            return None

        row = await self._nvrs.upsert(db, nvr, now)
        state = {"recovered": False, "missing_alert": False}
        if nvr.responded or any(winner is not None and winner.kind == "nvr" for winner in winners):
            state.update(self._nvrs.mark_seen(row, now))
        else:
            state["missing_alert"] = self._nvrs.mark_missed(row, self._settings.discovery_miss_threshold, now)
        report["nvr"] = dict(row.to_dict(), **state)
        return row.id

    async def _record(
        self, db: AsyncSession, candidate: CandidateDTO, winner: Optional[SourceDTO], nvr_id, report, now,
    ) -> Optional[int]:
        """Persist one verification result; returns the camera id if verified."""
        try:
            camera = await self._cameras.find_for_candidate(db, candidate)
        except CameraMatchConflict as exc:
            report["errors"].append(str(exc))
            return None
        if winner is None:
            report["unverified_candidates"].append(candidate.sources[0].source_identity)
            if camera is None:
                return None
        else:
            is_new = camera is None
            recovered = not is_new and not camera.is_present and camera.first_frame_at is not None
            camera, change = await self._cameras.save_verified(db, camera, candidate, winner, nvr_id, now)
            # DIRECT CAMERA EXCESS
            # change = {
            #     "identity": "serial:ABC123",
            #     "camera_uuid": None,
            #     "ip_address": "192.168.1.21",
            #     "source_url": "rtsp://192.168.1.21:554/Streaming/Channels/102",
            #     "previous_source_url": "rtsp://192.168.1.20:554/Streaming/Channels/102",
            #     "source_kind": "direct",
            #     "channel_no": None,
            #     "reason": "ip_change",
            # }
            # NVR BASED CAMERA 
            # change = {
            #     "identity": "serial:ABC123#3",
            #     "camera_uuid": None,
            #     "ip_address": "192.168.1.11",
            #     "source_url": "rtsp://192.168.1.11:554/Streaming/Channels/302",
            #     "previous_source_url": "rtsp://192.168.1.10:554/Streaming/Channels/302",
            #     "source_kind": "nvr",
            #     "channel_no": 3,
            #     "reason": "ip_change",
            # }
            # Camera Object
            # camera = Camera(
            #     id=7,                         # local database row ID
            #     identity="serial:ABC123",     # stable identity from when first saved
            #     dedupe_key="serial:ABC123",
            #     camera_uuid="550e8400-e29b-41d4-a716-446655440000",

            #     serial_number="ABC123",
            #     mac_address="aa:bb:cc:dd:ee:ff",
            #     model="DS-2CD...",
            #     firmware="V5.7...",
            #     device_name="Front Door",

            #     source_kind="nvr",            # currently verified route
            #     nvr_id=1,                     # local NVR database row ID
            #     channel_no=3,
            #     ip_address="192.168.1.10",    # NVR IP
            #     rtsp_port=554,
            #     source_url="rtsp://192.168.1.10:554/Streaming/Channels/302",

            #     is_present=True,
            #     jetson_state="pushed",
            #     # plus timestamps, miss counters, alert state, etc.
            # )
            if change is not None:
                report["ip_changes"].append(change)

        if winner is None:
            return None

        entry = {
            "identity": camera.identity,# this is source identity made by serial number with ip
            "ip": camera.ip_address, 
            "ip_address": camera.ip_address,
            "serial_number": camera.serial_number,
            "model": camera.model,
            "firmware": camera.firmware,
            "device_name": camera.device_name,
            "mac": camera.mac_address,
            "channel": camera.channel_no or 1,
            "rtsp_port": camera.rtsp_port,
            "is_nvr": camera.source_kind == "nvr",
            "source_kind": camera.source_kind,
            "nvr_id": camera.nvr_id,
            "channel_no": camera.channel_no,
            "source_url": camera.source_url,
        }
        report["present"].append(entry)
        if is_new:
            report["new_cameras"].append(dict(entry))
        if recovered:
            report["recovered_cameras"].append(entry)
        return camera.id

    # 
    # Reporting / state
    # ------------------------------------------------------------------
    def _publish(self, report: Dict[str, Any]) -> None:
        self._last_report = report
        self._last_finished_at = time.monotonic()
        if any(report[key] for key in CHANGE_KEYS) or (report["nvr"] or {}).get("missing_alert"):
            logger.info("Discovery: %d present, %d new, %d missing, %d recovered, %d route changes",
                        len(report["present"]), *(len(report[key]) for key in CHANGE_KEYS))
        for error in report["errors"]:
            logger.warning("Discovery: %s", error)

    def last_report(self) -> Optional[Dict[str, Any]]:
        """Shallow copy of the last complete report; nested values are shared."""
        return dict(self._last_report) if self._last_report else None

    def scan_in_progress(self) -> bool:
        return self._inflight is not None and not self._inflight.done()

    def _within_quiet_period(self) -> bool:
        return (self._last_finished_at is not None
                and time.monotonic() - self._last_finished_at < self._settings.discovery_interval_s)

    async def _roster_rows(self) -> List[Dict[str, Any]]:
        async with self._session_factory() as db:
            return [c.to_dict() for c in await self._cameras.list_roster(db)]

    async def roster(self) -> List[Dict[str, Any]]:
        """The durable camera inventory, as the cloud sees it (relay URLs)."""
        return [self._relay.export(row) for row in await self._roster_rows()]

    async def forget(self, identity: str) -> bool:
        """Drop a local roster entry. Jetson channel------------------------------------------------------------------s and relay paths are kept."""
        async with self._session_factory() as db:
            async with db.begin():
                return await self._cameras.forget(db, identity)

    def status(self) -> Dict[str, Any]:
        report = self.last_report()
        roster = report["roster"] if report else []
        return {
            "enabled": self._settings.discovery_enabled,
            "running": self._loop_task is not None and not self._loop_task.done(),
            "nvr_mode": self._settings.nvr_mode,
            "interval_s": self._settings.discovery_interval_s,
            "miss_threshold": self._settings.discovery_miss_threshold,
            "in_quiet_period": self._within_quiet_period(),
            "scan_in_progress": self.scan_in_progress(),
            "last_sweep_at": report["generated_at"] if report else None,
            "last_sweep_duration_s": report["duration_s"] if report else None,
            "present_count": len(report["present"]) if report else 0,
            "missing_count": sum(1 for r in roster if not r.get("is_present")),
        }


def _log_task_failure(task: asyncio.Task) -> None:
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.error("Requested discovery sweep failed; keeping the last report",
                     exc_info=error)
