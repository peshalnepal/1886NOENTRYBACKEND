"""Test doubles for the MiniPC roster: no network, no ffmpeg, no real cameras.
Scans return scripted HikDevice lists and verification is a predicate over the
rendered (credentialed) URL.

Run the suite from Backend/:
    python -m pytest NVR/tests -q
"""

import asyncio
import json
import os
import tempfile
import threading
import time
from typing import Callable, Dict, List, Optional
from unittest import mock

import httpx

from NVR import discovery as hik
from NVR.application.services.discovery_adapter import DiscoveryAdapter
from NVR.application.services.frame_verifier import VerifyResult
from NVR.application.services.sweep_service import SweepService
from NVR.config import Settings
from NVR.core.database import Database

BASE_ENV = {
    "HIK_USERNAME": "admin",
    "HIK_PASSWORD": "cam-pass",
    "NVR_USERNAME": "recorder",
    "NVR_PASSWORD": "nvr-pass",
    "DISCOVERY_INTERVAL_S": "60",
    "DISCOVERY_MISS_THRESHOLD": "2",
    "NVR_PUBLIC_RTSP_BASE_URL": "rtsp://reader:test@tower.example:8554",
}

NVR_HOST = "10.0.0.50"


def make_settings(**overrides) -> Settings:
    """Settings from an isolated environment: shell vars and NVR/.env never leak in."""
    env = dict(BASE_ENV, **{key: str(value) for key, value in overrides.items()})
    with mock.patch.dict(os.environ, env, clear=True):
        return Settings(_env_file=None)


def hikvision_env(**overrides) -> Dict[str, str]:
    return dict({"NVR_HOST": NVR_HOST}, **overrides)


# ---------------------------------------------------------------------------
# Devices, shaped exactly like discovery.probe_device() / scan_network()
# ---------------------------------------------------------------------------
def direct(ip: str, serial: str, mac: Optional[str] = None, name: str = "IP CAMERA") -> "hik.HikDevice":
    return hik.HikDevice(ip=ip, serial_number=serial, model="DS-2CD2143G0-I", firmware="V5.7",
                         device_name=name, mac=mac, channel=1, rtsp_port=554, is_nvr=False)


def nvr_channel(channel: int, serial: Optional[str] = None, mac: Optional[str] = None,
                host: str = NVR_HOST, name: Optional[str] = None,
                model: str = "DS-2CD2143G0-I") -> "hik.HikDevice":
    return hik.HikDevice(ip=host, serial_number=serial, model=model, firmware=None,
                         device_name=name or "Camera {}".format(channel), mac=mac,
                         channel=channel, rtsp_port=554, is_nvr=True)


def static_channel(channel: int, host: str = NVR_HOST) -> "hik.HikDevice":
    return nvr_channel(channel, host=host, name="{} ch{}".format(host, channel), model=hik.STATIC_CHANNEL_MODEL)


class ScriptedScan:
    """scan_fn replacement: returns the current device list; records calls and threads."""

    def __init__(self, devices: Optional[List["hik.HikDevice"]] = None, delay_s: float = 0.0):
        self.devices = list(devices or [])
        self.delay_s = delay_s
        self.calls = 0
        self.threads: List[str] = []

    def __call__(self, settings):
        self.calls += 1
        self.threads.append(threading.current_thread().name)
        if self.delay_s:
            time.sleep(self.delay_s)
        return list(self.devices)


class FakeVerifier:
    """Decodes a 'frame' when `predicate(url)` is true. Records every URL it was given."""

    def __init__(self, predicate: Optional[Callable[[str], bool]] = None):
        self.predicate = predicate or (lambda url: True)
        self.urls: List[str] = []

    def fail_hosts(self, *hosts: str) -> None:
        self.predicate = lambda url: not any("@{}:".format(h) in url or "//{}:".format(h) in url for h in hosts)

    def ok_all(self) -> None:
        self.predicate = lambda url: True

    async def verify(self, url: str) -> VerifyResult:
        self.urls.append(url)
        await asyncio.sleep(0)
        return VerifyResult(True) if self.predicate(url) else VerifyResult(False, "no frame")


class MemoryMediaMtx:
    """httpx handler for MediaMTX's v3 path API, backed by a dict."""

    def __init__(self):
        self.paths: Dict[str, dict] = {}
        self.mutations: List[tuple] = []
        self.fail = False

    def __call__(self, request):
        if self.fail:
            return httpx.Response(503, text="secret upstream details")
        path = request.url.path.rsplit("/", 1)[-1]
        if request.method == "GET":
            return httpx.Response(200, json=self.paths[path]) if path in self.paths else httpx.Response(404)
        self.paths[path] = json.loads(request.content)
        self.mutations.append((request.method, path))
        return httpx.Response(200)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


class PassThroughRelay:
    """Relay stand-in for roster tests: no MediaMTX, rows keep their LAN fields."""

    async def reconcile(self, identities):
        return []

    def export(self, row):
        return dict(row)


class NoAdoption:
    async def camera_snapshot(self):
        return []

    async def reconcile(self, identities, remote=None):
        return []


class TowerHarness:
    """A SweepService over a temp SQLite file, a scripted scan and a fake verifier."""

    def __init__(self, env: Optional[Dict[str, str]] = None, devices=None, delay_s: float = 0.0):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmp.name, "nvr.sqlite3")
        self.settings = make_settings(MINIPC_DB_PATH=self.db_path, **(env or {}))
        self.scan = ScriptedScan(devices, delay_s=delay_s)
        self.verifier = FakeVerifier()
        self.database: Optional[Database] = None
        self.adapter: Optional[DiscoveryAdapter] = None
        self.sweep: Optional[SweepService] = None

    async def start(self) -> "TowerHarness":
        self.database = Database(self.db_path)
        await self.database.init()
        self._build_service()
        return self

    def _build_service(self) -> None:
        self.adapter = DiscoveryAdapter(self.settings, scan_fn=self.scan)
        self.sweep = SweepService(self.settings, self.database.session_factory, self.adapter,
                                  self.verifier, PassThroughRelay(), NoAdoption())

    async def restart(self) -> None:
        """A MiniPC reboot: fresh services over the same database file."""
        await self.sweep.stop()
        self.adapter.close()
        await self.database.dispose()
        self.database = Database(self.db_path)
        self._build_service()

    async def close(self) -> None:
        if self.sweep is not None:
            await self.sweep.stop()
        if self.adapter is not None:
            self.adapter.close()
        if self.database is not None:
            await self.database.dispose()
        self._tmp.cleanup()

    async def run(self) -> dict:
        return await self.sweep.run_once()

    @staticmethod
    def by_identity(report: dict) -> Dict[str, dict]:
        return {row["identity"]: row for row in report["roster"]}
