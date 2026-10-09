"""Provision camera pulls on the MiniPC's MediaMTX, independently of Jetson."""

from hashlib import sha256
import re
from urllib.parse import urlsplit, urlunsplit

import httpx
from sqlalchemy import select

from .core.database_orm import Camera


class RelayError(Exception):
    """Safe error: upstream bodies and camera credentials are never included."""


def _same_setting(key, actual, expected):
    if key in {"recordPartDuration", "recordSegmentDuration", "recordDeleteAfter"}:
        # Go's duration JSON canonicalizes 168h to 168h0m0s. Comparing raw
        # strings would restart every recorder on every discovery sweep.
        def seconds(value):
            parts = re.findall(r"([0-9]+(?:\.[0-9]+)?)(h|m|s)", str(value))
            if not parts or "".join(n + unit for n, unit in parts) != str(value):
                return None
            return sum(float(n) * {"h": 3600, "m": 60, "s": 1}[unit] for n, unit in parts)
        return seconds(actual) is not None and seconds(actual) == seconds(expected)
    return actual == expected


class MediaRelay:
    def __init__(self, settings, sessions, adapter, *, transport=None):
        self.sessions = sessions
        self.adapter = adapter
        self.states = {}  # identity -> configured | error (absent = pending)
        self.settings = settings
        self.recording_states = {}
        self.public_base = settings.nvr_public_rtsp_base_url.rstrip("/")
        self.lan_base = (settings.mediamtx_lan_rtsp_base_url or settings.nvr_public_rtsp_base_url).rstrip("/")
        self.http = httpx.AsyncClient(
            base_url=str(settings.mediamtx_api_url).rstrip("/"),
            auth=(settings.mediamtx_api_user, settings.mediamtx_api_password)
                 if settings.mediamtx_api_user else None,
            timeout=10.0, trust_env=False, transport=transport,
        )

    async def close(self):
        await self.http.aclose()

    @staticmethod
    def path(identity):
        # Sticky discovery identity makes the path stable across route/IP changes.
        return "nvr-" + sha256(identity.encode()).hexdigest()

    def public_url(self, identity):
        return "{}/{}".format(self.public_base, self.path(identity))

    def jetson_url(self, identity):
        return "{}/{}".format(self.lan_base, self.path(identity))

    @staticmethod
    def recording_path(identity):
        return "record-" + sha256(identity.encode()).hexdigest()

    @staticmethod
    def stream_url(source, stream):
        """Select a Hikvision profile, retaining the channel and credentials."""
        parts = urlsplit(source)
        match = re.fullmatch(r"(/Streaming/Channels/)([0-9]+)(/?)", parts.path, re.IGNORECASE)
        if not match or int(match[2]) < 100:
            raise RelayError("Cannot select stream profile: unsupported camera URL")
        channel = int(match[2]) // 100
        return urlunsplit(parts._replace(path=f"{match[1]}{channel * 100 + stream}{match[3]}"))

    async def ensure(self, identity, source):
        """Add the path, patch it if its source changed, or leave it alone."""
        path = self.path(identity)
        payload = {"source": self.stream_url(source, 2), "sourceOnDemand": True,
                   "rtspTransport": "tcp", "record": False}
        await self._ensure_path(path, payload)

    async def ensure_recording(self, identity, source):
        await self._ensure_path(self.recording_path(identity), {
            "source": self.stream_url(source, 2), "sourceOnDemand": False,
            "rtspTransport": "tcp", "record": self.settings.recording_enabled,
            "recordPath": self.settings.recording_directory.rstrip("/") + "/%path/%Y-%m-%d_%H-%M-%S-%f",
            "recordFormat": "fmp4", "recordPartDuration": "1s",
            "recordSegmentDuration": "15s", "recordDeleteAfter": self.settings.recording_retention,
        })

    async def _ensure_path(self, path, payload):
        try:
            current = await self.http.get("/v3/config/paths/get/" + path)
            if current.status_code == 404:
                response = await self.http.post("/v3/config/paths/add/" + path, json=payload)
            elif current.is_success:
                existing = current.json()
                if not isinstance(existing, dict):
                    raise RelayError("MediaMTX returned invalid path configuration")
                if all(_same_setting(key, existing.get(key), value) for key, value in payload.items()):
                    return
                response = await self.http.patch("/v3/config/paths/patch/" + path, json=payload)
            else:
                raise RelayError("MediaMTX lookup failed (HTTP {})".format(current.status_code))
            if not response.is_success:
                raise RelayError("MediaMTX provisioning failed (HTTP {})".format(response.status_code))
        except (httpx.RequestError, ValueError):
            raise RelayError("MediaMTX is unavailable or returned invalid JSON") from None

    async def reconcile(self, identities):
        async with self.sessions() as db:
            cameras = list((await db.execute(select(Camera).where(Camera.identity.in_(identities)))).scalars())
        errors = []
        for camera in cameras:
            source = self.adapter.render(camera.source_kind, camera.source_url)
            try:
                await self.ensure(camera.identity, source)
                self.states[camera.identity] = "configured"
            except RelayError as exc:
                self.states[camera.identity] = "error"
                errors.append("{}: {}".format(camera.identity, exc))
            try:
                if self.settings.recording_enabled:
                    await self.ensure_recording(camera.identity, source)
                    self.recording_states[camera.identity] = "configured"
                else:
                    response = await self.http.delete("/v3/config/paths/delete/" + self.recording_path(camera.identity))
                    if response.status_code != 404:
                        response.raise_for_status()
                    self.recording_states[camera.identity] = "disabled"
            except (RelayError, httpx.HTTPError):
                self.recording_states[camera.identity] = "error"
                errors.append("{}: recording path provisioning failed".format(camera.identity))
        return errors

    def export(self, row):
        """A report/roster row as the cloud sees it: the public relay URL only."""
        row = dict(row)
        identity = row["identity"]
        row["source_url"] = self.public_url(identity)
        row["relay_state"] = self.states.get(identity, "pending")
        row["recording_state"] = self.recording_states.get(identity, "pending")
        row.pop("previous_source_url", None)
        return row

    async def translate(self, payload, *, outward):
        """Swap exact known relay URLs in camera API JSON: Jetson LAN <-> public."""
        async with self.sessions() as db:
            identities = list((await db.execute(select(Camera.identity))).scalars())
        pairs = [(self.jetson_url(i), self.public_url(i)) for i in identities]
        mapping = dict(pairs) if outward else {public: lan for lan, public in pairs}

        def visit(value):
            if isinstance(value, list):
                return [visit(item) for item in value]
            if isinstance(value, dict):
                return {key: mapping.get(item, item) if key == "source_url" and isinstance(item, str)
                        else visit(item) for key, item in value.items()}
            return value
        return visit(payload)
