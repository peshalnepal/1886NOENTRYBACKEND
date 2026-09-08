import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx

from core.env import env_bool, env_float
from core.source_url import is_rtsp_source

logger = logging.getLogger(__name__)


def _mediamtx_source_payload(source: str) -> Dict[str, Any]:
    """Build the MediaMTX path payload for a camera ``source``.

    ``rtspTransport`` only applies to RTSP/RTSPS sources; sending it for a
    WebRTC/HLS/RTMP/SRT source is meaningless, so it is only included for RTSP.
    """
    payload: Dict[str, Any] = {"source": source, "sourceOnDemand": True}
    if is_rtsp_source(source):
        payload["rtspTransport"] = "tcp"
    return payload


def get_public_webrtc_base() -> str:
    pub_host = os.getenv("PUBLIC_HOST", "localhost")
    pub_scheme = os.getenv("PUBLIC_SCHEME", "http")
    pub_port = os.getenv("WEBRTC_HTTP_PORT", "8889")
    return (os.getenv("WEBRTC_PUBLIC_BASE_URL") or f"{pub_scheme}://{pub_host}:{pub_port}").rstrip("/")


def derive_public_webrtc_url(stream_key: str) -> str:
    """
    Derive the WHEP endpoint URL for a stream.
    MediaMTX ≥ 1.4 uses: {base}/{stream_key}/whep
    """
    base = get_public_webrtc_base()
    return f"{base}/{stream_key}/whep"


def resolve_camera_webrtc_url(*, camera_code: Optional[str], stored_url: Optional[str]) -> Optional[str]:
    code = str(camera_code or "").strip()
    if code:
        return derive_public_webrtc_url(code)

    url = str(stored_url or "").strip()
    return url or None

class WebRTCGatewayClient:
    """
    Provisions (or updates) any-source -> WebRTC streams on a MediaMTX gateway
    using the WHEP protocol.

    The path ``source`` is the camera's ``source_url`` and may be any scheme
    MediaMTX can pull: rtsp/rtsps, rtmp/rtmps, srt, http(s) (HLS), or webrtc/whep.
    MediaMTX pulls it and re-broadcasts to the browser over WebRTC/WHEP — the
    output protocol is always WebRTC regardless of the input scheme. (MediaMTX
    does not transcode, so the source codec must be WebRTC-compatible — H264/
    VP8/VP9/AV1; e.g. H265/MJPEG sources can be recorded but not live-viewed
    over WebRTC.)

    Streams are read at {WEBRTC_PUBLIC_BASE_URL}/{stream_key}/whep, where the
    frontend POSTs an SDP offer. MediaMTX must have WHEP enabled and STUN
    configured.

    Two operational modes:
      1) Admin API available -> provisions streams via /v3/config/paths/add
      2) No admin API -> derives stable public WHEP URLs without provisioning

    Configured by the WEBRTC_*/MTX_API_* environment variables read in __init__.
    """

    def __init__(self):
        self.admin_api_enabled = env_bool("WEBRTC_ADMIN_API_ENABLED", True)
        self.admin_api_url = (
            (
                os.getenv("WEBRTC_ADMIN_API_URL")
                or "https://noentrymtxfdxidm.centralus.azurecontainer.io"
            ).rstrip("/")
            if self.admin_api_enabled
            else ""
        )

        self.public_base = get_public_webrtc_base()

        self.api_user = os.getenv("MTX_API_USER") or os.getenv("MEDIAMTX_API_USER", "api")
        # `is not None` rather than `or`: an empty string is a valid password.
        api_pass = os.getenv("MTX_API_PASS")
        if api_pass is None:
            api_pass = os.getenv("MEDIAMTX_API_PASS")
        self.api_pass = "api_pass_123" if api_pass is None else api_pass

        self._warn_interval_s = env_float("WEBRTC_WARN_INTERVAL_S", 60.0)
        self._last_warn: Dict[str, float] = {}

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                env_float("WEBRTC_ADMIN_TIMEOUT_S", 15.0),
                connect=env_float("WEBRTC_ADMIN_CONNECT_TIMEOUT_S", 5.0),
            ),
            verify=False,  # the gateway commonly uses a self-signed cert
        )

        logger.info(
            "WebRTCGatewayClient initialized: admin_api_enabled=%s, admin_api_url=%s, public_base=%s",
            self.admin_api_enabled,
            self.admin_api_url if self.admin_api_enabled else "(disabled)",
            self.public_base,
        )

    def _warn_throttled(self, key: str, message: str, *args: object) -> None:
        now = time.time()
        last = self._last_warn.get(key, 0.0)
        if (now - last) < self._warn_interval_s:
            return
        self._last_warn[key] = now
        logger.warning(message, *args)

    def _is_timeout_or_network_error(self, exc: Exception) -> bool:
        return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError))

    async def close(self) -> None:
        await self._client.aclose()
    
    def _auth(self) -> Tuple[str, str]:
        return (self.api_user, self.api_pass)

    def _derive_public_webrtc_url(self, stream_key: str) -> str:
        return derive_public_webrtc_url(stream_key)

    def _response_error_message(self, *, action: str, response: httpx.Response) -> str:
        body = (response.text or "").strip().replace("\n", " ")
        if len(body) > 200:
            body = body[:200] + "..."
        if body:
            return f"{action} returned HTTP {response.status_code}: {body}"
        return f"{action} returned HTTP {response.status_code}"

    async def _provision(
        self, *, action: str, method: str, stream_key: str, payload: Dict[str, Any]
    ) -> Optional[str]:
        """One provisioning attempt against the admin API.

        Returns None on success, or a human-readable reason on failure — the
        caller decides whether that failure is fatal.
        """
        url = f"{self.admin_api_url}/v3/config/paths/{action}/{quote(stream_key, safe='')}"
        try:
            response = await self._client.request(
                method, url, json=payload, auth=self._auth()
            )
            if response.status_code == 200:
                return None
            return self._response_error_message(
                action=f"MediaMTX {action}", response=response
            )
        except Exception as exc:
            if self._is_timeout_or_network_error(exc):
                self._warn_throttled(
                    f"ensure_stream_{action}_timeout",
                    "MediaMTX %s timed out/unreachable. stream_key=%s admin_api=%s error=%s",
                    action,
                    stream_key,
                    self.admin_api_url,
                    str(exc),
                )
                return f"MediaMTX {action} timed out/unreachable: {type(exc).__name__}: {exc}"
            logger.warning(
                "MediaMTX %s request failed. stream_key=%s error=%s",
                action,
                stream_key,
                exc,
                exc_info=True,
            )
            return f"MediaMTX {action} request failed: {type(exc).__name__}: {exc}"

    async def ensure_stream(self, *, stream_key: str, source_url: str) -> Optional[str]:
        """Ensure the stream exists in MediaMTX and return its stable WHEP URL.

        Tries `add` first, then `patch` (the path may already exist). With the
        admin API disabled the derived URL is returned without provisioning.
        """
        whep_url = self._derive_public_webrtc_url(stream_key)
        if not self.admin_api_url:
            logger.debug("Admin API disabled, returning derived WHEP URL: %s", whep_url)
            return whep_url

        payload = _mediamtx_source_payload(source_url)
        errors: Dict[str, str] = {}

        for action, method in (("add", "POST"), ("patch", "PATCH")):
            error = await self._provision(
                action=action, method=method, stream_key=stream_key, payload=payload
            )
            if error is None:
                logger.info(
                    "Stream provisioned via %s: stream_key=%s whep_url=%s",
                    action,
                    stream_key,
                    whep_url,
                )
                return whep_url
            errors[action] = error
            logger.warning(
                "%s. stream_key=%s admin_api=%s", error, stream_key, self.admin_api_url
            )

        raise RuntimeError(
            "Failed to provision MediaMTX stream '{}'. add_error={}; patch_error={}. "
            "This likely means: (1) MediaMTX is unreachable, (2) credentials are wrong, "
            "(3) WHEP protocol not enabled, or (4) stream format invalid.".format(
                stream_key,
                errors.get("add", "unknown"),
                errors.get("patch", "unknown"),
            )
        )

    async def update_stream(self, *, stream_key: str, source_url: str) -> None:
        if not self.admin_api_url:
            return
            
        safe_name = quote(stream_key, safe="")
        url = f"{self.admin_api_url}/v3/config/paths/patch/{safe_name}"
        payload = _mediamtx_source_payload(source_url)

        try:
            await self._client.patch(url, json=payload, auth=self._auth())
        except Exception as exc:
            if self._is_timeout_or_network_error(exc):
                self._warn_throttled(
                    "update_stream_timeout",
                    "MediaMTX update timed out/unreachable. stream_key=%s admin_api=%s",
                    stream_key,
                    self.admin_api_url,
                )
            else:
                logger.error("MediaMTX update failed for %s", stream_key, exc_info=True)

    async def delete_stream(self, *, stream_key: str) -> bool:
        if not self.admin_api_url:
            return False
            
        safe_name = quote(stream_key, safe="")
        url = f"{self.admin_api_url}/v3/config/paths/delete/{safe_name}"
        
        try:
            response = await self._client.delete(url, auth=self._auth())
            if response.status_code == 404:
                logger.info("MediaMTX stream already absent for %s", stream_key)
                return False
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                self._response_error_message(action="MediaMTX delete", response=exc.response)
            ) from exc
        except Exception as exc:
            if self._is_timeout_or_network_error(exc):
                self._warn_throttled(
                    "delete_stream_timeout",
                    "MediaMTX delete timed out/unreachable. stream_key=%s admin_api=%s",
                    stream_key,
                    self.admin_api_url,
                )
            else:
                logger.warning("MediaMTX delete failed for %s", stream_key, exc_info=True)
            raise


    async def _list_paths(self, endpoint: str, *, label: str) -> List[Dict[str, Any]]:
        """GET a MediaMTX list endpoint, returning its `items` array.

        Always degrades to `[]`: an unreachable gateway must not break the
        caller's reconcile, and the warnings are throttled so a persistently
        down MediaMTX cannot flood the log.
        """
        if not self.admin_api_url:
            return []

        try:
            response = await self._client.get(
                f"{self.admin_api_url}{endpoint}", auth=self._auth()
            )
            response.raise_for_status()
            items = (response.json() or {}).get("items") or []
            # Some versions include nulls in items; drop anything unusable.
            return [item for item in items if isinstance(item, dict)]
        except Exception as exc:
            if self._is_timeout_or_network_error(exc):
                self._warn_throttled(
                    f"{label}_timeout",
                    "MediaMTX %s timeout/unreachable. admin_api=%s",
                    label,
                    self.admin_api_url,
                )
            elif isinstance(exc, httpx.HTTPStatusError):
                self._warn_throttled(
                    f"{label}_status",
                    "MediaMTX %s HTTP error. status=%s admin_api=%s",
                    label,
                    exc.response.status_code,
                    self.admin_api_url,
                )
            else:
                logger.exception("MediaMTX %s failed", label)
            return []

    async def list_configured_paths(self) -> List[Dict[str, Any]]:
        """Paths configured via `/v3/config/paths/add` — the source of truth for
        which cameras this gateway is meant to serve."""
        return await self._list_paths(
            "/v3/config/paths/list", label="list_configured_paths"
        )

    async def list_active_paths(self) -> List[Dict[str, Any]]:
        """Runtime paths, including `readers` (viewers) and traffic stats."""
        return await self._list_paths("/v3/paths/list", label="list_active_paths")

    @staticmethod
    def _count_reader_types(readers: Any) -> Dict[str, int]:
        """
        MediaMTX 'readers' is a list of objects like:
          { "type": "webRTCSession" | "rtspSession" | "hlsMuxer" | ..., "id": "..." }
        """
        out: Dict[str, int] = {}
        if not isinstance(readers, list):
            return out
        for r in readers:
            if not isinstance(r, dict):
                continue
            t = str(r.get("type") or "unknown")
            out[t] = out.get(t, 0) + 1
        return out

    async def list_webrtc_cameras(self, *, include_active: bool = True) -> List[Dict[str, Any]]:
        """The gateway's camera list: configured paths (the source of truth),
        optionally merged with runtime stats (active readers/viewers)."""
        cfg_paths = await self.list_configured_paths()

        active_by_name: Dict[str, Dict[str, Any]] = {}
        if include_active:
            for p in await self.list_active_paths():
                name = p.get("name")
                if isinstance(name, str) and name:
                    active_by_name[name] = p

        out: List[Dict[str, Any]] = []
        for c in cfg_paths:
            name = c.get("name")
            if not isinstance(name, str) or not name:
                continue

            active = active_by_name.get(name) or {}
            readers = active.get("readers") or []
            counts = self._count_reader_types(readers)

            out.append(
                {
                    "stream_key": name,
                    "source_url": c.get("source"),                 # what you set in ensure_stream()
                    "webrtc_url": self._derive_public_webrtc_url(name),
                    "max_readers": c.get("maxReaders"),
                    "source_on_demand": c.get("sourceOnDemand"),
                    "ready": active.get("ready"),
                    "bytes_received": active.get("bytesReceived"),
                    "bytes_sent": active.get("bytesSent"),
                    "readers": sum(counts.values()),
                    "reader_types": counts,
                    "webrtc_readers": counts.get("webRTCSession", 0),
                }
            )

        return out
