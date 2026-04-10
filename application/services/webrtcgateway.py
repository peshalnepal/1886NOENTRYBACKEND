from typing import Any, Dict, List, Optional, Tuple
import httpx
import os 
from typing import Optional, Tuple,Literal

from urllib.parse import quote
import logging
import time
logger = logging.getLogger(__name__)


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
    Provisions (or updates) RTSP->WebRTC streams on a MediaMTX gateway using WHEP protocol.

    WHEP (WebRTC HTTP Egress Protocol) Requirements:
      - MediaMTX must have WHEP protocol enabled
      - Streams are accessed via: {WEBRTC_PUBLIC_BASE_URL}/{stream_key}/whep
      - The frontend will POST an SDP offer to establish P2P WebRTC connection
    
    Two operational modes:
      1) Admin API available -> provisions streams via /v3/config/paths/add
      2) No admin API -> derives stable public WHEP URLs
    
    Required Environment Variables:
      - WEBRTC_PUBLIC_BASE_URL: Base URL for public WHEP access (e.g., https://mtx.example.com)
        If not set, defaults to http://localhost:8889
      - WEBRTC_ADMIN_API_URL: MediaMTX admin API URL (via Caddy reverse proxy)
        Defaults to https://noentrymtxfdxidm.centralus.azurecontainer.io
      - WEBRTC_ADMIN_API_ENABLED: Set to 'false' to skip stream provisioning (default: true)
      - MTX_API_USER or MEDIAMTX_API_USER: Admin API username (default: api)
      - MTX_API_PASS or MEDIAMTX_API_PASS: Admin API password (default: api_pass_123)
      - WEBRTC_ADMIN_TIMEOUT_S: Request timeout in seconds (default: 15)
      - WEBRTC_ADMIN_CONNECT_TIMEOUT_S: Connection timeout in seconds (default: 5)
      - WEBRTC_WARN_INTERVAL_S: Throttle warnings to once per N seconds (default: 60)
    
    MediaMTX Configuration Required:
      - WHEP protocol must be enabled in MediaMTX config
      - Ensure environment has proper STUN servers configured
      - Example rtspTransport: "tcp" for reliability
    """

    def __init__(self):
        enabled_raw = str(os.getenv("WEBRTC_ADMIN_API_ENABLED", "true")).strip().lower()
        self.admin_api_enabled = enabled_raw in {"1", "true", "yes", "on"}

        self.admin_api_url = (
            os.getenv("WEBRTC_ADMIN_API_URL")
            or "https://noentrymtxfdxidm.centralus.azurecontainer.io"
        ).rstrip("/")
        if not self.admin_api_enabled:
            self.admin_api_url = ""

        self.public_base = get_public_webrtc_base()

        self.api_user = os.getenv("MTX_API_USER") or os.getenv("MEDIAMTX_API_USER", "api")
        # Use 'is not None' to allow empty-string passwords (empty string IS a valid password)
        mtx_pass = os.getenv("MTX_API_PASS","api_pass_123")
        if mtx_pass is None:
            mtx_pass = os.getenv("MEDIAMTX_API_PASS","api_pass_123")
        if mtx_pass is None:
            mtx_pass = "api_pass_123"
        self.api_pass = mtx_pass

        request_timeout_s = float(os.getenv("WEBRTC_ADMIN_TIMEOUT_S", "15"))
        connect_timeout_s = float(os.getenv("WEBRTC_ADMIN_CONNECT_TIMEOUT_S", "5"))
        self._warn_interval_s = float(os.getenv("WEBRTC_WARN_INTERVAL_S", "60"))
        self._last_warn: Dict[str, float] = {}

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout_s, connect=connect_timeout_s),
            verify=False  # Disable SSL verification for self-signed certs
        )
        
        # Log configuration on initialization
        logger.info(
            "WebRTCGatewayClient initialized: admin_api_enabled=%s, admin_api_url=%s, public_base=%s",
            self.admin_api_enabled,
            self.admin_api_url if self.admin_api_enabled else "(disabled)",
            self.public_base
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

    async def ensure_stream(self, *, stream_key: str, rtsp_url: str) -> Optional[str]:
        """
        Ensure stream exists in MediaMTX. Returns stable WHEP URL.
        
        Flow:
        1. If admin API disabled, derive and return WHEP URL immediately
        2. If admin API enabled, provision stream via /v3/config/paths/add
        3. If add fails, try /v3/config/paths/patch
        4. Return WHEP URL: {public_base}/{stream_key}/whep
        
        Returns the public WHEP URL that frontend can use to connect.
        """
        whep_url = self._derive_public_webrtc_url(stream_key)
        
        if not self.admin_api_url:
            logger.debug("Admin API disabled, returning derived WHEP URL: %s", whep_url)
            return whep_url

        safe_name = quote(stream_key, safe="")
        add_url = f"{self.admin_api_url}/v3/config/paths/add/{safe_name}"
        payload = {"source": rtsp_url, "rtspTransport": "tcp", "sourceOnDemand": True}
        add_error: Optional[str] = None

        # 1. Try Add
        try:
            logger.debug("Attempting to provision stream: add_url=%s, stream_key=%s, rtsp_url=%s", 
                        add_url, stream_key, rtsp_url)
            r = await self._client.post(add_url, json=payload, auth=self._auth())
            if r.status_code == 200:
                logger.info("Stream provisioned successfully via add: stream_key=%s, whep_url=%s", 
                           stream_key, whep_url)
                return whep_url
            add_error = self._response_error_message(action="MediaMTX add", response=r)
            logger.warning("%s. stream_key=%s admin_api=%s", add_error, stream_key, self.admin_api_url)
        except Exception as exc:
            if self._is_timeout_or_network_error(exc):
                add_error = f"MediaMTX add timed out/unreachable: {type(exc).__name__}: {exc}"
                self._warn_throttled(
                    "ensure_stream_add_timeout",
                    "MediaMTX add timed out/unreachable. stream_key=%s admin_api=%s error=%s",
                    stream_key,
                    self.admin_api_url,
                    str(exc),
                )
            else:
                add_error = f"MediaMTX add request failed: {type(exc).__name__}: {exc}"
                logger.warning("MediaMTX add request failed, trying patch. stream_key=%s error=%s", 
                             stream_key, str(exc), exc_info=True)

        patch_url = f"{self.admin_api_url}/v3/config/paths/patch/{safe_name}"
        patch_error: Optional[str] = None
        try:
            logger.debug("Attempting to provision stream: patch_url=%s, stream_key=%s", 
                        patch_url, stream_key)
            r = await self._client.patch(patch_url, json=payload, auth=self._auth())
            if r.status_code == 200:
                logger.info("Stream provisioned successfully via patch: stream_key=%s, whep_url=%s", 
                           stream_key, whep_url)
                return whep_url
            patch_error = self._response_error_message(action="MediaMTX patch", response=r)
            logger.error("%s. stream_key=%s admin_api=%s", patch_error, stream_key, self.admin_api_url)
        except Exception as exc:
            if self._is_timeout_or_network_error(exc):
                patch_error = f"MediaMTX patch timed out/unreachable: {type(exc).__name__}: {exc}"
                self._warn_throttled(
                    "ensure_stream_patch_timeout",
                    "MediaMTX patch timed out/unreachable. stream_key=%s admin_api=%s error=%s",
                    stream_key,
                    self.admin_api_url,
                    str(exc),
                )
            else:
                patch_error = f"MediaMTX patch request failed: {type(exc).__name__}: {exc}"
                logger.error("MediaMTX patch request failed. stream_key=%s error=%s", 
                           stream_key, str(exc), exc_info=True)

        raise RuntimeError(
            "Failed to provision MediaMTX stream '{}'. add_error={}; patch_error={}. "
            "This likely means: (1) MediaMTX is unreachable, (2) credentials are wrong, "
            "(3) WHEP protocol not enabled, or (4) stream format invalid.".format(
                stream_key,
                add_error or "unknown",
                patch_error or "unknown",
            )
        )

    async def update_stream(self, *, stream_key: str, rtsp_url: str) -> None:
        if not self.admin_api_url:
            return
            
        safe_name = quote(stream_key, safe="")
        url = f"{self.admin_api_url}/v3/config/paths/patch/{safe_name}"
        payload = {"source": rtsp_url, "rtspTransport": "tcp", "sourceOnDemand": True}
        
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

    async def delete_stream(self, *, stream_key: str) -> None:
        if not self.admin_api_url:
            return
            
        safe_name = quote(stream_key, safe="")
        url = f"{self.admin_api_url}/v3/config/paths/delete/{safe_name}"
        
        try:
            await self._client.delete(url, auth=self._auth())
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


    async def list_configured_paths(self) -> List[Dict[str, Any]]:
        """
        List configured paths in MediaMTX (i.e., what you've added via /v3/config/paths/add).
        Returns the raw 'items' array (filtered for non-null).
        """
        if not self.admin_api_url:
            return []

        url = f"{self.admin_api_url}/v3/config/paths/list"
        try:
            r = await self._client.get(url, auth=self._auth())
            r.raise_for_status()
            data = r.json() or {}
            items = data.get("items") or []
            # Some versions may include nulls in items; filter them out
            return [it for it in items if isinstance(it, dict)]
        except Exception as exc:
            if self._is_timeout_or_network_error(exc):
                self._warn_throttled(
                    "list_configured_paths_timeout",
                    "MediaMTX list_configured_paths timeout/unreachable. admin_api=%s",
                    self.admin_api_url,
                )
            elif isinstance(exc, httpx.HTTPStatusError):
                self._warn_throttled(
                    "list_configured_paths_status",
                    "MediaMTX list_configured_paths HTTP error. status=%s admin_api=%s",
                    exc.response.status_code,
                    self.admin_api_url,
                )
            else:
                logger.exception("MediaMTX list_configured_paths failed")
            return []

    async def list_active_paths(self) -> List[Dict[str, Any]]:
        """
        List active paths (runtime) including 'readers' (viewers) and other stats.
        """
        if not self.admin_api_url:
            return []

        url = f"{self.admin_api_url}/v3/paths/list"
        try:
            r = await self._client.get(url, auth=self._auth())
            r.raise_for_status()
            data = r.json() or {}
            items = data.get("items") or []
            return [it for it in items if isinstance(it, dict)]
        except Exception as exc:
            if self._is_timeout_or_network_error(exc):
                self._warn_throttled(
                    "list_active_paths_timeout",
                    "MediaMTX list_active_paths timeout/unreachable. admin_api=%s",
                    self.admin_api_url,
                )
            elif isinstance(exc, httpx.HTTPStatusError):
                self._warn_throttled(
                    "list_active_paths_status",
                    "MediaMTX list_active_paths HTTP error. status=%s admin_api=%s",
                    exc.response.status_code,
                    self.admin_api_url,
                )
            else:
                logger.exception("MediaMTX list_active_paths failed")
            return []

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
        """
        Your "camera list" for the MediaMTX server:
        - Uses config list as the source of truth (configured paths)
        - Optionally merges in runtime stats (active readers/viewers)
        """
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
                    "rtsp_url": c.get("source"),                 # what you set in ensure_stream()
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
