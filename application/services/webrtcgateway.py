from typing import Any, Dict, List, Optional, Tuple
import httpx
import os 
from typing import Optional, Tuple,Literal

from urllib.parse import quote
import logging
import time
logger = logging.getLogger(__name__)

class WebRTCGatewayClient:
    """
    Provisions (or updates) RTSP->WebRTC streams on an Azure-hosted gateway (MediaMTX/go2rtc/etc).

    We support two modes:
      1) Admin API available -> call it to upsert streams.
      2) No admin API -> derive a stable public webrtc_url from WEBRTC_PUBLIC_BASE_URL + stream_key.

    Environment:
      - WEBRTC_ADMIN_API_URL (optional)
      - WEBRTC_ADMIN_UPSERT_PATH (default: /streams)
      - WEBRTC_ADMIN_UPDATE_PATH (default: /streams/{stream_key})
      - WEBRTC_ADMIN_DELETE_PATH (default: /streams/{stream_key})
      - WEBRTC_PUBLIC_BASE_URL (required for derivation if admin doesn't return a url)
      - WEBRTC_ADMIN_API_KEY (optional header: x-api-key)
    """

    def __init__(self):
        enabled_raw = str(os.getenv("WEBRTC_ADMIN_API_ENABLED", "true")).strip().lower()
        self.admin_api_enabled = enabled_raw in {"1", "true", "yes", "on"}

        self.admin_api_url = (
            os.getenv("WEBRTC_ADMIN_API_URL")
            or "https://noentrymtxfdxidm.centralus.azurecontainer.io:9997"
        ).rstrip("/")
        if not self.admin_api_enabled:
            self.admin_api_url = ""

        pub_host = os.getenv("PUBLIC_HOST", "localhost")
        pub_scheme = os.getenv("PUBLIC_SCHEME", "http")
        pub_port = os.getenv("WEBRTC_HTTP_PORT", "8889")
        self.public_base = (os.getenv("WEBRTC_PUBLIC_BASE_URL") or f"{pub_scheme}://{pub_host}:{pub_port}").rstrip("/")

        self.api_user = os.getenv("MTX_API_USER") or os.getenv("MEDIAMTX_API_USER", "api")
        self.api_pass = os.getenv("MTX_API_PASS") or os.getenv("MEDIAMTX_API_PASS", "api_pass_123")

        request_timeout_s = float(os.getenv("WEBRTC_ADMIN_TIMEOUT_S", "15"))
        connect_timeout_s = float(os.getenv("WEBRTC_ADMIN_CONNECT_TIMEOUT_S", "5"))
        self._warn_interval_s = float(os.getenv("WEBRTC_WARN_INTERVAL_S", "60"))
        self._last_warn: Dict[str, float] = {}

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(request_timeout_s, connect=connect_timeout_s)
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
        return f"{self.public_base}/{stream_key}"

    def _response_error_message(self, *, action: str, response: httpx.Response) -> str:
        body = (response.text or "").strip().replace("\n", " ")
        if len(body) > 200:
            body = body[:200] + "..."
        if body:
            return f"{action} returned HTTP {response.status_code}: {body}"
        return f"{action} returned HTTP {response.status_code}"

    async def ensure_stream(self, *, stream_key: str, rtsp_url: str) -> Optional[str]:
        """
        Ensure stream exists in MediaMTX. Returns webrtc_url (stable).
        """
        if not self.admin_api_url:
            return self._derive_public_webrtc_url(stream_key)

        safe_name = quote(stream_key, safe="")
        add_url = f"{self.admin_api_url}/v3/config/paths/add/{safe_name}"
        payload = {"source": rtsp_url, "rtspTransport": "tcp"}
        add_error: Optional[str] = None

        # 1. Try Add
        try:
            r = await self._client.post(add_url, json=payload, auth=self._auth())
            if r.status_code == 200:
                return self._derive_public_webrtc_url(stream_key)
            add_error = self._response_error_message(action="MediaMTX add", response=r)
            logger.warning("%s. stream_key=%s admin_api=%s", add_error, stream_key, self.admin_api_url)
        except Exception as exc:
            if self._is_timeout_or_network_error(exc):
                add_error = f"MediaMTX add timed out/unreachable: {type(exc).__name__}: {exc}"
                self._warn_throttled(
                    "ensure_stream_add_timeout",
                    "MediaMTX add timed out/unreachable. stream_key=%s admin_api=%s",
                    stream_key,
                    self.admin_api_url,
                )
            else:
                add_error = f"MediaMTX add request failed: {type(exc).__name__}: {exc}"
                logger.warning("MediaMTX add request failed, trying patch. stream_key=%s", stream_key, exc_info=True)

        patch_url = f"{self.admin_api_url}/v3/config/paths/patch/{safe_name}"
        patch_error: Optional[str] = None
        try:
            r = await self._client.patch(patch_url, json=payload, auth=self._auth())
            if r.status_code == 200:
                return self._derive_public_webrtc_url(stream_key)
            patch_error = self._response_error_message(action="MediaMTX patch", response=r)
            logger.error("%s. stream_key=%s admin_api=%s", patch_error, stream_key, self.admin_api_url)
        except Exception as exc:
            if self._is_timeout_or_network_error(exc):
                patch_error = f"MediaMTX patch timed out/unreachable: {type(exc).__name__}: {exc}"
                self._warn_throttled(
                    "ensure_stream_patch_timeout",
                    "MediaMTX patch timed out/unreachable. stream_key=%s admin_api=%s",
                    stream_key,
                    self.admin_api_url,
                )
            else:
                patch_error = f"MediaMTX patch request failed: {type(exc).__name__}: {exc}"
                logger.error("MediaMTX patch request failed. stream_key=%s", stream_key, exc_info=True)

        raise RuntimeError(
            "Failed to provision MediaMTX stream '{}'. add_error={}; patch_error={}".format(
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
        payload = {"source": rtsp_url}
        
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
