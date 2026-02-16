from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import httpx
import os 
from urllib.parse import quote
import logging
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
        self.admin_api_url = (os.getenv("WEBRTC_ADMIN_API_URL") or "https://noentrymtxfdxidm.centralus.azurecontainer.io:9997").rstrip("/")
        pub_host = os.getenv("PUBLIC_HOST", "localhost")
        pub_scheme = os.getenv("PUBLIC_SCHEME", "http")
        pub_port = os.getenv("WEBRTC_HTTP_PORT", "8889")
        self.public_base = (os.getenv("WEBRTC_PUBLIC_BASE_URL") or f"{pub_scheme}://{pub_host}:{pub_port}").rstrip("/")

        self.api_user = os.getenv("MTX_API_USER", "api")
        self.api_pass = os.getenv("MTX_API_PASS", "api_pass_123")

        self._client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))

    async def close(self) -> None:
        await self._client.aclose()
    
    def _auth(self) -> Tuple[str, str]:
        return (self.api_user, self.api_pass)

    def _derive_public_webrtc_url(self, stream_key: str) -> str:
        return f"{self.public_base}/{stream_key}"


    async def ensure_stream(self, *, stream_key: str, rtsp_url: str) -> Optional[str]:
        """
        Ensure stream exists in MediaMTX. Returns webrtc_url (stable).
        """
        if not self.admin_api_url:
            return self._derive_public_webrtc_url(stream_key)

        safe_name = quote(stream_key, safe="")
        add_url = f"{self.admin_api_url}/v3/config/paths/add/{safe_name}"
        payload = {"source": rtsp_url, "rtspTransport": "tcp"}

        # 1. Try Add
        try:
            r = await self._client.post(add_url, json=payload, auth=self._auth())
            if r.status_code == 200:
                return self._derive_public_webrtc_url(stream_key)
        except Exception:
            logger.warning("MediaMTX add request failed, trying patch or ignoring", exc_info=True)

        patch_url = f"{self.admin_api_url}/v3/config/paths/patch/{safe_name}"
        try:
            r = await self._client.patch(patch_url, json=payload, auth=self._auth())
            if r.status_code == 200:
                return self._derive_public_webrtc_url(stream_key)
        except Exception:
            logger.error("MediaMTX patch request failed", exc_info=True)
            
        return self._derive_public_webrtc_url(stream_key)

    async def update_stream(self, *, stream_key: str, rtsp_url: str) -> None:
        if not self.admin_api_url:
            return
            
        safe_name = quote(stream_key, safe="")
        url = f"{self.admin_api_url}/v3/config/paths/patch/{safe_name}"
        payload = {"source": rtsp_url}
        
        try:
            await self._client.patch(url, json=payload, auth=self._auth())
        except Exception:
             logger.error(f"MediaMTX update failed for {stream_key}", exc_info=True)

    async def delete_stream(self, *, stream_key: str) -> None:
        if not self.admin_api_url:
            return
            
        safe_name = quote(stream_key, safe="")
        url = f"{self.admin_api_url}/v3/config/paths/delete/{safe_name}"
        
        try:
            await self._client.delete(url, auth=self._auth())
        except Exception:
             logger.warning(f"MediaMTX delete failed for {stream_key}", exc_info=True)
