import asyncio
import json
import httpx
import logging
from typing import Any, AsyncGenerator, Dict, Optional, List, Tuple
from urllib.parse import urljoin
from application.channels.channel_config import VideoChannelConfig
from core.env import env_bool

logger = logging.getLogger(__name__)

_EDGE_HTTP_TRUST_ENV = env_bool(
    "EDGE_CHANNEL_HTTP_TRUST_ENV",
    env_bool("EDGE_HTTP_TRUST_ENV", False),
)

_http = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=3.0, read=3.0, write=3.0, pool=3.0),
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=40,
        keepalive_expiry=30.0,
    ),
    trust_env=_EDGE_HTTP_TRUST_ENV,
)

class VideoChannel:
    def __init__(self, config: VideoChannelConfig):
        self.config = config
        self._last_good_detection_url: Optional[str] = None
        self._last_good_stream_url: Optional[str] = None
        self._last_error_sig: Optional[str] = None
        self._device_reachable: bool = True

    def key(self) -> str:
        return str(self.config.camera_uuid)

    def _build_detection_url(self, path_template: str) -> str:
        base = (self.config.device_url or "").rstrip("/") + "/"
        path = str(path_template).format(camera_uuid=str(self.config.camera_uuid)).lstrip("/")
        return urljoin(base, path)

    def detection_urls(self) -> List[str]:
        preferred_tpl = str(getattr(self.config, "detection_path_template", None) or "").strip()
        templates: List[str] = [
            "/api/cameras/{camera_uuid}/latest",
            "/api/detections/{camera_uuid}",
        ]
        if preferred_tpl:
            templates.append(preferred_tpl)
        templates.extend([
            "/cameras/{camera_uuid}/latest",
            "/api/detection/{camera_uuid}",
            "/detections/{camera_uuid}",
            "/detection/{camera_uuid}",
        ])

        urls = [self._build_detection_url(tpl) for tpl in templates]
        if self._last_good_detection_url:
            urls.insert(0, self._last_good_detection_url)
        return list(dict.fromkeys(urls))

    def detection_stream_urls(self) -> List[str]:
        templates = [
            "/api/cameras/{camera_uuid}/detections/stream",
            "/cameras/{camera_uuid}/detections/stream",
        ]
        urls = [self._build_detection_url(tpl) for tpl in templates]
        if self._last_good_stream_url:
            urls.insert(0, self._last_good_stream_url)
        return list(dict.fromkeys(urls))

    def _detection_enabled(self) -> bool:
        return getattr(self.config, "detection_enabled", True) is not False

    def _request_timeout(self, *, minimum: float = 0.0) -> httpx.Timeout:
        timeout_s = max(
            minimum,
            float(getattr(self.config, "request_timeout_s", 3.0) or 3.0),
        )
        return httpx.Timeout(
            timeout_s,
            connect=min(3.0, timeout_s),
            read=timeout_s,
            write=timeout_s,
            pool=timeout_s,
        )

    def snapshot_urls(self) -> List[str]:
        base = str(self.config.device_url or "").rstrip("/")
        if not base:
            return []

        camera_id = str(self.config.camera_uuid)
        urls = [f"{base}/cameras/{camera_id}/snapshot.jpg"]
        if not base.endswith("/api"):
            urls.insert(0, f"{base}/api/cameras/{camera_id}/snapshot.jpg")
        return list(dict.fromkeys(urls))
        
    async def fetch_detection_json(self) -> Optional[Dict[str, Any]]:
        """
        Keep one-shot GET for refresh/fallback paths.
        """
        if not self._detection_enabled():
            return None
        if not self.config.device_url:
            logger.warning("Jetson device_url not configured for camera=%s", self.key())
            return None

        urls = self.detection_urls()
        last_err_sig: Optional[str] = None
        attempted = 0

        for url in urls:
            attempted += 1
            try:
                r = await _http.get(url, timeout=self._request_timeout())
                if r.status_code in (404, 405):
                    last_err_sig = f"http:{r.status_code}:{url}"
                    continue
                r.raise_for_status()

                data = r.json()
                self._last_good_detection_url = url
                self._last_error_sig = None
                self._device_reachable = True
                return data

            except (httpx.ConnectTimeout, httpx.ConnectError, httpx.PoolTimeout):
                last_err_sig = f"connect_error:{url}"
                self._device_reachable = False
                break
            except httpx.ReadTimeout:
                last_err_sig = f"read_timeout:{url}"
                self._device_reachable = False
                break
            except Exception as e:
                last_err_sig = f"exc:{type(e).__name__}:{url}"
                continue

        if last_err_sig and last_err_sig != self._last_error_sig:
            logger.warning(
                "Jetson latest fetch failed camera=%s tried=%d last=%s",
                self.key(),
                attempted,
                last_err_sig,
            )
            self._last_error_sig = last_err_sig
        return None

    async def stream_detections(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Long-lived SSE reader. Yields detection payloads from Jetson.
        Reconnect is handled by ModelPipeline, not here.
        """
        if not self._detection_enabled():
            return
        if not self.config.device_url:
            logger.warning("Jetson device_url not configured for camera=%s", self.key())
            return

        urls = self.detection_stream_urls()
        last_err_sig: Optional[str] = None
        attempted = 0

        for url in urls:
            attempted += 1
            try:
                # Jetson SSE emits keepalive comments every ~1s, so a finite
                # read timeout is okay and helps detect dead sockets.
                async with _http.stream(
                    "GET",
                    url,
                    timeout=self._request_timeout(minimum=15.0),
                    headers={"Accept": "text/event-stream"},
                ) as r:
                    if r.status_code in (404, 405):
                        last_err_sig = f"http:{r.status_code}:{url}"
                        continue
                    r.raise_for_status()

                    self._last_good_stream_url = url
                    self._last_error_sig = None
                    self._device_reachable = True

                    data_lines: List[str] = []

                    async for raw_line in r.aiter_lines():
                        line = (raw_line or "").strip()

                        # event boundary
                        if not line:
                            if not data_lines:
                                continue
                            try:
                                payload = json.loads("\n".join(data_lines))
                            except Exception:
                                data_lines = []
                                continue
                            data_lines = []
                            if isinstance(payload, dict):
                                yield payload
                            continue

                        # SSE keepalive/comment
                        if line.startswith(":"):
                            continue

                        if line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())

                    # stream closed normally; let outer loop reconnect
                    return

            except asyncio.CancelledError:
                raise
            except (httpx.ConnectTimeout, httpx.ConnectError, httpx.PoolTimeout):
                last_err_sig = f"connect_error:{url}"
                self._device_reachable = False
                break
            except httpx.ReadTimeout:
                last_err_sig = f"read_timeout:{url}"
                self._device_reachable = False
                break
            except Exception as e:
                last_err_sig = f"exc:{type(e).__name__}:{url}"
                continue

        if last_err_sig and last_err_sig != self._last_error_sig:
            logger.warning(
                "Jetson detection stream failed camera=%s tried=%d last=%s",
                self.key(),
                attempted,
                last_err_sig,
            )
            self._last_error_sig = last_err_sig


    async def fetch_snapshot_bytes(self) -> Optional[Tuple[bytes, str]]:
        if not self._detection_enabled():
            return None
        if not self.config.device_url:
            return None

        timeout = self._request_timeout()

        for url in self.snapshot_urls():
            try:
                resp = await _http.get(
                    url,
                    timeout=timeout,
                    headers={"Accept": "image/jpeg,image/*;q=0.9,*/*;q=0.1"},
                )
                if resp.status_code in (404, 405):
                    continue
                if resp.status_code >= 400:
                    continue

                payload = bytes(resp.content or b"")
                if not payload:
                    continue

                content_type = str(resp.headers.get("content-type") or "image/jpeg")
                return payload, content_type
            except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout):
                break
            except Exception:
                continue

        return None