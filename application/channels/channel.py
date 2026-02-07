# agents/application/channels/channel.py
"""
Azure-side Camera Channel (NO RTSP ingest).

Each channel represents one camera and knows:
- camera_uuid
- rtsp_url
- webrtc_url (immutable in edits)
- site_uuid / device_uuid
- device_url (Jetson base URL)

It can pull latest detections from Jetson:
GET {device_url}/detection/{camera_uuid}

This file intentionally contains NO OpenCV/GStreamer code.
"""

import asyncio
import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, Optional,List
from urllib.parse import urljoin
from application.channels.channel_config import VideoChannelConfig
logger = logging.getLogger(__name__)


async def _run_blocking(fn, *args, **kwargs):
    """Python 3.7+ friendly replacement for asyncio.to_thread()."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

class VideoChannel:
    """
    A camera "channel" that pulls detection metadata from Jetson.
    """

    def __init__(self, config: VideoChannelConfig):
        self.config = config
        self._last_good_detection_url: Optional[str] = None
        self._last_error_sig: Optional[str] = None

    def key(self) -> str:
        return str(self.config.camera_uuid)
    
    def cam_url(self)->str:
        return str(self.config.webrtc_url)

    def _build_detection_url(self, path_template: str) -> str:
        base = (self.config.device_url or "").rstrip("/") + "/"
        path = str(path_template).format(camera_uuid=str(self.config.camera_uuid)).lstrip("/")
        # urljoin needs base to end with '/'
        return urljoin(base, path)

    def detection_urls(self) -> List[str]:
        templates: List[str] = []

        preferred_tpl = getattr(self.config, "detection_path_template", None)
        if preferred_tpl:
            templates.append(str(preferred_tpl))

        # Compatibility with different Jetson services.
        templates.extend(
            [
                "/detections/{camera_uuid}",
                "/detection/{camera_uuid}",
                "/cameras/{camera_uuid}/latest",
            ]
        )

        urls = [self._build_detection_url(tpl) for tpl in templates]
        if self._last_good_detection_url:
            urls = [self._last_good_detection_url] + urls

        # Preserve order while removing duplicates.
        return list(dict.fromkeys(urls))
        
    async def fetch_detection_json(self) -> Optional[Dict[str, Any]]:
        return await self.stream()

    async def stream(self) -> Optional[Dict[str, Any]]:
        """
        Returns parsed JSON (dict) or None on network/parse errors.
        """
        if not self.config.enabled:
            return None
        if not self.config.device_url:
            logger.warning("Jetson device_url not configured for camera=%s", self.key())
            return None

        def _http_get_json(url: str):
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=float(self.config.request_timeout_s)) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8", errors="replace"))

        urls = self.detection_urls()
        last_err_sig: Optional[str] = None

        for url in urls:
            try:
                data = await _run_blocking(_http_get_json, url)
                self._last_good_detection_url = url
                self._last_error_sig = None
                return data
            except urllib.error.HTTPError as e:
                # 404/405 usually mean "wrong endpoint path"; try next candidate.
                last_err_sig = "http:{}:{}".format(getattr(e, "code", "unknown"), url)
                continue
            except urllib.error.URLError as e:
                last_err_sig = "url:{}:{}".format(str(e), url)
                continue
            except Exception as e:
                last_err_sig = "exc:{}:{}".format(type(e).__name__, url)
                continue

        if last_err_sig and last_err_sig != self._last_error_sig:
            logger.warning(
                "Jetson detection fetch failed camera=%s tried=%d last=%s",
                self.key(),
                len(urls),
                last_err_sig,
            )
            self._last_error_sig = last_err_sig
        return None
