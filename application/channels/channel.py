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
import httpx
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, Optional,List, Tuple
from urllib.parse import urljoin
from application.channels.channel_config import VideoChannelConfig
logger = logging.getLogger(__name__)

_http = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=8.0, read=8.0, write=5.0, pool=8.0),
    limits=httpx.Limits(
        max_connections=50,
        max_keepalive_connections=20,
        keepalive_expiry=30.0,
    ),
)

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
        return await self.stream()
    
    async def stream(self) -> Optional[Dict[str, Any]]:
        if not self.config.enabled:
            return None
        if hasattr(self.config, "is_scheduled_now") and not self.config.is_scheduled_now():
            return None
        if not self.config.device_url:
            logger.warning("Jetson device_url not configured for camera=%s", self.key())
            return None

        urls = self.detection_urls()
        last_err_sig: Optional[str] = None

        for url in urls:
            try:
                timeout_s = float(self.config.request_timeout_s or 6.0)
                t = httpx.Timeout(timeout_s, connect=min(8.0, timeout_s), read=timeout_s, write=timeout_s, pool=timeout_s)
                r = await _http.get(url, timeout=t)
                if r.status_code in (404, 405):
                    last_err_sig = f"http:{r.status_code}:{url}"
                    continue
                r.raise_for_status()
                data = r.json()

                self._last_good_detection_url = url
                self._last_error_sig = None
                return data

            except (httpx.ConnectTimeout, httpx.ConnectError):
                last_err_sig = f"connect_error:{url}"
                break  # device likely down; don't try other paths
            except httpx.ReadTimeout:
                last_err_sig = f"read_timeout:{url}"
                # I'd continue here (server slow on that path; other path might work)
                continue
            except Exception as e:
                last_err_sig = f"exc:{type(e).__name__}:{url}"
                continue

        if last_err_sig and last_err_sig != self._last_error_sig:
            logger.warning("Jetson detection fetch failed camera=%s tried=%d last=%s", self.key(), len(urls), last_err_sig)
            self._last_error_sig = last_err_sig
        return None

    async def fetch_snapshot_bytes(self) -> Optional[Tuple[bytes, str]]:
        if not self.config.enabled:
            return None
        if hasattr(self.config, "is_scheduled_now") and not self.config.is_scheduled_now():
            return None
        if not self.config.device_url:
            return None

        timeout_s = float(self.config.request_timeout_s or 6.0)
        timeout = httpx.Timeout(
            timeout_s,
            connect=min(8.0, timeout_s),
            read=timeout_s,
            write=timeout_s,
            pool=timeout_s,
        )

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
                return None
            except Exception:
                continue

        return None
