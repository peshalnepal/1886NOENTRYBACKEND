"""MiniPC settings: the single config for the whole service, discovery included.

Read from the environment and `NVR/.env` (or the file named by NVR_ENV_FILE).
Real environment variables win over the file.
"""

import ipaddress
import os
import re
from pathlib import Path
from typing import List, Literal, Optional, Tuple
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = os.getenv("NVR_ENV_FILE") or Path(__file__).resolve().parent / ".env"


def parse_channels(spec: str) -> List[int]:
    """'1-4,8' -> [1, 2, 3, 4, 8]. Raises ValueError on a malformed spec."""
    channels = set()
    for item in filter(None, (part.strip() for part in spec.split(","))):
        first, _, last = item.partition("-")
        first, last = int(first), int(last or first)
        if not 1 <= first <= last <= 512:
            raise ValueError("invalid channel range {!r}".format(item))
        channels.update(range(first, last + 1))
    return sorted(channels)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    # This service's own HTTP API and database (not the physical NVR_* box below).
    minipc_db_path: str = "/var/lib/nvr/nvr.sqlite3"
    minipc_bind_host: str = "0.0.0.0"
    minipc_port: int = Field(8000, ge=1, le=65535)

    jetson_base_url: AnyHttpUrl = "http://127.0.0.1:8080"
    jetson_timeout_s: float = Field(30.0, ge=1.0)

    # MediaMTX relay on this MiniPC. The public base must be reachable from cloud;
    # the LAN base points to MediaMTX on this MiniPC (defaults to the public base).
    mediamtx_api_url: AnyHttpUrl = "http://127.0.0.1:9997"
    mediamtx_api_user: str = ""
    mediamtx_api_password: str = ""
    nvr_public_rtsp_base_url: str
    mediamtx_lan_rtsp_base_url: str = ""

    recording_enabled: bool = True
    recording_playback_url: AnyHttpUrl = "http://127.0.0.1:9996"
    recording_directory: str = "/var/lib/nvr/recordings"
    recording_retention: str = "168h"
    recording_clip_directory: str = "/var/lib/nvr/clips"
    recording_clip_retention_s: int = Field(604800, ge=120)
    recording_max_exports: int = Field(2, ge=1, le=8)
    recording_max_pending: int = Field(16, ge=1, le=128)
    recording_max_clip_bytes: int = Field(536870912, ge=1048576)
    recording_public_base_url: str = ""
    recording_api_key: str = ""

    @field_validator("recording_public_base_url")
    @classmethod
    def _recording_public_url(cls, value: str) -> str:
        if value:
            parts = urlsplit(value)
            if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.query or parts.fragment:
                raise ValueError("must be an HTTP(S) base URL without credentials or query")
        return value.rstrip("/")

    @field_validator("recording_retention")
    @classmethod
    def _recording_retention(cls, value: str) -> str:
        if not re.fullmatch(r"[1-9][0-9]*(?:h|m|s)", value):
            raise ValueError("use a positive MediaMTX duration, for example 168h")
        return value

    discovery_enabled: bool = True
    discovery_interval_s: int = Field(60, ge=10)
    # Avoid marking cameras missing after a single dropped probe.
    discovery_miss_threshold: int = Field(2, ge=2)
    # Probe the LAN for direct cameras (WS-Discovery + subnet sweep).
    discovery_local_enabled: bool = True
    # Comma-separated; empty = the /24 of this host's IPv4. /16 at most.
    discovery_subnets: str = ""

    # Direct LAN cameras.
    hik_username: str = "admin"
    hik_password: str = ""
    hik_rtsp_port: int = Field(554, ge=1, le=65535)
    hik_http_port: int = Field(80, ge=1, le=65535)
    hik_rtsp_stream: int = Field(2, ge=1, le=3)  # 1 = main, 2 = sub

    # The tower's one physical NVR. Empty host = the MiniPC is the NVR.
    nvr_host: str = ""
    nvr_rtsp_port: int = Field(554, ge=1, le=65535)
    nvr_http_port: int = Field(80, ge=1, le=65535)
    # Empty = every enabled channel ISAPI reports. Set (e.g. "1-16") = only these
    # channels, and they stay candidates over RTSP even when ISAPI is unreachable.
    nvr_channels: str = ""
    # Empty username = use the HIK_* account.
    nvr_username: str = ""
    nvr_password: str = ""
    nvr_channel_offset: int = Field(0, ge=0, le=64)

    # Which route wins when one camera is reachable directly and via the NVR.
    source_preference: Literal["direct", "nvr"] = "direct"

    # Frame verification (ffmpeg decodes one frame per candidate route).
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    verify_timeout_s: float = Field(12.0, ge=1.0)
    verify_concurrency: int = Field(4, ge=1, le=64)
    # Recorders cap concurrent RTSP sessions; keep per-host probes low.
    verify_per_host: int = Field(2, ge=1, le=16)

    @field_validator("nvr_public_rtsp_base_url", "mediamtx_lan_rtsp_base_url")
    @classmethod
    def _rtsp_origin(cls, value: str) -> str:
        if value:
            parts = urlsplit(value)
            if (parts.scheme not in {"rtsp", "rtsps"} or not parts.hostname
                    or parts.query or parts.fragment or parts.path not in {"", "/"}):
                raise ValueError("must be an RTSP(S) origin without a path or query")
            parts.port  # raises ValueError for an out-of-range port
        return value

    @field_validator("nvr_host")
    @classmethod
    def _host(cls, value: str) -> str:
        value = value.strip()
        if value and not re.fullmatch(r"[\w.-]+", value, re.ASCII):
            raise ValueError("must be a bare hostname or IP address")
        return value

    @field_validator("nvr_channels")
    @classmethod
    def _channels(cls, value: str) -> str:
        parse_channels(value)
        return value

    @field_validator("discovery_subnets")
    @classmethod
    def _subnets(cls, value: str) -> str:
        for subnet in filter(None, (s.strip() for s in value.split(","))):
            if ipaddress.IPv4Network(subnet, strict=False).prefixlen < 16:
                raise ValueError("subnet {} is larger than /16".format(subnet))
        return value

    @model_validator(mode="after")
    def _channels_need_host(self):
        if self.nvr_channels and not self.nvr_host:
            raise ValueError("NVR_CHANNELS is set but NVR_HOST is empty")
        if not self.nvr_public_rtsp_base_url:
            raise ValueError("NVR_PUBLIC_RTSP_BASE_URL is required")
        return self

    @property
    def nvr_mode(self) -> str:
        return "hikvision" if self.nvr_host else "minipc"

    @property
    def camera_account(self) -> Tuple[str, str]:
        return self.hik_username, self.hik_password

    @property
    def nvr_account(self) -> Tuple[str, str]:
        return (self.nvr_username, self.nvr_password) if self.nvr_username else self.camera_account

    @property
    def nvr_channel_list(self) -> Optional[List[int]]:
        return parse_channels(self.nvr_channels) or None

    @property
    def subnet_list(self) -> List[str]:
        return [s.strip() for s in self.discovery_subnets.split(",") if s.strip()]
