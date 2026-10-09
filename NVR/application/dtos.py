"""Internal DTOs passed between the discovery adapter, the sweep and repositories."""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SourceKind = Literal["direct", "nvr"]


class SourceDTO(BaseModel):
    """One route to a camera, as discovery found it. `source_url` has no credentials."""

    model_config = ConfigDict(frozen=True)

    source_identity: str
    kind: SourceKind
    host: str
    rtsp_port: int
    channel_no: Optional[int] = None  # NVR input number; None for a direct camera
    source_url: str
    serial_number: Optional[str] = None
    mac_address: Optional[str] = None
    model: Optional[str] = None
    firmware: Optional[str] = None
    device_name: Optional[str] = None


class CandidateDTO(BaseModel):
    """One physical camera: every route that reaches it, best route first."""

    model_config = ConfigDict(frozen=True)

    sources: tuple[SourceDTO, ...] = Field(min_length=1)

    @property
    def dedupe_key(self) -> str:
        """Lowest serial key, else lowest MAC key, else lowest route identity."""
        for prefix, field in (("serial:", "serial_number"), ("mac:", "mac_address")):
            value = min(filter(None, (getattr(source, field) for source in self.sources)), default=None)
            if value:
                return prefix + value
        return min(self.source_identities)

    @property
    def source_identities(self) -> List[str]:
        return [source.source_identity for source in self.sources]


class NvrDTO(BaseModel):
    model_config = ConfigDict(frozen=True)

    host: str
    rtsp_port: int
    http_port: int
    origin: str  # "isapi" | "static" (NVR_CHANNELS set)
    channels_spec: Optional[str] = None
    # ISAPI answered this sweep. Otherwise presence comes from frame verification.
    responded: bool = False


class ScanResultDTO(BaseModel):
    nvr: Optional[NvrDTO] = None
    candidates: List[CandidateDTO] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
