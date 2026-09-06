"""Azure Blob storage for alert snapshot images."""

import base64
import binascii
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobSasPermissions, ContentSettings, generate_blob_sas
from azure.storage.blob.aio import BlobServiceClient

from application.services.storage_common import parse_connection_string as _parse_connection_string
from core.env import env_int

_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+);base64,(?P<data>.+)$",
    re.DOTALL,
)
_IMAGE_MIME_EXTENSIONS = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}

# Payload sub-dicts that may carry the image reference, in priority order.
_IMAGE_PAYLOAD_SECTIONS = ("extra", "msg")


def extract_image_storage_key(payload: Any) -> str:
    """The alert's blob key, from wherever the payload happens to carry it."""
    if not isinstance(payload, dict):
        return ""

    for section in _IMAGE_PAYLOAD_SECTIONS:
        block = payload.get(section)
        if isinstance(block, dict):
            key = str(block.get("image_storage_key") or "").strip()
            if key:
                return key
    return ""


def strip_image_fields(payload: Any) -> Any:
    """Copy of `payload` with the inline image references removed."""
    if not isinstance(payload, dict):
        return payload

    out = dict(payload)
    for section, drop_keys in (
        ("msg", ("image_url", "image_storage_key")),
        ("extra", ("image_url", "image_storage_key", "thumbnail_url", "image")),
    ):
        block = out.get(section)
        if isinstance(block, dict):
            out[section] = {k: v for k, v in block.items() if k not in drop_keys}
    return out


class AlertImageStorageService:
    SAS_TTL_HOURS = env_int("ALERT_IMAGE_SAS_TTL_HOURS", 720, minimum=1)

    def __init__(self) -> None:
        self.connection_string = (
            os.getenv("ALERT_IMAGE_BLOB_CONNECTION_STRING")
            or os.getenv("VIDEO_CLIP_BLOB_CONNECTION_STRING")
            or ""
        ).strip()
        self.container_name = (
            os.getenv("ALERT_IMAGE_BLOB_CONTAINER")
            or "alert-images"
        ).strip() or "alert-images"
        self.enabled = bool(self.connection_string)
        self._blob_service: Optional[BlobServiceClient] = None
        self._container_ready = False

        conn_parts = _parse_connection_string(self.connection_string)
        self._sas_account_name = conn_parts.get("accountname", "")
        self._sas_account_key = conn_parts.get("accountkey", "")

    async def close(self) -> None:
        if self._blob_service is not None:
            await self._blob_service.close()
            self._blob_service = None
        self._container_ready = False

    async def delete_blob(self, *, blob_name: str) -> bool:
        blob_key = str(blob_name or "").strip()
        if not blob_key or not self.enabled:
            return False

        blob_service = await self._get_blob_service()
        blob = blob_service.get_blob_client(container=self.container_name, blob=blob_key)
        try:
            await blob.delete_blob(delete_snapshots="include")
        except ResourceNotFoundError:
            return False
        return True

    async def store_image_data_url(
        self,
        *,
        image_data_url: str,
        camera_uuid: str,
        ts_ms: int,
    ) -> Optional[Dict[str, str]]:
        if not self.enabled:
            return None

        parsed = self._decode_data_url(image_data_url)
        if parsed is None:
            return None

        content_type, payload, ext = parsed
        ts_dt = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
        blob_name = self._build_storage_key(
            camera_uuid=str(camera_uuid),
            ts_dt=ts_dt,
            ext=ext,
        )
        blob_url = await self._upload_blob(
            blob_name=blob_name,
            payload=payload,
            content_type=content_type,
        )
        return {
            "image_url": blob_url,
            "image_storage_key": blob_name,
        }

    async def _get_blob_service(self) -> BlobServiceClient:
        if self._blob_service is None:
            self._blob_service = BlobServiceClient.from_connection_string(self.connection_string)
        if not self._container_ready:
            container = self._blob_service.get_container_client(self.container_name)
            try:
                await container.create_container()
            except ResourceExistsError:
                pass
            self._container_ready = True
        return self._blob_service

    def _decode_data_url(self, raw: str) -> Optional[Tuple[str, bytes, str]]:
        text = str(raw or "").strip()
        if not text.startswith("data:"):
            return None

        match = _DATA_URL_RE.match(text)
        if not match:
            return None

        content_type = match.group("mime").lower()
        ext = _IMAGE_MIME_EXTENSIONS.get(content_type)
        if not ext:
            return None

        try:
            payload = base64.b64decode(match.group("data"), validate=True)
        except (ValueError, binascii.Error):
            return None
        if not payload:
            return None

        return content_type, payload, ext

    def _build_storage_key(self, *, camera_uuid: str, ts_dt: datetime, ext: str) -> str:
        day = ts_dt.astimezone(timezone.utc).strftime("%Y/%m/%d")
        token = uuid.uuid4().hex
        return f"alerts/{camera_uuid}/{day}/{token}.{ext}"

    def _signed_url(self, *, blob_name: str, blob_url: str) -> str:
        if not self._sas_account_name or not self._sas_account_key:
            return blob_url

        token = generate_blob_sas(
            account_name=self._sas_account_name,
            container_name=self.container_name,
            blob_name=blob_name,
            account_key=self._sas_account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=self.SAS_TTL_HOURS),
        )
        return f"{blob_url}?{token}" if token else blob_url

    async def _upload_blob(
        self,
        *,
        blob_name: str,
        payload: bytes,
        content_type: str,
    ) -> str:
        blob_service = await self._get_blob_service()
        blob = blob_service.get_blob_client(container=self.container_name, blob=blob_name)
        await blob.upload_blob(
            payload,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
        return self._signed_url(blob_name=blob_name, blob_url=blob.url)
