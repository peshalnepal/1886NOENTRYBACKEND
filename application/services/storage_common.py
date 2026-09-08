"""Shared helpers for the Azure blob storage services."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict

from azure.storage.blob import BlobSasPermissions, generate_blob_sas


def parse_connection_string(raw: str) -> Dict[str, str]:
    """Split an Azure storage connection string into its lower-cased keys."""
    parts: Dict[str, str] = {}
    for item in str(raw or "").split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parts[key.strip().lower()] = value.strip()
    return parts


def signed_blob_url(
    *,
    blob_url: str,
    blob_name: str,
    container_name: str,
    account_name: str,
    account_key: str,
    ttl_hours: int,
) -> str:
    if not account_name or not account_key:
        return blob_url

    token = generate_blob_sas(
        account_name=account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(hours=ttl_hours),
    )
    return f"{blob_url}?{token}" if token else blob_url
