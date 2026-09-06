"""Shared helpers for the Azure blob storage services."""

from __future__ import annotations

from typing import Dict


def parse_connection_string(raw: str) -> Dict[str, str]:
    """Split an Azure storage connection string into its lower-cased keys."""
    parts: Dict[str, str] = {}
    for item in str(raw or "").split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parts[key.strip().lower()] = value.strip()
    return parts
