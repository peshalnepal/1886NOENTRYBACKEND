from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from uuid import UUID

from domain.channel import ChannelConfig


@dataclass
class Template:
    """
    The platform-agnostic blueprint for an Agent. It holds all
    user-configurable settings in a generic format. It is abstract and does
    not know about specific channel implementations.
    """

    id: UUID
    configs: ChannelConfig

