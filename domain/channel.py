from abc import ABC, abstractmethod
from typing import AsyncGenerator

from domain.events import ChannelEvent


class Channel(ABC):
    """Contract for real-time, bidirectional communication with an external
    agent service. Implementations consume a `ChannelEvent` and stream back any
    number of resulting events."""

    @abstractmethod
    async def stream(self, event: ChannelEvent) -> AsyncGenerator[ChannelEvent, None]:
        # The bare yield is what makes this an async generator for type checkers.
        yield
