import uuid
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import AsyncGenerator, Optional,Literal,Tuple

# Import the base ChannelEvent for generic type hinting
from domain.events import ChannelEvent


class ChannelConfig(ABC):
    """
    Abstract base class for channel-specific configurations. This serves as a
    marker interface for all concrete channel config implementations.
    """

    channel_id: str
    camera_uuid: Optional[uuid.UUID]
    rtsp_url: str
    enabled: bool
    sample_fps: float
    decode_backend: Literal["gstreamer", "opencv"]
    resize: Optional[Tuple[int, int]]
    reconnect_base_ms: int
    reconnect_max_ms: int
    emit_format: Literal["raw", "jpeg"]
    jpeg_quality: int



class Channel(ABC):
    """
    An abstract interface for a communication channel.

    This defines the contract for any class that handles the real-time,
    bidirectional communication with an external agent service (like OpenAI
    or VAPI). It operates by processing and yielding generic ChannelEvents.
    """

    @abstractmethod
    async def stream(self, event: ChannelEvent) -> AsyncGenerator[ChannelEvent, None]:
        """
        Processes an incoming ChannelEvent and streams back resulting events.

        This generic method allows different channel implementations to handle
        various event types (e.g., user messages, system webhooks) and yield
        any number of corresponding response events.

        Args:
            event: A ChannelEvent subclass representing the incoming data.

        Yields:
            An asynchronous generator of ChannelEvent objects.
        """
        # This is an abstract method; the yield is needed to satisfy the
        # type checker for an async generat or.
        yield