"""WebSocket protocol definitions for streaming video transfer."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class MessageType(str, Enum):
    """WebSocket message types."""

    # Session lifecycle
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    SESSION_ABORT = "session_abort"

    # Frame transfer
    FRAME = "frame"
    FRAME_ACK = "frame_ack"

    # Signal data
    SIGNAL = "signal"
    SIGNAL_ACK = "signal_ack"

    # Processing
    PROCESS_START = "process_start"
    PROCESS_PROGRESS = "process_progress"
    PROCESS_COMPLETE = "process_complete"

    # Errors
    ERROR = "error"

    # Health
    PING = "ping"
    PONG = "pong"


class StreamMessage:
    """Base class for stream messages."""

    def __init__(self):
        self.type: MessageType = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StreamMessage:
        """Create message from dictionary."""
        msg_type = MessageType(data.get("type", ""))

        if msg_type == MessageType.SESSION_START:
            return SessionStartMessage.from_dict(data)
        elif msg_type == MessageType.SESSION_END:
            return SessionEndMessage.from_dict(data)
        elif msg_type == MessageType.FRAME:
            return FrameMessage.from_dict(data)
        elif msg_type == MessageType.SIGNAL:
            return SignalMessage.from_dict(data)
        elif msg_type == MessageType.ERROR:
            return ErrorMessage.from_dict(data)
        else:
            return StreamMessage(type=msg_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {"type": self.type.value}


@dataclass
class SessionStartMessage:
    """Session start message."""

    session_id: str
    fps: float
    width: int
    height: int
    expected_frames: Optional[int] = None
    video_format: str = "bgr24"  # OpenCV default
    type: MessageType = None

    def __post_init__(self):
        self.type = MessageType.SESSION_START

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SessionStartMessage:
        return cls(
            session_id=data["session_id"],
            fps=data["fps"],
            width=data["width"],
            height=data["height"],
            expected_frames=data.get("expected_frames"),
            video_format=data.get("video_format", "bgr24"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "expected_frames": self.expected_frames,
            "video_format": self.video_format,
        }


@dataclass
class SessionEndMessage:
    """Session end message."""

    session_id: str
    total_frames: int
    type: MessageType = None

    def __post_init__(self):
        self.type = MessageType.SESSION_END

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SessionEndMessage:
        return cls(
            session_id=data["session_id"],
            total_frames=data["total_frames"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "total_frames": self.total_frames,
        }


@dataclass
class FrameMessage:
    """Frame data message."""

    session_id: str
    frame_number: int
    timestamp: float
    encoding: str = "jpeg"  # jpeg, png, or raw
    quality: int = 85  # for jpeg
    size: int = 0  # bytes
    type: MessageType = None

    def __post_init__(self):
        self.type = MessageType.FRAME

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FrameMessage:
        return cls(
            session_id=data["session_id"],
            frame_number=data["frame_number"],
            timestamp=data["timestamp"],
            encoding=data.get("encoding", "jpeg"),
            quality=data.get("quality", 85),
            size=data.get("size", 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "encoding": self.encoding,
            "quality": self.quality,
            "size": self.size,
        }


@dataclass
class FrameAckMessage:
    """Frame acknowledgment message."""

    session_id: str
    frame_number: int
    status: str = "received"  # received, error, duplicate
    type: MessageType = None

    def __post_init__(self):
        self.type = MessageType.FRAME_ACK

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "frame_number": self.frame_number,
            "status": self.status,
        }


@dataclass
class SignalMessage:
    """Signal file data message."""

    session_id: str
    filename: str
    size: int
    type: MessageType = None

    def __post_init__(self):
        self.type = MessageType.SIGNAL

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SignalMessage:
        return cls(
            session_id=data["session_id"],
            filename=data["filename"],
            size=data["size"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "filename": self.filename,
            "size": self.size,
        }


@dataclass
class SignalAckMessage:
    """Signal acknowledgment message."""

    session_id: str
    status: str = "received"
    type: MessageType = None

    def __post_init__(self):
        self.type = MessageType.SIGNAL_ACK

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "status": self.status,
        }


@dataclass
class ProcessProgressMessage:
    """Processing progress update."""

    session_id: str
    stage: str  # fisheye, yolo, decision, overlay
    progress: float  # 0.0 to 1.0
    message: str = ""
    type: MessageType = None

    def __post_init__(self):
        self.type = MessageType.PROCESS_PROGRESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
        }


@dataclass
class ProcessCompleteMessage:
    """Processing complete message with results."""

    session_id: str
    result: Dict[str, Any]
    type: MessageType = None

    def __post_init__(self):
        self.type = MessageType.PROCESS_COMPLETE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "result": self.result,
        }


@dataclass
class ErrorMessage:
    """Error message."""

    session_id: Optional[str]
    error_code: str
    error_message: str
    recoverable: bool = False
    type: MessageType = None

    def __post_init__(self):
        self.type = MessageType.ERROR

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ErrorMessage:
        return cls(
            session_id=data.get("session_id"),
            error_code=data["error_code"],
            error_message=data["error_message"],
            recoverable=data.get("recoverable", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "recoverable": self.recoverable,
        }
