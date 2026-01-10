"""Streaming session management."""
from __future__ import annotations

import asyncio
import io
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("streaming.session")


@dataclass
class FrameData:
    """Data for a single frame."""

    frame_number: int
    timestamp: float
    encoding: str
    data: bytes
    received_at: float = field(default_factory=time.time)
    decoded: bool = False
    _decoded_frame: Optional[np.ndarray] = None

    def decode(self) -> np.ndarray:
        """Decode frame data to numpy array."""
        if self.decoded and self._decoded_frame is not None:
            return self._decoded_frame

        try:
            if self.encoding in ("jpeg", "jpg", "png"):
                # Decode compressed image
                img = Image.open(io.BytesIO(self.data))
                # Convert to BGR for OpenCV compatibility
                frame = np.array(img)
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # RGB to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif len(frame.shape) == 2:
                    # Grayscale to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                self._decoded_frame = frame
                self.decoded = True
                return frame
            elif self.encoding == "raw":
                # Raw bytes (should be numpy array)
                # Format: height, width, channels encoded in first 12 bytes
                if len(self.data) < 12:
                    raise ValueError("Raw frame data too short")

                h = int.from_bytes(self.data[0:4], "little")
                w = int.from_bytes(self.data[4:8], "little")
                c = int.from_bytes(self.data[8:12], "little")

                expected_size = 12 + h * w * c
                if len(self.data) != expected_size:
                    raise ValueError(
                        f"Raw frame size mismatch: expected {expected_size}, got {len(self.data)}"
                    )

                frame_bytes = self.data[12:]
                frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h, w, c))

                self._decoded_frame = frame
                self.decoded = True
                return frame
            else:
                raise ValueError(f"Unknown encoding: {self.encoding}")

        except Exception as exc:
            logger.error(f"Failed to decode frame {self.frame_number}: {exc}")
            raise

    def memory_size(self) -> int:
        """Estimate memory usage in bytes."""
        size = len(self.data)
        if self._decoded_frame is not None:
            size += self._decoded_frame.nbytes
        return size


class StreamingSession:
    """Manages a single streaming session."""

    def __init__(
        self,
        session_id: str,
        fps: float,
        width: int,
        height: int,
        expected_frames: Optional[int] = None,
        max_frame_gap: int = 10,
        session_timeout: float = 300.0,  # 5 minutes
    ):
        self.session_id = session_id
        self.fps = fps
        self.width = width
        self.height = height
        self.expected_frames = expected_frames
        self.max_frame_gap = max_frame_gap
        self.session_timeout = session_timeout

        self.frames: Dict[int, FrameData] = {}
        self.signal_data: Optional[bytes] = None
        self.signal_filename: Optional[str] = None

        self.started_at = time.time()
        self.last_activity = time.time()
        self.completed = False
        self.aborted = False

        self.total_frames_received = 0
        self.duplicate_frames = 0
        self.out_of_order_frames = 0
        self.highest_frame_number = -1

        self._lock = asyncio.Lock()

    async def add_frame(self, frame_data: FrameData) -> Dict[str, any]:
        """
        Add a frame to the session.

        Returns:
            Status dict with validation results
        """
        async with self._lock:
            self.last_activity = time.time()

            frame_num = frame_data.frame_number

            # Check for duplicates
            if frame_num in self.frames:
                self.duplicate_frames += 1
                logger.warning(
                    f"Session {self.session_id}: Duplicate frame {frame_num}"
                )
                return {
                    "status": "duplicate",
                    "frame_number": frame_num,
                    "total_frames": len(self.frames),
                }

            # Check for out-of-order (informational, we still accept it)
            if frame_num < self.highest_frame_number:
                self.out_of_order_frames += 1
                logger.info(
                    f"Session {self.session_id}: Out-of-order frame {frame_num} "
                    f"(highest: {self.highest_frame_number})"
                )

            # Check for large gaps (potential issue)
            if self.highest_frame_number >= 0:
                gap = frame_num - self.highest_frame_number
                if gap > self.max_frame_gap:
                    logger.warning(
                        f"Session {self.session_id}: Large frame gap detected: "
                        f"{gap} frames between {self.highest_frame_number} and {frame_num}"
                    )

            # Store frame
            self.frames[frame_num] = frame_data
            self.total_frames_received += 1
            self.highest_frame_number = max(self.highest_frame_number, frame_num)

            return {
                "status": "received",
                "frame_number": frame_num,
                "total_frames": len(self.frames),
                "memory_mb": self.memory_usage() / 1024 / 1024,
            }

    async def set_signal_data(self, data: bytes, filename: str):
        """Set signal file data."""
        async with self._lock:
            self.signal_data = data
            self.signal_filename = filename
            self.last_activity = time.time()
            logger.info(
                f"Session {self.session_id}: Signal data received "
                f"({len(data)} bytes, {filename})"
            )

    def get_missing_frames(self) -> List[int]:
        """Get list of missing frame numbers."""
        if not self.frames:
            return []

        frame_numbers = sorted(self.frames.keys())
        min_frame = frame_numbers[0]
        max_frame = frame_numbers[-1]

        all_frames = set(range(min_frame, max_frame + 1))
        received_frames = set(frame_numbers)
        missing = sorted(all_frames - received_frames)

        return missing

    def validate(self) -> Dict[str, any]:
        """
        Validate session is ready for processing.

        Returns:
            Validation result with status and any issues
        """
        issues = []

        if not self.frames:
            issues.append("No frames received")

        if self.signal_data is None:
            issues.append("No signal data received")

        missing_frames = self.get_missing_frames()
        if missing_frames:
            if len(missing_frames) <= 5:
                issues.append(f"Missing frames: {missing_frames}")
            else:
                issues.append(
                    f"Missing {len(missing_frames)} frames "
                    f"(first: {missing_frames[0]}, last: {missing_frames[-1]})"
                )

        if self.expected_frames is not None:
            if len(self.frames) < self.expected_frames:
                issues.append(
                    f"Expected {self.expected_frames} frames, "
                    f"received {len(self.frames)}"
                )

        # Check for reasonable frame count
        if len(self.frames) < 10:
            issues.append(f"Too few frames: {len(self.frames)} (minimum 10)")

        # Check if session timed out
        if time.time() - self.last_activity > self.session_timeout:
            issues.append(
                f"Session timeout: no activity for "
                f"{time.time() - self.last_activity:.1f}s"
            )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "frame_count": len(self.frames),
            "missing_frames": len(missing_frames),
            "duplicate_frames": self.duplicate_frames,
            "out_of_order_frames": self.out_of_order_frames,
            "has_signal": self.signal_data is not None,
        }

    def write_video(self, output_path: Path) -> Path:
        """
        Write buffered frames to video file.

        Args:
            output_path: Path to write video file

        Returns:
            Path to written video file
        """
        if not self.frames:
            raise ValueError("No frames to write")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get sorted frame numbers
        frame_numbers = sorted(self.frames.keys())

        logger.info(
            f"Session {self.session_id}: Writing video to {output_path} "
            f"({len(frame_numbers)} frames, {self.fps} FPS)"
        )

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, self.fps, (self.width, self.height)
        )

        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {output_path}")

        try:
            # Write frames in order
            for frame_num in frame_numbers:
                frame_data = self.frames[frame_num]
                frame = frame_data.decode()

                # Validate frame dimensions
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    logger.warning(
                        f"Frame {frame_num} size mismatch: "
                        f"expected {self.width}x{self.height}, "
                        f"got {frame.shape[1]}x{frame.shape[0]} - resizing"
                    )
                    frame = cv2.resize(frame, (self.width, self.height))

                writer.write(frame)

            writer.release()
            logger.info(f"Session {self.session_id}: Video written successfully")

            return output_path

        except Exception as exc:
            writer.release()
            logger.error(f"Failed to write video: {exc}")
            raise

    def write_signal(self, output_path: Path) -> Path:
        """Write signal data to file."""
        if self.signal_data is None:
            raise ValueError("No signal data available")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(self.signal_data)

        logger.info(
            f"Session {self.session_id}: Signal file written to {output_path}"
        )

        return output_path

    def memory_usage(self) -> int:
        """Estimate total memory usage in bytes."""
        total = 0

        # Frame data
        for frame in self.frames.values():
            total += frame.memory_size()

        # Signal data
        if self.signal_data:
            total += len(self.signal_data)

        # Overhead (rough estimate)
        total += 1024 * len(self.frames)  # Dict overhead

        return total

    def cleanup(self):
        """Release memory."""
        logger.info(
            f"Session {self.session_id}: Cleaning up "
            f"({len(self.frames)} frames, {self.memory_usage() / 1024 / 1024:.1f} MB)"
        )
        self.frames.clear()
        self.signal_data = None

    def get_stats(self) -> Dict[str, any]:
        """Get session statistics."""
        duration = time.time() - self.started_at
        idle_time = time.time() - self.last_activity

        return {
            "session_id": self.session_id,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "expected_frames": self.expected_frames,
            "frames_received": len(self.frames),
            "total_frames_received": self.total_frames_received,
            "duplicate_frames": self.duplicate_frames,
            "out_of_order_frames": self.out_of_order_frames,
            "missing_frames": len(self.get_missing_frames()),
            "has_signal": self.signal_data is not None,
            "duration_seconds": duration,
            "idle_seconds": idle_time,
            "memory_mb": self.memory_usage() / 1024 / 1024,
            "completed": self.completed,
            "aborted": self.aborted,
        }


class SessionManager:
    """Manages multiple streaming sessions."""

    def __init__(
        self,
        max_sessions: int = 10,
        max_memory_mb: int = 2048,  # 2 GB
        cleanup_interval: float = 60.0,  # 1 minute
        session_timeout: float = 300.0,  # 5 minutes
    ):
        self.max_sessions = max_sessions
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cleanup_interval = cleanup_interval
        self.session_timeout = session_timeout

        self.sessions: Dict[str, StreamingSession] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def create_session(
        self,
        session_id: str,
        fps: float,
        width: int,
        height: int,
        expected_frames: Optional[int] = None,
    ) -> StreamingSession:
        """Create a new streaming session."""
        async with self._lock:
            # Check if session already exists
            if session_id in self.sessions:
                raise ValueError(f"Session {session_id} already exists")

            # Check session limit
            if len(self.sessions) >= self.max_sessions:
                # Try to cleanup old sessions first
                await self._cleanup_old_sessions()

                if len(self.sessions) >= self.max_sessions:
                    raise RuntimeError(
                        f"Maximum concurrent sessions reached ({self.max_sessions})"
                    )

            # Check memory limit
            total_memory = sum(s.memory_usage() for s in self.sessions.values())
            if total_memory > self.max_memory_bytes:
                await self._cleanup_old_sessions()

                total_memory = sum(s.memory_usage() for s in self.sessions.values())
                if total_memory > self.max_memory_bytes:
                    raise RuntimeError(
                        f"Maximum memory limit reached "
                        f"({total_memory / 1024 / 1024:.1f} MB)"
                    )

            # Create session
            session = StreamingSession(
                session_id=session_id,
                fps=fps,
                width=width,
                height=height,
                expected_frames=expected_frames,
                session_timeout=self.session_timeout,
            )

            self.sessions[session_id] = session

            logger.info(
                f"Created session {session_id} "
                f"({len(self.sessions)} active sessions)"
            )

            return session

    async def get_session(self, session_id: str) -> Optional[StreamingSession]:
        """Get a session by ID."""
        async with self._lock:
            return self.sessions.get(session_id)

    async def remove_session(self, session_id: str) -> bool:
        """Remove and cleanup a session."""
        async with self._lock:
            session = self.sessions.pop(session_id, None)
            if session:
                session.cleanup()
                logger.info(
                    f"Removed session {session_id} "
                    f"({len(self.sessions)} active sessions)"
                )
                return True
            return False

    async def _cleanup_old_sessions(self):
        """Remove timed out or completed sessions."""
        now = time.time()
        to_remove = []

        for session_id, session in self.sessions.items():
            idle_time = now - session.last_activity

            if session.completed or session.aborted:
                to_remove.append(session_id)
                logger.info(
                    "Cleaning up completed/aborted session: %s (frames=%d, signal=%s)",
                    session_id,
                    len(session.frames),
                    "yes" if session.signal_data else "no",
                )
            elif idle_time > self.session_timeout:
                to_remove.append(session_id)
                logger.warning(
                    "Cleaning up timed out session: %s (idle %.1fs, frames=%d, signal=%s)",
                    session_id,
                    idle_time,
                    len(session.frames),
                    "yes" if session.signal_data else "no",
                )

        for session_id in to_remove:
            await self.remove_session(session_id)

    async def start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is not None:
            logger.warning("Cleanup task already running")
            return

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    async with self._lock:
                        await self._cleanup_old_sessions()
                except asyncio.CancelledError:
                    logger.info("Cleanup task cancelled")
                    break
                except Exception as exc:
                    logger.exception(f"Error in cleanup task: {exc}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Started cleanup task")

    async def stop_cleanup_task(self):
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Stopped cleanup task")

    def get_stats(self) -> Dict[str, any]:
        """Get manager statistics."""
        total_memory = sum(s.memory_usage() for s in self.sessions.values())
        total_frames = sum(len(s.frames) for s in self.sessions.values())

        return {
            "active_sessions": len(self.sessions),
            "max_sessions": self.max_sessions,
            "total_memory_mb": total_memory / 1024 / 1024,
            "max_memory_mb": self.max_memory_bytes / 1024 / 1024,
            "total_frames": total_frames,
            "sessions": {sid: s.get_stats() for sid, s in self.sessions.items()},
        }
