# Bug Fix: Dataclass Inheritance Error

## Problem

When starting the server, the following error occurred:

```
TypeError: SessionStartMessage.__init__() missing 1 required positional argument: 'type'
```

## Root Cause

The issue was with Python dataclass inheritance. When a dataclass inherits from another dataclass that has a field with a default value, **all fields in the child class must also have default values**.

Original problematic structure:
```python
@dataclass
class StreamMessage:
    type: MessageType = None  # Has default

@dataclass
class SessionStartMessage(StreamMessage):
    session_id: str  # NO default - ERROR!
    fps: float       # NO default - ERROR!
    ...
```

Python's dataclass mechanism requires that if any field in the inheritance hierarchy has a default value, all subsequent fields must have defaults too.

## Solution

Removed the inheritance from `StreamMessage` for all message classes and added the `type` field directly to each dataclass with a default value:

```python
# No inheritance
@dataclass
class SessionStartMessage:
    session_id: str
    fps: float
    width: int
    height: int
    expected_frames: Optional[int] = None
    video_format: str = "bgr24"
    type: MessageType = None  # Added with default

    def __post_init__(self):
        self.type = MessageType.SESSION_START  # Set in post-init
```

This pattern was applied to all message classes:
- `SessionStartMessage`
- `SessionEndMessage`
- `FrameMessage`
- `FrameAckMessage`
- `SignalMessage`
- `SignalAckMessage`
- `ProcessProgressMessage`
- `ProcessCompleteMessage`
- `ErrorMessage`

## Files Modified

- `/workspace/streaming/protocol.py`

## Changes Made

1. Changed `StreamMessage` from `@dataclass` to regular class
2. Added `type: MessageType = None` field to each message dataclass
3. Kept `__post_init__` methods to set the correct type value

## Verification

```bash
# Test imports
python3 -c "from streaming.protocol import SessionStartMessage; print('OK')"

# Test creation
python3 -c "
from streaming.protocol import SessionStartMessage
msg = SessionStartMessage(session_id='test', fps=30.0, width=1920, height=1080)
print(msg.type)
"

# Test from_dict
python3 -c "
from streaming.protocol import SessionStartMessage
msg = SessionStartMessage.from_dict({'session_id': 'test', 'fps': 30, 'width': 1920, 'height': 1080})
print(msg.session_id)
"
```

All tests pass ✅

## Status

**FIXED** - Server should now start without errors.

## How to Test

Restart the server:
```bash
cd /workspace
python3 referee_service.py
```

Should see:
```
INFO: Loading YOLO model from yolov8m-pose.pt
INFO: YOLO model loaded successfully
INFO: Session manager cleanup task started
INFO: Uvicorn running on http://0.0.0.0:8080
```

Then test with the streaming client from your laptop.
