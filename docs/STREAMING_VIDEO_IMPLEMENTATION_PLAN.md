# Streaming Video Transfer Implementation Plan

## Executive Summary

**VERDICT: HIGHLY VIABLE AND RECOMMENDED**

Implementing streaming frame-by-frame video transfer is not only feasible but will significantly reduce latency in the AI referee pipeline. Instead of waiting for the entire video to record, then transfer, we can stream frames during recording, eliminating the transfer bottleneck.

**Estimated time savings**: 50-90% reduction in transfer time (frames transfer during recording, not after)

---

## Current Architecture Analysis

### 1. Current Flow (BLOCKING)
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Record     │ ──> │   Transfer   │ ──> │   Process   │
│  Video      │     │   Video      │     │   Video     │
│  (30-60s)   │     │   (10-30s)   │     │   (20-40s)  │
└─────────────┘     └──────────────┘     └─────────────┘
Total: 60-130 seconds
```

### 2. Streaming Flow (CONCURRENT)
```
┌─────────────────────────────────────┐
│  Record + Stream Frames (30-60s)    │
└─────────────────────────────────────┘
                    │
                    ├─> Frames arrive at server
                    │
                    └─> Buffer in memory
                            │
                            └─> Process when complete (20-40s)
Total: 50-100 seconds (30-50% faster)
```

---

## Current Implementation Details

### Client Side (`referee_client.py`)
**Current behavior**:
- Lines 78-79: Opens entire video file from disk
- Sends via `requests.post()` with multipart/form-data
- Waits for entire file to upload before processing begins

**Key limitation**: Video must be fully recorded before transfer begins

### Server Side (`referee_service.py`)
**Current behavior**:
- Lines 82-86: `/analyze` endpoint accepts UploadFile for video
- Lines 107-109: Saves entire video to disk via `shutil.copyfileobj()`
- Then processes from disk

**Key insight**: Already uses streaming under the hood (copyfileobj), but only after entire upload completes

### Processing Pipeline (`AI_Referee.py`)

#### 1. Fisheye Correction (`correct_fisheye_video`, lines 432-501)
- Opens video with `cv2.VideoCapture` (line 461)
- Reads frame-by-frame (line 477)
- Applies correction to each frame independently
- Writes corrected frames to new video file

**Streaming compatibility**: ✅ EXCELLENT - Each frame processed independently

#### 2. YOLO Tracking (`extract_tracks_from_video`, lines 592-638)
- Opens video with `cv2.VideoCapture` (line 597)
- Reads frame-by-frame (line 608)
- Runs YOLO pose estimation on each frame (line 612)
- Maintains state via `TwoFencerTracker` object

**Streaming compatibility**: ✅ EXCELLENT - Sequential frame processing with state tracking

#### 3. Decision Making (`referee_decision`)
- Operates on collected keypoint data
- Needs complete video data for temporal analysis

**Streaming compatibility**: ⚠️ REQUIRES BUFFERING - Needs all frames before final decision

---

## Streaming Implementation Options

### Option 1: WebSocket Frame Streaming (RECOMMENDED)

**Architecture**:
```
Local Laptop                          VM Server
┌──────────────────┐                 ┌──────────────────┐
│  Camera Capture  │                 │  WebSocket       │
│  (OpenCV)        │                 │  Server          │
│                  │                 │                  │
│  ┌────────────┐  │    WebSocket   │  ┌────────────┐  │
│  │ Grab Frame │  │ ─────────────> │  │ Receive    │  │
│  │ Encode JPG │  │   JSON + Bytes │  │ Buffer     │  │
│  │ Send       │  │ <────────────  │  │ ACK        │  │
│  └────────────┘  │                 │  └────────────┘  │
│       │          │                 │       │          │
│       │ (Loop)   │                 │       │          │
│       └──────────┤                 │       ├──────────┤
│                  │                 │  ┌────▼──────┐   │
│  Signal Monitor  │   HTTP POST    │  │  Process   │   │
│  ┌────────────┐  │ ─────────────> │  │  Pipeline  │   │
│  │ Send .txt  │  │                 │  └────────────┘  │
│  └────────────┘  │                 │                  │
└──────────────────┘                 └──────────────────┘
```

**Protocol Design**:

```python
# Message format (JSON + Binary)
{
    "session_id": "phrase_20251106_123045",
    "frame_number": 42,
    "timestamp": 1699276800.123,
    "width": 1920,
    "height": 1080,
    "format": "jpeg",  # or "png", "raw"
    "size": 65432,
    "is_last_frame": false
}
# Followed by: <JPEG bytes>
```

**Advantages**:
- ✅ Bidirectional communication (can ACK frames)
- ✅ Low latency
- ✅ Can send metadata with each frame
- ✅ Connection monitoring (know if server dies)
- ✅ Can request retransmission of frames
- ✅ Built into FastAPI via `websockets` library

**Disadvantages**:
- ❌ More complex than HTTP
- ❌ Need to handle reconnection logic
- ❌ Binary data handling needs care

**Implementation Complexity**: Medium

---

### Option 2: HTTP Chunked Transfer Encoding

**Architecture**:
```
Local Laptop                          VM Server
┌──────────────────┐                 ┌──────────────────┐
│  Camera Capture  │   POST          │  FastAPI         │
│                  │   multipart     │  Endpoint        │
│  Streaming       │   chunked       │                  │
│  MJPEG Generator │ ─────────────> │  Async Stream    │
│                  │                 │  Consumer        │
└──────────────────┘                 └──────────────────┘
```

**Protocol Design**:

```python
# HTTP POST with chunked transfer
POST /stream_analyze HTTP/1.1
Transfer-Encoding: chunked
Content-Type: multipart/x-mixed-replace; boundary=frame

--frame
Content-Type: image/jpeg
Content-Length: 65432

<JPEG bytes>
--frame
Content-Type: image/jpeg
Content-Length: 67123

<JPEG bytes>
--frame--
```

**Advantages**:
- ✅ Standard HTTP (no WebSocket complexity)
- ✅ Works with existing HTTP infrastructure
- ✅ Simple client implementation
- ✅ No persistent connection state

**Disadvantages**:
- ❌ One-way only (no ACKs)
- ❌ Can't recover from errors easily
- ❌ Server can't signal client

**Implementation Complexity**: Low-Medium

---

### Option 3: gRPC Bidirectional Streaming

**Architecture**:
```
Local Laptop                          VM Server
┌──────────────────┐                 ┌──────────────────┐
│  gRPC Client     │   Bidirectional │  gRPC Server     │
│                  │   Stream        │                  │
│  Stream Frames   │ ══════════════> │  Receive         │
│                  │ <══════════════ │  Send ACKs       │
└──────────────────┘                 └──────────────────┘
```

**Advantages**:
- ✅ Highly efficient binary protocol
- ✅ Built-in streaming support
- ✅ Bidirectional
- ✅ HTTP/2 multiplexing

**Disadvantages**:
- ❌ Most complex to implement
- ❌ Requires `.proto` definitions
- ❌ Separate port/service from FastAPI
- ❌ Overkill for this use case

**Implementation Complexity**: High

---

## Recommended Approach: Phased Implementation

### Phase 1: WebSocket Buffered Streaming (MVP)

**Goal**: Eliminate transfer bottleneck by streaming during recording

**Client Changes** (`referee_client_streaming.py` - NEW FILE):

```python
import asyncio
import websockets
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional

class StreamingRefereeClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.ws_url = server_url.replace('http://', 'ws://').replace('https://', 'wss://')

    async def stream_video_from_camera(
        self,
        camera_index: int,
        signal_path: Path,
        duration_seconds: float = 60.0
    ):
        """Stream frames from camera to server in real-time"""

        # Open camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        # Get camera properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        session_id = f"phrase_{int(time.time())}"

        async with websockets.connect(f"{self.ws_url}/stream") as websocket:
            # Send session start
            await websocket.send(json.dumps({
                "type": "session_start",
                "session_id": session_id,
                "fps": fps,
                "width": width,
                "height": height,
                "duration_seconds": duration_seconds
            }))

            frame_count = 0
            start_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Check duration
                elapsed = time.time() - start_time
                if elapsed >= duration_seconds:
                    break

                # Encode frame as JPEG (compress for network transfer)
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    continue

                jpeg_bytes = jpeg.tobytes()

                # Send metadata
                metadata = {
                    "type": "frame",
                    "session_id": session_id,
                    "frame_number": frame_count,
                    "timestamp": time.time(),
                    "size": len(jpeg_bytes)
                }
                await websocket.send(json.dumps(metadata))

                # Send frame data
                await websocket.send(jpeg_bytes)

                # Wait for ACK (optional, for reliability)
                ack = await websocket.recv()

                frame_count += 1

                # Maintain frame rate
                await asyncio.sleep(max(0, 1.0/fps - 0.001))

            cap.release()

            # Send session end
            await websocket.send(json.dumps({
                "type": "session_end",
                "session_id": session_id,
                "total_frames": frame_count
            }))

            # Send signal file
            with open(signal_path, 'rb') as f:
                signal_data = f.read()

            await websocket.send(json.dumps({
                "type": "signal",
                "session_id": session_id,
                "filename": signal_path.name,
                "size": len(signal_data)
            }))
            await websocket.send(signal_data)

            # Wait for analysis result
            result = await websocket.recv()
            return json.loads(result)

    async def stream_video_from_file(self, video_path: Path, signal_path: Path):
        """Stream frames from existing video file (for testing/migration)"""
        cap = cv2.VideoCapture(str(video_path))
        # Similar logic to stream_video_from_camera...
        # (can reuse most code)
```

**Server Changes** (`referee_service.py` modifications):

```python
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
from collections import defaultdict
from typing import Dict, List
import io
import PIL.Image

class StreamingSession:
    def __init__(self, session_id: str, fps: float, width: int, height: int):
        self.session_id = session_id
        self.fps = fps
        self.width = width
        self.height = height
        self.frames: List[np.ndarray] = []
        self.signal_data: Optional[bytes] = None
        self.complete = False
        self.timestamp = datetime.utcnow()

    def add_frame(self, jpeg_bytes: bytes):
        """Decode and store frame"""
        # Decode JPEG to numpy array
        img = PIL.Image.open(io.BytesIO(jpeg_bytes))
        frame = np.array(img)
        self.frames.append(frame)

    def get_temp_video_path(self) -> Path:
        """Write buffered frames to temporary video file for processing"""
        temp_dir = Path("/tmp/streaming_sessions") / self.session_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        video_path = temp_dir / "buffered_video.mp4"

        # Write frames to video file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(video_path),
            fourcc,
            self.fps,
            (self.width, self.height)
        )

        for frame in self.frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        return video_path

# Add to create_app():
app.state.streaming_sessions: Dict[str, StreamingSession] = {}

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_session: Optional[StreamingSession] = None

    try:
        while True:
            # Receive metadata first
            metadata_json = await websocket.receive_text()
            metadata = json.loads(metadata_json)

            msg_type = metadata.get("type")

            if msg_type == "session_start":
                session_id = metadata["session_id"]
                current_session = StreamingSession(
                    session_id=session_id,
                    fps=metadata["fps"],
                    width=metadata["width"],
                    height=metadata["height"]
                )
                app.state.streaming_sessions[session_id] = current_session
                logger.info(f"Started streaming session: {session_id}")
                await websocket.send_text(json.dumps({"status": "started"}))

            elif msg_type == "frame":
                if current_session is None:
                    raise ValueError("No active session")

                # Receive frame data
                jpeg_bytes = await websocket.receive_bytes()
                current_session.add_frame(jpeg_bytes)

                # Send ACK
                await websocket.send_text(json.dumps({
                    "status": "frame_received",
                    "frame_number": metadata["frame_number"]
                }))

            elif msg_type == "session_end":
                if current_session is None:
                    raise ValueError("No active session")

                logger.info(f"Session ended: {current_session.session_id}, "
                           f"frames: {len(current_session.frames)}")
                await websocket.send_text(json.dumps({"status": "ended"}))

            elif msg_type == "signal":
                if current_session is None:
                    raise ValueError("No active session")

                # Receive signal data
                signal_bytes = await websocket.receive_bytes()
                current_session.signal_data = signal_bytes

                # Now we have everything, process!
                result = await process_streaming_session(current_session, app)

                # Send result back
                await websocket.send_text(json.dumps(result))

                # Cleanup
                del app.state.streaming_sessions[current_session.session_id]
                current_session = None

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        if current_session:
            # Cleanup incomplete session
            del app.state.streaming_sessions[current_session.session_id]
    except Exception as exc:
        logger.exception(f"WebSocket error: {exc}")
        await websocket.close(code=1011, reason=str(exc))


async def process_streaming_session(
    session: StreamingSession,
    app: FastAPI
) -> Dict:
    """Process buffered streaming session"""

    # Create artifact directory
    artifact_dir = app.state.output_root / session.session_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Write video from buffered frames
    video_path = session.get_temp_video_path()

    # Copy to artifact dir
    final_video_path = artifact_dir / "streamed_video.mp4"
    shutil.copy(video_path, final_video_path)

    # Write signal file
    signal_path = artifact_dir / "signal.txt"
    signal_path.write_bytes(session.signal_data)

    # Now use existing analyze pipeline
    phrase = parse_txt_file(str(signal_path))

    loop = asyncio.get_running_loop()
    async with app.state.model_lock:
        analysis_result: AnalysisResult = await loop.run_in_executor(
            None,
            functools.partial(
                analyze_video_signal,
                str(final_video_path),
                str(signal_path),
                model=app.state.model,
                model_path=app.state.model_path,
                return_keypoints=False,
                output_dir=artifact_dir,
                save_excel=True,
                save_overlay=True,
                overlay_draw_skeleton=True,
                overlay_draw_labels=False,
                overlay_show_progress=False,
                phrase=phrase,
                fisheye_enabled=app.state.fisheye_config["enabled"],
                fisheye_strength=app.state.fisheye_config["strength"],
                fisheye_balance=app.state.fisheye_config["balance"],
                fisheye_keep_audio=app.state.fisheye_config["keep_audio"],
                fisheye_progress=app.state.fisheye_config["progress"],
            ),
        )

    return analysis_result.to_dict(include_keypoints=False)
```

---

### Phase 2: Real-Time Processing (ADVANCED)

**Goal**: Process frames as they arrive, provide early feedback

**Key Changes**:

1. **Streaming YOLO Processing**:
   - Process each frame immediately upon receipt
   - Build keypoint data incrementally
   - No need to write video file first

2. **Progressive Analysis**:
   - Detect hits in real-time
   - Provide immediate feedback
   - Final decision after all frames received

**Modified Processing Flow**:

```python
class StreamingAnalyzer:
    def __init__(self, model: YOLO, fps: float, width: int, height: int):
        self.model = model
        self.fps = fps
        self.tracker = TwoFencerTracker(frame_w=width, frame_h=height)
        self.tracks_per_frame = []
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process single frame as it arrives"""

        # Run YOLO
        results = self.model(frame, verbose=False)
        detections = []
        if len(results) > 0 and results[0].keypoints is not None:
            kpts = results[0].keypoints.xy
            if hasattr(kpts, "cpu"):
                kpts = kpts.cpu().numpy()
            else:
                kpts = np.array(kpts)
            for i in range(kpts.shape[0]):
                detections.append(kpts[i])

        # Update tracker
        self.tracker.update(detections)

        # Store tracks
        frame_tracks = []
        for tid in (0, 1):
            kpts = self.tracker.get_track(tid)
            bbox = bbox_from_keypoints(kpts) if kpts is not None else None
            frame_tracks.append({
                'track_id': tid,
                'keypoints': kpts.copy() if kpts is not None else None,
                'box': np.array(bbox, dtype=float) if bbox else None,
            })

        self.tracks_per_frame.append(frame_tracks)
        self.frame_count += 1

        return {
            "frame_number": self.frame_count,
            "detections": len(detections),
            "tracked": sum(1 for t in frame_tracks if t['keypoints'] is not None)
        }

    def finalize(self, phrase: FencingPhrase) -> AnalysisResult:
        """Complete analysis after all frames received"""

        # Use existing process_video_and_extract_data
        (
            left_xdata,
            left_ydata,
            right_xdata,
            right_ydata,
            normalisation_constant,
            video_angle,
        ) = process_video_and_extract_data(self.tracks_per_frame)

        # Make decision
        decision = referee_decision(
            phrase,
            left_xdata,
            left_ydata,
            right_xdata,
            right_ydata,
            normalisation_constant=normalisation_constant,
        )

        # ... rest of analysis result creation
```

**Benefits**:
- ⚡ Faster results (start processing immediately)
- 📊 Progressive feedback (see tracking in real-time)
- 💾 Less disk I/O (no temporary video file)

**Challenges**:
- 🔄 More complex state management
- 🐛 Harder to debug without video file
- 📹 Still need to save overlay video somehow

---

## Implementation Checklist

### Phase 1 Tasks

#### Client Side
- [ ] Create `referee_client_streaming.py`
- [ ] Implement `StreamingRefereeClient` class
- [ ] Add `stream_video_from_camera()` method
- [ ] Add `stream_video_from_file()` method (for testing)
- [ ] Add error handling and reconnection logic
- [ ] Add progress reporting
- [ ] Test with local video file first
- [ ] Test with camera capture

#### Server Side
- [ ] Add WebSocket route `/stream` to `referee_service.py`
- [ ] Implement `StreamingSession` class
- [ ] Add frame buffering logic
- [ ] Implement `process_streaming_session()` function
- [ ] Add session cleanup (TTL, max memory)
- [ ] Add error handling for disconnections
- [ ] Add monitoring/logging
- [ ] Test with simulated client

#### Testing
- [ ] Test with small video (10 frames)
- [ ] Test with full video (1800 frames)
- [ ] Test frame loss scenarios
- [ ] Test reconnection
- [ ] Measure latency improvements
- [ ] Load test (multiple concurrent streams)

### Phase 2 Tasks (Real-Time Processing)

- [ ] Implement `StreamingAnalyzer` class
- [ ] Modify server to process frames immediately
- [ ] Add progressive result updates
- [ ] Test accuracy vs buffered approach
- [ ] Optimize memory usage
- [ ] Add real-time overlay generation option

---

## Technical Considerations

### 1. Network Bandwidth

**Frame size estimation**:
- Raw frame: 1920×1080×3 = 6.2 MB
- JPEG (quality 85): ~200 KB per frame
- Video (30 FPS, 60s): 200 KB × 30 × 60 = 360 MB

**Transfer time**:
- 100 Mbps: 29 seconds
- 1 Gbps: 2.9 seconds

**Optimization**:
- Use JPEG compression (quality 80-90)
- Consider H.264 encoding for even better compression
- Adaptive quality based on network speed

### 2. Frame Synchronization

**Challenge**: Ensure frames align with signal data timestamps

**Solution**:
- Client sends timestamp with each frame
- Server timestamps on receipt
- Clock synchronization between client/server
- Signal file includes frame numbers or timestamps

### 3. Memory Management

**Challenge**: Buffering 1800 frames × 6 MB = 10.8 GB raw

**Solutions**:
- Keep JPEG compressed in buffer: 1800 × 200 KB = 360 MB ✅
- Process and discard frames immediately (Phase 2)
- Streaming write to disk as frames arrive
- Set max buffer size, reject new sessions if full

### 4. Fisheye Correction

**Current**: Reads video file, corrects all frames, writes new file

**Streaming options**:
1. **Correct on arrival**: Apply correction to each frame as buffered
2. **Correct during playback**: Correct when writing to final video
3. **Skip for streaming**: Optional optimization (may affect accuracy)

**Recommendation**: Correct each frame as it's added to buffer

### 5. Error Handling

**Scenarios**:
- Network disconnection mid-stream
- Frame corruption
- Out-of-order frames
- Client crash
- Server crash

**Mitigations**:
- Frame numbering to detect gaps
- ACK messages to detect loss
- Retransmission protocol
- Session timeout and cleanup
- Graceful degradation (process partial video)

### 6. Signal Data Coordination

**Options**:

1. **Send after video**: Current approach, simple
   ```
   Stream frames → End session → Send signal → Process
   ```

2. **Send before video**: Pre-upload signal
   ```
   Send signal → Stream frames → Process
   ```

3. **Send during video**: Parallel streams
   ```
   Stream frames (WebSocket) + Upload signal (HTTP POST)
   ```

4. **Real-time signal**: Stream signal data during recording
   ```
   Frame 1 → Frame 2 → Signal event → Frame 3 → ...
   ```

**Recommendation**: Start with Option 1 (send after), migrate to Option 4 for fully real-time system

---

## Performance Projections

### Current System
```
Recording:  60s  ████████████████████████████████████
Transfer:   25s  ████████████
Processing: 35s  ██████████████████
Total:     120s
```

### Phase 1 (Buffered Streaming)
```
Recording + Transfer (concurrent): 60s  ████████████████████████████████████
Processing:                         35s  ██████████████████
Total:                             ~75s  (38% faster)
```

### Phase 2 (Real-Time Processing)
```
Recording + Transfer + Processing (concurrent): 60s  ████████████████████████████████████
Finalization:                                    5s  ███
Total:                                         ~65s  (46% faster)
```

---

## Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Network instability | Failed uploads | Medium | Implement retries, buffering, graceful degradation |
| Memory overflow | Server crash | Low | Set max buffer size, monitoring, cleanup |
| Frame corruption | Analysis errors | Low | Validate frames, checksums, frame numbering |
| Increased complexity | Bugs | High | Thorough testing, phased rollout, keep old endpoint |
| YOLO accuracy | Wrong decisions | Low | Validation testing vs current approach |
| Synchronization issues | Timing errors | Medium | Timestamp validation, clock sync |

---

## Migration Strategy

### Step 1: Parallel Systems (Weeks 1-2)
- Deploy streaming endpoint alongside existing `/analyze`
- Both systems operational
- Test streaming with subset of users
- Compare results for accuracy

### Step 2: Default to Streaming (Week 3)
- Make streaming the default
- Keep `/analyze` as fallback
- Monitor error rates

### Step 3: Deprecate Old System (Week 4+)
- After validation period
- Remove `/analyze` endpoint
- Clean up old code

---

## Alternative Architectures (Future Considerations)

### 1. Edge Processing
- Run YOLO on local laptop
- Stream only keypoints (not frames)
- Pros: Much lower bandwidth (KB vs MB)
- Cons: Requires powerful laptop, YOLO model on client

### 2. Hybrid Approach
- Low-res stream for real-time preview
- High-res upload for final analysis
- Pros: Best of both worlds
- Cons: More complex

### 3. P2P with WebRTC
- Direct peer-to-peer streaming
- Bypass server for video transfer
- Pros: Lower latency, less server load
- Cons: Complex NAT traversal, security

---

## Dependencies

### New Python Packages Needed

**Client**:
```
websockets>=11.0
opencv-python>=4.8.0  # (already have)
Pillow>=10.0
```

**Server**:
```
websockets>=11.0  # (FastAPI includes this)
Pillow>=10.0
```

### No changes needed:
- FastAPI (already supports WebSockets)
- OpenCV (already used)
- YOLO (no changes)

---

## Code Organization

```
/workspace/
├── referee_service.py              # Modified: Add WebSocket endpoint
├── referee_client.py               # Keep as-is (backward compat)
├── referee_client_streaming.py     # NEW: Streaming client
├── AI_Referee.py                   # Minimal changes
├── streaming/                      # NEW: Streaming modules
│   ├── __init__.py
│   ├── session.py                  # StreamingSession class
│   ├── analyzer.py                 # StreamingAnalyzer (Phase 2)
│   └── protocol.py                 # Message format definitions
└── tests/
    ├── test_streaming_client.py    # NEW
    └── test_streaming_server.py    # NEW
```

---

## Testing Plan

### Unit Tests
- [ ] Frame encoding/decoding
- [ ] Session lifecycle
- [ ] Buffer management
- [ ] Error conditions

### Integration Tests
- [ ] End-to-end streaming flow
- [ ] WebSocket connection/disconnection
- [ ] Frame synchronization
- [ ] Signal data handling

### Performance Tests
- [ ] Latency measurement
- [ ] Memory usage profiling
- [ ] Network bandwidth usage
- [ ] Concurrent sessions

### Accuracy Tests
- [ ] Compare decisions: streaming vs current
- [ ] Frame loss scenarios
- [ ] Compression quality impact

---

## Monitoring and Observability

### Metrics to Track
- Frames per second (throughput)
- Frame loss rate
- Average frame latency
- Memory usage per session
- Active session count
- Processing time (buffered vs current)
- Decision accuracy (vs ground truth)

### Logging
```python
logger.info(
    "Streaming session",
    session_id=session.session_id,
    frames_received=len(session.frames),
    duration_seconds=elapsed,
    avg_fps=len(session.frames) / elapsed,
    memory_mb=session.memory_usage() / 1024 / 1024
)
```

---

## Conclusion

**Streaming frame transfer is HIGHLY VIABLE and RECOMMENDED.**

The current architecture is already well-suited for streaming:
- ✅ Frame-by-frame processing
- ✅ Independent frame operations
- ✅ Sequential state tracking
- ✅ Existing async infrastructure (FastAPI)

**Expected benefits**:
- 🚀 30-50% reduction in total latency
- 📊 Real-time progress visibility
- 🔄 Better error recovery
- 📈 Scalable to multiple concurrent sessions

**Recommended approach**:
1. Start with Phase 1 (WebSocket buffered streaming)
2. Validate accuracy and performance
3. Migrate to Phase 2 (real-time processing) if needed

**Estimated development time**:
- Phase 1: 2-3 days (MVP), 1 week (production-ready)
- Phase 2: 1-2 weeks

**Risk level**: Low-Medium (well-understood patterns, minimal changes to core logic)

---

## Next Steps

1. **Review this document** - Confirm approach aligns with requirements
2. **Prototype Phase 1** - Build MVP with test video file
3. **Validate accuracy** - Ensure streaming produces same results as current
4. **Performance testing** - Measure actual latency improvements
5. **Production deployment** - Phased rollout with monitoring

