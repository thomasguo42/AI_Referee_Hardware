# Bug Analysis: WebSocket Disconnect After SESSION_START

## Symptoms

```
INFO:streaming.session:Created session 20251106_170116_phrase04 (1 active sessions)
ERROR:referee_service:Expected text message, got bytes
ERROR:referee_service:WebSocket error: Unexpected ASGI message 'websocket.send'...
INFO:streaming.session:Session 20251106_170116_phrase04: Cleaning up (0 frames, 0.0 MB)
```

**Key observation:** Session created successfully, but 0 frames received before disconnect.

## Root Cause

The error has **two parts**:

### 1. Client Disconnect (PRIMARY ISSUE)
The client is disconnecting immediately after SESSION_START succeeds. This is evidenced by:
- Session created ✅
- 0 frames received ❌
- Cleanup shows 0 frames, 0.0 MB ❌

**Likely causes:**
- Client error/exception after sending SESSION_START
- Network issue causing disconnect
- Client timeout
- Client validation failure
- Missing dependencies on client

### 2. Error Handler Bug (FIXED)
When the client disconnects, the server was catching the `WebSocketDisconnect` exception in the wrong place and trying to send an error message after the connection was already closed.

**Fixed by:** Catching `WebSocketDisconnect` separately and re-raising it, so it's handled by the outer exception handler that doesn't try to send messages.

## Server-Side Fix Applied

### Before:
```python
try:
    message_text = await websocket.receive_text()
except Exception:
    # This catches WebSocketDisconnect too!
    logger.error("Expected text message, got bytes")
    await websocket.send_json(...)  # FAILS - connection closed!
    break
```

### After:
```python
try:
    message_text = await websocket.receive_text()
except WebSocketDisconnect:
    # Let outer handler deal with it
    logger.info(f"Client disconnected gracefully for session {session_id}")
    raise
except Exception as exc:
    # Only other errors
    logger.error(f"Error receiving message: {exc}")
    try:
        await websocket.send_json(...)
    except:
        pass  # Connection might be closed
    break
```

## Client-Side Troubleshooting

**The main issue is on the CLIENT side** - something is causing the client to disconnect after SESSION_START.

### Check these on the client:

#### 1. Check Client Logs
Look for errors or exceptions in the client output right after "Session started successfully"

#### 2. Verify Video File
```bash
# Can the client read the video?
python3 -c "
import cv2
cap = cv2.VideoCapture('your_video.avi')
print('Opened:', cap.isOpened())
ret, frame = cap.read()
print('Read frame:', ret)
if ret:
    print('Frame shape:', frame.shape)
cap.release()
"
```

#### 3. Check Dependencies
```bash
# On client machine
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
python3 -c "import websockets; print('websockets:', websockets.__version__)"
python3 -c "from PIL import Image; print('Pillow OK')"
```

#### 4. Test with Verbose Mode
```bash
python3 referee_client_streaming.py \
    http://server:8080 \
    --video test.avi \
    --signal test.txt \
    --verbose
```

Look for error messages after "Session started successfully"

#### 5. Check Network Connection
```bash
# From client
ping server-ip

# Test WebSocket
curl http://server-ip:8080/health
```

#### 6. Try with Small Video First
```bash
# Create a short test video (10 frames)
python3 -c "
import cv2
import numpy as np

out = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
for i in range(10):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out.write(frame)
out.release()
print('Created test.mp4 with 10 frames')
"

# Try streaming it
python3 referee_client_streaming.py \
    http://server:8080 \
    --video test.mp4 \
    --signal signal.txt
```

#### 7. Check Firewall/Proxy
Some networks block WebSocket connections. Try:
```bash
# If you're behind a proxy or firewall
ssh -L 8080:localhost:8080 user@server

# Then connect to localhost
python3 referee_client_streaming.py \
    http://localhost:8080 \
    --video test.avi \
    --signal test.txt
```

## Debugging Steps

### On Server:
```bash
# Enable debug logging
export REFEREE_LOG_LEVEL=DEBUG
python3 referee_service.py
```

### On Client:
```bash
# Run with verbose mode
python3 referee_client_streaming.py \
    http://server:8080 \
    --video test.avi \
    --signal test.txt \
    --verbose 2>&1 | tee client.log
```

Then look at `client.log` for the exact error.

## Common Client Errors

### 1. "Cannot open video"
```python
# Client error:
RuntimeError: Failed to open video: /path/to/video.avi

# Fix: Check video file exists and is readable
```

### 2. "Failed to encode frame"
```python
# Client error during frame encoding
# Fix: Check frame dimensions, format
```

### 3. "Connection timeout"
```python
# Client timeout after SESSION_START
# Fix: Check network, increase timeout
```

### 4. "Signal file not found"
```python
# Client can't read signal file
# Fix: Check signal file path
```

## Expected Behavior

When working correctly, you should see:

**Server:**
```
INFO: Created session ... (1 active sessions)
INFO: Session ...: Received frame 0, total: 1, memory: 0.2 MB
INFO: Session ...: Received frame 100, total: 100, memory: 20.5 MB
INFO: Session ...: Received frame 200, total: 200, memory: 41.0 MB
...
INFO: Session ... ended: 1800 frames sent
INFO: Session ...: Signal data received, starting processing...
INFO: Session ...: Processing complete
INFO: Removed session ... (0 active sessions)
```

**Client:**
```
INFO - Connected to ws://server:8080/stream
INFO - Sent session start
INFO - Session started successfully
INFO - Progress: 100/1800 frames (5.6%), 28.3 FPS, 45.2 Mbps
INFO - Progress: 200/1800 frames (11.1%), 29.1 FPS, 46.1 Mbps
...
INFO - Sent session end (1800 frames)
INFO - Session ended successfully
INFO - Sent signal data (1234 bytes)
INFO - Processing: video_reconstruction (100%)
INFO - Processing: analysis (0%)
INFO - Processing complete!
```

## Next Steps

1. **Check client logs** with `--verbose` flag
2. **Test with a small video** (10-20 frames)
3. **Verify video file** can be opened by OpenCV
4. **Check network connectivity**
5. If still failing, share the **client error logs**

## Status

**Server-side error handling: FIXED ✅**
**Client-side issue: NEEDS INVESTIGATION ⚠️**

The server is now more robust and won't crash when clients disconnect, but we need to fix whatever is causing the client to disconnect after SESSION_START.
