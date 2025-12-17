"""FastAPI service exposing the AI fencing referee pipeline."""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from ultralytics import YOLO

from src.referee.analysis import AnalysisResult, analyze_video_signal, parse_txt_file
from src.streaming.session import SessionManager, StreamingSession, FrameData
from src.streaming.protocol import (
    MessageType,
    StreamMessage,
    SessionStartMessage,
    SessionEndMessage,
    FrameMessage,
    FrameAckMessage,
    SignalMessage,
    SignalAckMessage,
    ProcessProgressMessage,
    ProcessCompleteMessage,
    ErrorMessage,
)

logger = logging.getLogger("referee_service")
logging.basicConfig(level=os.environ.get("REFEREE_LOG_LEVEL", "INFO"))

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


async def _process_streaming_session(
    session: StreamingSession, app: FastAPI, websocket: WebSocket
) -> Dict[str, Any]:
    """Process a completed streaming session."""
    request_started = time.time()

    # Create artifact directory
    artifact_dir = app.state.output_root / session.session_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Write video from buffered frames
    video_path = artifact_dir / "streamed_video.mp4"
    session.write_video(video_path)

    # Write signal file
    signal_path = artifact_dir / (session.signal_filename or "signal.txt")
    session.write_signal(signal_path)

    # Parse signal
    phrase = parse_txt_file(str(signal_path))

    # Send progress update
    await websocket.send_json(
        ProcessProgressMessage(
            session_id=session.session_id,
            stage="video_reconstruction",
            progress=1.0,
            message="Video reconstructed from frames",
        ).to_dict()
    )

    # Run analysis
    loop = asyncio.get_running_loop()
    async with app.state.model_lock:
        await websocket.send_json(
            ProcessProgressMessage(
                session_id=session.session_id,
                stage="analysis",
                progress=0.0,
                message="Starting YOLO analysis",
            ).to_dict()
        )

        analysis_result: AnalysisResult = await loop.run_in_executor(
            None,
            functools.partial(
                analyze_video_signal,
                str(video_path),
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

    # Prepare result
    analysis_video = (
        analysis_result.artifacts.get("analysis_video")
        or analysis_result.artifacts.get("corrected_video")
        or analysis_result.input_video_path
        or str(video_path)
    )
    analysis_result.artifacts["video"] = analysis_video
    analysis_result.artifacts.setdefault("signal", str(signal_path))

    payload = analysis_result.to_dict(include_keypoints=False)
    payload["artifact_dir"] = str(artifact_dir)
    payload["wall_time_seconds"] = round(time.time() - request_started, 4)
    payload["streaming_stats"] = session.get_stats()

    result_path = artifact_dir / "analysis_result.json"
    payload.setdefault("artifacts", {})["result_json"] = str(result_path)
    result_path.write_text(json.dumps(payload, indent=2))

    return payload


def create_app() -> FastAPI:
    app = FastAPI(title="AI Fencing Referee Service", version="1.0.0")

    model_path = os.environ.get("REFEREE_YOLO_MODEL", "models/yolo11x-pose.pt")
    app.state.model_path = model_path
    app.state.model: Optional[YOLO] = None
    app.state.model_lock = asyncio.Lock()
    output_root = Path(os.environ.get("REFEREE_OUTPUT_DIR", "results/processed_phrases"))
    output_root.mkdir(parents=True, exist_ok=True)
    app.state.output_root = output_root

    # Streaming session management (disabled by default)
    streaming_enabled = _env_bool("REFEREE_STREAMING_ENABLED", False)
    max_streaming_sessions = int(os.environ.get("REFEREE_MAX_STREAMING_SESSIONS", "10"))
    max_streaming_memory_mb = int(os.environ.get("REFEREE_MAX_STREAMING_MEMORY_MB", "2048"))
    streaming_session_timeout = float(
        os.environ.get("REFEREE_STREAMING_SESSION_TIMEOUT", "300")
    )
    app.state.streaming_enabled = streaming_enabled
    app.state.session_manager = (
        SessionManager(
            max_sessions=max_streaming_sessions,
            max_memory_mb=max_streaming_memory_mb,
            session_timeout=streaming_session_timeout,
        )
        if streaming_enabled
        else None
    )



    fisheye_enabled = _env_bool("REFEREE_FISHEYE_ENABLED", True)
    fisheye_strength = float(os.environ.get("REFEREE_FISHEYE_STRENGTH", "-0.18"))
    fisheye_balance = float(os.environ.get("REFEREE_FISHEYE_BALANCE", "0.0"))
    fisheye_keep_audio = _env_bool("REFEREE_FISHEYE_KEEP_AUDIO", True)
    fisheye_progress = _env_bool("REFEREE_FISHEYE_PROGRESS", False)

    app.state.fisheye_config = {
        "enabled": fisheye_enabled,
        "strength": fisheye_strength,
        "balance": fisheye_balance,
        "keep_audio": fisheye_keep_audio,
        "progress": fisheye_progress,
    }

    @app.on_event("startup")
    async def _load_model() -> None:
        try:
            logger.info("Loading YOLO model from %s", app.state.model_path)
            app.state.model = YOLO(app.state.model_path)
            logger.info("YOLO model loaded successfully")

            # Start session cleanup task (if streaming enabled)
            if app.state.streaming_enabled and app.state.session_manager:
                await app.state.session_manager.start_cleanup_task()
                logger.info("Session manager cleanup task started")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to load YOLO model: %s", exc)
            raise

    @app.on_event("shutdown")
    async def _cleanup_model() -> None:
        logger.info("Shutting down referee service")

        # Stop session cleanup task
        if app.state.streaming_enabled and app.state.session_manager:
            await app.state.session_manager.stop_cleanup_task()
            logger.info("Session manager cleanup task stopped")

    @app.get("/health")
    async def healthcheck() -> Dict[str, Any]:
        model_loaded = app.state.model is not None
        return {
            "status": "ok" if model_loaded else "model_not_loaded",
            "model_path": app.state.model_path,
            "model_loaded": model_loaded,
        }

    @app.post("/analyze")
    async def analyze(
        video: UploadFile = File(..., description="Fencing phrase .avi video"),
        signal: UploadFile = File(..., description="Electric signal .txt file"),
        include_keypoints: bool = Form(False),
        save_overlay: bool = Form(True),
    ) -> JSONResponse:
        if app.state.model is None:
            raise HTTPException(status_code=500, detail="YOLO model not loaded")

        if not video.filename:
            raise HTTPException(status_code=400, detail="Video file must have a filename")
        if not signal.filename:
            raise HTTPException(status_code=400, detail="Signal file must have a filename")

        request_started = time.time()

        try:
            safe_stem_source = Path(signal.filename or video.filename).stem or "phrase"
            safe_stem = re.sub(r"[^A-Za-z0-9_-]", "_", safe_stem_source) or "phrase"
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            artifact_dir = app.state.output_root / f"{safe_stem}_{timestamp}"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            video_path = artifact_dir / Path(video.filename).name
            signal_path = artifact_dir / Path(signal.filename).name

            video.file.seek(0)
            with open(video_path, "wb") as outfile:
                shutil.copyfileobj(video.file, outfile)

            signal.file.seek(0)
            with open(signal_path, "wb") as outfile:
                shutil.copyfileobj(signal.file, outfile)

            phrase = parse_txt_file(str(signal_path))

            loop = asyncio.get_running_loop()
            async with app.state.model_lock:
                analysis_result: AnalysisResult = await loop.run_in_executor(
                    None,
                    functools.partial(
                        analyze_video_signal,
                        str(video_path),
                        str(signal_path),
                        model=app.state.model,
                        model_path=app.state.model_path,
                        return_keypoints=include_keypoints,
                        output_dir=artifact_dir,
                        save_excel=True,
                        save_overlay=save_overlay,
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

            analysis_video = (
                analysis_result.artifacts.get("analysis_video")
                or analysis_result.artifacts.get("corrected_video")
                or analysis_result.input_video_path
                or str(video_path)
            )
            analysis_result.artifacts["video"] = analysis_video
            analysis_result.artifacts.setdefault("signal", str(signal_path))

            payload = analysis_result.to_dict(include_keypoints=include_keypoints)
            payload["artifact_dir"] = str(artifact_dir)
            payload["wall_time_seconds"] = round(time.time() - request_started, 4)

            result_path = artifact_dir / "analysis_result.json"
            payload.setdefault("artifacts", {})["result_json"] = str(result_path)
            result_path.write_text(json.dumps(payload, indent=2))

            return JSONResponse(content=payload)

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected error during analysis: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

    @app.websocket("/stream")
    async def websocket_stream_endpoint(websocket: WebSocket):
        """WebSocket endpoint (disabled unless REFEREE_STREAMING_ENABLED=1)."""
        if not app.state.streaming_enabled or app.state.session_manager is None:
            await websocket.accept()
            logger.warning(
                "Streaming endpoint disabled - instructing client to use /analyze upload"
            )
            await websocket.send_json(
                ErrorMessage(
                    session_id=None,
                    error_code="STREAMING_DISABLED",
                    error_message="Real-time streaming is disabled. Please upload a complete video via /analyze.",
                    recoverable=False,
                ).to_dict()
            )
            await websocket.close()
            return

        current_session: Optional[StreamingSession] = None
        session_id: Optional[str] = None

        try:
            while True:
                # Receive message (text = metadata, bytes = frame/signal data)
                try:
                    message_text = await websocket.receive_text()
                except WebSocketDisconnect:
                    # Client disconnected gracefully
                    logger.info(f"Client disconnected gracefully for session {session_id}")
                    raise
                except Exception as exc:
                    # Other error (not disconnect)
                    logger.error(f"Error receiving message: {exc}")
                    try:
                        await websocket.send_json(
                            ErrorMessage(
                                session_id=session_id,
                                error_code="PROTOCOL_ERROR",
                                error_message="Expected text message with metadata",
                                recoverable=False,
                            ).to_dict()
                        )
                    except Exception:
                        pass
                    break

                # Parse message
                try:
                    message_data = json.loads(message_text)
                    msg_type = MessageType(message_data.get("type", ""))
                    logger.debug(f"Received message type: {msg_type}")
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.error(f"Invalid message format: {exc}")
                    await websocket.send_json(
                        ErrorMessage(
                            session_id=session_id,
                            error_code="INVALID_MESSAGE",
                            error_message=f"Invalid message format: {exc}",
                            recoverable=False,
                        ).to_dict()
                    )
                    break

                # Handle message types
                if msg_type == MessageType.SESSION_START:
                    try:
                        start_msg = SessionStartMessage.from_dict(message_data)
                        session_id = start_msg.session_id

                        logger.info(
                            f"Starting session {session_id}: "
                            f"{start_msg.width}x{start_msg.height} @ {start_msg.fps} FPS"
                        )

                        current_session = await app.state.session_manager.create_session(
                            session_id=session_id,
                            fps=start_msg.fps,
                            width=start_msg.width,
                            height=start_msg.height,
                            expected_frames=start_msg.expected_frames,
                        )

                        await websocket.send_json(
                            {"type": "session_started", "session_id": session_id}
                        )

                    except Exception as exc:
                        logger.exception(f"Failed to start session: {exc}")
                        await websocket.send_json(
                            ErrorMessage(
                                session_id=session_id,
                                error_code="SESSION_START_FAILED",
                                error_message=str(exc),
                                recoverable=False,
                            ).to_dict()
                        )
                        break

                elif msg_type == MessageType.FRAME:
                    if current_session is None:
                        await websocket.send_json(
                            ErrorMessage(
                                session_id=session_id,
                                error_code="NO_ACTIVE_SESSION",
                                error_message="No active session, send SESSION_START first",
                                recoverable=False,
                            ).to_dict()
                        )
                        break

                    try:
                        frame_msg = FrameMessage.from_dict(message_data)

                        # Receive frame data (binary)
                        frame_bytes = await websocket.receive_bytes()

                        # Validate size
                        if len(frame_bytes) != frame_msg.size:
                            logger.warning(
                                f"Frame size mismatch: expected {frame_msg.size}, "
                                f"got {len(frame_bytes)}"
                            )

                        # Create frame data
                        frame_data = FrameData(
                            frame_number=frame_msg.frame_number,
                            timestamp=frame_msg.timestamp,
                            encoding=frame_msg.encoding,
                            data=frame_bytes,
                        )

                        # Add to session
                        result = await current_session.add_frame(frame_data)

                        # Send ACK
                        ack = FrameAckMessage(
                            session_id=session_id,
                            frame_number=frame_msg.frame_number,
                            status=result["status"],
                        )
                        await websocket.send_json(ack.to_dict())

                        # Log progress every 100 frames
                        if frame_msg.frame_number % 100 == 0:
                            logger.info(
                                f"Session {session_id}: Received frame {frame_msg.frame_number}, "
                                f"total: {result['total_frames']}, "
                                f"memory: {result['memory_mb']:.1f} MB"
                            )

                    except Exception as exc:
                        logger.exception(f"Failed to process frame: {exc}")
                        await websocket.send_json(
                            ErrorMessage(
                                session_id=session_id,
                                error_code="FRAME_PROCESSING_FAILED",
                                error_message=str(exc),
                                recoverable=True,
                            ).to_dict()
                        )

                elif msg_type == MessageType.SESSION_END:
                    if current_session is None:
                        await websocket.send_json(
                            ErrorMessage(
                                session_id=session_id,
                                error_code="NO_ACTIVE_SESSION",
                                error_message="No active session",
                                recoverable=False,
                            ).to_dict()
                        )
                        break

                    try:
                        end_msg = SessionEndMessage.from_dict(message_data)
                        logger.info(
                            f"Session {session_id} ended: {end_msg.total_frames} frames sent"
                        )

                        # Validate session
                        validation = current_session.validate()
                        if not validation["valid"]:
                            logger.warning(
                                f"Session {session_id} validation issues: "
                                f"{validation['issues']}"
                            )

                        await websocket.send_json(
                            {
                                "type": "session_ended",
                                "session_id": session_id,
                                "validation": validation,
                            }
                        )

                    except Exception as exc:
                        logger.exception(f"Failed to end session: {exc}")
                        await websocket.send_json(
                            ErrorMessage(
                                session_id=session_id,
                                error_code="SESSION_END_FAILED",
                                error_message=str(exc),
                                recoverable=False,
                            ).to_dict()
                        )
                        break

                elif msg_type == MessageType.SIGNAL:
                    if current_session is None:
                        await websocket.send_json(
                            ErrorMessage(
                                session_id=session_id,
                                error_code="NO_ACTIVE_SESSION",
                                error_message="No active session",
                                recoverable=False,
                            ).to_dict()
                        )
                        break

                    try:
                        signal_msg = SignalMessage.from_dict(message_data)

                        # Receive signal data
                        signal_bytes = await websocket.receive_bytes()

                        # Validate size
                        if len(signal_bytes) != signal_msg.size:
                            logger.warning(
                                f"Signal size mismatch: expected {signal_msg.size}, "
                                f"got {len(signal_bytes)}"
                            )

                        await current_session.set_signal_data(
                            signal_bytes, signal_msg.filename
                        )

                        # Send ACK
                        ack = SignalAckMessage(session_id=session_id, status="received")
                        await websocket.send_json(ack.to_dict())

                        logger.info(
                            f"Session {session_id}: Signal data received, "
                            f"starting processing..."
                        )

                        # Now process the session
                        await websocket.send_json(
                            {
                                "type": "process_start",
                                "session_id": session_id,
                            }
                        )

                        # Process in background
                        result = await _process_streaming_session(
                            current_session, app, websocket
                        )

                        # Send completion
                        complete_msg = ProcessCompleteMessage(
                            session_id=session_id,
                            result=result,
                        )
                        await websocket.send_json(complete_msg.to_dict())

                        logger.info(f"Session {session_id}: Processing complete")

                        # Cleanup session
                        await app.state.session_manager.remove_session(session_id)
                        current_session = None
                        session_id = None

                    except Exception as exc:
                        logger.exception(f"Failed to process signal/analyze: {exc}")
                        await websocket.send_json(
                            ErrorMessage(
                                session_id=session_id,
                                error_code="PROCESSING_FAILED",
                                error_message=str(exc),
                                recoverable=False,
                            ).to_dict()
                        )
                        break

                elif msg_type == MessageType.PING:
                    await websocket.send_json({"type": "pong"})

                else:
                    logger.warning(f"Unknown message type: {msg_type}")
                    await websocket.send_json(
                        ErrorMessage(
                            session_id=session_id,
                            error_code="UNKNOWN_MESSAGE_TYPE",
                            error_message=f"Unknown message type: {msg_type}",
                            recoverable=True,
                        ).to_dict()
                    )

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session {session_id}")
            if current_session:
                await app.state.session_manager.remove_session(session_id)
        except Exception as exc:
            logger.exception(f"WebSocket error: {exc}")
            try:
                await websocket.send_json(
                    ErrorMessage(
                        session_id=session_id,
                        error_code="INTERNAL_ERROR",
                        error_message=str(exc),
                        recoverable=False,
                    ).to_dict()
                )
            except:
                pass
            if current_session and session_id:
                await app.state.session_manager.remove_session(session_id)
        finally:
            logger.info(f"WebSocket connection closed for session {session_id}")



    @app.get("/streaming/stats")
    async def streaming_stats() -> Dict[str, Any]:
        """Get streaming session statistics."""
        if not app.state.streaming_enabled or app.state.session_manager is None:
            return {
                "streaming_enabled": False,
                "message": "Streaming disabled. Use POST /analyze uploads.",
            }

        stats = app.state.session_manager.get_stats()
        stats["streaming_enabled"] = True
        return stats

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("REFEREE_HOST", "0.0.0.0")
    port = int(os.environ.get("REFEREE_PORT", "8080"))
    uvicorn.run("referee_service:app", host=host, port=port, reload=False)
