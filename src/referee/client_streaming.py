#!/usr/bin/env python3
"""Capture-or-upload client that sends full videos to the AI referee service."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("legacy_client")


class LegacyUploadClient:
    """Uploads complete videos to the /analyze endpoint."""

    def __init__(self, server_url: str, request_timeout: float = 600.0):
        self.server_url = server_url.rstrip("/")
        self.request_timeout = request_timeout

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------
    def record_camera(
        self,
        camera_index: int,
        duration_seconds: float,
        fps: float,
        output_path: Path,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Path:
        logger.info(
            "Recording camera %s for %.1fs at %.1f FPS → %s",
            camera_index,
            duration_seconds,
            fps,
            output_path,
        )

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_index}")

        try:
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps:
                cap.set(cv2.CAP_PROP_FPS, fps)

            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or (width or 1280))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or (height or 720))
            actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30.0
            target_fps = min(fps or actual_fps, actual_fps or fps or 30.0)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (frame_w, frame_h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open writer for {output_path}")

            frames_written = 0
            start = time.time()
            last_log = start
            try:
                while True:
                    elapsed = time.time() - start
                    if elapsed >= duration_seconds:
                        break

                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Camera read failed at frame %d", frames_written)
                        break

                    writer.write(frame)
                    frames_written += 1

                    if time.time() - last_log >= 2.0:
                        logger.info(
                            "Capture progress: %.1fs / %.1fs (%d frames)",
                            elapsed,
                            duration_seconds,
                            frames_written,
                        )
                        last_log = time.time()

                    if target_fps > 0:
                        expected = frames_written / target_fps
                        sleep_for = start + expected - time.time()
                        if sleep_for > 0:
                            time.sleep(min(0.02, sleep_for))
            finally:
                writer.release()

            logger.info(
                "Camera capture complete: %d frames, %.1f seconds",
                frames_written,
                time.time() - start,
            )
            return output_path
        finally:
            cap.release()

    def transcode_video(self, source: Path, target: Path, fps_override: Optional[float] = None) -> Path:
        logger.info("Transcoding %s → %s", source, target)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {source} for transcoding")

        writer = None
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            target.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(target), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open writer for {target}")

            frames = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
                frames += 1
            logger.info("Transcoding complete (%d frames)", frames)
            return target
        finally:
            cap.release()
            if writer is not None:
                writer.release()

    # ------------------------------------------------------------------
    # Upload helper
    # ------------------------------------------------------------------
    def upload_phrase(
        self,
        video_path: Path,
        signal_path: Path,
        include_keypoints: bool,
        save_overlay: bool,
    ) -> dict:
        url = f"{self.server_url}/analyze"
        logger.info(
            "Uploading %s (%.2f MB) with signal %s",
            video_path,
            video_path.stat().st_size / 1024 / 1024,
            signal_path,
        )

        files = {
            "video": (video_path.name, open(video_path, "rb"), "video/mp4"),
            "signal": (signal_path.name, open(signal_path, "rb"), "text/plain"),
        }
        data = {}
        if include_keypoints:
            data["include_keypoints"] = "true"
        if not save_overlay:
            data["save_overlay"] = "false"

        try:
            response = requests.post(
                url,
                files=files,
                data=data,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            logger.info("Upload finished (%d)", response.status_code)
            return response.json()
        finally:
            for file_tuple in files.values():
                file_tuple[1].close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture or upload full videos to the AI referee service",
    )
    parser.add_argument(
        "server",
        help="Base URL of the referee service (e.g., http://server:8080)",
    )
    parser.add_argument(
        "--signal",
        type=Path,
        required=True,
        help="Path to electric signal .txt file",
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Path to an existing video file to upload",
    )
    parser.add_argument(
        "--camera",
        type=int,
        help="Camera index to record from (alternative to --video)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Recording duration in seconds when using --camera",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Target FPS for camera capture or transcoding",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Optional width hint for camera capture",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Optional height hint for camera capture",
    )
    parser.add_argument(
        "--transcode-video",
        action="store_true",
        help="Re-encode --video input to mp4 before uploading",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Request timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--include-keypoints",
        action="store_true",
        help="Ask server to include raw keypoints in response",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Skip overlay rendering on server",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.video and args.camera is None:
        logger.error("Either --video or --camera must be provided")
        sys.exit(2)
    if args.video and args.camera is not None:
        logger.error("Use either --video or --camera, not both")
        sys.exit(2)

    signal_path = args.signal.expanduser().resolve()
    if not signal_path.is_file():
        logger.error("Signal file not found: %s", signal_path)
        sys.exit(2)

    client = LegacyUploadClient(args.server, request_timeout=args.timeout)

    with tempfile.TemporaryDirectory(prefix="referee_upload_") as tmpdir:
        work_dir = Path(tmpdir)

        if args.camera is not None:
            output_path = work_dir / f"capture_{int(time.time())}.mp4"
            try:
                video_path = client.record_camera(
                    camera_index=args.camera,
                    duration_seconds=args.duration,
                    fps=args.fps,
                    output_path=output_path,
                    width=args.width,
                    height=args.height,
                )
            except Exception as exc:
                logger.error("Camera recording failed: %s", exc)
                sys.exit(1)
        else:
            video_path = args.video.expanduser().resolve()
            if not video_path.is_file():
                logger.error("Video file not found: %s", video_path)
                sys.exit(2)
            if args.transcode_video:
                target = work_dir / f"{video_path.stem}_compressed.mp4"
                try:
                    video_path = client.transcode_video(video_path, target, fps_override=args.fps)
                except Exception as exc:
                    logger.error("Transcoding failed: %s", exc)
                    sys.exit(1)

        try:
            result = client.upload_phrase(
                video_path=video_path,
                signal_path=signal_path,
                include_keypoints=args.include_keypoints,
                save_overlay=not args.no_overlay,
            )
        except requests.HTTPError as exc:
            logger.error("Server returned error %s: %s", exc.response.status_code if exc.response else "?", exc)
            if exc.response is not None:
                try:
                    print(json.dumps(exc.response.json(), indent=2))
                except Exception:
                    print(exc.response.text)
            sys.exit(1)
        except requests.RequestException as exc:
            logger.error("Upload failed: %s", exc)
            sys.exit(1)

        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
