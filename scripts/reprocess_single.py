#!/usr/bin/env python3
"""
Re-run the full fisheye + YOLO pipeline on a single video.

If a phrase TXT is present (or provided), the referee decision step runs; otherwise,
the script stops after fisheye correction, tracking, overlay, and keypoints export.

Usage:
  python3 scripts/reprocess_single.py --video-path path/to/video.mp4 --model-path models/yolo11x-pose.pt
  # Optional: --txt-path path/to/phrase.txt to enable decision logic.
"""

import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent))
from src.referee.analysis import (  # type: ignore
    correct_fisheye_video,
    extract_tracks_from_video,
    parse_txt_file,
    process_video_and_extract_data,
    referee_decision,
    render_overlay_video,
    sanitize_for_json,
    save_keypoints_to_excel,
)


def _find_txt_default(video_path: Path) -> Path | None:
    """Pick the first .txt in the same directory if present."""
    candidates = list(video_path.parent.glob("*.txt"))
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Reprocess a single video with fisheye correction and data extraction.")
    parser.add_argument("--video-path", type=Path, required=True, help="Path to the input video (mp4/avi).")
    parser.add_argument("--txt-path", type=Path, help="Path to the phrase TXT; defaults to first .txt in the same folder.")
    parser.add_argument("--model-path", type=Path, default=Path("models/yolo11x-pose.pt"), help="Path to YOLO weights.")
    parser.add_argument("--progress", action="store_true", help="Show progress during fisheye correction.")
    args = parser.parse_args()

    video_path: Path = args.video_path
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    txt_path: Path | None = args.txt_path or _find_txt_default(video_path)
    if txt_path is not None and not txt_path.exists():
        raise FileNotFoundError(f"Phrase TXT not found at {txt_path}")

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {args.model_path}")

    folder = video_path.parent

    # Clean prior artifacts from the folder.
    for f in folder.glob("*keypoints.xlsx"):
        f.unlink()
    for f in folder.glob("*_overlay.mp4"):
        f.unlink()
    for f in folder.glob("*_corrected.mp4"):
        f.unlink()

    print(f"Loading YOLO model: {args.model_path}")
    model = YOLO(str(args.model_path))

    corrected_path = folder / f"{video_path.stem}_corrected.mp4"
    corrected_video_path = correct_fisheye_video(
        input_path=str(video_path),
        output_path=str(corrected_path),
        progress=args.progress,
    )

    tracks_per_frame = extract_tracks_from_video(corrected_video_path, model)
    left_xdata, left_ydata, right_xdata, right_ydata, norm_constant, video_angle = process_video_and_extract_data(
        tracks_per_frame
    )

    excel_path = folder / f"{video_path.stem}_keypoints.xlsx"
    save_keypoints_to_excel(left_xdata, left_ydata, right_xdata, right_ydata, str(excel_path))

    overlay_path = folder / f"{video_path.stem}_overlay.mp4"
    render_overlay_video(
        video_path=Path(corrected_video_path),
        output_path=overlay_path,
        left_xdata=left_xdata,
        left_ydata=left_ydata,
        right_xdata=right_xdata,
        right_ydata=right_ydata,
        normalisation_constant=norm_constant if norm_constant else 1.0,
        draw_skeleton=True,
        draw_labels=True,
        show_progress=args.progress,
    )

    # If no TXT, skip decision logic and JSON update.
    if txt_path is None:
        print("No TXT file found; skipping referee decision. Outputs are corrected video, overlay, and keypoints.")
    else:
        # Load existing JSON if present for preservation of additional metadata.
        json_path = folder / "analysis_result.json"
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    result_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Corrupt JSON found at {json_path}; starting fresh.")
                result_data = {}
        else:
            result_data = {}

        phrase = parse_txt_file(str(txt_path))
        decision = referee_decision(
            phrase,
            left_xdata,
            left_ydata,
            right_xdata,
            right_ydata,
            normalisation_constant=norm_constant,
        )

        result_data["normalisation_constant"] = norm_constant
        result_data["video_angle"] = video_angle
        result_data["winner"] = decision.get("winner")
        result_data["reason"] = decision.get("reason")
        result_data["left_pauses"] = decision.get("left_pauses")
        result_data["right_pauses"] = decision.get("right_pauses")
        result_data["blade_analysis"] = decision.get("blade_analysis")
        result_data["blade_details"] = decision.get("blade_details")
        result_data["speed_comparison"] = decision.get("speed_comparison")
        result_data["lunge_detected"] = decision.get("lunge_detected")

        result_data = sanitize_for_json(result_data)

        with open(json_path, "w") as f:
            json.dump(result_data, f, indent=2)

    print("Reprocess complete.")
    print(f"Corrected video: {corrected_video_path}")
    print(f"Overlay: {overlay_path}")
    print(f"Keypoints Excel: {excel_path}")
    if txt_path is not None:
        print(f"Analysis JSON: {json_path}")


if __name__ == "__main__":
    main()
