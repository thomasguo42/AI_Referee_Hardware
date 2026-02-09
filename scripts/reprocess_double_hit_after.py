#!/usr/bin/env python3
"""
Batch reprocess phrase folders under new_data/double_hit after a given folder name.

This mirrors scripts/reprocess_single.py behavior (fisheye -> YOLO tracks -> Excel -> overlay -> decision -> JSON),
but loads the YOLO model once for speed and iterates many folders.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

# Allow running from repo root without installing as a package.
import sys

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


def _find_video(folder: Path) -> Path | None:
    avis = sorted(folder.glob("*.avi"))
    if avis:
        return avis[0]
    mp4s = sorted([p for p in folder.glob("*.mp4") if "_overlay" not in p.name and "_corrected" not in p.name])
    if mp4s:
        return mp4s[0]
    return None


def _find_txt(folder: Path) -> Path | None:
    txts = sorted(folder.glob("*.txt"))
    return txts[0] if txts else None


def _cleanup(folder: Path) -> None:
    for f in folder.glob("*keypoints.xlsx"):
        f.unlink(missing_ok=True)
    for f in folder.glob("*_overlay.mp4"):
        f.unlink(missing_ok=True)
    for f in folder.glob("*_corrected.mp4"):
        f.unlink(missing_ok=True)
    (folder / "analysis_result.json").unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch reprocess new_data/double_hit after a given folder name.")
    ap.add_argument("--base-dir", type=Path, default=Path("new_data/double_hit"))
    ap.add_argument("--start-after", type=str, required=True, help="Only process folders with name > this value.")
    ap.add_argument("--model-path", type=Path, default=Path("models/yolo11x-pose.pt"))
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on processed folders.")
    ap.add_argument("--progress", action="store_true", help="Show fisheye/overlay progress bars.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip folders that already have overlay+excel+json.")
    ap.add_argument("--log", type=Path, default=None, help="JSONL log path (default: <base-dir>/_batch_reprocess_log.jsonl).")
    args = ap.parse_args()

    base: Path = args.base_dir
    if not base.exists():
        raise SystemExit(f"Base dir not found: {base}")

    if not args.model_path.exists():
        raise SystemExit(f"Model weights not found at {args.model_path}")

    log_path = args.log or (base / "_batch_reprocess_log.jsonl")

    folders = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)
    to_process = [p for p in folders if p.name > args.start_after]
    if args.limit is not None:
        to_process = to_process[: args.limit]

    if not to_process:
        print("No folders to process (check --start-after).")
        return 0

    print(f"Loading YOLO model: {args.model_path}")
    model = YOLO(str(args.model_path))

    counts = {"processed": 0, "skipped": 0, "errors": 0}
    t0 = time.time()

    total = len(to_process)
    with log_path.open("a", encoding="utf-8") as log:
        for i, folder in enumerate(to_process, 1):
            rec = {"folder": str(folder), "name": folder.name, "status": None}
            try:
                video = _find_video(folder)
                txt = _find_txt(folder)
                if video is None or txt is None:
                    rec["status"] = "skipped_missing_inputs"
                    counts["skipped"] += 1
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(f"[{i}/{total}] {folder.name}: {rec['status']}")
                    continue

                corrected = folder / f"{video.stem}_corrected.mp4"
                overlay = folder / f"{video.stem}_overlay.mp4"
                excel = folder / f"{video.stem}_keypoints.xlsx"
                json_path = folder / "analysis_result.json"

                if args.skip_existing and corrected.exists() and overlay.exists() and excel.exists() and json_path.exists():
                    rec["status"] = "skipped_existing"
                    counts["skipped"] += 1
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(f"[{i}/{total}] {folder.name}: {rec['status']}")
                    continue

                _cleanup(folder)

                corrected_video_path = correct_fisheye_video(
                    input_path=str(video),
                    output_path=str(corrected),
                    progress=args.progress,
                )

                tracks = extract_tracks_from_video(corrected_video_path, model)
                left_x, left_y, right_x, right_y, norm_constant, video_angle = process_video_and_extract_data(tracks)

                save_keypoints_to_excel(left_x, left_y, right_x, right_y, str(excel))
                render_overlay_video(
                    video_path=Path(corrected_video_path),
                    output_path=overlay,
                    left_xdata=left_x,
                    left_ydata=left_y,
                    right_xdata=right_x,
                    right_ydata=right_y,
                    normalisation_constant=norm_constant if norm_constant else 1.0,
                    draw_skeleton=True,
                    draw_labels=True,
                    show_progress=args.progress,
                )

                phrase = parse_txt_file(str(txt))
                decision = referee_decision(
                    phrase,
                    left_x,
                    left_y,
                    right_x,
                    right_y,
                    normalisation_constant=norm_constant,
                )

                result_data = {
                    "normalisation_constant": norm_constant,
                    "video_angle": video_angle,
                    "winner": decision.get("winner"),
                    "reason": decision.get("reason"),
                    "left_pauses": decision.get("left_pauses"),
                    "right_pauses": decision.get("right_pauses"),
                    "blade_analysis": decision.get("blade_analysis"),
                    "blade_details": decision.get("blade_details"),
                    "speed_comparison": decision.get("speed_comparison"),
                    "lunge_detected": decision.get("lunge_detected"),
                }
                result_data = sanitize_for_json(result_data)
                json_path.write_text(json.dumps(result_data, indent=2), encoding="utf-8")

                rec["status"] = "ok"
                rec["video"] = str(video)
                rec["txt"] = str(txt)
                rec["corrected"] = str(corrected)
                rec["overlay"] = str(overlay)
                rec["excel"] = str(excel)
                rec["json"] = str(json_path)
                counts["processed"] += 1
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"[{i}/{total}] {folder.name}: ok")

            except Exception as e:
                rec["status"] = "error"
                rec["error"] = f"{type(e).__name__}: {e}"
                counts["errors"] += 1
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"[{i}/{total}] {folder.name}: error ({rec['error']})")
                # keep going

    dt = time.time() - t0
    print(json.dumps({"counts": counts, "seconds": dt, "log": str(log_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
