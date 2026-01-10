#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
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

def get_actual_winner(txt_path):
    try:
        with open(txt_path, 'r') as f:
            content = f.read()
            
            # Priority 1: Confirmed result winner
            match = re.search(r"Confirmed result winner: (Right|Left)", content, re.IGNORECASE)
            if match:
                return match.group(1).lower()
            
            # Priority 2: Manual selection winner
            match = re.search(r"Manual selection winner: (Right|Left)", content, re.IGNORECASE)
            if match:
                return match.group(1).lower()
                
            return None
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Re-run the full pipeline on recorded phrases (fisheye correction + tracking).")
    parser.add_argument('--training-dir', type=Path, default=Path('data/training_data'))
    parser.add_argument('--mismatch-dir', type=Path, default=Path('results/mismatched_results'))
    parser.add_argument('--model-path', type=Path, default=Path('models/yolo11x-pose.pt'))
    parser.add_argument('--limit', type=int, default=None, help='Optional limit on number of phrases to process')
    args = parser.parse_args()

    training_dir = args.training_dir
    mismatch_dir = args.mismatch_dir
    
    # Clean mismatch directory
    if mismatch_dir.exists():
        shutil.rmtree(mismatch_dir)
    mismatch_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Model
    if args.limit == 0:
        print("Limit set to 0; exiting without processing.")
        return

    if not args.model_path.exists():
        print(f"Model weights not found at {args.model_path}. Skipping reprocess run.")
        return

    print(f"Loading model: {args.model_path}...")
    model = YOLO(str(args.model_path))
        
    total_checked = 0
    mismatches_found = 0
    errors = 0
    
    print(f"Scanning {training_dir}...")
    
    subdirs = [d for d in training_dir.iterdir() if d.is_dir()]
    
    processed = 0
    for item in tqdm(subdirs, desc="Processing videos"):
        if args.limit is not None and processed >= args.limit:
            break
        # Find necessary files
        txt_files = list(item.glob("*.txt"))
        
        # Find video files (avi or mp4, excluding overlay)
        video_files = list(item.glob("*.avi"))
        if not video_files:
            video_files = [f for f in item.glob("*.mp4") if "_overlay" not in f.name and "_corrected" not in f.name]
            
        json_path = item / "analysis_result.json"
        
        if not txt_files or not video_files:
            continue
            
        txt_path = txt_files[0]
        video_path = video_files[0]
        
        try:
            # 0. CLEANUP: Remove old excel, overlay, and corrected files
            for f in item.glob("*keypoints.xlsx"):
                f.unlink()
            for f in item.glob("*_overlay.mp4"):
                f.unlink()
            for f in item.glob("*_corrected.mp4"):
                f.unlink()

            # 1. Correct Fisheye
            # We will generate a temporary corrected video path
            corrected_video_path = item / (video_path.stem + "_corrected.mp4")
            
            # If the input is already corrected (unlikely given our glob), skip
            # But we should just run correction.
            # Note: correct_fisheye_video returns the path as a string
            corrected_path_str = correct_fisheye_video(
                input_path=str(video_path),
                output_path=str(corrected_video_path),
                progress=False
            )
            
            # 2. Run YOLO Tracking & Process Data on CORRECTED video
            tracks_per_frame = extract_tracks_from_video(corrected_path_str, model)
            left_xdata, left_ydata, right_xdata, right_ydata, c, video_angle = \
                process_video_and_extract_data(tracks_per_frame)
            
            # 3. Save new Excel
            output_filename = video_path.stem + "_keypoints.xlsx"
            excel_path = item / output_filename
            save_keypoints_to_excel(left_xdata, left_ydata, right_xdata, right_ydata, str(excel_path))
            
            # 4. Load existing JSON for normalization constant (if available)
            norm_constant = None
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        old_data = json.load(f)
                        norm_constant = old_data.get("normalisation_constant")
                except json.JSONDecodeError:
                    print(f"Warning: Corrupt JSON found at {json_path}. Ignoring.")
                    old_data = {}
            else:
                old_data = {}
            
            # Use 'c' from tracking if available, otherwise fallback
            if c is not None:
                norm_constant = c

            # 5. Generate Overlay Video using CORRECTED video
            overlay_filename = video_path.stem + "_overlay.mp4"
            overlay_path = item / overlay_filename
            render_overlay_video(
                video_path=Path(corrected_path_str),
                output_path=overlay_path,
                left_xdata=left_xdata,
                left_ydata=left_ydata,
                right_xdata=right_xdata,
                right_ydata=right_ydata,
                normalisation_constant=norm_constant if norm_constant else 1.0,
                draw_skeleton=True,
                draw_labels=True,
                show_progress=False
            )

            # 5. Run Referee Decision
            phrase = parse_txt_file(str(txt_path))
            
            decision = referee_decision(
                phrase, 
                left_xdata, left_ydata, 
                right_xdata, right_ydata, 
                normalisation_constant=norm_constant
            )
            
            # 6. Update JSON
            result_data = old_data
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

            # Sanitize the entire dictionary
            result_data = sanitize_for_json(result_data)

            with open(json_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            # 7. Check Mismatch
            actual_winner = get_actual_winner(txt_path)
            processed_winner = decision.get("winner")
            
            if actual_winner and processed_winner:
                total_checked += 1
                if actual_winner != processed_winner:
                    # Copy to mismatch folder
                    destination = mismatch_dir / item.name
                    if destination.exists():
                        shutil.rmtree(destination)
                    shutil.copytree(str(item), str(destination))
                    mismatches_found += 1
            else:
                # print(f"Skipping comparison for {item.name}: Actual={actual_winner}, Processed={processed_winner}")
                pass
        
        except Exception as e:
            print(f"Error processing {item.name}: {e}")
            import traceback
            traceback.print_exc()
            errors += 1
        finally:
            processed += 1
            
    print(f"Total checked: {total_checked}")
    print(f"Mismatches found and copied: {mismatches_found}")
    print(f"Errors: {errors}")

if __name__ == "__main__":
    main()
