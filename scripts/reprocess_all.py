#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent))
from src.referee.analysis import (  # type: ignore
    extract_tracks_from_video,
    parse_txt_file,
    process_video_and_extract_data,
    referee_decision,
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
    parser = argparse.ArgumentParser(description="Re-run tracking + decision logic without fisheye correction.")
    parser.add_argument('--training-dir', type=Path, default=Path('data/training_data'))
    parser.add_argument('--mismatch-dir', type=Path, default=Path('results/mismatched_results'))
    parser.add_argument('--model-path', type=Path, default=Path('models/yolo11x-pose.pt'))
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    if args.limit == 0:
        print("Limit set to 0; exiting without processing.")
        return

    if not args.model_path.exists():
        print(f"Model weights not found at {args.model_path}. Skipping reprocess run.")
        return

    training_dir = args.training_dir
    mismatch_dir = args.mismatch_dir
    
    # Clean mismatch directory
    if mismatch_dir.exists():
        shutil.rmtree(mismatch_dir)
    mismatch_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Model
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
            video_files = [f for f in item.glob("*.mp4") if "_overlay" not in f.name]
            
        json_path = item / "analysis_result.json"
        
        if not txt_files or not video_files:
            # print(f"Skipping {item.name}: Missing txt or video")
            continue
            
        txt_path = txt_files[0]
        video_path = video_files[0]
        
        try:
            # 1. Run YOLO Tracking & Process Data
            tracks_per_frame = extract_tracks_from_video(str(video_path), model)
            left_xdata, left_ydata, right_xdata, right_ydata, c, video_angle = \
                process_video_and_extract_data(tracks_per_frame)
            
            # 2. Save new Excel
            output_filename = video_path.stem + "_keypoints.xlsx"
            excel_path = item / output_filename
            save_keypoints_to_excel(left_xdata, left_ydata, right_xdata, right_ydata, str(excel_path))
            
            # 3. Load existing JSON for normalization constant (if available)
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
            
            # If no norm constant, we might need to recalculate it or default?
            # The original code calculated it. Let's assume it persists or we can't easily recalc without the full pipeline
            # Actually, process_video_and_extract_data returns 'c' which IS the normalization constant (or related?)
            # No, 'c' in process_video_and_extract_data is likely confidence or something else?
            # Checking AI_Referee.py: process_video_and_extract_data returns (..., c, video_angle)
            # And in main(), c is used as normalisation_constant. So we have it!
            norm_constant = c

            print(f"Processed {item.name}: {len(tracks_per_frame)} frames tracked.")
            if not tracks_per_frame:
                print(f"Warning: No tracks found for {item.name}")

            # 4. Run Referee Decision
            phrase = parse_txt_file(str(txt_path))
            
            decision = referee_decision(
                phrase, 
                left_xdata, left_ydata, 
                right_xdata, right_ydata, 
                normalisation_constant=norm_constant
            )
            
            # 5. Update JSON
            result_data = old_data if json_path.exists() else {}
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
            
            # 6. Check Mismatch
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
                print(f"Skipping comparison for {item.name}: Actual={actual_winner}, Processed={processed_winner}")

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
