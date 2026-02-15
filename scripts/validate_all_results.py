#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

# Import the debug referee logic
sys.path.append(str(Path(__file__).parent))
import debug_referee as debug_referee_module
from debug_referee import (
    extract_side_hit_events,
    load_keypoints_from_excel,
    parse_txt_file,
    referee_decision,
    sanitize_for_json,
)

WINNER_PRIORITIES = [
    "Confirmed result winner",
    "Remote referee winner overrides manual selection",
    "Remote referee winner",
    "Manual selection winner",
]

def has_relevant_blade_contact(phrase) -> bool:
    """Check if any blade contact is within 1s before hit and before lockout."""
    hit_time = phrase.simultaneous_hit_time
    if hit_time is None:
        return False
    for bc in phrase.blade_contacts:
        if (hit_time - 1.0) <= bc.time < hit_time:
            if phrase.lockout_start is None or bc.time < phrase.lockout_start:
                return True
    return False

def uses_blade_contact_for_judging(decision: dict) -> bool:
    """True only when blade contact analysis actually influenced the decision path."""
    return decision.get("blade_details") is not None


def _extract_winner(content: str, label: str) -> Optional[str]:
    pattern = rf"{re.escape(label)}:\s*(?P<winner>Right|Left|Abstain)(?:\\s+Fencer)?(?:\\s*\\([^)]*\\))?"
    match = re.search(pattern, content, re.IGNORECASE)
    if not match:
        return None
    winner = match.group("winner").strip().lower()
    return winner if winner in {"left", "right", "abstain"} else None


def get_actual_winner(txt_path: str) -> Optional[str]:
    """Extract the actual winner from txt file across historical formats."""
    try:
        with open(txt_path, 'r') as f:
            content = f.read()

        for label in WINNER_PRIORITIES:
            winner = _extract_winner(content, label)
            if winner:
                return winner

        return None
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
        return None


def prediction_matches(actual: str, predicted: Optional[str]) -> bool:
    if predicted is None:
        return False
    if actual == 'abstain':
        return predicted in {'left', 'right', 'abstain'}
    return actual == predicted

def main():
    parser = argparse.ArgumentParser(description="Re-run debug_referee across existing training data and collect mismatches.")
    parser.add_argument('--root', type=Path, default=Path('data/training_data'))
    parser.add_argument('--correct-dir', type=Path, default=Path('results/correct_results'))
    parser.add_argument('--mismatch-dir', type=Path, default=Path('results/mismatched_results'))
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    training_dir = args.root
    mismatch_dir = args.mismatch_dir
    correct_dir = args.correct_dir
    
    # Clean up directories
    if mismatch_dir.exists():
        shutil.rmtree(mismatch_dir)
    mismatch_dir.mkdir(parents=True, exist_ok=True)
    
    if correct_dir.exists():
        shutil.rmtree(correct_dir)
    correct_dir.mkdir(parents=True, exist_ok=True)

    mismatch_blade_dir = mismatch_dir / "blade_contact"
    mismatch_other_dir = mismatch_dir / "no_blade_contact"
    correct_blade_dir = correct_dir / "blade_contact"
    correct_other_dir = correct_dir / "no_blade_contact"
    for d in [mismatch_blade_dir, mismatch_other_dir, correct_blade_dir, correct_other_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_checked = 0
    mismatches_found = 0
    errors = 0
    skipped_no_winner = 0
    skipped_no_excel = 0
    
    print(f"Scanning {training_dir}...")
    print("=" * 80)
    
    subdirs = sorted([d for d in training_dir.iterdir() if d.is_dir()])
    
    processed = 0
    for item in subdirs:
        if args.limit is not None and processed >= args.limit:
            break
        # Find necessary files
        txt_files = list(item.glob("*.txt"))
        # Match scripts/debug_referee.py discovery behavior.
        excel_files = list(item.glob("*.xlsx"))
        json_path = item / "analysis_result.json"
        
        if not txt_files:
            continue
        
        if not excel_files:
            skipped_no_excel += 1
            continue
        
        txt_path = txt_files[0]
        excel_path = excel_files[0]
        
        try:
            total_processed += 1
            
            # Get actual winner from txt
            actual_winner = get_actual_winner(str(txt_path))
            
            if actual_winner is None:
                skipped_no_winner += 1
                print(f"[SKIP] {item.name}: No winner found in txt")
                continue
            
            # Load keypoints from Excel
            left_xdata, left_ydata, right_xdata, right_ydata = load_keypoints_from_excel(str(excel_path))
            
            # Parse phrase from txt
            phrase = parse_txt_file(str(txt_path))
            if left_xdata and 16 in left_xdata:
                max_frame = len(left_xdata[16]) - 1
                debug_referee_module._trim_phrase_to_frames(phrase, max_frame)
            side_hit_events = extract_side_hit_events(str(txt_path), fps=phrase.fps)
            # Load normalization constant from JSON if available
            norm_constant = None
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        old_data = json.load(f)
                        norm_constant = old_data.get("normalisation_constant")
                except:
                    pass
            
            # Run referee decision (now includes arm extension logic)
            decision = referee_decision(
                phrase,
                left_xdata, left_ydata,
                right_xdata, right_ydata,
                normalisation_constant=norm_constant,
                side_hit_events=side_hit_events,
            )
            blade_relevant = uses_blade_contact_for_judging(decision)
            
            predicted_winner = decision.get("winner")
            
            # Update JSON with new decision
            result_data = {}
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        result_data = json.load(f)
                except:
                    pass
            
            result_data["winner"] = predicted_winner
            result_data["reason"] = decision.get("reason")
            result_data["left_pauses"] = decision.get("left_pauses")
            result_data["right_pauses"] = decision.get("right_pauses")
            result_data["blade_analysis"] = decision.get("blade_analysis")
            result_data["blade_details"] = decision.get("blade_details")
            result_data["speed_comparison"] = decision.get("speed_comparison")
            result_data["lunge_detected"] = decision.get("lunge_detected")
            result_data["left_arm_extensions"] = decision.get("left_arm_extensions", [])
            result_data["right_arm_extensions"] = decision.get("right_arm_extensions", [])
            
            # Sanitize and save
            result_data = sanitize_for_json(result_data)
            with open(json_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            
            # Compare with actual winner
            total_checked += 1
            
            if not prediction_matches(actual_winner, predicted_winner):
                # Mismatch found!
                mismatches_found += 1
                
                # Copy entire folder to mismatched_results
                destination_root = mismatch_blade_dir if blade_relevant else mismatch_other_dir
                destination = destination_root / item.name
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(str(item), str(destination))
                
                print(f"[MISMATCH] {item.name}")
                print(f"  Actual: {actual_winner}, Predicted: {predicted_winner}")
                print(f"  Reason: {decision.get('reason')}")
                print()
            else:
                # Match found - copy to correct_results
                destination_root = correct_blade_dir if blade_relevant else correct_other_dir
                destination = destination_root / item.name
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(str(item), str(destination))
                
                match_label = actual_winner
                if actual_winner == 'abstain' and predicted_winner is not None:
                    match_label = f"abstain (predicted {predicted_winner})"
                print(f"[MATCH] {item.name}: {match_label} ✓")
        
        except Exception as e:
            errors += 1
            print(f"[ERROR] {item.name}: {e}")
        finally:
            processed += 1
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 80)
    print(f"\n=== SUMMARY ===")
    print(f"Total folders processed: {total_processed}")
    print(f"Skipped (no Excel): {skipped_no_excel}")
    print(f"Skipped (no winner in txt): {skipped_no_winner}")
    print(f"Total checked: {total_checked}")
    print(f"Matches: {total_checked - mismatches_found}")
    print(f"Mismatches: {mismatches_found}")
    print(f"Errors: {errors}")
    print(f"\nAccuracy: {((total_checked - mismatches_found) / total_checked * 100):.2f}%" if total_checked > 0 else "N/A")
    print(f"\nCorrect results copied to: {correct_dir}")
    print(f"Mismatched results copied to: {mismatch_dir}")

if __name__ == "__main__":
    main()
