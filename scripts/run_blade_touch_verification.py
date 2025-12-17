#!/usr/bin/env python3
"""Evaluate debug_referee over blade_touch_data and copy phrases into correctness buckets."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

import sys
sys.path.append('scripts')

import debug_referee  # type: ignore

DATA_DIR = Path('data/blade_touch_data')
OUTPUT_ROOT = Path('results/blade_touch_verification')
WINNER_PATTERN = re.compile(r'(Confirmed result winner|Manual selection winner):\s*(Right|Left)', re.IGNORECASE)


def extract_winner(txt_path: Path) -> str:
    winner = None
    for line in txt_path.read_text(encoding='utf-8').splitlines():
        match = WINNER_PATTERN.search(line)
        if match:
            winner = match.group(2).lower()
    if not winner:
        raise ValueError(f"No declared winner in {txt_path}")
    return winner


def evaluate_folder(folder: Path):
    txt = next(folder.glob('*.txt'))
    excel = next(folder.glob('*keypoints.xlsx'))
    json_path = folder / 'analysis_result.json'
    phrase = debug_referee.parse_txt_file(str(txt))
    left_x, left_y, right_x, right_y = debug_referee.load_keypoints_from_excel(str(excel))
    norm = None
    if json_path.exists():
        try:
            import json
            with json_path.open('r', encoding='utf-8') as handle:
                norm = json.load(handle).get('normalisation_constant')
        except Exception:
            norm = None
    result = debug_referee.referee_decision(phrase, left_x, left_y, right_x, right_y, normalisation_constant=norm)
    winner = extract_winner(txt)
    return winner, result


def main():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    (OUTPUT_ROOT / 'correct').mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / 'mismatch').mkdir(parents=True, exist_ok=True)

    rows = []
    total = 0
    ok = 0
    for folder in sorted(DATA_DIR.iterdir()):
        if not folder.is_dir():
            continue
        try:
            winner, result = evaluate_folder(folder)
        except Exception as exc:
            print(f"Skipping {folder.name}: {exc}")
            continue
        total += 1
        predicted = (result.get('winner') or '').lower()
        correct = predicted == winner.lower()
        ok += int(correct)
        bucket = 'correct' if correct else 'mismatch'
        dest = OUTPUT_ROOT / bucket / folder.name
        shutil.copytree(folder, dest, dirs_exist_ok=True)
        rows.append({
            'name': folder.name,
            'winner': winner,
            'predicted': predicted,
            'correct': correct,
            'reason': result.get('reason'),
            'blade_analysis': result.get('blade_analysis'),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_ROOT / 'verification_results.csv', index=False)
    accuracy = ok / total if total else 0.0
    print(f"Evaluated {total} phrases. Accuracy = {accuracy:.3f}")
    print(f"Results written to {OUTPUT_ROOT}")


if __name__ == '__main__':
    main()
