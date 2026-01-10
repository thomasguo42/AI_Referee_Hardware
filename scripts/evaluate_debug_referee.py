#!/usr/bin/env python3
"""Run debug_referee logic over blade_touch_data and report accuracy."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

# Ensure scripts package is available
import sys
sys.path.append('scripts')

import debug_referee  # type: ignore

WINNER_PATTERN = re.compile(r'(Confirmed result winner|Manual selection winner):\s*(Right|Left)', re.IGNORECASE)


def extract_winner(txt_path: Path) -> str:
    winner = None
    with txt_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            match = WINNER_PATTERN.search(line)
            if match:
                winner = match.group(2).lower()
    if not winner:
        raise ValueError(f"No declared winner in {txt_path}")
    return winner

DATA_DIR = Path('data/blade_touch_data')


def evaluate_folder(folder: Path):
    txt = next(folder.glob('*.txt'))
    excel = next(folder.glob('*keypoints.xlsx'))
    phrase = debug_referee.parse_txt_file(str(txt))
    left_x, left_y, right_x, right_y = debug_referee.load_keypoints_from_excel(str(excel))
    result = debug_referee.referee_decision(phrase, left_x, left_y, right_x, right_y)
    winner = extract_winner(txt)
    return winner, result


def main():
    rows = []
    ok = 0
    total = 0
    for folder in sorted(DATA_DIR.iterdir()):
        if not folder.is_dir():
            continue
        try:
            winner, result = evaluate_folder(folder)
        except Exception as exc:
            print(f"Skipping {folder.name}: {exc}")
            continue
        total += 1
        pred = result['winner']
        if pred is None:
            pred = 'none'
        correct = (pred.lower() == winner.lower())
        ok += int(correct)
        rows.append({
            'name': folder.name,
            'winner': winner,
            'predicted': pred,
            'correct': correct,
            'reason': result['reason'],
            'blade_analysis': result.get('blade_analysis'),
        })
    df = pd.DataFrame(rows)
    df.to_csv('results/debug_referee_eval.csv', index=False)
    accuracy = ok / total if total else 0
    print(f"Evaluated {total} phrases, accuracy {accuracy:.3f}")


if __name__ == '__main__':
    main()
