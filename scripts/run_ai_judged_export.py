#!/usr/bin/env python3
"""Run the ensemble pipeline and export predicted videos into correct/mismatch folders."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.referee.pipeline import build_feature_table, train_ensemble_model  # type: ignore

LABEL_MAP = {0: "left", 1: "right"}


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_phrase_videos(phrase_dir: Path, dest_dir: Path) -> List[str]:
    copied = []
    if not phrase_dir.exists():
        return copied
    for filename in os.listdir(phrase_dir):
        if filename.endswith(".mp4"):
            src = phrase_dir / filename
            dst = dest_dir / filename
            shutil.copy2(src, dst)
            copied.append(filename)
    return copied


def export_judged_phrases(root: Path, output_correct: Path, output_mismatch: Path, max_phrases: int | None, dry_run: bool) -> None:
    print("Building feature table...")
    df, _ = build_feature_table(str(root))
    df = df[df["label"].notna()].reset_index(drop=True)

    if max_phrases is not None:
        df = df.head(max_phrases).copy()

    print("Training ensemble and collecting predictions...")
    label_counts = df['label'].value_counts()
    if label_counts.size < 2 or label_counts.min() < 5:
        print("Not enough balanced labeled phrases to train ensemble; need at least five samples per class.")
        return
    model_report = train_ensemble_model(df)
    preds = model_report["predictions"].astype(int)
    threshold = model_report.get("threshold")

    df = df.iloc[: len(preds)].copy()
    df["pred_label"] = preds
    df["pred_label_text"] = df["pred_label"].map(LABEL_MAP)

    if dry_run:
        print("Dry run requested; not copying any files.")
        return

    _ensure_clean_dir(output_correct)
    _ensure_clean_dir(output_mismatch)

    total = len(df)
    correct = 0

    for idx, row in df.iterrows():
        folder = row["folder"]
        actual = row["label"]
        predicted = row["pred_label_text"]
        phrase_dir = root / folder
        is_correct = actual == predicted
        target_root = output_correct if is_correct else output_mismatch
        dest_dir = target_root / folder
        dest_dir.mkdir(parents=True, exist_ok=True)

        copied = _copy_phrase_videos(phrase_dir, dest_dir)
        summary_path = dest_dir / "prediction.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Folder: {folder}\n")
            f.write(f"Actual Winner: {actual}\n")
            f.write(f"Predicted Winner: {predicted}\n")
            if threshold is not None:
                f.write(f"Meta Threshold: {threshold:.3f}\n")
            f.write(f"Videos copied: {', '.join(copied) if copied else 'none'}\n")

        if is_correct:
            correct += 1

        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{total} phrases...")

    accuracy = correct / total if total else 0.0
    print(f"Done. Accuracy on training_data (using ensemble predictions): {accuracy:.3f}")
    print(f"Correct phrases stored in: {OUTPUT_CORRECT}")
    print(f"Mismatched phrases stored in: {OUTPUT_MISMATCH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/training_data'))
    parser.add_argument('--output-correct', type=Path, default=Path('results/AI_Judged_Correct'))
    parser.add_argument('--output-mismatch', type=Path, default=Path('results/AI_Judged_Mismatched'))
    parser.add_argument('--max-phrases', type=int, help='Limit number of phrases used when training/exporting')
    parser.add_argument('--dry-run', action='store_true', help='Train the ensemble but skip copying artifacts')
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(f"Training directory {args.root} does not exist")

    export_judged_phrases(args.root, args.output_correct, args.output_mismatch, args.max_phrases, args.dry_run)


if __name__ == "__main__":
    main()
