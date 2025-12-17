#!/usr/bin/env python3
"""Run blade-touch and non-blade pipelines over training data and report accuracy."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# Ensure project modules are reachable
doc_root = Path(__file__).resolve().parent.parent
sys.path.append(str(doc_root))
sys.path.append(str(doc_root / 'blade_touch_rule'))

import scripts.blade_touch_referee as btr  # type: ignore
from blade_touch_rule import non_blade_rules as nbr  # type: ignore


def load_blade_samples(blade_dir: Path) -> Dict[str, btr.PhraseSample]:  # type: ignore
    samples = btr.collect_samples(blade_dir)
    return {sample.name: sample for sample in samples}


def load_non_blade_df(non_dir: Path) -> pd.DataFrame:
    records = nbr.build_records(non_dir)
    rows = []
    for rec in records:
        row = {'folder': rec.folder.name, 'winner': rec.winner}
        row.update(rec.features)
        rows.append(row)
    return pd.DataFrame(rows)


def split_folders(blade_names: List[str], non_blade_names: List[str], test_ratio: float, seed: int) -> Tuple[Set[str], Set[str]]:
    all_names = blade_names + non_blade_names
    rng = np.random.default_rng(seed)
    rng.shuffle(all_names)
    split_idx = int(len(all_names) * (1 - test_ratio))
    train_set = set(all_names[:split_idx])
    test_set = set(all_names[split_idx:])
    return train_set, test_set


def evaluate_blade(samples: Dict[str, btr.PhraseSample], train_names: Set[str], test_names: Set[str]) -> Tuple[int, int, float]:
    def filter_samples(name_set: Set[str]) -> List[btr.PhraseSample]:
        return [sample for name, sample in samples.items() if name in name_set]

    train_samples = filter_samples(train_names)
    test_samples = filter_samples(test_names)
    if not train_samples or not test_samples:
        return 0, 0, 0.0

    train_df, train_X, train_y, feature_cols = btr.build_dataset(train_samples)
    scaler, clf, log_scores = btr.train_logistic(train_X, train_y, feature_cols)
    gb_model, gb_scores = btr.train_gradient_boost(train_X, train_y)

    def predict(samples_subset: List[btr.PhraseSample]) -> Tuple[int, int]:
        df, X, y, _ = btr.build_dataset(samples_subset)
        if gb_model and np.mean(gb_scores) >= np.mean(log_scores):
            preds = gb_model.predict(X)
        else:
            preds = clf.predict(scaler.transform(X))
        correct = int((preds == y).sum())
        return correct, len(y)

    train_correct, train_total = predict(train_samples)
    test_correct, test_total = predict(test_samples)
    return test_correct, test_total, test_correct / test_total if test_total else 0.0


def evaluate_non_blade(df: pd.DataFrame, train_names: Set[str], test_names: Set[str]) -> Tuple[int, int, float]:
    train_df = df[df['folder'].isin(train_names)].reset_index(drop=True)
    test_df = df[df['folder'].isin(test_names)].reset_index(drop=True)
    if train_df.empty or test_df.empty:
        return 0, 0, 0.0
    params = nbr.tune(train_df)
    train_acc, _, _ = nbr.evaluate(train_df, params)
    test_acc, _, _ = nbr.evaluate(test_df, params)
    test_total = len(test_df)
    test_correct = int(round(test_acc * test_total))
    return test_correct, test_total, test_acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir', type=Path, default=Path('blade_touch_rule/training_data'))
    parser.add_argument('--blade-dir', type=Path, default=Path('blade_touch_rule/blade_touch_data'))
    parser.add_argument('--non-blade-dir', type=Path, default=Path('blade_touch_rule/non_blade_data'))
    parser.add_argument('--test-ratio', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()

    blade_samples = load_blade_samples(args.blade_dir)
    non_blade_df = load_non_blade_df(args.non_blade_dir)
    blade_names = list(blade_samples.keys())
    non_blade_names = non_blade_df['folder'].tolist()

    train_names, test_names = split_folders(blade_names, non_blade_names, args.test_ratio, args.seed)
    train_blade = train_names.intersection(blade_names)
    test_blade = test_names.intersection(blade_names)
    train_non_blade = train_names.intersection(non_blade_names)
    test_non_blade = test_names.intersection(non_blade_names)

    print(f"Train/Test split: {len(train_names)} train / {len(test_names)} test")
    blade_correct, blade_total, blade_acc = evaluate_blade(blade_samples, train_blade, test_blade)
    print(f"Blade-touch test: {blade_total} phrases, accuracy={blade_acc:.3%}")

    nb_correct, nb_total, nb_acc = evaluate_non_blade(non_blade_df, train_non_blade, test_non_blade)
    print(f"Non-blade test: {nb_total} phrases, accuracy={nb_acc:.3%}")

    combined_total = blade_total + nb_total
    combined_correct = blade_correct + nb_correct
    combined_acc = combined_correct / combined_total if combined_total else 0.0
    print(f"Combined test accuracy={combined_acc:.3%} over {combined_total} phrases")


if __name__ == '__main__':
    main()
