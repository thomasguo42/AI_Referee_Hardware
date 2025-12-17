#!/usr/bin/env python3
"""Blade-touch right-of-way heuristics based on keypoint dynamics."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts import blade_touch_referee as btr  # type: ignore

FPS = 15.0
TREE_THRESHOLDS = {
    "right_back_progress": 1.08,
    "left_weapon_lead_progress": -0.12,
    "front_gap": 0.65,
    "right_weapon_lead_progress": 0.37,
    "right_front_knee_velocity": 0.01,
}
THRESHOLD_TOLERANCE = 0.02


@dataclass
class RowDecision:
    folder: str
    predicted: Optional[str]
    reason: str
    winner: Optional[str]
    diagnostics: Dict[str, float]


def _load_phrase(folder: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int, str]:
    txt_files = sorted(folder.glob("*.txt"))
    excel_files = sorted(folder.glob("*keypoints.xlsx"))
    if not txt_files or not excel_files:
        raise FileNotFoundError(f"Missing TXT or Excel in {folder}")
    contact_time, winner = btr.parse_txt(txt_files[0])
    data = btr.load_keypoints(excel_files[0])
    contact_frame = int(round(contact_time * FPS))
    return data, contact_frame, winner


def _tree_decision(features: Dict[str, float]) -> Tuple[Optional[str], str]:
    rbp = features["right_back_progress"]
    lwlp = features["left_weapon_lead_progress"]
    fg = features["front_gap"]
    rwlp = features["right_weapon_lead_progress"]
    rfkv = features["right_front_knee_velocity"]

    close_to_split = any(
        abs(val - TREE_THRESHOLDS[key]) < THRESHOLD_TOLERANCE
        for key, val in (
            ("right_back_progress", rbp),
            ("left_weapon_lead_progress", lwlp),
            ("front_gap", fg),
            ("right_weapon_lead_progress", rwlp),
            ("right_front_knee_velocity", rfkv),
        )
    )

    if rbp <= TREE_THRESHOLDS["right_back_progress"]:
        if lwlp <= TREE_THRESHOLDS["left_weapon_lead_progress"]:
            reason = (
                "Right held ROW: left wrist was retracting while right back leg never committed."
            )
            return (None if close_to_split else "right"), reason
        if fg <= TREE_THRESHOLDS["front_gap"]:
            reason = (
                "Right held ROW: distance gap stayed tight so defender never seized space."
            )
            return (None if close_to_split else "right"), reason
        if rwlp <= TREE_THRESHOLDS["right_weapon_lead_progress"]:
            reason = (
                "Left beat ROW: right hand stalled after a loose distance, so left's blade took initiative."
            )
            return (None if close_to_split else "left"), reason
        reason = (
            "Right held ROW: even with space, right's blade kept progressing ahead of the hand."
        )
        return (None if close_to_split else "right"), reason

    # Right back foot drove through > 1.08 m
    if rfkv <= TREE_THRESHOLDS["right_front_knee_velocity"]:
        reason = (
            "Left riposte: right drove deep but front knee stalled, indicating a stop-hit opportunity."
        )
        return (None if close_to_split else "left"), reason
    reason = "Right attack completed: back leg and front knee stayed active through the contact."
    return (None if close_to_split else "right"), reason


def evaluate_phrase(folder: Path) -> RowDecision:
    data, contact_frame, winner = _load_phrase(folder)
    left_feat = btr.compute_fencer_features(data["left_x"], data["left_y"], contact_frame, +1.0)
    right_feat = btr.compute_fencer_features(data["right_x"], data["right_y"], contact_frame, -1.0)
    features = {**{f"left_{k}": v for k, v in left_feat.items()}, **{f"right_{k}": v for k, v in right_feat.items()}}
    features["front_gap"] = right_feat["front_now"] - left_feat["front_now"]
    predicted, reason = _tree_decision({
        "right_back_progress": features["right_back_progress"],
        "left_weapon_lead_progress": features["left_weapon_lead_progress"],
        "front_gap": features["front_gap"],
        "right_weapon_lead_progress": features["right_weapon_lead_progress"],
        "right_front_knee_velocity": features["right_front_knee_velocity"],
    })
    return RowDecision(
        folder=folder.name,
        predicted=predicted,
        reason=reason,
        winner=winner,
        diagnostics={
            "right_back_progress": features["right_back_progress"],
            "left_weapon_lead_progress": features["left_weapon_lead_progress"],
            "front_gap": features["front_gap"],
            "right_weapon_lead_progress": features["right_weapon_lead_progress"],
            "right_front_knee_velocity": features["right_front_knee_velocity"],
        },
    )


def run_directory(data_dir: Path) -> Tuple[List[RowDecision], float]:
    decisions: List[RowDecision] = []
    correct = 0
    total = 0
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        try:
            decision = evaluate_phrase(sub)
        except Exception as exc:
            print(f"[WARN] Skipping {sub.name}: {exc}")
            continue
        decisions.append(decision)
        if decision.predicted is None:
            continue
        total += 1
        if decision.predicted == decision.winner:
            correct += 1
    accuracy = correct / total if total else 0.0
    return decisions, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/blade_touch_data"))
    parser.add_argument("--output", type=Path, help="Optional JSON dump of per-phrase decisions")
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"Data directory {args.data_dir} not found")

    decisions, accuracy = run_directory(args.data_dir)
    total_preds = sum(1 for d in decisions if d.predicted)
    print(f"Evaluated {len(decisions)} phrases ({total_preds} firm ROW calls).")
    print(f"Right-of-way accuracy: {accuracy:.3%}")

    if args.output:
        export = []
        for d in decisions:
            export.append({
                "folder": d.folder,
                "predicted": d.predicted,
                "winner": d.winner,
                "reason": d.reason,
                "diagnostics": d.diagnostics,
            })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(export, handle, indent=2)
        print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()

