"""Heuristic referee overrides derived from debug_referee logic."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from data_loader import FencingPhraseLoader
from debug_referee import (
    BladeContact,
    FencingPhrase,
    detect_arm_extension,
    detect_pause_retreat_intervals,
    referee_decision,
)

def _phrase_to_fencing_phrase(phrase: Dict) -> FencingPhrase:
    events = phrase["events"]
    start_time = None
    lockout_start = None
    simultaneous_time = None
    for ev in events:
        if ev.get("type") == "start" and start_time is None:
            start_time = ev["time"]
        elif ev.get("type") == "lockout_start":
            lockout_start = ev["time"]
        elif ev.get("type") == "double_hit" and simultaneous_time is None:
            simultaneous_time = ev["time"]
    if simultaneous_time is None:
        hit_times = [ev["time"] for ev in events if ev.get("type") in ("hit", "double_hit")]
        if len(hit_times) >= 2:
            simultaneous_time = max(hit_times)
        elif hit_times:
            simultaneous_time = hit_times[0]
    blade_contacts = [
        BladeContact(time=ev["time"], frame=ev["frame"])
        for ev in events
        if ev.get("type") == "blade_contact"
    ]
    fps = 15.0
    start_frame = int(start_time * fps) if start_time is not None else 0
    sim_frame = int(simultaneous_time * fps) if simultaneous_time is not None else None
    return FencingPhrase(
        start_time=start_time or 0.0,
        start_frame=start_frame,
        simultaneous_hit_time=simultaneous_time,
        simultaneous_hit_frame=sim_frame,
        blade_contacts=blade_contacts,
        lockout_start=lockout_start,
        declared_winner=phrase.get("winner", ""),
        fps=fps,
    )


def _dataframe_to_dict(df) -> Dict[int, List[float]]:
    return {i: df[f"kp_{i}"].tolist() for i in range(17)}


def heuristic_predict(phrase: Dict) -> str:
    fencing_phrase = _phrase_to_fencing_phrase(phrase)
    left_x = _dataframe_to_dict(phrase["left_x"])
    left_y = _dataframe_to_dict(phrase["left_y"])
    right_x = _dataframe_to_dict(phrase["right_x"])
    right_y = _dataframe_to_dict(phrase["right_y"])
    result = referee_decision(fencing_phrase, left_x, left_y, right_x, right_y)
    winner = result.get("winner")
    if winner is None:
        winner = "right"
    winner = winner.lower()
    if winner not in {"left", "right"}:
        winner = "right"
    return winner


def evaluate_heuristic(root: str = "/workspace/training_data") -> float:
    loader = FencingPhraseLoader(root)
    phrases = loader.load_all()
    total = 0
    correct = 0
    for phrase in phrases:
        pred = heuristic_predict(phrase)
        actual = phrase["winner"]
        total += 1
        if pred == actual:
            correct += 1
    return correct / total if total else 0.0

if __name__ == "__main__":
    acc = evaluate_heuristic()
    print(f"Heuristic accuracy: {acc:.3f}")
