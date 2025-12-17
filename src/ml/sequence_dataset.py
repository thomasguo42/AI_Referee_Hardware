"""Utilities for building per-frame sequence tensors from phrase data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_loader import FencingPhraseLoader
from .feature_extraction import (
    compute_pair_timeseries,
    normalize_keypoints,
    phrase_keypoints_to_arrays,
)

EVENT_CHANNELS = ["blade_contact", "hit_left", "hit_right", "double_hit", "lockout_start"]


def _stack_positions(norm_data: np.ndarray) -> np.ndarray:
    frames = norm_data.shape[0]
    return norm_data.reshape(frames, -1)


def _compute_velocity(arr: np.ndarray, fps: float) -> np.ndarray:
    vel = np.zeros_like(arr)
    vel[1:] = (arr[1:] - arr[:-1]) * fps
    return vel


def _event_matrix(events: List[Dict], n_frames: int) -> np.ndarray:
    mat = np.zeros((n_frames, len(EVENT_CHANNELS)), dtype=np.float32)
    channel_map = {name: idx for idx, name in enumerate(EVENT_CHANNELS)}
    for ev in events:
        etype = ev.get("type")
        attrs = ev.get("attributes", {})
        frame = int(ev.get("frame", 0))
        frame = max(0, min(n_frames - 1, frame))
        if etype == "hit":
            scorer = attrs.get("scorer")
            if scorer == "left":
                mat[frame, channel_map["hit_left"]] = 1.0
            elif scorer == "right":
                mat[frame, channel_map["hit_right"]] = 1.0
        elif etype == "double_hit":
            mat[frame, channel_map["double_hit"]] = 1.0
        elif etype == "blade_contact":
            mat[frame, channel_map["blade_contact"]] = 1.0
        elif etype == "lockout_start":
            mat[frame, channel_map["lockout_start"]] = 1.0
    return mat


def build_phrase_sequence(phrase: Dict, fps: float) -> Tuple[np.ndarray, np.ndarray]:
    left_arr, right_arr = phrase_keypoints_to_arrays(phrase)
    norm = normalize_keypoints(left_arr, right_arr)
    left = norm.left
    right = norm.right
    frames = left.shape[0]

    left_pos = _stack_positions(left)
    right_pos = _stack_positions(right)
    left_vel = _compute_velocity(left_pos, fps)
    right_vel = _compute_velocity(right_pos, fps)

    pair_series = compute_pair_timeseries(norm, fps)
    gap = pair_series["front_foot_gap"].reshape(frames, 1)
    center = pair_series["center_advancement"].reshape(frames, 1)

    events = _event_matrix(phrase["events"], frames)

    features = np.concatenate([
        left_pos,
        right_pos,
        left_vel,
        right_vel,
        gap,
        center,
        events,
    ], axis=1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    mask = np.ones(frames, dtype=np.float32)
    return features.astype(np.float32), mask


@dataclass
class SequenceEntry:
    seq: np.ndarray
    mask: np.ndarray
    label: int
    folder: str
    session: str


def build_sequence_entries(root_dir: str) -> List[SequenceEntry]:
    loader = FencingPhraseLoader(root_dir)
    phrases = loader.load_all()
    entries: List[SequenceEntry] = []
    for phrase in phrases:
        seq, mask = build_phrase_sequence(phrase, loader.fps)
        label = 0 if phrase["winner"] == "left" else 1
        entries.append(SequenceEntry(
            seq=seq,
            mask=mask,
            label=label,
            folder=phrase["folder"],
            session=phrase["folder"].split('_')[0],
        ))
    return entries


class PhraseSequenceDataset(Dataset):
    def __init__(self, entries: Sequence[SequenceEntry], max_len: int):
        self.entries = entries
        self.max_len = max_len
        self.feature_dim = entries[0].seq.shape[1] if entries else 0

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        seq = entry.seq
        mask = entry.mask
        length = min(len(seq), self.max_len)
        padded_seq = np.zeros((self.max_len, self.feature_dim), dtype=np.float32)
        padded_mask = np.zeros(self.max_len, dtype=np.float32)
        padded_seq[:length] = seq[:length]
        padded_mask[:length] = mask[:length]
        return {
            "sequence": torch.from_numpy(padded_seq),
            "mask": torch.from_numpy(padded_mask),
            "label": torch.tensor(entry.label, dtype=torch.long),
            "length": length,
            "folder": entry.folder,
            "session": entry.session,
        }


def max_sequence_length(entries: Sequence[SequenceEntry]) -> int:
    return max((entry.seq.shape[0] for entry in entries), default=0)
