"""Utilities for normalizing keypoints and extracting motion features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

FRONT_FOOT_IDX = 16
BACK_FOOT_IDX = 15
FRONT_KNEE_IDX = 14
BACK_KNEE_IDX = 13
HIP_LEFT_IDX = 11
HIP_RIGHT_IDX = 12
SHOULDER_LEFT_IDX = 5
SHOULDER_RIGHT_IDX = 6
WRIST_LEFT_IDX = 9
WRIST_RIGHT_IDX = 10
ELBOW_LEFT_IDX = 7
ELBOW_RIGHT_IDX = 8

SWORD_HAND_IDX = {
    "left": WRIST_RIGHT_IDX,   # left fencer holds saber in right hand
    "right": WRIST_LEFT_IDX,   # right fencer holds saber in left hand
}
SWORD_ELBOW_IDX = {
    "left": ELBOW_RIGHT_IDX,
    "right": ELBOW_LEFT_IDX,
}
SWORD_SHOULDER_IDX = {
    "left": SHOULDER_RIGHT_IDX,
    "right": SHOULDER_LEFT_IDX,
}

KEYPOINT_COLUMNS = [f"kp_{i}" for i in range(17)]


def _dataframes_to_array(x_df: pd.DataFrame, y_df: pd.DataFrame) -> np.ndarray:
    """Stack X/Y coordinate DataFrames (frames x keypoints) into array."""
    # Ensure consistent column ordering even if Excel capitalization changes
    cols = [c for c in KEYPOINT_COLUMNS if c in x_df.columns]
    if len(cols) != len(KEYPOINT_COLUMNS):
        raise ValueError("Missing expected keypoint columns in Excel file")
    x_vals = x_df[cols].to_numpy(dtype=float)
    y_vals = y_df[cols].to_numpy(dtype=float)
    stacked = np.stack([x_vals, y_vals], axis=-1)  # shape (frames, kpts, 2)
    return stacked


@dataclass
class NormalizedKeypoints:
    left: np.ndarray
    right: np.ndarray
    center_x: float
    center_y: float
    scale: float
    reference_frame: int


def phrase_keypoints_to_arrays(phrase: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Convert pandas DataFrames in phrase dict to dense numpy arrays."""
    left = _dataframes_to_array(phrase["left_x"], phrase["left_y"])
    right = _dataframes_to_array(phrase["right_x"], phrase["right_y"])
    return left, right


def _first_valid_frame(*arrays: np.ndarray, kp_indices: Tuple[int, ...] = (FRONT_FOOT_IDX,)) -> Optional[int]:
    frames = arrays[0].shape[0]
    for i in range(frames):
        valid = True
        for arr in arrays:
            for kp in kp_indices:
                if np.isnan(arr[i, kp, 0]) or np.isnan(arr[i, kp, 1]):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return i
    return None


def normalize_keypoints(left: np.ndarray, right: np.ndarray) -> NormalizedKeypoints:
    """Translate/scale coordinates so strip center is at 0 and distance normalized."""
    frame_idx = _first_valid_frame(left, right, kp_indices=(FRONT_FOOT_IDX,))
    if frame_idx is None:
        raise ValueError("No valid frame with both front feet present")

    left_front = left[frame_idx, FRONT_FOOT_IDX, 0]
    right_front = right[frame_idx, FRONT_FOOT_IDX, 0]
    center_x = np.nanmean([left_front, right_front])
    initial_gap = np.abs(right_front - left_front)
    scale = initial_gap if initial_gap > 1e-6 else 1.0

    # Use hip midpoint for vertical centering if available
    hips = []
    for arr in (left, right):
        hip_vals = []
        for kp in (HIP_LEFT_IDX, HIP_RIGHT_IDX):
            if not np.isnan(arr[frame_idx, kp, 1]):
                hip_vals.append(arr[frame_idx, kp, 1])
        if hip_vals:
            hips.append(np.mean(hip_vals))
    center_y = np.nanmean(hips) if hips else 0.0

    def _normalize(arr: np.ndarray) -> np.ndarray:
        normed = arr.copy()
        normed[:, :, 0] = (normed[:, :, 0] - center_x) / scale
        normed[:, :, 1] = (normed[:, :, 1] - center_y) / scale
        return normed

    return NormalizedKeypoints(
        left=_normalize(left),
        right=_normalize(right),
        center_x=center_x,
        center_y=center_y,
        scale=scale,
        reference_frame=frame_idx,
    )


def _nan_safe_velocity(series: np.ndarray, fps: float) -> np.ndarray:
    velocity = np.full_like(series, np.nan, dtype=float)
    valid_frames = np.where(~np.isnan(series))[0]
    if valid_frames.size < 2:
        return velocity
    valid_values = series[valid_frames]
    diffs = np.diff(valid_values) * fps
    vel_valid = np.concatenate([[0.0], diffs])
    velocity[valid_frames] = vel_valid
    return velocity


def _nan_safe_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.full_like(a, np.nan, dtype=float)
    mask = (~np.isnan(a)) & (~np.isnan(b))
    diff[mask] = a[mask] - b[mask]
    return diff


def compute_fencer_timeseries(norm: NormalizedKeypoints, side: str, fps: float) -> Dict[str, np.ndarray]:
    data = norm.left if side == "left" else norm.right
    sword_hand = SWORD_HAND_IDX[side]
    sword_shoulder = SWORD_SHOULDER_IDX[side]
    sword_elbow = SWORD_ELBOW_IDX[side]

    front_foot_x = data[:, FRONT_FOOT_IDX, 0]
    back_foot_x = data[:, BACK_FOOT_IDX, 0]
    sword_x = data[:, sword_hand, 0]
    sword_y = data[:, sword_hand, 1]

    # Distances in normalized coordinate space
    extension = np.linalg.norm(data[:, sword_hand, :] - data[:, sword_shoulder, :], axis=1)
    elbow_angle_proxy = np.linalg.norm(data[:, sword_hand, :] - data[:, sword_elbow, :], axis=1)
    stance_width = _nan_safe_difference(front_foot_x, back_foot_x)

    return {
        "front_foot_x": front_foot_x,
        "back_foot_x": back_foot_x,
        "front_foot_velocity": _nan_safe_velocity(front_foot_x, fps),
        "sword_x": sword_x,
        "sword_y": sword_y,
        "sword_velocity": _nan_safe_velocity(sword_x, fps),
        "sword_extension": extension,
        "elbow_extension": elbow_angle_proxy,
        "stance_width": stance_width,
    }


def compute_pair_timeseries(norm: NormalizedKeypoints, fps: float) -> Dict[str, np.ndarray]:
    left = norm.left
    right = norm.right
    left_front = left[:, FRONT_FOOT_IDX, 0]
    right_front = right[:, FRONT_FOOT_IDX, 0]
    gap = _nan_safe_difference(right_front, left_front)
    center_advancement = (left_front + right_front) / 2.0
    return {
        "front_foot_gap": gap,
        "center_advancement": center_advancement,
        "gap_velocity": _nan_safe_velocity(gap, fps),
    }


def summarize_timeseries(series: Dict[str, np.ndarray]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for name, values in series.items():
        valid = values[~np.isnan(values)]
        if valid.size == 0:
            summary[f"{name}_nan"] = 1.0
            continue
        summary[f"{name}_mean"] = float(np.mean(valid))
        summary[f"{name}_max"] = float(np.max(valid))
        summary[f"{name}_min"] = float(np.min(valid))
        summary[f"{name}_std"] = float(np.std(valid))
    return summary


def extract_phrase_feature_bundle(phrase: Dict, fps: float) -> Dict[str, Dict[str, np.ndarray]]:
    left_arr, right_arr = phrase_keypoints_to_arrays(phrase)
    norm = normalize_keypoints(left_arr, right_arr)
    left_series = compute_fencer_timeseries(norm, "left", fps)
    right_series = compute_fencer_timeseries(norm, "right", fps)
    pair_series = compute_pair_timeseries(norm, fps)
    return {
        "timeseries": {
            "left": left_series,
            "right": right_series,
            "pair": pair_series,
        },
        "summary": {
            "left": summarize_timeseries(left_series),
            "right": summarize_timeseries(right_series),
            "pair": summarize_timeseries(pair_series),
        },
        "normalization": {
            "center_x": norm.center_x,
            "center_y": norm.center_y,
            "scale": norm.scale,
            "reference_frame": norm.reference_frame,
        },
    }


def bucket_events(events: List[Dict]) -> Dict[str, List[Dict]]:
    buckets: Dict[str, List[Dict]] = {
        "hit_left": [],
        "hit_right": [],
        "double_hit": [],
        "blade_contact": [],
        "lockout_start": [],
    }
    for ev in events:
        etype = ev.get("type")
        attrs = ev.get("attributes", {})
        if etype == "hit" and attrs.get("scorer") == "left":
            buckets["hit_left"].append(ev)
        elif etype == "hit" and attrs.get("scorer") == "right":
            buckets["hit_right"].append(ev)
        elif etype == "double_hit":
            buckets["double_hit"].append(ev)
        elif etype == "blade_contact":
            buckets["blade_contact"].append(ev)
        elif etype == "lockout_start":
            buckets["lockout_start"].append(ev)
    return buckets
