#!/usr/bin/env python3
"""Rule-based referee for phrases without blade contacts (motion-only cues)."""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

FPS = 15.0
FRONT_FOOT = 16
BACK_FOOT = 15
FRONT_WRIST = 10
FRONT_HIP = 12
CORE_POINTS = [5, 6, 11, 12]
BASELINE_WINDOW = 12
VELOCITY_LAG = 2
MOMENTUM_WINDOW = 12
RECENT_WINDOW = 6
PAUSE_VEL = 0.015
RETREAT_VEL = 0.025
BACK_FOOT_THRESH = 0.03
Y_VAR_THRESH = 8e-4
MIN_PAUSE_FRAMES = 4
ARM_THRESHOLD = 0.05
MIN_ARM_FRAMES = 3

PARAM_GRID = {
    'pause_margin': [0.4, 0.6],
    'pause_recent': [1.5],
    'initiative_high': [1.4],
    'initiative_margin': [0.6, 0.8],
    'baseline_margin': [0.2, 0.3],
    'wrist_margin': [0.8],
    'gap_margin': [0.3],
    'pressure_margin': [0.8],
    'arm_gap_margin': [0.05, 0.1],
    'arm_support_margin': [0.1, 0.3],
    'arm_velocity_margin': [1.5],
    'extension_margin': [0.3],
    'extension_fresh': [0.4, 0.5],
    'retreat_margin': [0.5, 0.6],
    'logit_threshold': [0.45, 0.5, 0.55],
}


@dataclass
class PhraseRecord:
    folder: Path
    hit_frame: int
    hit_time: float
    winner: str
    features: Dict[str, float]


@dataclass
class Prediction:
    folder: str
    prediction: Optional[str]
    reason: str
    winner: str
    tag: str


@dataclass
class FallbackModel:
    scaler: StandardScaler
    clf: LogisticRegression


HIT_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)s\s*\|\s*HIT", re.IGNORECASE)
WINNER_PREFIXES = (
    'manual selection winner:',
    'confirmed result winner:',
)
VALID_WINNERS = {'left', 'right', 'abstain'}


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _safe_mean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if values.size else 0.0


def parse_txt(txt_path: Path) -> Tuple[float, str]:
    hit_times: List[float] = []
    winner: Optional[str] = None
    with txt_path.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            hit_match = HIT_PATTERN.search(line)
            if hit_match:
                hit_times.append(float(hit_match.group(1)))
            lower = line.lower()
            for prefix in WINNER_PREFIXES:
                if prefix in lower:
                    tail = lower.split(prefix, 1)[1].strip()
                    if 'left' in tail:
                        winner = 'left'
                    elif 'right' in tail:
                        winner = 'right'
                    elif 'abstain' in tail:
                        winner = 'abstain'
                    break
            if winner is None and 'winner:' in lower and 'remote referee' not in lower:
                tail = lower.split('winner:', 1)[1].strip()
                if 'left' in tail:
                    winner = 'left'
                elif 'right' in tail:
                    winner = 'right'
                elif 'abstain' in tail:
                    winner = 'abstain'
    if not hit_times or winner not in VALID_WINNERS:
        raise ValueError('Missing hit time or winner in TXT')
    return hit_times[-1], winner  # last scoring hit


def load_keypoints(excel_path: Path) -> Dict[str, np.ndarray]:
    xls = pd.ExcelFile(excel_path)
    sheets = {name.strip().lower(): name for name in xls.sheet_names}
    required = ['left_x', 'left_y', 'right_x', 'right_y']
    missing = [sheet for sheet in required if sheet not in sheets]
    if missing:
        raise ValueError(f'Missing sheets {missing}')

    data: Dict[str, np.ndarray] = {}
    for logical in required:
        df = xls.parse(sheets[logical])
        ordered = sorted(df.columns, key=lambda col: int(col.split('_')[-1]))
        arr = df[ordered].to_numpy(dtype=float)
        filled = (
            pd.DataFrame(arr)
            .interpolate(limit_direction='both')
            .bfill()
            .ffill()
            .fillna(0.0)
        )
        data[logical] = filled.to_numpy()
    return data


def _window_drive(series: np.ndarray, frame_idx: int, direction: float) -> Tuple[float, float]:
    start = max(0, frame_idx - MOMENTUM_WINDOW)
    window = series[start:frame_idx + 1]
    if window.size < 2:
        return 0.0, 0.0
    diffs = direction * np.diff(window)
    forward = float(np.mean(np.clip(diffs, 0.0, None))) * FPS
    retreat = float(np.mean(np.clip(-diffs, 0.0, None))) * FPS
    return forward, retreat


def _detect_pause_segments(front_x: np.ndarray, front_y: np.ndarray, back_x: np.ndarray,
                           direction: float) -> List[Dict[str, float]]:
    intervals: List[Dict[str, float]] = []
    start_idx: Optional[int] = None

    def finalize(start: int, end: int) -> None:
        if end - start + 1 < MIN_PAUSE_FRAMES:
            return
        seg = slice(start, end + 1)
        y_var = float(np.nanvar(front_y[seg])) if front_y[seg].size else 0.0
        if y_var > Y_VAR_THRESH:
            return
        diffs = []
        for idx in range(start + 1, end + 1):
            cur = front_x[idx]
            prev = front_x[idx - 1]
            if np.isnan(cur) or np.isnan(prev):
                continue
            diffs.append((cur - prev) * direction)
        if not diffs:
            return
        avg_vel = float(np.mean(diffs))
        back_forward = 0.0
        for idx in range(start + 1, end + 1):
            cur = back_x[idx]
            prev = back_x[idx - 1]
            if np.isnan(cur) or np.isnan(prev):
                continue
            back_forward = max(back_forward, (cur - prev) * direction)
        if back_forward > BACK_FOOT_THRESH:
            return
        kind = 'retreat' if avg_vel < -RETREAT_VEL else 'pause'
        intervals.append({'start': start, 'end': end, 'kind': kind})

    for idx in range(1, len(front_x)):
        cur = front_x[idx]
        prev = front_x[idx - 1]
        vel = 0.0
        if not (np.isnan(cur) or np.isnan(prev)):
            vel = (cur - prev) * direction
        paused = abs(vel) < PAUSE_VEL or vel < -RETREAT_VEL
        if paused:
            if start_idx is None:
                start_idx = idx - 1
        elif start_idx is not None:
            finalize(start_idx, idx - 1)
            start_idx = None
    if start_idx is not None:
        finalize(start_idx, len(front_x) - 1)
    return intervals


def _detect_arm_extensions(xdata: np.ndarray, ydata: np.ndarray) -> List[Tuple[int, int]]:
    wrist_x = xdata[:, FRONT_WRIST]
    wrist_y = ydata[:, FRONT_WRIST]
    hip_x = xdata[:, FRONT_HIP]
    hip_y = ydata[:, FRONT_HIP]
    dists = np.sqrt((wrist_x - hip_x) ** 2 + (wrist_y - hip_y) ** 2)
    dists = np.nan_to_num(dists)
    base = np.nanmedian(dists[:BASELINE_WINDOW]) if dists.size else 0.0
    base = base if base > 0 else np.nanmedian(dists)
    normalized = dists - (base if base > 0 else 0.0)
    intervals: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, value in enumerate(normalized):
        if value > ARM_THRESHOLD:
            if start is None:
                start = idx
        elif start is not None:
            if idx - start >= MIN_ARM_FRAMES:
                intervals.append((start, idx - 1))
            start = None
    if start is not None and len(normalized) - start >= MIN_ARM_FRAMES:
        intervals.append((start, len(normalized) - 1))
    return intervals


def _latest_interval(intervals: List[Tuple[int, int]], frame_idx: int) -> Optional[Tuple[int, int]]:
    candidates = [interval for interval in intervals if interval[0] <= frame_idx]
    return max(candidates, key=lambda item: item[0]) if candidates else None


def compute_features(xdata: np.ndarray, ydata: np.ndarray, frame_idx: int, direction: float,
                     hit_time: float) -> Dict[str, float]:
    idx = _clamp(frame_idx, 0, xdata.shape[0] - 1)
    prev_idx = _clamp(idx - VELOCITY_LAG, 0, xdata.shape[0] - 1)
    base_end = max(1, min(BASELINE_WINDOW, idx))
    base_slice = slice(0, base_end)
    recent_idx = max(0, idx - RECENT_WINDOW)

    def _series(axis: str, kp: int) -> np.ndarray:
        source = xdata if axis == 'x' else ydata
        return source[:, kp]

    front_x = _series('x', FRONT_FOOT)
    front_y = _series('y', FRONT_FOOT)
    back_x = _series('x', BACK_FOOT)
    wrist_x = _series('x', FRONT_WRIST)
    wrist_y = _series('y', FRONT_WRIST)
    com_series = np.nanmean(xdata[:, CORE_POINTS], axis=1)

    front_progress = direction * (front_x[idx] - _safe_mean(front_x[base_slice]))
    front_velocity = direction * (front_x[idx] - front_x[prev_idx]) * FPS / max(1, idx - prev_idx)
    front_recent = direction * (front_x[idx] - front_x[recent_idx])
    forward_drive, retreat_drive = _window_drive(front_x, idx, direction)
    wrist_progress = direction * (wrist_x[idx] - _safe_mean(wrist_x[base_slice]))
    wrist_velocity = direction * (wrist_x[idx] - wrist_x[prev_idx]) * FPS / max(1, idx - prev_idx)
    wrist_recent = direction * (wrist_x[idx] - wrist_x[recent_idx])
    com_velocity = direction * (com_series[idx] - com_series[prev_idx]) * FPS / max(1, idx - prev_idx)
    com_recent = direction * (com_series[idx] - com_series[recent_idx])
    stance_now = abs(front_x[idx] - back_x[idx])
    stance_base = _safe_mean(np.abs(front_x[base_slice] - back_x[base_slice]))
    stance_expansion = stance_now - stance_base

    attack_threshold = max(0.04, 0.2 * abs(front_progress))
    attack_start = idx
    for frame in range(idx):
        if direction * (front_x[frame] - _safe_mean(front_x[base_slice])) >= attack_threshold:
            attack_start = frame
            break
    frames_since = max(1, idx - attack_start)
    attack_rate = front_progress / (frames_since / FPS)
    attack_start_time = attack_start / FPS

    pause_segments = _detect_pause_segments(front_x, front_y, back_x, direction)
    current_time = frame_idx / FPS
    if pause_segments:
        last = pause_segments[-1]
        pause_end_time = last['end'] / FPS
        pause_kind = 1.0 if last['kind'] == 'pause' else -1.0
        pause_duration = (last['end'] - last['start']) / FPS
        time_since_pause = current_time - pause_end_time
    else:
        pause_kind = 0.0
        pause_duration = 0.0
        time_since_pause = current_time + 5.0

    extensions = _detect_arm_extensions(xdata, ydata)
    ext_interval = _latest_interval(extensions, frame_idx)
    if ext_interval:
        start_frame = ext_interval[0]
        ext_time = start_frame / FPS
        ext_age = current_time - ext_time
        extension_flag = 1.0
    else:
        ext_time = -1.0
        ext_age = current_time + 5.0
        extension_flag = 0.0
    arm_gap = ext_time - attack_start_time if ext_time >= 0 else 1.0

    pressure = (
        1.0 * front_progress +
        0.8 * front_recent +
        0.8 * forward_drive +
        0.5 * wrist_recent +
        0.4 * com_recent -
        0.6 * retreat_drive
    )

    return {
        'front_progress': float(front_progress),
        'front_velocity': float(front_velocity),
        'forward_drive': float(forward_drive),
        'retreat_drive': float(retreat_drive),
        'wrist_progress': float(wrist_progress),
        'wrist_velocity': float(wrist_velocity),
        'wrist_recent': float(wrist_recent),
        'front_recent': float(front_recent),
        'com_recent': float(com_recent),
        'attack_rate': float(attack_rate),
        'attack_start_time': float(attack_start_time),
        'com_velocity': float(com_velocity),
        'stance_expansion': float(stance_expansion),
        'pressure': float(pressure),
        'time_since_pause': float(time_since_pause),
        'pause_kind': float(pause_kind),
        'pause_duration': float(pause_duration),
        'extension_flag': float(extension_flag),
        'extension_age': float(ext_age),
        'extension_time': float(ext_time),
        'arm_foot_gap': float(arm_gap),
        'front_abs': float(front_x[idx]),
        'front_baseline': float(_safe_mean(front_x[base_slice])),
        'com_abs': float(com_series[idx]),
    }


def build_records(root: Path) -> List[PhraseRecord]:
    records: List[PhraseRecord] = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        txt_files = sorted(folder.glob('*.txt'))
        excel_files = sorted(folder.glob('*keypoints.xlsx'))
        if not txt_files or not excel_files:
            continue
        try:
            hit_time, winner = parse_txt(txt_files[0])
            data = load_keypoints(excel_files[0])
        except Exception as exc:  # pragma: no cover - diagnostic prints
            print(f"[WARN] {folder.name}: {exc}")
            continue
        total_frames = data['left_x'].shape[0]
        hit_frame = _clamp(int(round(hit_time * FPS)), 0, total_frames - 1)
        left_feat = compute_features(data['left_x'], data['left_y'], hit_frame, +1.0, hit_time)
        right_feat = compute_features(data['right_x'], data['right_y'], hit_frame, -1.0, hit_time)
        front_gap = right_feat['front_abs'] - left_feat['front_abs']
        gap_base = right_feat['front_baseline'] - left_feat['front_baseline']
        gap_progress = gap_base - front_gap
        com_gap = right_feat['com_abs'] - left_feat['com_abs']
        features = {
            'left_front_progress': left_feat['front_progress'],
            'right_front_progress': right_feat['front_progress'],
            'left_front_velocity': left_feat['front_velocity'],
            'right_front_velocity': right_feat['front_velocity'],
            'left_forward_drive': left_feat['forward_drive'],
            'right_forward_drive': right_feat['forward_drive'],
            'left_retreat_drive': left_feat['retreat_drive'],
            'right_retreat_drive': right_feat['retreat_drive'],
            'left_wrist_velocity': left_feat['wrist_velocity'],
            'right_wrist_velocity': right_feat['wrist_velocity'],
            'left_wrist_progress': left_feat['wrist_progress'],
            'right_wrist_progress': right_feat['wrist_progress'],
            'left_wrist_recent': left_feat['wrist_recent'],
            'right_wrist_recent': right_feat['wrist_recent'],
            'left_attack_rate': left_feat['attack_rate'],
            'right_attack_rate': right_feat['attack_rate'],
            'left_attack_start': left_feat['attack_start_time'],
            'right_attack_start': right_feat['attack_start_time'],
            'left_com_velocity': left_feat['com_velocity'],
            'right_com_velocity': right_feat['com_velocity'],
            'left_com_recent': left_feat['com_recent'],
            'right_com_recent': right_feat['com_recent'],
            'left_recent_progress': left_feat['front_recent'],
            'right_recent_progress': right_feat['front_recent'],
            'left_pressure': left_feat['pressure'],
            'right_pressure': right_feat['pressure'],
            'left_arm_gap': left_feat['arm_foot_gap'],
            'right_arm_gap': right_feat['arm_foot_gap'],
            'left_stance_expansion': left_feat['stance_expansion'],
            'right_stance_expansion': right_feat['stance_expansion'],
            'left_time_since_pause': left_feat['time_since_pause'],
            'right_time_since_pause': right_feat['time_since_pause'],
            'left_pause_kind': left_feat['pause_kind'],
            'right_pause_kind': right_feat['pause_kind'],
            'left_pause_duration': left_feat['pause_duration'],
            'right_pause_duration': right_feat['pause_duration'],
            'left_extension_flag': left_feat['extension_flag'],
            'right_extension_flag': right_feat['extension_flag'],
            'left_extension_age': left_feat['extension_age'],
            'right_extension_age': right_feat['extension_age'],
            'left_extension_time': left_feat['extension_time'],
            'right_extension_time': right_feat['extension_time'],
            'front_gap': front_gap,
            'gap_progress': gap_progress,
            'com_gap': com_gap,
        }
        records.append(PhraseRecord(folder=folder, hit_frame=hit_frame, hit_time=hit_time,
                                    winner=winner, features=features))
    return records


def _logistic_matrix(df: pd.DataFrame) -> np.ndarray:
    columns = [
        df['left_front_progress'] - df['right_front_progress'],
        df['left_forward_drive'] - df['right_forward_drive'],
        df['right_retreat_drive'] - df['left_retreat_drive'],
        df['left_wrist_velocity'] - df['right_wrist_velocity'],
        df['left_wrist_progress'] - df['right_wrist_progress'],
        df['left_wrist_recent'] - df['right_wrist_recent'],
        df['left_attack_rate'] - df['right_attack_rate'],
        df['left_attack_start'] - df['right_attack_start'],
        df['left_com_velocity'] - df['right_com_velocity'],
        df['left_com_recent'] - df['right_com_recent'],
        df['left_stance_expansion'] - df['right_stance_expansion'],
        df['left_time_since_pause'] - df['right_time_since_pause'],
        df['left_extension_age'] - df['right_extension_age'],
        df['left_extension_flag'] - df['right_extension_flag'],
        df['left_front_velocity'] - df['right_front_velocity'],
        df['left_recent_progress'] - df['right_recent_progress'],
        df['left_pressure'] - df['right_pressure'],
        df['left_arm_gap'] - df['right_arm_gap'],
        df['gap_progress'],
    ]
    matrix = np.column_stack(columns)
    return np.nan_to_num(matrix)


def train_fallback(df: pd.DataFrame) -> Optional[FallbackModel]:
    df = augment_with_mirror(df)
    mask = df['winner'].isin(('left', 'right'))
    if mask.sum() < 10:
        return None
    feature_df = df.loc[mask]
    X = _logistic_matrix(feature_df)
    y = (feature_df['winner'] == 'left').astype(int).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=500, class_weight='balanced')
    clf.fit(X_scaled, y)
    return FallbackModel(scaler=scaler, clf=clf)


def _fallback_predict(series: pd.Series, fallback: FallbackModel) -> float:
    feature_df = pd.DataFrame([series])
    matrix = _logistic_matrix(feature_df)
    scaled = fallback.scaler.transform(matrix)
    probs = fallback.clf.predict_proba(scaled)[0]
    return float(probs[1])


LOGISTIC_OVERRIDE_TAGS = {
    'arm_init_left', 'arm_init_right', 'arm_ext_left', 'arm_ext_right',
    'arm_only_left', 'arm_only_right', 'initiative_high_left', 'initiative_high_right',
    'initiative_wrist_left', 'initiative_wrist_right', 'gap_left', 'gap_right',
    'pressure_left', 'pressure_right', 'retreat_break_left', 'retreat_break_right',
    'net_left', 'net_right', 'pause_left', 'pause_right'
}

OVERRIDE_THRESHOLDS = {
    'pause_left': 0.8,
    'pause_right': 0.8,
}

SOFT_TAGS = {'arm_init_left', 'arm_init_right', 'initiative_high_left', 'initiative_high_right'}


def _override_with_logistic(pred: Optional[str], reason: str, tag: str, prob_left: float) -> Tuple[Optional[str], str]:
    if pred is None or tag not in LOGISTIC_OVERRIDE_TAGS:
        return pred, reason
    prob_right = 1.0 - prob_left
    thresh_left = OVERRIDE_THRESHOLDS.get(tag, 0.6)
    thresh_right = OVERRIDE_THRESHOLDS.get(tag, 0.6)
    if prob_left >= thresh_left and pred != 'left':
        return 'left', f"{reason} -> logistic override ({prob_left:.2f})"
    if prob_right >= thresh_right and pred != 'right':
        return 'right', f"{reason} -> logistic override ({prob_right:.2f})"
    return pred, reason


def _initiative_score(row: pd.Series) -> float:
    diff = (
        0.35 * (row['left_front_progress'] - row['right_front_progress']) +
        0.35 * (row['left_forward_drive'] - row['right_forward_drive']) +
        0.20 * (row['right_retreat_drive'] - row['left_retreat_drive']) +
        0.15 * (row['left_front_velocity'] - row['right_front_velocity']) +
        0.15 * (row['left_wrist_velocity'] - row['right_wrist_velocity']) +
        0.15 * (row['left_wrist_recent'] - row['right_wrist_recent']) +
        0.10 * (row['left_attack_rate'] - row['right_attack_rate']) +
        0.10 * (row['left_com_velocity'] - row['right_com_velocity']) +
        0.10 * (row['left_com_recent'] - row['right_com_recent']) +
        0.10 * (row['left_recent_progress'] - row['right_recent_progress']) +
        0.15 * (row['left_pressure'] - row['right_pressure']) +
        0.10 * (row['left_stance_expansion'] - row['right_stance_expansion']) -
        0.10 * (row['left_arm_gap'] - row['right_arm_gap'])
    )
    return float(diff)


def _compare_winner(pred: Optional[str], winner: str) -> bool:
    if pred is None:
        return False
    if winner == 'abstain':
        return pred in ('left', 'right')
    return pred == winner


def _decision(row: pd.Series, params: Dict[str, float]) -> Tuple[Optional[str], str, str]:
    pause_gap = row['left_time_since_pause'] - row['right_time_since_pause']
    min_pause = min(row['left_time_since_pause'], row['right_time_since_pause'])
    front_support = row['left_front_progress'] - row['right_front_progress']
    pause_hint: Optional[str] = None
    if min_pause < params['pause_recent']:
        left_active = row['left_forward_drive'] >= 0.8 * max(0.1, row['right_forward_drive'])
        right_active = row['right_forward_drive'] >= 0.8 * max(0.1, row['left_forward_drive'])
        left_blade = row['left_wrist_velocity'] >= 0.8 * max(0.1, row['right_wrist_velocity'])
        right_blade = row['right_wrist_velocity'] >= 0.8 * max(0.1, row['left_wrist_velocity'])
        if pause_gap > params['pause_margin'] and left_active and left_blade and front_support >= params['arm_support_margin']:
            pause_hint = 'left'
        elif pause_gap < -params['pause_margin'] and right_active and right_blade and front_support <= -params['arm_support_margin']:
            pause_hint = 'right'
    if (
        row['right_pause_kind'] < 0
        and row['right_time_since_pause'] < params['retreat_margin']
        and (row['left_front_progress'] - row['right_front_progress']) >= params['arm_support_margin']
        and row['left_forward_drive'] >= row['right_forward_drive']
    ):
        return 'left', 'Right still recovering from retreat', 'retreat_left'
    if (
        row['left_pause_kind'] < 0
        and row['left_time_since_pause'] < params['retreat_margin']
        and (row['right_front_progress'] - row['left_front_progress']) >= params['arm_support_margin']
        and row['right_forward_drive'] >= row['left_forward_drive']
    ):
        return 'right', 'Left still recovering from retreat', 'retreat_right'

    # Arm extension precedence
    if row['left_extension_flag'] and row['right_extension_flag']:
        arm_gap_diff = row['left_arm_gap'] - row['right_arm_gap']
        if arm_gap_diff < -params['arm_gap_margin'] and (
            front_support >= params['arm_support_margin'] or
            (row['left_wrist_velocity'] - row['right_wrist_velocity']) >= params['arm_velocity_margin']
        ):
            return 'left', 'Left arm initiated first', 'arm_init_left'
        if arm_gap_diff > params['arm_gap_margin'] and (
            front_support <= -params['arm_support_margin'] or
            (row['right_wrist_velocity'] - row['left_wrist_velocity']) >= params['arm_velocity_margin']
        ):
            return 'right', 'Right arm initiated first', 'arm_init_right'
        ext_gap = row['right_extension_time'] - row['left_extension_time']
        if abs(ext_gap) > params['extension_margin']:
            if ext_gap > 0 and front_support >= params['arm_support_margin']:
                return 'left', 'Left extended first', 'arm_ext_left'
            if ext_gap < 0 and front_support <= -params['arm_support_margin']:
                return 'right', 'Right extended first', 'arm_ext_right'
    elif (
        row['left_extension_flag']
        and row['left_arm_gap'] <= params['arm_gap_margin']
        and front_support >= params['arm_support_margin']
        and (row['left_wrist_velocity'] - row['right_wrist_velocity']) >= params['arm_velocity_margin']
    ):
        return 'left', 'Only left extended', 'arm_only_left'
    elif (
        row['right_extension_flag']
        and row['right_arm_gap'] <= params['arm_gap_margin']
        and front_support <= -params['arm_support_margin']
        and (row['right_wrist_velocity'] - row['left_wrist_velocity']) >= params['arm_velocity_margin']
    ):
        return 'right', 'Only right extended', 'arm_only_right'

    score = _initiative_score(row)
    if score >= params['initiative_high']:
        return 'left', 'Left sustained initiative', 'initiative_high_left'
    if score <= -params['initiative_high']:
        return 'right', 'Right sustained initiative', 'initiative_high_right'

    if score >= params['initiative_margin']:
        wrist_gap = row['left_wrist_progress'] - row['right_wrist_progress']
        if wrist_gap > params['wrist_margin']:
            return 'left', 'Left initiative + blade work', 'initiative_wrist_left'
        gap_gain = row['gap_progress']
        if gap_gain > params['gap_margin']:
            return 'left', 'Left closed distance', 'gap_left'
    if score <= -params['initiative_margin']:
        wrist_gap = row['right_wrist_progress'] - row['left_wrist_progress']
        if wrist_gap > params['wrist_margin']:
            return 'right', 'Right initiative + blade work', 'initiative_wrist_right'
        if -row['gap_progress'] > params['gap_margin']:
            return 'right', 'Right closed distance', 'gap_right'

    pressure_diff = row['left_pressure'] - row['right_pressure']
    if pressure_diff > params['pressure_margin']:
        return 'left', 'Left pressure advantage', 'pressure_left'
    if pressure_diff < -params['pressure_margin']:
        return 'right', 'Right pressure advantage', 'pressure_right'

    retreat_gap = row['right_retreat_drive'] - row['left_retreat_drive']
    if retreat_gap > params['retreat_margin'] and row['left_forward_drive'] > 0:
        return 'left', 'Right is retreating', 'retreat_break_left'
    if retreat_gap < -params['retreat_margin'] and row['right_forward_drive'] > 0:
        return 'right', 'Left is retreating', 'retreat_break_right'

    if pause_hint == 'left':
        return 'left', 'Right paused closer to hit', 'pause_left'
    if pause_hint == 'right':
        return 'right', 'Left paused closer to hit', 'pause_right'

    if abs(score) < params['baseline_margin']:
        return None, 'Insufficient separation', 'undecided'
    return ('left', 'Left by net initiative', 'net_left') if score > 0 else ('right', 'Right by net initiative', 'net_right')


def records_to_df(records: List[PhraseRecord]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = {'folder': rec.folder.name, 'winner': rec.winner}
        row.update(rec.features)
        rows.append(row)
    return pd.DataFrame(rows)


def augment_with_mirror(df: pd.DataFrame) -> pd.DataFrame:
    mirrored_rows = []
    for row in df.to_dict(orient='records'):
        mirrored = dict(row)
        for col, value in row.items():
            if col.startswith('left_'):
                counterpart = 'right_' + col[len('left_'):]
                if counterpart in row:
                    mirrored[col] = row.get(counterpart, value)
                    mirrored[counterpart] = value
        if 'front_gap' in row:
            mirrored['front_gap'] = -row['front_gap']
        if 'gap_progress' in row:
            mirrored['gap_progress'] = -row['gap_progress']
        if 'com_gap' in row:
            mirrored['com_gap'] = -row['com_gap']
        winner = row.get('winner')
        if winner == 'left':
            mirrored['winner'] = 'right'
        elif winner == 'right':
            mirrored['winner'] = 'left'
        mirrored['folder'] = f"{row.get('folder', 'sample')}__mirror"
        mirrored_rows.append(mirrored)
    if mirrored_rows:
        return pd.concat([df, pd.DataFrame(mirrored_rows)], ignore_index=True)
    return df.copy()


def derive_soft_tags(df: pd.DataFrame, params: Dict[str, float], min_samples: int = 3, threshold: float = 0.7) -> Set[str]:
    tag_counts: Dict[str, List[int]] = {}
    for row in df.itertuples(index=False):
        series = pd.Series(row._asdict())
        pred, _, tag = _decision(series, params)
        if pred is None:
            continue
        total, correct = tag_counts.get(tag, [0, 0])
        total += 1
        if _compare_winner(pred, series['winner']):
            correct += 1
        tag_counts[tag] = [total, correct]
    soft: Set[str] = set()
    for tag, (total, correct) in tag_counts.items():
        if total >= min_samples:
            acc = correct / total if total else 0.0
            if acc < threshold:
                soft.add(tag)
    return soft


def evaluate(
    df: pd.DataFrame,
    params: Dict[str, float],
    fallback: Optional[FallbackModel] = None,
    soft_tags: Optional[Set[str]] = None,
) -> Tuple[float, float, List[Prediction]]:
    preds: List[Prediction] = []
    decided = 0
    correct = 0
    for row in df.itertuples(index=False):
        series = pd.Series(row._asdict())
        pred, reason, tag = _decision(series, params)
        log_pred = None
        log_prob = None
        if fallback is not None:
            log_prob = _fallback_predict(series, fallback)
            threshold = params.get('logit_threshold', 0.5)
            log_pred = 'left' if log_prob >= threshold else 'right'
        active_soft_tags = soft_tags or SOFT_TAGS
        if tag in active_soft_tags:
            reason = f"{reason} (soft hint)"
            pred = None
        if pred is None and fallback is not None:
            pred = log_pred
            reason = f"{reason} -> logistic fallback ({log_prob:.2f})"
        elif pred is not None and fallback is not None and log_prob is not None:
            pred, reason = _override_with_logistic(pred, reason, tag, log_prob)
        preds.append(Prediction(folder=series['folder'], prediction=pred, reason=reason,
                                winner=series['winner'], tag=tag))
        if pred is None:
            continue
        decided += 1
        if _compare_winner(pred, series['winner']):
            correct += 1
    accuracy = correct / decided if decided else 0.0
    coverage = decided / len(df) if len(df) else 0.0
    return accuracy, coverage, preds


def tune(df: pd.DataFrame) -> Dict[str, float]:
    df = augment_with_mirror(df)
    if df.empty:
        raise RuntimeError('Empty training data')
    best_params: Optional[Dict[str, float]] = None
    best_score = (-1.0, -1.0)
    folds: List[Tuple[pd.DataFrame, Optional[FallbackModel]]]
    if len(df) < 12:
        folds = [(df.reset_index(drop=True), train_fallback(df))]
    else:
        kfold = KFold(n_splits=4, shuffle=True, random_state=42)
        folds = []
        indices = np.arange(len(df))
        for train_idx, val_idx in kfold.split(indices):
            train_fold = df.iloc[train_idx].reset_index(drop=True)
            val_fold = df.iloc[val_idx].reset_index(drop=True)
            folds.append((val_fold, train_fallback(train_fold)))

    for pause_margin in PARAM_GRID['pause_margin']:
        for pause_recent in PARAM_GRID['pause_recent']:
            for initiative_high in PARAM_GRID['initiative_high']:
                for initiative_margin in PARAM_GRID['initiative_margin']:
                    for baseline_margin in PARAM_GRID['baseline_margin']:
                        for wrist_margin in PARAM_GRID['wrist_margin']:
                            for gap_margin in PARAM_GRID['gap_margin']:
                                for pressure_margin in PARAM_GRID['pressure_margin']:
                                    for arm_gap_margin in PARAM_GRID['arm_gap_margin']:
                                        for arm_support_margin in PARAM_GRID['arm_support_margin']:
                                            for arm_velocity_margin in PARAM_GRID['arm_velocity_margin']:
                                                for extension_margin in PARAM_GRID['extension_margin']:
                                                    for extension_fresh in PARAM_GRID['extension_fresh']:
                                                        for retreat_margin in PARAM_GRID['retreat_margin']:
                                                            for logit_threshold in PARAM_GRID['logit_threshold']:
                                                                params = {
                                                                'pause_margin': pause_margin,
                                                                'pause_recent': pause_recent,
                                                                'initiative_high': initiative_high,
                                                                'initiative_margin': initiative_margin,
                                                                'baseline_margin': baseline_margin,
                                                                'wrist_margin': wrist_margin,
                                                                'gap_margin': gap_margin,
                                                                'pressure_margin': pressure_margin,
                                                                'arm_gap_margin': arm_gap_margin,
                                                                'arm_support_margin': arm_support_margin,
                                                                'arm_velocity_margin': arm_velocity_margin,
                                                                'extension_margin': extension_margin,
                                                                'extension_fresh': extension_fresh,
                                                                'retreat_margin': retreat_margin,
                                                                'logit_threshold': logit_threshold,
                                                            }
                                                    accs: List[float] = []
                                                    covs: List[float] = []
                                                    for val_fold, fallback in folds:
                                                        acc, cov, _ = evaluate(val_fold, params, fallback)
                                                        accs.append(acc)
                                                        covs.append(cov)
                                                    score = (float(np.mean(accs)), float(np.mean(covs)))
                                                    if score > best_score:
                                                        best_score = score
                                                        best_params = params
    if best_params is None:
        raise RuntimeError('Unable to tune parameters')
    return best_params


def split(records: List[PhraseRecord], test_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for rec in records:
        row = {'folder': rec.folder.name, 'winner': rec.winner}
        row.update(rec.features)
        rows.append(row)
    df = pd.DataFrame(rows)
    stratify = df['winner'] if df['winner'].nunique() > 1 else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=seed,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('blade_touch_rule/non_blade_data'))
    parser.add_argument('--test-ratio', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    records = build_records(args.data_dir)
    if not records:
        raise SystemExit('No phrases available')
    train_df, test_df = split(records, args.test_ratio, args.seed)
    print(f'Dataset: {len(train_df)} train / {len(test_df)} test')
    params = tune(train_df)
    fallback = train_fallback(train_df)
    soft_tags = SOFT_TAGS.union(derive_soft_tags(train_df, params))
    print('Tuned params:', params)
    if fallback is None:
        print('Warning: fallback model unavailable (insufficient labels)')
    train_acc, train_cov, _ = evaluate(train_df, params, fallback, soft_tags)
    print(f'Train accuracy={train_acc:.3%} coverage={train_cov:.3%}')
    test_acc, test_cov, preds = evaluate(test_df, params, fallback, soft_tags)
    print(f'Test accuracy={test_acc:.3%} coverage={test_cov:.3%}')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([p.__dict__ for p in preds]).to_json(args.output, orient='records', indent=2)


if __name__ == '__main__':
    main()
