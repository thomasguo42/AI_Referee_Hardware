from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.append(str(SCRIPTS_DIR))

import blade_touch_referee as btr  # type: ignore

BLEND_WINDOW = btr.BLEND_WINDOW
VELOCITY_LAG = btr.VELOCITY_LAG
MOMENTUM_WINDOW = btr.MOMENTUM_WINDOW

LOGISTIC_MODEL_CACHE = None

DROP_TAIL_FRAMES = 1
LUNGE_BACKWARD_THRESHOLD = 0.01

DEBUG_LOGGING = True

def _debug(message: str) -> None:
    if DEBUG_LOGGING:
        print(message)

def _decide_attack_by_arm_and_speed(
    left_xdata: Dict[int, List[float]],
    left_ydata: Dict[int, List[float]],
    right_xdata: Dict[int, List[float]],
    right_ydata: Dict[int, List[float]],
    left_slow_start: Optional[PauseInterval],
    right_slow_start: Optional[PauseInterval],
    window_start: int,
    window_end: int,
    fps: float,
) -> Tuple[str, str, Dict, List[ArmExtensionInterval], List[ArmExtensionInterval]]:
    window_hit_frame = window_end
    left_extensions = detect_arm_extension(
        left_xdata,
        left_ydata,
        is_left_fencer=True,
        fps=fps,
        hit_frame=window_hit_frame,
        start_frame=window_start,
        end_frame=window_end,
        debug=True,
    )
    right_extensions = detect_arm_extension(
        right_xdata,
        right_ydata,
        is_left_fencer=False,
        fps=fps,
        hit_frame=window_hit_frame,
        start_frame=window_start,
        end_frame=window_end,
        debug=True,
    )

    left_latest_ext = left_extensions[-1] if (left_extensions and left_extensions[-1].near_hit) else None
    right_latest_ext = right_extensions[-1] if (right_extensions and right_extensions[-1].near_hit) else None

    if left_latest_ext and right_latest_ext:
        left_start_frame = left_latest_ext.effective_start_frame
        right_start_frame = right_latest_ext.effective_start_frame
        left_start = left_latest_ext.effective_start_time
        right_start = right_latest_ext.effective_start_time

        if left_start_frame == right_start_frame:
            if left_slow_start and not right_slow_start:
                return (
                    "right",
                    f'Arm extensions simultaneous ({left_start:.2f}s). Left penalized for slow start.',
                    {},
                    left_extensions,
                    right_extensions,
                )
            if right_slow_start and not left_slow_start:
                return (
                    "left",
                    f'Arm extensions simultaneous ({left_start:.2f}s). Right penalized for slow start.',
                    {},
                    left_extensions,
                    right_extensions,
                )

            left_speed, left_accel = calculate_speed_acceleration(
                left_xdata, left_ydata, start_frame=window_start, end_frame=window_end
            )
            right_speed, right_accel = calculate_speed_acceleration(
                right_xdata, right_ydata, start_frame=window_start, end_frame=window_end
            )

            speed_info = {
                'left_speed': left_speed,
                'left_accel': left_accel,
                'right_speed': right_speed,
                'right_accel': right_accel
            }

            if left_speed > right_speed:
                return (
                    "left",
                    f'Arm extensions simultaneous ({left_start:.2f}s vs {right_start:.2f}s). '
                    f'Left faster (speed: {left_speed:.3f} vs {right_speed:.3f})',
                    speed_info,
                    left_extensions,
                    right_extensions,
                )
            return (
                "right",
                f'Arm extensions simultaneous ({left_start:.2f}s vs {right_start:.2f}s). '
                f'Right faster (speed: {right_speed:.3f} vs {left_speed:.3f})',
                speed_info,
                left_extensions,
                right_extensions,
            )
        if left_start_frame < right_start_frame:
            return (
                "left",
                f'Left extended arm first ({left_start:.2f}s vs {right_start:.2f}s)',
                {},
                left_extensions,
                right_extensions,
            )
        return (
            "right",
            f'Right extended arm first ({right_start:.2f}s vs {left_start:.2f}s)',
            {},
            left_extensions,
            right_extensions,
        )

    if left_latest_ext:
        left_start = left_latest_ext.effective_start_time
        return (
            "left",
            f'Only left extended arm (at {left_start:.2f}s)',
            {},
            left_extensions,
            right_extensions,
        )

    if right_latest_ext:
        right_start = right_latest_ext.effective_start_time
        return (
            "right",
            f'Only right extended arm (at {right_start:.2f}s)',
            {},
            left_extensions,
            right_extensions,
        )

    left_speed, left_accel = calculate_speed_acceleration(
        left_xdata, left_ydata, start_frame=window_start, end_frame=window_end
    )
    right_speed, right_accel = calculate_speed_acceleration(
        right_xdata, right_ydata, start_frame=window_start, end_frame=window_end
    )

    speed_info = {
        'left_speed': left_speed,
        'left_accel': left_accel,
        'right_speed': right_speed,
        'right_accel': right_accel
    }

    if left_speed > right_speed:
        return (
            "left",
            f'No arm extensions detected. Left faster (speed: {left_speed:.3f} vs {right_speed:.3f})',
            speed_info,
            left_extensions,
            right_extensions,
        )
    return (
        "right",
        f'No arm extensions detected. Right faster (speed: {right_speed:.3f} vs {left_speed:.3f})',
        speed_info,
        left_extensions,
        right_extensions,
    )

def _pause_overlap_ok(
    left_pauses: List[PauseInterval],
    right_pauses: List[PauseInterval],
    fps: float,
    max_total_seconds: float = 1.0,
    min_overlap_ratio: float = 0.5,
    max_end_frame_delta: int = 3,
    left_last_end: Optional[int] = None,
    right_last_end: Optional[int] = None,
) -> Tuple[bool, float]:
    left_frames = set()
    for interval in left_pauses:
        left_frames.update(range(interval.start_frame, interval.end_frame + 1))
    right_frames = set()
    for interval in right_pauses:
        right_frames.update(range(interval.start_frame, interval.end_frame + 1))

    if not left_frames or not right_frames:
        return False, 0.0

    left_total = len(left_frames)
    right_total = len(right_frames)
    max_total_frames = fps * max_total_seconds
    if left_total >= max_total_frames or right_total >= max_total_frames:
        return False, 0.0

    overlap_frames = len(left_frames & right_frames)
    overlap_ratio = overlap_frames / max(left_total, right_total)
    if overlap_ratio < min_overlap_ratio:
        return False, overlap_ratio

    if left_last_end is not None and right_last_end is not None:
        if abs(left_last_end - right_last_end) > max_end_frame_delta:
            return False, overlap_ratio

    _debug(
        "[PauseOverlap] totals "
        f"left={left_total} right={right_total} overlap={overlap_frames} "
        f"ratio={overlap_ratio:.2f}"
    )
    return True, overlap_ratio

def _load_logistic_model():
    global LOGISTIC_MODEL_CACHE
    if LOGISTIC_MODEL_CACHE is None:
        model_path = PROJECT_ROOT / 'results' / 'blade_touch_referee_model.joblib'
        if not model_path.exists():
            return None
        payload = joblib.load(model_path)
        LOGISTIC_MODEL_CACHE = payload
    return LOGISTIC_MODEL_CACHE

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# --- Dataclasses ---

@dataclass
class BladeContact:
    """Represents a blade-to-blade contact"""
    time: float
    frame: int

@dataclass
class PauseInterval:
    """Represents a pause/retreat interval"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    is_retreat: bool = False

@dataclass
class LungeInterval:
    """Represents a lunge interval based on front/back foot distance."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float

@dataclass
class FencingPhrase:
    """
    Contains all data for a fencing phrase
    Important: Fencer 1 = Right fencer, Fencer 2 = Left fencer
    """
    start_time: float
    start_frame: int
    simultaneous_hit_time: Optional[float]  # Both fencers hit at this time
    simultaneous_hit_frame: Optional[int]
    blade_contacts: List[BladeContact]
    lockout_start: Optional[float]
    declared_winner: str
    fps: float = 15.0

# --- Helper Functions ---

def sanitize_for_json(value):
    """Convert numpy/dataclass values into JSON-serialisable primitives."""
    if value is None:
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    if is_dataclass(value):
        return sanitize_for_json(asdict(value))
    if hasattr(value, "tolist") and not isinstance(value, str):
        return sanitize_for_json(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value

def parse_txt_file(txt_path: str) -> FencingPhrase:
    """Parse the TXT file to extract timing information."""
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    start_time = None
    simultaneous_hit_time = None
    blade_contacts = []
    lockout_start = None
    declared_winner = None
    scoreboard_f1 = None
    scoreboard_f2 = None
    hit_events: List[float] = []
    
    for line in lines:
        if "Phrase recording started" in line:
            match = re.search(r'(\d+\.\d+)s', line)
            if match:
                start_time = float(match.group(1))
        
        if "Simultaneous valid hits" in line:
            match = re.search(r'(\d+\.\d+)s', line)
            if match:
                simultaneous_hit_time = float(match.group(1))

        if "Off-Target: Blade-to-blade contact" in line:
            match = re.search(r'(\d+\.\d+)s', line)
            if match:
                time = float(match.group(1))
                blade_contacts.append(BladeContact(time=time, frame=int(time * 15)))

        if "| HIT:" in line:
            match = re.search(r'(\d+\.\d+)s', line)
            if match:
                hit_events.append(float(match.group(1)))

        if "Lockout period started" in line:
            match = re.search(r'(\d+\.\d+)s', line)
            if match:
                lockout_start = float(match.group(1))

        if "Winner:" in line:
            match = re.search(r'Winner: (\w+)', line)
            if match:
                declared_winner = match.group(1)

        if "Scores ->" in line:
            match = re.search(
                r"Scores -> Fencer 1:\s*([A-Za-z]+),\s*Fencer 2:\s*([A-Za-z]+)",
                line,
            )
            if match:
                scoreboard_f1 = match.group(1).strip().upper()
                scoreboard_f2 = match.group(2).strip().upper()
    
    both_hit = (scoreboard_f1 == "HIT" and scoreboard_f2 == "HIT")

    if both_hit:
        if simultaneous_hit_time is None:
            if hit_events:
                simultaneous_hit_time = max(hit_events)
            elif lockout_start is not None:
                simultaneous_hit_time = lockout_start
            elif start_time is not None:
                simultaneous_hit_time = start_time
    else:
        simultaneous_hit_time = None

    fps = 15.0
    start_frame = int(start_time * fps) if start_time else 0
    simultaneous_hit_frame = int(simultaneous_hit_time * fps) if simultaneous_hit_time else None

    return FencingPhrase(
        start_time=start_time or 0.0,
        start_frame=start_frame,
        simultaneous_hit_time=simultaneous_hit_time,
        simultaneous_hit_frame=simultaneous_hit_frame,
        blade_contacts=blade_contacts,
        lockout_start=lockout_start,
        declared_winner=declared_winner,
        fps=fps
    )

def load_keypoints_from_excel(excel_path: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load keypoint data from Excel file"""
    df_left_x = pd.read_excel(excel_path, sheet_name='left_x')
    df_left_y = pd.read_excel(excel_path, sheet_name='left_y')
    df_right_x = pd.read_excel(excel_path, sheet_name='right_x')
    df_right_y = pd.read_excel(excel_path, sheet_name='right_y')
    
    left_xdata = {i: df_left_x[f'kp_{i}'].tolist() for i in range(17)}
    left_ydata = {i: df_left_y[f'kp_{i}'].tolist() for i in range(17)}
    right_xdata = {i: df_right_x[f'kp_{i}'].tolist() for i in range(17)}
    right_ydata = {i: df_right_y[f'kp_{i}'].tolist() for i in range(17)}
    
    if DROP_TAIL_FRAMES > 0:
        left_xdata = _trim_keypoint_data(left_xdata, DROP_TAIL_FRAMES)
        left_ydata = _trim_keypoint_data(left_ydata, DROP_TAIL_FRAMES)
        right_xdata = _trim_keypoint_data(right_xdata, DROP_TAIL_FRAMES)
        right_ydata = _trim_keypoint_data(right_ydata, DROP_TAIL_FRAMES)

    return left_xdata, left_ydata, right_xdata, right_ydata


def _dict_to_array(data: Dict[int, List[float]]) -> np.ndarray:
    return np.array([data[kp] for kp in range(17)], dtype=float).T

def _trim_keypoint_data(data: Dict[int, List[float]], drop_frames: int) -> Dict[int, List[float]]:
    if drop_frames <= 0:
        return data
    trimmed = {}
    for kp, values in data.items():
        trimmed[kp] = values[:-drop_frames] if len(values) > drop_frames else []
    return trimmed

def _trim_phrase_to_frames(phrase: FencingPhrase, max_frame: int) -> None:
    if max_frame < 0:
        phrase.blade_contacts = []
        phrase.simultaneous_hit_time = None
        phrase.simultaneous_hit_frame = None
        phrase.lockout_start = None
        return

    max_time = max_frame / phrase.fps if phrase.fps else None

    phrase.blade_contacts = [
        bc for bc in phrase.blade_contacts if bc.frame <= max_frame
    ]

    if max_time is not None:
        if phrase.simultaneous_hit_time is not None and phrase.simultaneous_hit_time > max_time:
            phrase.simultaneous_hit_time = None
            phrase.simultaneous_hit_frame = None
        if phrase.lockout_start is not None and phrase.lockout_start > max_time:
            phrase.lockout_start = None

    if phrase.simultaneous_hit_time is not None:
        phrase.simultaneous_hit_frame = int(phrase.simultaneous_hit_time * phrase.fps)

def calculate_center_of_mass(xdata: Dict, ydata: Dict, frame_idx: int) -> Tuple[float, float]:
    """Calculate center of mass from key body points (hips and shoulders)"""
    max_frame = len(xdata[5]) - 1
    if frame_idx > max_frame or frame_idx < 0:
        return np.nan, np.nan
    
    key_points = [5, 6, 11, 12]
    valid_x = []
    valid_y = []
    
    for kp in key_points:
        x = xdata[kp][frame_idx]
        y = ydata[kp][frame_idx]
        if not np.isnan(x) and not np.isnan(y):
            valid_x.append(x)
            valid_y.append(y)
    
    if not valid_x:
        return np.nan, np.nan
    
    return np.mean(valid_x), np.mean(valid_y)



def calculate_speed_acceleration(
    xdata: Dict,
    ydata: Dict,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> Tuple[float, float]:
    """Calculate average speed and acceleration for the dataset slice."""
    max_frame = len(xdata[16]) - 1
    if end_frame is None or end_frame > max_frame:
        end_frame = max_frame
    start_frame = max(0, min(start_frame, end_frame))
    
    front_foot_x = []
    com_x = []
    
    for i in range(start_frame, end_frame + 1):
        ff_x = xdata[16][i]
        if not np.isnan(ff_x):
            front_foot_x.append(ff_x)
        
        com_x_val, _ = calculate_center_of_mass(xdata, ydata, i)
        if not np.isnan(com_x_val):
            com_x.append(com_x_val)
    
    velocities = []
    for positions in [front_foot_x, com_x]:
        if len(positions) > 1:
            for i in range(1, len(positions)):
                vel = abs(positions[i] - positions[i-1])
                velocities.append(vel)
    
    avg_speed = np.mean(velocities) if velocities else 0
    
    if len(velocities) > 1:
        accelerations = [abs(velocities[i] - velocities[i-1]) for i in range(1, len(velocities))]
        avg_acceleration = np.mean(accelerations)
    else:
        avg_acceleration = 0
    
    return avg_speed, avg_acceleration

def detect_lunge_intervals(
    xdata: Dict[int, List[float]],
    ydata: Dict[int, List[float]],
    is_left_fencer: bool,
    fps: float = 15.0,
    threshold: float = 1.0,
    min_consecutive: int = 3,
    end_frame_buffer: int = 4,
) -> List[LungeInterval]:
    """Detect lunge intervals based on front/back foot distance."""
    max_frame = len(xdata[16]) - 1
    intervals: List[LungeInterval] = []
    if max_frame < 0:
        return intervals

    side = "left" if is_left_fencer else "right"
    _debug(
        f"[Lunge:{side}] start max_frame={max_frame} threshold={threshold} "
        f"min_consecutive={min_consecutive} end_frame_buffer={end_frame_buffer}"
    )

    lunge_frames = []
    for i in range(0, max_frame + 1):
        if i >= len(xdata[15]) or i >= len(xdata[16]):
            _debug(f"[Lunge:{side}] frame={i} missing keypoints")
            lunge_frames.append(False)
            continue
        x_rear = xdata[15][i]
        y_rear = ydata[15][i]
        x_front = xdata[16][i]
        y_front = ydata[16][i]
        if any(np.isnan([x_rear, y_rear, x_front, y_front])):
            _debug(f"[Lunge:{side}] frame={i} nan keypoints")
            lunge_frames.append(False)
            continue
        dist = math.hypot(x_front - x_rear, y_front - y_rear)
        is_lunge = dist > threshold
        _debug(f"[Lunge:{side}] frame={i} dist={dist:.3f} lunge={is_lunge}")
        lunge_frames.append(is_lunge)

    expected_direction = 1 if is_left_fencer else -1
    start = None
    for i, is_lunge in enumerate(lunge_frames):
        if is_lunge and start is None:
            start = i
            _debug(f"[Lunge:{side}] interval start frame={start}")
        elif not is_lunge and start is not None:
            end = i - 1
            if end - start + 1 >= min_consecutive:
                _debug(f"[Lunge:{side}] interval candidate {start}-{end} accepted length={end - start + 1}")
                intervals.append(
                    LungeInterval(
                        start_frame=start,
                        end_frame=end,
                        start_time=start / fps,
                        end_time=end / fps,
                    )
                )
            else:
                _debug(f"[Lunge:{side}] interval candidate {start}-{end} rejected length={end - start + 1}")
            start = None

    if start is not None:
        end = len(lunge_frames) - 1
        if end - start + 1 >= min_consecutive:
            _debug(f"[Lunge:{side}] interval candidate {start}-{end} accepted length={end - start + 1}")
            intervals.append(
                LungeInterval(
                    start_frame=start,
                    end_frame=end,
                    start_time=start / fps,
                    end_time=end / fps,
                )
            )
        else:
            _debug(f"[Lunge:{side}] interval candidate {start}-{end} rejected length={end - start + 1}")

    valid_intervals = []
    for interval in intervals:
        if interval.end_frame > (max_frame - end_frame_buffer):
            _debug(
                f"[Lunge:{side}] interval {interval.start_frame}-{interval.end_frame} "
                f"rejected end_frame within buffer"
            )
            continue
        has_backward = False
        for f_idx in range(interval.start_frame + 1, interval.end_frame + 1):
            if f_idx >= len(xdata[15]):
                continue
            curr_bf = xdata[15][f_idx]
            prev_bf = xdata[15][f_idx - 1]
            if np.isnan(curr_bf) or np.isnan(prev_bf):
                continue
            bf_vel = (curr_bf - prev_bf) * expected_direction
            if bf_vel < -LUNGE_BACKWARD_THRESHOLD:
                _debug(
                    f"[Lunge:{side}] interval {interval.start_frame}-{interval.end_frame} "
                    f"rejected backward frame={f_idx} bf_vel={bf_vel:.4f}"
                )
                has_backward = True
                break
        if not has_backward:
            _debug(
                f"[Lunge:{side}] interval {interval.start_frame}-{interval.end_frame} accepted"
            )
            valid_intervals.append(interval)
    return valid_intervals

# --- Core Logic Functions (Editable for Debugging) ---

def detect_pause_retreat_intervals(xdata: Dict, ydata: Dict, is_left_fencer: bool, 
                                   fps: float = 15.0) -> List[PauseInterval]:
    """
    Detect pause/retreat intervals for a fencer using simplified logic:
    - Use only front foot (keypoint 16)
    - No smoothing
    - Raw velocity check
    - Y-variance filter on front foot
    - Back foot (keypoint 15) movement filter
    - Assumes entire dataset range
    """
    intervals = []
    
    start_frame = 0
    end_frame = len(xdata[16]) - 1
    
    if start_frame >= end_frame:
        return intervals
    
    # Get front foot positions
    front_foot_x = [xdata[16][i] for i in range(start_frame, end_frame + 1)]
    front_foot_y = [ydata[16][i] for i in range(start_frame, end_frame + 1)]
    
    # Calculate raw velocities of front foot
    velocities = []
    for i in range(1, len(front_foot_x)):
        if not np.isnan(front_foot_x[i]) and not np.isnan(front_foot_x[i-1]):
            vel = front_foot_x[i] - front_foot_x[i-1]
        else:
            vel = 0
        velocities.append(vel)

    # print (velocities) # Removed raw velocity print to reduce noise
    
    expected_direction = 1 if is_left_fencer else -1
    
    # --- TUNABLE PARAMS ---
    pause_threshold = 0.035
    retreat_threshold = 0.035 # Threshold to distinguish Retreat from Pause
    min_pause_frames = 4
    y_variance_threshold = 0.001
    back_foot_threshold = 0.05 # Threshold for back foot movement
    # ----------------------
    
    pause_frames: List[Tuple[List[int], bool]] = []
    current_pause_frames = []

    side = "left" if is_left_fencer else "right"

    def process_and_filter_interval(frames):
        # print ("Processing frames:", frames) # Clean up
        if len(frames) < min_pause_frames:
            return

        # 1. Determine if Pause or Retreat
        # Get velocities for these frames
        interval_vels = []
        for f_idx in frames:
            v_idx = f_idx - start_frame - 1
            if 0 <= v_idx < len(velocities):
                interval_vels.append(abs(velocities[v_idx]))
        
        if not interval_vels:
            return

        avg_abs_vel = np.mean(interval_vels)
        _debug(f"[PauseDetect:{side}] interval {frames[0]}-{frames[-1]} avg_abs_vel={avg_abs_vel:.4f}")
        
        # If average velocity is high, it's a retreat (valid break of ROW)
        # We skip variance and back foot checks for retreats
        if avg_abs_vel > retreat_threshold:
            _debug(f"[PauseDetect:{side}] classified=RETREAT threshold={retreat_threshold:.4f}")
            pause_frames.append((frames, True))
            return

        _debug(f"[PauseDetect:{side}] classified=PAUSE threshold={retreat_threshold:.4f}")
        
        # Filter: Back Foot Movement (Keypoint 15)
        # Identify valid frames where back foot velocity is within threshold
        valid_frames = []
        for f_idx in frames:
            # Calculate bf_vel for this frame
            bf_vel = 0.0
            if f_idx > 0 and f_idx < len(xdata[15]):
                curr_bf = xdata[15][f_idx]
                prev_bf = xdata[15][f_idx-1]
                if not np.isnan(curr_bf) and not np.isnan(prev_bf):
                    bf_vel = (curr_bf - prev_bf) * expected_direction
            
            # Check threshold (max limit for forward movement)
            # If bf_vel is high positive (moving forward), we reject the frame.
            # If bf_vel is low or negative (retreating), we keep it.
            if bf_vel < back_foot_threshold:
                valid_frames.append(f_idx)
            else:
                _debug(
                    f"[PauseDetect:{side}] reject frame={f_idx} back_foot_vel={bf_vel:.4f} "
                    f"threshold={back_foot_threshold:.4f}"
                )

        # Split valid_frames into continuous segments
        if not valid_frames:
            return

        segments = []
        if valid_frames:
            current_segment = [valid_frames[0]]
            for i in range(1, len(valid_frames)):
                if valid_frames[i] == valid_frames[i-1] + 1:
                    current_segment.append(valid_frames[i])
                else:
                    segments.append(current_segment)
                    current_segment = [valid_frames[i]]
            segments.append(current_segment)
        
        # Check each segment
        for segment in segments:
            # Check Length
            if len(segment) < min_pause_frames:
                continue
            
            # Check Y-Variance
            y_coords = []
            for f_idx in segment:
                idx = f_idx - start_frame
                if 0 <= idx < len(front_foot_y):
                    y = front_foot_y[idx]
                    if not np.isnan(y):
                        y_coords.append(y)
            
            if len(y_coords) > 1:
                y_var = np.var(y_coords)
                _debug(
                    f"[PauseDetect:{side}] segment {segment[0]}-{segment[-1]} "
                    f"y_var={y_var:.6f} threshold={y_variance_threshold:.6f}"
                )
                if y_var >= y_variance_threshold:
                    _debug(f"[PauseDetect:{side}] reject segment=high_y_variance")
                    continue # Failed Y-var check
            
            # Passed checks
            _debug(f"[PauseDetect:{side}] accept segment {segment[0]}-{segment[-1]}")
            pause_frames.append((segment, False))
    
    for i, vel in enumerate(velocities):
        frame_idx = start_frame + i + 1
        
        # Check if paused (near zero velocity) or retreating (opposite direction)
        is_paused = (abs(vel) < pause_threshold) or (vel * expected_direction < 0)
        
        if is_paused:
            current_pause_frames.append(frame_idx)
        else:
            # print ("Processing interval:", current_pause_frames) # Clean up
            process_and_filter_interval(current_pause_frames)
            current_pause_frames = []
    
    # Handle end of loop
    process_and_filter_interval(current_pause_frames)
    
    for pf, is_retreat in pause_frames:
        intervals.append(PauseInterval(
            start_frame=pf[0],
            end_frame=pf[-1],
            start_time=pf[0] / fps,
            end_time=pf[-1] / fps,
            duration=(pf[-1] - pf[0]) / fps,
            is_retreat=is_retreat
        ))
    
    return intervals


def _extract_slow_starts(
    pauses: List[PauseInterval],
    fps: float,
) -> Tuple[List[PauseInterval], Optional[PauseInterval]]:
    """Reclassify opening short pauses as slow starts."""
    if not pauses:
        return pauses, None

    max_frames = int(fps * 1.0)
    remaining: List[PauseInterval] = []
    slow_start: Optional[PauseInterval] = None

    for interval in pauses:
        length_frames = interval.end_frame - interval.start_frame + 1
        if interval.start_frame <= 2 and length_frames < max_frames:
            if slow_start is None or interval.end_frame > slow_start.end_frame:
                slow_start = interval
            continue
        remaining.append(interval)

    return remaining, slow_start

@dataclass
class ArmExtensionInterval:
    """Represents an arm extension interval"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    avg_distance: float
    avg_angle: float
    effective_start_frame: int
    effective_start_time: float
    near_hit: bool

def detect_arm_extension(xdata: Dict, ydata: Dict, is_left_fencer: bool,
                        fps: float = 15.0,
                        hit_frame: Optional[int] = None,
                        start_frame: int = 0,
                        end_frame: Optional[int] = None,
                        debug: bool = False) -> List[ArmExtensionInterval]:
    """Detect arm extension intervals based on reach duration and elbow straightening."""
    intervals: List[ArmExtensionInterval] = []

    hand_kp = 10  # weapon wrist
    hip_kp = 12   # front hip
    shoulder_kp = 6  # weapon shoulder
    elbow_kp = 8     # weapon elbow

    distance_threshold = 0.5
    min_extension_frames = 1
    max_hit_frame_gap = 4
    straight_angle_threshold = 140.0  # degrees

    total_frames = len(xdata[hand_kp])
    if end_frame is None or end_frame >= total_frames:
        end_frame = total_frames - 1
    start_frame = max(0, min(start_frame, end_frame))
    if end_frame - start_frame < 1:
        return intervals

    distance_by_frame: Dict[int, float] = {}
    reach_mask: List[bool] = []
    window_frames = list(range(start_frame, end_frame + 1))

    for frame_idx in window_frames:
        hand_x, hand_y = xdata[hand_kp][frame_idx], ydata[hand_kp][frame_idx]
        hip_x, hip_y = xdata[hip_kp][frame_idx], ydata[hip_kp][frame_idx]

        valid_distance = not (
            np.isnan(hand_x) or np.isnan(hand_y) or np.isnan(hip_x) or np.isnan(hip_y)
        )
        dist = math.hypot(hand_x - hip_x, hand_y - hip_y) if valid_distance else 0.0
        distance_by_frame[frame_idx] = dist
        reach_mask.append(valid_distance and dist >= distance_threshold)

    extension_frames: List[List[int]] = []
    current_frames: List[int] = []
    for idx, extended in enumerate(reach_mask):
        frame_idx = window_frames[idx]
        if extended:
            current_frames.append(frame_idx)
        else:
            if len(current_frames) >= min_extension_frames:
                extension_frames.append(current_frames.copy())
            current_frames = []
    if len(current_frames) >= min_extension_frames:
        extension_frames.append(current_frames)

    if not extension_frames:
        return intervals

    latest_frames = extension_frames[-1]
    if not latest_frames:
        return intervals

    if hit_frame is None:
        hit_frame = end_frame

    near_hit = True
    if hit_frame - latest_frames[-1] > max_hit_frame_gap:
        near_hit = False

    def _elbow_angle(frame_idx: int) -> Optional[float]:
        shoulder_x, shoulder_y = xdata[shoulder_kp][frame_idx], ydata[shoulder_kp][frame_idx]
        elbow_x, elbow_y = xdata[elbow_kp][frame_idx], ydata[elbow_kp][frame_idx]
        hand_x, hand_y = xdata[hand_kp][frame_idx], ydata[hand_kp][frame_idx]
        if (
            np.isnan(shoulder_x) or np.isnan(shoulder_y)
            or np.isnan(elbow_x) or np.isnan(elbow_y)
            or np.isnan(hand_x) or np.isnan(hand_y)
        ):
            return None
        upper = np.array([shoulder_x - elbow_x, shoulder_y - elbow_y])
        lower = np.array([hand_x - elbow_x, hand_y - elbow_y])
        if np.linalg.norm(upper) == 0 or np.linalg.norm(lower) == 0:
            return None
        cos_theta = np.dot(upper, lower) / (np.linalg.norm(upper) * np.linalg.norm(lower))
        cos_theta = max(-1.0, min(1.0, float(cos_theta)))
        return math.degrees(math.acos(cos_theta))

    effective_start = None
    angle_samples: List[float] = []
    for frame in latest_frames:
        angle = _elbow_angle(frame)
        if angle is None:
            continue
        angle_samples.append(angle)
        if effective_start is None and angle >= straight_angle_threshold:
            effective_start = frame

    debug_side = 'left' if is_left_fencer else 'right'
    if effective_start is None:
        if debug:
            _debug(f"[ArmExt:{debug_side}] no straightening >= {straight_angle_threshold}deg in latest reach interval")
        return intervals

    avg_dist = float(np.mean([distance_by_frame[f] for f in latest_frames])) if latest_frames else 0.0
    avg_angle = float(np.mean(angle_samples)) if angle_samples else 0.0

    if debug:
        _debug(
            f"[ArmExt:{debug_side}] reach interval {latest_frames[0]}-{latest_frames[-1]} (len={len(latest_frames)}) "
            f"avg distance={avg_dist:.3f} (threshold {distance_threshold})"
        )
        _debug(
            f"[ArmExt:{debug_side}] effective start frame {effective_start} angle >= {straight_angle_threshold}°"
        )

    intervals.append(ArmExtensionInterval(
        start_frame=latest_frames[0],
        end_frame=latest_frames[-1],
        start_time=latest_frames[0] / fps,
        end_time=latest_frames[-1] / fps,
        duration=(latest_frames[-1] - latest_frames[0]) / fps,
        avg_distance=avg_dist,
        avg_angle=avg_angle,
        effective_start_frame=effective_start,
        effective_start_time=effective_start / fps,
        near_hit=near_hit
    ))

    return intervals


def analyze_blade_contact(left_xdata: Dict, left_ydata: Dict, right_xdata: Dict,
                         right_ydata: Dict, contact_frame: int, 
                         current_right_of_way: str = 'none',
                         attack_variance_threshold: float = 0.1) -> Tuple[str, Dict]:
    """Determine blade priority via learned logistic feature scoring."""

    left_x_arr, left_y_arr = _dict_to_array(left_xdata), _dict_to_array(left_ydata)
    right_x_arr, right_y_arr = _dict_to_array(right_xdata), _dict_to_array(right_ydata)

    left_feat = btr.compute_fencer_features(left_x_arr, left_y_arr, contact_frame, direction=+1.0)
    right_feat = btr.compute_fencer_features(right_x_arr, right_y_arr, contact_frame, direction=-1.0)

    features = {f'left_{k}': v for k, v in left_feat.items()}
    features.update({f'right_{k}': v for k, v in right_feat.items()})
    features['front_gap'] = right_feat['front_now'] - left_feat['front_now']
    features['front_gap_change'] = right_feat['front_progress'] - left_feat['front_progress']
    features['front_velocity_gap'] = right_feat['front_velocity'] - left_feat['front_velocity']
    features['stance_gap'] = right_feat['stance_now'] - left_feat['stance_now']

    diff_fields = [
        'front_progress',
        'front_velocity',
        'front_velocity_mean_window',
        'front_velocity_peak_window',
        'front_wrist_progress',
        'front_knee_progress',
        'front_height_change',
        'weapon_lead',
        'weapon_lead_progress',
        'weapon_vs_com',
        'stance_progress',
        'com_progress',
        'attack_lead_time',
        'attack_progress_rate',
    ]
    for key in diff_fields:
        lkey = f'left_{key}'
        rkey = f'right_{key}'
        if lkey in features and rkey in features:
            features[f'delta_{key}'] = features[lkey] - features[rkey]

    model_payload = _load_logistic_model()
    rationale = ''
    winner = current_right_of_way if current_right_of_way in {'left', 'right'} else 'right'

    if model_payload:
        feature_names = model_payload['features']
        missing = [name for name in feature_names if name not in features]
        if not missing:
            vector = np.array([[features[name] for name in feature_names]], dtype=float)
            scaler = model_payload['scaler']
            model = model_payload['model']
            if scaler is not None:
                vector = scaler.transform(vector)
            probs = model.predict_proba(vector)[0]
            pred_label = 'left' if probs[1] >= 0.5 else 'right'
            winner = pred_label
            rationale = f'logistic momentum vote (p_left={probs[1]:.2f})'
        else:
            rationale = f'missing features for logistic model: {missing}'
    else:
        rationale = 'logistic model unavailable, falling back to pause ROW'

    details = {
        'contact_frame': contact_frame,
        'left_features': left_feat,
        'right_features': right_feat,
        'logistic_available': bool(model_payload),
        'rationale': rationale,
    }

    _debug(f"[BladeContact] frame={contact_frame} rationale={rationale} winner={winner}")
    return winner, details

def referee_decision(phrase: FencingPhrase, left_xdata: Dict, left_ydata: Dict,
                    right_xdata: Dict, right_ydata: Dict,
                    normalisation_constant: Optional[float] = None) -> Dict:
    """Main refereeing logic"""
    result = {
        'winner': None,
        'reason': '',
        'left_pauses': [],
        'right_pauses': [],
        'left_slow_start': None,
        'right_slow_start': None,
        'blade_analysis': None,
        'blade_details': None,
        'speed_comparison': None,
        'lunge_detected': {'left': [], 'right': [], 'latest': None},
    }
    
    # Removed hit_frame usage for pause detection range
    
    left_pauses_raw = detect_pause_retreat_intervals(
        left_xdata, left_ydata, is_left_fencer=True, fps=phrase.fps
    )
    right_pauses_raw = detect_pause_retreat_intervals(
        right_xdata, right_ydata, is_left_fencer=False, fps=phrase.fps
    )

    left_pauses = left_pauses_raw
    right_pauses = right_pauses_raw
    left_pauses, left_slow_start = _extract_slow_starts(left_pauses, phrase.fps)
    right_pauses, right_slow_start = _extract_slow_starts(right_pauses, phrase.fps)
    result['left_slow_start'] = asdict(left_slow_start) if left_slow_start else None
    result['right_slow_start'] = asdict(right_slow_start) if right_slow_start else None
    
    result['left_pauses'] = left_pauses
    result['right_pauses'] = right_pauses
    
    left_last_pause_end = max([p.end_time for p in left_pauses]) if left_pauses else None
    right_last_pause_end = max([p.end_time for p in right_pauses]) if right_pauses else None
    left_last_pause_end_frame = max([p.end_frame for p in left_pauses]) if left_pauses else None
    right_last_pause_end_frame = max([p.end_frame for p in right_pauses]) if right_pauses else None

    left_lunges = detect_lunge_intervals(left_xdata, left_ydata, is_left_fencer=True, fps=phrase.fps)
    right_lunges = detect_lunge_intervals(right_xdata, right_ydata, is_left_fencer=False, fps=phrase.fps)
    result['lunge_detected'] = {
        'left': [asdict(l) for l in left_lunges],
        'right': [asdict(l) for l in right_lunges],
        'latest': None,
    }

    latest_left = left_lunges[-1] if left_lunges else None
    latest_right = right_lunges[-1] if right_lunges else None
    latest_side = None
    latest_interval = None
    if latest_left and latest_right:
        if latest_left.end_frame > latest_right.end_frame:
            latest_side, latest_interval = 'left', latest_left
        elif latest_right.end_frame > latest_left.end_frame:
            latest_side, latest_interval = 'right', latest_right
    elif latest_left:
        latest_side, latest_interval = 'left', latest_left
    elif latest_right:
        latest_side, latest_interval = 'right', latest_right

    if latest_side and latest_interval:
        result['lunge_detected']['latest'] = {
            'side': latest_side,
            'start_frame': latest_interval.start_frame,
            'end_frame': latest_interval.end_frame,
            'start_time': latest_interval.start_time,
            'end_time': latest_interval.end_time,
        }

        other_pause_end_frame = (
            right_last_pause_end_frame if latest_side == 'left' else left_last_pause_end_frame
        )
        if other_pause_end_frame is None or other_pause_end_frame < (latest_interval.end_frame + 4):
            winner = 'right' if latest_side == 'left' else 'left'
            result['winner'] = winner
            result['reason'] = (
                f'{winner.capitalize()} wins: pause end before lunge window end '
                f'({latest_interval.end_frame + 4}) after {latest_side} lunge'
            )
            return result
    
    hit_time = phrase.simultaneous_hit_time
    hit_frame = phrase.simultaneous_hit_frame
    if hit_frame is None and hit_time is not None:
        hit_frame = int(hit_time * phrase.fps)

    valid_blade_contacts = []
    if hit_time is not None:
        valid_blade_contacts = [
            bc for bc in phrase.blade_contacts
            if (hit_time - 1.0) <= bc.time < hit_time
            and (phrase.lockout_start is None or bc.time < phrase.lockout_start)
        ]
    
    last_blade_contact = valid_blade_contacts[-1] if valid_blade_contacts else None

    arm_debug = not left_pauses and not right_pauses and not last_blade_contact
    left_extensions = detect_arm_extension(
        left_xdata, left_ydata, is_left_fencer=True, fps=phrase.fps, hit_frame=hit_frame, debug=arm_debug
    )
    right_extensions = detect_arm_extension(
        right_xdata, right_ydata, is_left_fencer=False, fps=phrase.fps, hit_frame=hit_frame, debug=arm_debug
    )

    result['left_arm_extensions'] = [asdict(e) for e in left_extensions]
    result['right_arm_extensions'] = [asdict(e) for e in right_extensions]

    overlap_ok, overlap_ratio = _pause_overlap_ok(
        left_pauses,
        right_pauses,
        phrase.fps,
        left_last_end=left_last_pause_end_frame,
        right_last_end=right_last_pause_end_frame,
    )
    if overlap_ok:
        window_start = max(left_last_pause_end_frame, right_last_pause_end_frame)
        window_end = len(left_xdata[16]) - 1
        winner, detail, speed_info, left_ext, right_ext = _decide_attack_by_arm_and_speed(
            left_xdata,
            left_ydata,
            right_xdata,
            right_ydata,
            left_slow_start,
            right_slow_start,
            window_start=window_start,
            window_end=window_end,
            fps=phrase.fps,
        )
        result['left_arm_extensions'] = [asdict(e) for e in left_ext]
        result['right_arm_extensions'] = [asdict(e) for e in right_ext]
        if speed_info:
            result['speed_comparison'] = speed_info
        result['winner'] = winner
        result['reason'] = (
            f'Pauses overlap (>50%, <1s, ratio={overlap_ratio:.2f}). {detail}'
        )
        return result

    if not left_pauses and not right_pauses and not last_blade_contact:
        window_start = 0
        window_end = len(left_xdata[16]) - 1
        winner, detail, speed_info, left_ext, right_ext = _decide_attack_by_arm_and_speed(
            left_xdata,
            left_ydata,
            right_xdata,
            right_ydata,
            left_slow_start,
            right_slow_start,
            window_start=window_start,
            window_end=window_end,
            fps=phrase.fps,
        )
        result['left_arm_extensions'] = [asdict(e) for e in left_ext]
        result['right_arm_extensions'] = [asdict(e) for e in right_ext]
        if speed_info:
            result['speed_comparison'] = speed_info
        result['winner'] = winner
        result['reason'] = f'No pauses/blade contacts. {detail}'

        return result
    
    left_extension_reset_time = None
    if left_extensions:
        left_latest_any = left_extensions[-1]
        if not left_latest_any.near_hit:
            left_extension_reset_time = left_latest_any.effective_start_time

    right_extension_reset_time = None
    if right_extensions:
        right_latest_any = right_extensions[-1]
        if not right_latest_any.near_hit:
            right_extension_reset_time = right_latest_any.effective_start_time

    def _compose_reset(time_pause, time_ext):
        source = None
        latest_time = None
        if time_pause is not None:
            latest_time = time_pause
            source = 'pause'
        if time_ext is not None and (latest_time is None or time_ext > latest_time):
            latest_time = time_ext
            source = 'arm extension'
        return latest_time, source

    left_last_reset, left_reset_source = _compose_reset(left_last_pause_end, left_extension_reset_time)
    right_last_reset, right_reset_source = _compose_reset(right_last_pause_end, right_extension_reset_time)

    if left_last_reset is not None and right_last_reset is not None:
        if left_last_reset < right_last_reset:
            right_of_way = 'left'
            pause_row = (
                f"Left has right-of-way ({left_reset_source or 'action'} at {left_last_reset:.2f}s vs "
                f"{right_reset_source or 'action'} at {right_last_reset:.2f}s)"
            )
        else:
            right_of_way = 'right'
            pause_row = (
                f"Right has right-of-way ({right_reset_source or 'action'} at {right_last_reset:.2f}s vs "
                f"{left_reset_source or 'action'} at {left_last_reset:.2f}s)"
            )
    elif left_last_reset is not None:
        right_of_way = 'right'
        pause_row = f"Right has right-of-way ({left_reset_source or 'action'} by left at {left_last_reset:.2f}s)"
    elif right_last_reset is not None:
        right_of_way = 'left'
        pause_row = f"Left has right-of-way ({right_reset_source or 'action'} by right at {right_last_reset:.2f}s)"
    else:
        right_of_way = 'none'
        pause_row = 'No pauses detected'
    
    if last_blade_contact:
        time_diff = hit_time - last_blade_contact.time

        if time_diff > 1.0:
            result['blade_analysis'] = f'Blade contact at {last_blade_contact.time:.2f}s ignored (>1s before hit)'
            result['winner'] = right_of_way
            result['reason'] = f'{pause_row}. Blade contact ignored (too early)'
        else:
            blade_beater, blade_details = analyze_blade_contact(
                left_xdata, left_ydata, right_xdata, right_ydata,
                last_blade_contact.frame,
                current_right_of_way=right_of_way
            )

            result['blade_details'] = blade_details
            rationale = blade_details.get('rationale', 'logistic evaluation') if blade_details else 'logistic evaluation'
            result['blade_analysis'] = f'Blade contact at {last_blade_contact.time:.2f}s - {rationale}'

            result['winner'] = blade_beater

            if blade_beater == right_of_way:
                result['reason'] = f'{pause_row}. {blade_beater.capitalize()} maintained right-of-way (logistic momentum confirmed)'
            else:
                result['reason'] = f'{pause_row}. Logistic momentum favored {blade_beater} despite pause context'
    else:
        result['winner'] = right_of_way
        result['reason'] = pause_row
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Debug AI Referee Logic")
    parser.add_argument("subfolder", help="Name of the subfolder to analyze")
    args = parser.parse_args()
    
    base_dir = Path("/workspace/data/training_data")
    target_dir = base_dir / args.subfolder
    
    if not target_dir.exists():
        print(f"Subfolder not found in training_data: {target_dir}")
        # Try mismatched_results just in case
        base_dir = Path("/workspace/mismatched_results")
        target_dir = base_dir / args.subfolder
        if not target_dir.exists():
            print(f"Subfolder not found in mismatched_results either.")
            sys.exit(1)
    
    debug_log_path = PROJECT_ROOT / "debug.txt"
    with open(debug_log_path, "w") as log_file, contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
        print(f"Analyzing {target_dir}...")

        txt_files = list(target_dir.glob("*.txt"))
        excel_files = list(target_dir.glob("*.xlsx"))
        json_path = target_dir / "analysis_result.json"
    
        if not txt_files or not excel_files or not json_path.exists():
            print("Missing required files (txt, xlsx, or json)")
            sys.exit(1)
            
        txt_path = txt_files[0]
        excel_path = excel_files[0]
        
        # Load normalization constant from existing JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
            norm_constant = data.get("normalisation_constant")
            
        phrase = parse_txt_file(str(txt_path))
        left_x, left_y, right_x, right_y = load_keypoints_from_excel(str(excel_path))
        if left_x and 16 in left_x:
            max_frame = len(left_x[16]) - 1
            _trim_phrase_to_frames(phrase, max_frame)
        
        decision = referee_decision(
            phrase, 
            left_x, left_y, 
            right_x, right_y, 
            normalisation_constant=norm_constant
        )
        
        print("\n" + "="*60)
        print("DEBUG RESULT")
        print("="*60)
        print(json.dumps(sanitize_for_json(decision), indent=2))
        print("="*60)

if __name__ == "__main__":
    main()
