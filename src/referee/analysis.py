from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Tuple
from ultralytics import YOLO
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDy_RRq8hd8rTYILt_mYtMH8GtM41GFp6I")
GEMINI_MODEL = "models/gemini-2.5-flash-lite"

def valid_mask(kpts_xy: np.ndarray):
    return (kpts_xy[:, 0] > 0) & (kpts_xy[:, 1] > 0)

def valid_points(kpts_xy: np.ndarray):
    return kpts_xy[valid_mask(kpts_xy)]

def kpt_centroid(kpts_xy: np.ndarray):
    vp = valid_points(kpts_xy)
    if len(vp) == 0:
        return (0.0, 0.0)
    return (float(vp[:, 0].mean()), float(vp[:, 1].mean()))

def bbox_from_keypoints(kpts_xy: np.ndarray):
    vp = valid_points(kpts_xy)
    if len(vp) < 2:
        return None
    x1, y1 = vp[:, 0].min(), vp[:, 1].min()
    x2, y2 = vp[:, 0].max(), vp[:, 1].max()
    return (int(x1), int(y1), int(x2), int(y2))

def bbox_center(bbox: Tuple[int, int, int, int]):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def bbox_area(bbox: Tuple[int, int, int, int]):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def bbox_iou(b1: Optional[Tuple[int, int, int, int]],
             b2: Optional[Tuple[int, int, int, int]]):
    if b1 is None or b2 is None:
        return 0.0
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = bbox_area(b1)
    area2 = bbox_area(b2)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def torso_scale(kpts: np.ndarray):
    if len(kpts) < 17:
        return 1.0
    left_shoulder = kpts[5]
    right_shoulder = kpts[6]
    left_hip = kpts[11]
    right_hip = kpts[12]
    pts = [left_shoulder, right_shoulder, left_hip, right_hip]
    valid = [p for p in pts if p[0] > 0 and p[1] > 0]
    if len(valid) < 2:
        return 1.0
    coords = np.array(valid)
    dists = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    return float(dists.max())

def robust_kpt_distance(k1: np.ndarray, k2: np.ndarray):
    mask = valid_mask(k1) & valid_mask(k2)
    if not mask.any():
        return float('inf')
    diff = k1[mask] - k2[mask]
    return float(np.linalg.norm(diff, axis=1).mean())

def composite_track_cost(
    k_det: np.ndarray,
    k_prev: np.ndarray,
    prev_bbox: Optional[Tuple[int, int, int, int]],
    pred_center: Optional[Tuple[float, float]],
    frame_diag: float,
):
    kpt_dist = robust_kpt_distance(k_det, k_prev)
    det_bbox = bbox_from_keypoints(k_det)
    if det_bbox is None or prev_bbox is None:
        bbox_cost = 0.5
    else:
        iou = bbox_iou(det_bbox, prev_bbox)
        bbox_cost = 1.0 - iou
    if pred_center is not None and det_bbox is not None:
        det_center = bbox_center(det_bbox)
        motion_dist = math.hypot(det_center[0] - pred_center[0],
                                 det_center[1] - pred_center[1])
        motion_cost = motion_dist / frame_diag if frame_diag > 0 else 0.0
    else:
        motion_cost = 0.0
    scale = torso_scale(k_prev)
    kpt_cost = kpt_dist / scale if scale > 0 else kpt_dist
    return 0.5 * kpt_cost + 0.3 * bbox_cost + 0.2 * motion_cost

class _TrackState:
    def __init__(self, tid: int, kpts: np.ndarray, frame_idx: int):
        self.id = tid
        self.kpts = kpts
        self.bbox = bbox_from_keypoints(kpts)
        self.centroid = kpt_centroid(kpts)
        self.prev_centroid: Optional[Tuple[float, float]] = None
        self.last_seen = frame_idx

    def predict_center(self):
        if self.prev_centroid is None:
            return self.centroid
        dx = self.centroid[0] - self.prev_centroid[0]
        dy = self.centroid[1] - self.prev_centroid[1]
        return (self.centroid[0] + dx, self.centroid[1] + dy)

class TwoFencerTracker:
    """Two-fencer keypoint tracker with robust matching."""
    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        max_miss: int = 25,
        edge_margin_frac: float = 0.06,
        top_margin_frac: float = 0.12,
        bottom_margin_frac: float = 0.20,
        hcenter_sigma_frac: float = 0.22,
    ):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.max_miss = max_miss
        self.frame_diag = math.hypot(frame_w, frame_h)
        self.edge_margin = int(frame_w * edge_margin_frac)
        self.top_margin = int(frame_h * top_margin_frac)
        self.bottom_margin = int(frame_h * bottom_margin_frac)
        self.hcenter_sigma = frame_w * hcenter_sigma_frac
        self.tracks: Dict[int, Optional[_TrackState]] = {0: None, 1: None}
        self.miss_count = {0: 0, 1: 0}
        self.frame_idx = 0
        self.initialized = False

    def _edge_safe(self, center: Tuple[float, float]):
        x, y = center
        if x < self.edge_margin or x > (self.frame_w - self.edge_margin):
            return False
        if y < self.top_margin or y > (self.frame_h - self.bottom_margin):
            return False
        return True

    def _init_score(self, kpts: np.ndarray, side: str):
        c = kpt_centroid(kpts)
        if not self._edge_safe(c):
            return -1e9
        hcenter = self.frame_w / 2
        dist_from_center = abs(c[0] - hcenter)
        gaussian = math.exp(-0.5 * (dist_from_center / self.hcenter_sigma) ** 2)
        if side == 'left':
            side_score = 1.0 if c[0] < hcenter else 0.0
        else:
            side_score = 1.0 if c[0] >= hcenter else 0.0
        return gaussian + side_score

    def _pick_initial_tracks(self, detections: List[np.ndarray]):
        if len(detections) < 2:
            return None, None
        scores_left = [(i, self._init_score(d, 'left')) for i, d in enumerate(detections)]
        scores_right = [(i, self._init_score(d, 'right')) for i, d in enumerate(detections)]
        scores_left.sort(key=lambda x: x[1], reverse=True)
        scores_right.sort(key=lambda x: x[1], reverse=True)
        best_left_idx = scores_left[0][0]
        best_right_idx = scores_right[0][0]
        if best_left_idx == best_right_idx:
            if len(scores_left) > 1:
                best_left_idx = scores_left[1][0]
            else:
                return None, None
        return detections[best_left_idx], detections[best_right_idx]

    def initialize(self, detections: List[np.ndarray]):
        left_kpts, right_kpts = self._pick_initial_tracks(detections)
        if left_kpts is None or right_kpts is None:
            return
        self.tracks[0] = _TrackState(0, left_kpts, self.frame_idx)
        self.tracks[1] = _TrackState(1, right_kpts, self.frame_idx)
        self.miss_count = {0: 0, 1: 0}
        self.initialized = True

    def update(self, detections: List[np.ndarray]):
        if not self.initialized:
            self.initialize(detections)
            self.frame_idx += 1
            return
        for tid in (0, 1):
            if self.tracks[tid] is not None:
                self.miss_count[tid] += 1
        if len(detections) == 0:
            for tid in (0, 1):
                if self.miss_count[tid] > self.max_miss:
                    self.tracks[tid] = None
            self.frame_idx += 1
            return
        costs = {}
        for tid in (0, 1):
            track = self.tracks[tid]
            if track is None:
                continue
            pred_center = track.predict_center()
            for d_idx, det_kpts in enumerate(detections):
                c = composite_track_cost(
                    det_kpts, track.kpts, track.bbox, pred_center, self.frame_diag
                )
                costs[(tid, d_idx)] = c
        assignments = {}
        used_dets = set()
        for tid in (0, 1):
            if self.tracks[tid] is None:
                continue
            candidates = [(d_idx, costs.get((tid, d_idx), 1e9))
                         for d_idx in range(len(detections))
                         if d_idx not in used_dets]
            if not candidates:
                continue
            candidates.sort(key=lambda x: x[1])
            best_d_idx, best_cost = candidates[0]
            if best_cost < 2.0:
                assignments[tid] = best_d_idx
                used_dets.add(best_d_idx)
        for tid in (0, 1):
            if tid in assignments:
                d_idx = assignments[tid]
                self._overwrite_track(self.tracks[tid], detections[d_idx])
                self.miss_count[tid] = 0
            else:
                if self.miss_count[tid] > self.max_miss:
                    self.tracks[tid] = None
        self.frame_idx += 1

    def _overwrite_track(self, track: _TrackState, kpts: np.ndarray):
        track.prev_centroid = track.centroid
        track.kpts = kpts
        track.bbox = bbox_from_keypoints(kpts)
        track.centroid = kpt_centroid(kpts)
        track.last_seen = self.frame_idx

    def _maybe_update_track(self, track: Optional[_TrackState], kpts: np.ndarray, cost: float):
        if track is None or cost > 2.0:
            return
        self._overwrite_track(track, kpts)

    def get_track(self, tid: int):
        t = self.tracks.get(tid)
        return t.kpts if t is not None else None


# ---------------------------------------------------------------------------
# Fisheye video correction helpers
# ---------------------------------------------------------------------------
def _fisheye_safe_cap(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap

def _fisheye_meta(path: str):
    cap = _fisheye_safe_cap(path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, frame_count

def _build_fisheye_maps(width: int, height: int, strength: float = -0.18, balance: float = 0.0):
    cx, cy = width / 2, height / 2
    fx = fy = max(width, height)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([strength, 0, 0, 0], dtype=np.float64)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (width, height), np.eye(3), balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (width, height), cv2.CV_16SC2
    )
    return map1, map2

def _auto_crop_from_maps(map1: np.ndarray, map2: np.ndarray, border_mode=cv2.BORDER_CONSTANT):
    h, w = map1.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    warped = cv2.remap(mask, map1, map2, cv2.INTER_LINEAR, borderMode=border_mode, borderValue=0)
    coords = cv2.findNonZero(warped)
    if coords is None:
        return 0, 0, w, h
    x, y, cw, ch = cv2.boundingRect(coords)
    return x, y, cw, ch

def _mux_audio_if_any(original_path: str, corrected_path: str, out_path: str):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1", original_path],
        capture_output=True, text=True
    )
    has_audio = (probe.returncode == 0 and probe.stdout.strip() == "audio")
    if not has_audio:
        if corrected_path != out_path:
            shutil.move(corrected_path, out_path)
        return out_path
    temp_out = out_path + ".temp.mp4"
    cmd = [
        "ffmpeg", "-y", "-i", corrected_path, "-i", original_path,
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac",
        "-shortest", temp_out
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("Audio mux failed, keeping video-only: %s", result.stderr)
        if corrected_path != out_path:
            shutil.move(corrected_path, out_path)
        return out_path
    if os.path.exists(corrected_path) and corrected_path != temp_out:
        os.remove(corrected_path)
    shutil.move(temp_out, out_path)
    return out_path

def correct_fisheye_video(
    input_path: str,
    output_path: Optional[str] = None,
    strength: float = -0.18,
    balance: float = 0.0,
    keep_audio: bool = True,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: Tuple[int, int, int] = (0, 0, 0),
    progress: bool = False,
) -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    width, height, fps, frame_count = _fisheye_meta(input_path)
    logger.info(
        "Fisheye correction: %s (%dx%d @ %.3f FPS, %d frames, strength=%s, balance=%s)",
        input_path,
        width,
        height,
        fps,
        frame_count,
        strength,
        balance,
    )

    map1, map2 = _build_fisheye_maps(width, height, strength=strength, balance=balance)
    crop_x, crop_y, crop_w, crop_h = _auto_crop_from_maps(map1, map2, border_mode=border_mode)
    logger.debug("Fisheye crop ROI: x=%d, y=%d, w=%d, h=%d", crop_x, crop_y, crop_w, crop_h)

    cap = _fisheye_safe_cap(input_path)

    if output_path is None:
        out_dir = Path(input_path).parent
        out_name = f"{Path(input_path).stem}_corrected.mp4"
        output_path = str(out_dir / out_name)

    tmp_out = output_path if output_path.endswith(".mp4") else output_path + ".mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out, fourcc, fps, (crop_w, crop_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer for {tmp_out}")

    iterator = range(frame_count)
    for _ in tqdm(iterator, desc="FisheyeUndistort", disable=not progress):
        ok, frame = cap.read()
        if not ok:
            break
        undistorted = cv2.remap(
            frame,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=border_mode,
            borderValue=border_value,
        )
        roi = undistorted[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        writer.write(roi)

    cap.release()
    writer.release()

    if keep_audio:
        logger.debug("Attempting audio mux for %s", input_path)
        final_path = _mux_audio_if_any(input_path, tmp_out, output_path)
    else:
        final_path = tmp_out if tmp_out == output_path else shutil.move(tmp_out, output_path)

    logger.info("Fisheye correction complete: %s", final_path)
    return str(final_path)

SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

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

def fill_with_linear_regression(data, c):
    """
    Fill NaN values using linear regression from neighboring valid points
    """
    filled = data.copy()
    for col in filled.columns:
        series = filled[col]
        if series.isna().all():
            continue
        nan_indices = series.index[series.isna()].tolist()
        if not nan_indices:
            continue
        valid_indices = series.index[~series.isna()].tolist()
        if len(valid_indices) < 2:
            filled[col].fillna(0, inplace=True)
            continue
        valid_x = np.array(valid_indices)
        valid_y = series.loc[valid_indices].values
        for nan_idx in nan_indices:
            before = [i for i in valid_indices if i < nan_idx]
            after = [i for i in valid_indices if i > nan_idx]
            if before and after:
                x1, y1 = before[-1], series.loc[before[-1]]
                x2, y2 = after[0], series.loc[after[0]]
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                filled.loc[nan_idx, col] = y1 + slope * (nan_idx - x1)
            elif before:
                filled.loc[nan_idx, col] = series.loc[before[-1]]
            elif after:
                filled.loc[nan_idx, col] = series.loc[after[0]]
            else:
                filled.loc[nan_idx, col] = 0
    return filled

def extract_tracks_from_video(video_path, model):
    """
    Extract YOLO pose tracks from video using persistent two-fencer tracking.
    Returns: list of tracks per frame
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = TwoFencerTracker(frame_w=width, frame_h=height)
    tracks_per_frame = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        detections = []
        if len(results) > 0 and results[0].keypoints is not None:
            kpts = results[0].keypoints.xy
            if hasattr(kpts, "cpu"):
                kpts = kpts.cpu().numpy()
            else:
                kpts = np.array(kpts)
            for i in range(kpts.shape[0]):
                detections.append(kpts[i])

        tracker.update(detections)

        frame_tracks = []
        for tid in (0, 1):
            kpts = tracker.get_track(tid)
            bbox = bbox_from_keypoints(kpts) if kpts is not None else None
            frame_tracks.append({
                'track_id': tid,
                'keypoints': kpts.copy() if kpts is not None else None,
                'box': np.array(bbox, dtype=float) if bbox else None,
            })

        tracks_per_frame.append(frame_tracks)

    cap.release()
    return tracks_per_frame

def process_video_and_extract_data(tracks_per_frame):
    """
    Process tracks and extract normalized keypoint data
    Ensures consistent left/right fencer assignment and foot swapping
    """
    left_xdata = {k: [] for k in range(17)}
    left_ydata = {k: [] for k in range(17)}
    right_xdata = {k: [] for k in range(17)}
    right_ydata = {k: [] for k in range(17)}
    video_angle = ''
    c = None
    
    # Find the first frame with keypoints for both tracks
    for tracks in tracks_per_frame:
        track_map = {t['track_id']: t for t in tracks}
        left_track = track_map.get(0)
        right_track = track_map.get(1)
        if not left_track or not right_track:
            continue

        k0 = left_track.get('keypoints')
        k1 = right_track.get('keypoints')
        if k0 is None or k1 is None:
            continue
        if len(k0) < 17 or len(k1) < 17:
            continue
        if k0[15][0] <= 0 or k0[16][0] <= 0 or k1[15][0] <= 0 or k1[16][0] <= 0:
            continue

        values = [k0[15][0], k0[16][0], k1[15][0], k1[16][0]]
        sorted_values = sorted(values, reverse=True)
        b = sorted_values[1]
        a = sorted_values[2]
        c = abs((b - a) / 4)

        bbox_left = left_track.get('box')
        bbox_right = right_track.get('box')
        if bbox_left is None and k0 is not None:
            bbox_left = np.array(bbox_from_keypoints(k0), dtype=float)
        if bbox_right is None and k1 is not None:
            bbox_right = np.array(bbox_from_keypoints(k1), dtype=float)

        if bbox_left is not None and bbox_right is not None:
            left_box_area = (bbox_left[2] - bbox_left[0]) * (bbox_left[3] - bbox_left[1])
            right_box_area = (bbox_right[2] - bbox_right[0]) * (bbox_right[3] - bbox_right[1])

            if left_box_area >= 1.75 * right_box_area:
                video_angle = 'left'
            elif right_box_area >= 1.75 * left_box_area:
                video_angle = 'right'
            else:
                video_angle = 'middle'
        break
    
    if c is None:
        raise ValueError("No valid frame with keypoints for both tracks found in the video")
    
    # Extract data for all frames
    def _append_nan_row():
        for j in range(17):
            left_xdata[j].append(np.nan)
            left_ydata[j].append(np.nan)
            right_xdata[j].append(np.nan)
            right_ydata[j].append(np.nan)

    for tracks in tracks_per_frame:
        track_map = {t['track_id']: t for t in tracks}
        left_track = track_map.get(0)
        right_track = track_map.get(1)

        left_kpts = left_track.get('keypoints') if left_track else None
        right_kpts = right_track.get('keypoints') if right_track else None

        if left_kpts is None or right_kpts is None:
            _append_nan_row()
            continue

        if len(left_kpts) < 17 or len(right_kpts) < 17:
            _append_nan_row()
            continue

        bbox_left = left_track.get('box')
        if bbox_left is None and left_kpts is not None:
            lb = bbox_from_keypoints(left_kpts)
            bbox_left = np.array(lb, dtype=float) if lb is not None else None

        bbox_right = right_track.get('box')
        if bbox_right is None and right_kpts is not None:
            rb = bbox_from_keypoints(right_kpts)
            bbox_right = np.array(rb, dtype=float) if rb is not None else None

        if bbox_left is None or bbox_right is None:
            _append_nan_row()
            continue

        center_left = (bbox_left[0] + bbox_left[2]) / 2
        center_right = (bbox_right[0] + bbox_right[2]) / 2

        if center_left > center_right:
            left_kpts, right_kpts = right_kpts, left_kpts
            bbox_left, bbox_right = bbox_right, bbox_left

        pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

        for keypoints, xdata, ydata, is_left in [
            (np.array(left_kpts, copy=True), left_xdata, left_ydata, True),
            (np.array(right_kpts, copy=True), right_xdata, right_ydata, False),
        ]:
            if len(keypoints) < 17:
                for j in range(17):
                    xdata[j].append(np.nan)
                    ydata[j].append(np.nan)
                continue

            for kp1, kp2 in pairs:
                if kp1 >= len(keypoints) or kp2 >= len(keypoints):
                    continue
                if is_left and keypoints[kp1][0] > keypoints[kp2][0]:
                    keypoints[[kp1, kp2]] = keypoints[[kp2, kp1]]
                elif not is_left and keypoints[kp1][0] < keypoints[kp2][0]:
                    keypoints[[kp1, kp2]] = keypoints[[kp2, kp1]]

            for j in range(17):
                if keypoints[j][0] > 0 and keypoints[j][1] > 0:
                    xdata[j].append(keypoints[j][0] / c)
                    ydata[j].append(keypoints[j][1] / c)
                else:
                    xdata[j].append(np.nan)
                    ydata[j].append(np.nan)
    
    return left_xdata, left_ydata, right_xdata, right_ydata, c, video_angle

def save_keypoints_to_excel(left_xdata, left_ydata, right_xdata, right_ydata, output_path):
    """
    Save keypoint data to Excel file with 4 sheets
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        pd.DataFrame(left_xdata).to_excel(writer, sheet_name='Left_X', index=False)
        pd.DataFrame(left_ydata).to_excel(writer, sheet_name='Left_Y', index=False)
        pd.DataFrame(right_xdata).to_excel(writer, sheet_name='Right_X', index=False)
        pd.DataFrame(right_ydata).to_excel(writer, sheet_name='Right_Y', index=False)

def _get_skeleton_connections():
    """Return COCO keypoint connections."""
    return [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
        (3, 5), (4, 6)
    ]

def _denormalize_keypoint(x: float, y: float, c: float):
    if np.isnan(x) or np.isnan(y):
        return None
    return (int(x * c), int(y * c))

def _draw_keypoints_on_frame(
    frame: np.ndarray,
    left_xdata: Dict[int, List[float]],
    left_ydata: Dict[int, List[float]],
    right_xdata: Dict[int, List[float]],
    right_ydata: Dict[int, List[float]],
    frame_idx: int,
    c_value: float,
    draw_skeleton: bool,
    draw_labels: bool,
):
    """Draw keypoints and skeleton on a single frame."""
    skeleton = _get_skeleton_connections()
    
    # Draw left fencer (blue)
    if frame_idx < len(left_xdata[0]):
        points = {}
        for kp_idx in range(17):
            x = left_xdata[kp_idx][frame_idx]
            y = left_ydata[kp_idx][frame_idx]
            pt = _denormalize_keypoint(x, y, c_value)
            if pt is not None:
                points[kp_idx] = pt
                cv2.circle(frame, pt, 3, (255, 0, 0), -1)
                if draw_labels:
                    cv2.putText(frame, str(kp_idx), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        if draw_skeleton:
            for kp1, kp2 in skeleton:
                if kp1 in points and kp2 in points:
                    cv2.line(frame, points[kp1], points[kp2], (255, 0, 0), 2)
    
    # Draw right fencer (green)
    if frame_idx < len(right_xdata[0]):
        points = {}
        for kp_idx in range(17):
            x = right_xdata[kp_idx][frame_idx]
            y = right_ydata[kp_idx][frame_idx]
            pt = _denormalize_keypoint(x, y, c_value)
            if pt is not None:
                points[kp_idx] = pt
                cv2.circle(frame, pt, 3, (0, 255, 0), -1)
                if draw_labels:
                    cv2.putText(frame, str(kp_idx), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        if draw_skeleton:
            for kp1, kp2 in skeleton:
                if kp1 in points and kp2 in points:
                    cv2.line(frame, points[kp1], points[kp2], (0, 255, 0), 2)
    
    return frame

def render_overlay_video(
    video_path: Path,
    output_path: Path,
    left_xdata: Dict[int, List[float]],
    left_ydata: Dict[int, List[float]],
    right_xdata: Dict[int, List[float]],
    right_ydata: Dict[int, List[float]],
    normalisation_constant: float,
    draw_skeleton: bool = True,
    draw_labels: bool = False,
    show_progress: bool = False,
):
    """Generate an overlay video with keypoints drawn on each frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer for {output_path}")
    
    frame_idx = 0
    iterator = range(frame_count) if not show_progress else tqdm(range(frame_count), desc="Rendering overlay")
    
    for _ in iterator:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = _draw_keypoints_on_frame(
            frame, left_xdata, left_ydata, right_xdata, right_ydata,
            frame_idx, normalisation_constant, draw_skeleton, draw_labels
        )
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()

def build_decision_summary(decision: Dict[str, Any], phrase: FencingPhrase):
    """Prepare structured context for natural language explanation."""
    winner = decision.get("winner", "unknown")
    reason = decision.get("reason", "")
    
    left_pauses = decision.get("left_pauses", [])
    right_pauses = decision.get("right_pauses", [])
    
    blade_analysis = decision.get("blade_analysis", "")
    speed_comparison = decision.get("speed_comparison", {})
    
    summary = f"Winner: {winner}\nReason: {reason}\n"
    summary += f"Left pauses: {len(left_pauses)}, Right pauses: {len(right_pauses)}\n"
    summary += f"Blade analysis: {blade_analysis}\n"
    summary += f"Speed comparison: {speed_comparison}\n"
    
    return summary

def generate_gemini_reason(decision: Dict[str, Any], phrase: FencingPhrase):
    """Use Gemini to craft a one-sentence fencing explanation."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        summary = build_decision_summary(decision, phrase)
        prompt = f"Based on this fencing analysis, provide a one-sentence explanation of the decision:\n{summary}"
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.warning(f"Gemini API call failed: {e}")
        return decision.get("reason", "")

def process_all_videos(base_path, model):
    """
    Process all videos in the training_data folder structure
    """
    base_path = Path(base_path)
    results = []
    
    for video_dir in base_path.iterdir():
        if not video_dir.is_dir():
            continue
        
        video_files = list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mp4"))
        txt_files = list(video_dir.glob("*.txt"))
        
        if not video_files or not txt_files:
            continue
        
        video_path = video_files[0]
        txt_path = txt_files[0]
        
        try:
            tracks = extract_tracks_from_video(str(video_path), model)
            left_x, left_y, right_x, right_y, c, angle = process_video_and_extract_data(tracks)
            
            excel_path = video_dir / f"{video_path.stem}_keypoints.xlsx"
            save_keypoints_to_excel(left_x, left_y, right_x, right_y, str(excel_path))
            
            phrase = parse_txt_file(str(txt_path))
            decision = referee_decision(phrase, left_x, left_y, right_x, right_y, c)
            
            results.append({
                'video': str(video_path),
                'decision': decision,
                'phrase': phrase
            })
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
    
    return results


"""
AI Fencing Referee Analysis System

IMPORTANT POSITION MAPPING:
- Fencer 1 = Right fencer = Right side of screen = right_xdata/right_ydata
- Fencer 2 = Left fencer = Left side of screen = left_xdata/left_ydata

In TXT files:
- "Right Fencer" = Fencer 1 = right_xdata/right_ydata
- "Left Fencer" = Fencer 2 = left_xdata/left_ydata

Movement directions:
- Left fencer (Fencer 2) advances right (+x direction)
- Right fencer (Fencer 1) advances left (-x direction)

Weapon hands (fencers face each other):
- Left fencer (Fencer 2): weapon in right hand (keypoint 10)
- Right fencer (Fencer 1): weapon in left hand (keypoint 9)
"""

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

@dataclass
class FencingPhrase:
    """Contains all data for a fencing phrase
    Important: Fencer 1 = Right fencer, Fencer 2 = Left fencer"""
    start_time: float
    start_frame: int
    simultaneous_hit_time: Optional[float]
    simultaneous_hit_frame: Optional[int]
    blade_contacts: List[BladeContact]
    lockout_start: Optional[float]
    declared_winner: str
    fps: float = 15.0

@dataclass
class AnalysisResult:
    """Container for a single video/txt analysis run."""
    phrase: FencingPhrase
    decision: Optional[Dict[str, Any]]
    frames_processed: int
    processing_time: float
    video_angle: Optional[str]
    normalisation_constant: Optional[float]
    left_xdata: Optional[Dict[int, List[float]]] = None
    left_ydata: Optional[Dict[int, List[float]]] = None
    right_xdata: Optional[Dict[int, List[float]]] = None
    right_ydata: Optional[Dict[int, List[float]]] = None
    video_path: Optional[str] = None
    txt_path: Optional[str] = None
    excel_path: Optional[str] = None
    input_signal_path: Optional[str] = None
    natural_reason: Optional[str] = None
    lunge_detected: Optional[Dict[str, bool]] = None
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(
        self,
        include_keypoints: bool = False,
    ):
        """Convert to dictionary for JSON serialization."""
        result = {
            'phrase': asdict(self.phrase),
            'decision': self.decision,
            'frames_processed': self.frames_processed,
            'processing_time': self.processing_time,
            'video_angle': self.video_angle,
            'normalisation_constant': self.normalisation_constant,
            'video_path': str(self.video_path) if self.video_path else None,
            'txt_path': str(self.txt_path) if self.txt_path else None,
            'excel_path': str(self.excel_path) if self.excel_path else None,
            'input_signal_path': str(self.input_signal_path) if self.input_signal_path else None,
            'natural_reason': self.natural_reason,
            'lunge_detected': self.lunge_detected,
            'artifacts': dict(self.artifacts),
        }
        
        if include_keypoints:
            result['left_xdata'] = self.left_xdata
            result['left_ydata'] = self.left_ydata
            result['right_xdata'] = self.right_xdata
            result['right_ydata'] = self.right_ydata
        
        return sanitize_for_json(result)

def parse_txt_file(txt_path: str) -> FencingPhrase:
    """Parse the TXT file to extract timing information.
    
    Updated scoring rule:
    - Treat the phrase as a double touch only when the scoreboard line reports
      ``Scores -> Fencer 1: HIT, Fencer 2: HIT`` (case-insensitive).
    - Any other scoreboard combination is considered a single-light phrase and
      should be skipped by downstream processing.
    """
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
                t = float(match.group(1))
                blade_contacts.append(BladeContact(time=t, frame=int(t * 15.0)))

        if "Lockout period started" in line:
            match = re.search(r'(\d+\.\d+)s', line)
            if match:
                lockout_start = float(match.group(1))

        if "HIT:" in line:
            match = re.search(r'(\d+\.\d+)s', line)
            if match:
                hit_events.append(float(match.group(1)))

        if "Scores ->" in line:
            match_f1 = re.search(r'Fencer 1:\s*(\w+)', line, re.IGNORECASE)
            match_f2 = re.search(r'Fencer 2:\s*(\w+)', line, re.IGNORECASE)
            if match_f1:
                scoreboard_f1 = match_f1.group(1).strip().upper()
            if match_f2:
                scoreboard_f2 = match_f2.group(1).strip().upper()

        if re.search(r'(Confirmed result winner|Manual selection winner):\s*(Right|Left)', line, re.IGNORECASE):
            match = re.search(r'(Right|Left)', line, re.IGNORECASE)
            if match:
                declared_winner = match.group(1).lower()

    if start_time is None:
        raise ValueError(f"Could not find start time in {txt_path}")

    if scoreboard_f1 != "HIT" or scoreboard_f2 != "HIT":
        raise ValueError(
            f"Skipping {txt_path}: Not a double-touch phrase "
            f"(Fencer 1: {scoreboard_f1}, Fencer 2: {scoreboard_f2})"
        )

    if simultaneous_hit_time is None:
        if hit_events:
            simultaneous_hit_time = hit_events[0]
        else:
            raise ValueError(f"Could not find simultaneous hit time in {txt_path}")

    start_frame = int(start_time * 15.0)
    simultaneous_hit_frame = int(simultaneous_hit_time * 15.0)

    return FencingPhrase(
        start_time=start_time,
        start_frame=start_frame,
        simultaneous_hit_time=simultaneous_hit_time,
        simultaneous_hit_frame=simultaneous_hit_frame,
        blade_contacts=blade_contacts,
        lockout_start=lockout_start,
        declared_winner=declared_winner or "unknown",
        fps=15.0
    )

def load_keypoints_from_excel(excel_path: str):
    """
    Load keypoint data from Excel file
    
    Returns: (left_xdata, left_ydata, right_xdata, right_ydata)
    Where:
    - left_xdata, left_ydata = Fencer 2 (left side of screen)
    - right_xdata, right_ydata = Fencer 1 (right side of screen)
    """
    xl = pd.ExcelFile(excel_path)
    left_x = xl.parse('Left_X')
    left_y = xl.parse('Left_Y')
    right_x = xl.parse('Right_X')
    right_y = xl.parse('Right_Y')
    
    left_xdata = {i: left_x[str(i)].tolist() for i in range(17)}
    left_ydata = {i: left_y[str(i)].tolist() for i in range(17)}
    right_xdata = {i: right_x[str(i)].tolist() for i in range(17)}
    right_ydata = {i: right_y[str(i)].tolist() for i in range(17)}
    
    return left_xdata, left_ydata, right_xdata, right_ydata

def calculate_center_of_mass(xdata: Dict, ydata: Dict, frame_idx: int):
    """
    Calculate center of mass from key body points (hips and shoulders)
    Keypoints: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip
    """
    key_points = [5, 6, 11, 12]
    x_coords = []
    y_coords = []
    
    for kp in key_points:
        x = xdata[kp][frame_idx]
        y = ydata[kp][frame_idx]
        if not np.isnan(x) and not np.isnan(y):
            x_coords.append(x)
            y_coords.append(y)
    
    if not x_coords:
        return None, None
    
    return np.mean(x_coords), np.mean(y_coords)


def detect_lunge(xdata: Dict[int, List[float]],
                 ydata: Dict[int, List[float]],
                 normalisation_constant: float,
                 frames_to_check: int = 5,
                 threshold_m: float = 0.8) -> bool:
    """
    Detect if a lunge occurred by measuring front-to-back foot distance increase.
    Operates on entire dataset range.
    """
    start_frame = 0
    end_frame = len(xdata[16]) - 1
    
    if end_frame - start_frame < frames_to_check:
        return False
    
    distance_samples = []
    for i in range(start_frame, min(start_frame + frames_to_check, end_frame + 1)):
        front_x = xdata[16][i]
        front_y = ydata[16][i]
        back_x = xdata[15][i]
        back_y = ydata[15][i]
        
        if not (np.isnan(front_x) or np.isnan(front_y) or np.isnan(back_x) or np.isnan(back_y)):
            dist = math.hypot(front_x - back_x, front_y - back_y)
            dist_m = dist * normalisation_constant
            distance_samples.append(dist_m)
    
    if len(distance_samples) < 2:
        return False
    
    avg_dist = np.mean(distance_samples)
    increasing = distance_samples[-1] > distance_samples[0]
    return avg_dist >= threshold_m and increasing

def detect_pause_retreat_intervals(xdata: Dict, ydata: Dict, is_left_fencer: bool, 
                                   fps: float = 15.0) -> List[PauseInterval]:
    """
    Detect pause/retreat intervals for a fencer using simplified logic:
    - Use only front foot (keypoint 16)
    - No smoothing
    - Raw velocity check
    - Y-variance filter on front foot
    - Back foot (keypoint 15) movement filter with fragment-based filtering
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
    
    expected_direction = 1 if is_left_fencer else -1
    
    # --- TUNABLE PARAMS ---
    pause_threshold = 0.03
    retreat_threshold = 0.03 # Threshold to distinguish Retreat from Pause
    min_pause_frames = 4
    y_variance_threshold = 0.001
    back_foot_threshold = 0.05 # Threshold for back foot movement
    # ----------------------
    
    pause_frames = []
    current_pause_frames = []

    def process_and_filter_interval(frames):
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
        
        # If average velocity is high, it's a retreat (valid break of ROW)
        # We skip variance and back foot checks for retreats
        if avg_abs_vel > retreat_threshold:
            pause_frames.append(frames)
            return

        # Else, it's a Pause. Apply strict filters.
        
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
                if y_var >= y_variance_threshold:
                    continue # Failed Y-var check
            
            # Passed checks
            pause_frames.append(segment)
    
    for i, vel in enumerate(velocities):
        frame_idx = start_frame + i + 1
        
        # Check if paused (near zero velocity) or retreating (opposite direction)
        is_paused = (abs(vel) < pause_threshold) or (vel * expected_direction < 0)
        
        if is_paused:
            current_pause_frames.append(frame_idx)
        else:
            process_and_filter_interval(current_pause_frames)
            current_pause_frames = []
    
    # Handle end of loop
    process_and_filter_interval(current_pause_frames)
    
    for pf in pause_frames:
        intervals.append(PauseInterval(
            start_frame=pf[0],
            end_frame=pf[-1],
            start_time=pf[0] / fps,
            end_time=pf[-1] / fps,
            duration=(pf[-1] - pf[0]) / fps
        ))
    
    return intervals

def analyze_blade_contact(left_xdata: Dict, left_ydata: Dict, right_xdata: Dict,
                         right_ydata: Dict, contact_frame: int, 
                         current_right_of_way: str = 'none',
                         attack_variance_threshold: float = 0.1) -> Tuple[str, Dict]:
    """Determine blade priority around contact."""

    window_before = 8
    window_after = 3

    max_frame_left = len(left_xdata[10]) - 1
    max_frame_right = len(right_xdata[9]) - 1
    max_frame = min(max_frame_left, max_frame_right)

    start_f = max(0, contact_frame - window_before)
    end_f = min(contact_frame + window_after, max_frame)

    if start_f >= end_f:
        return 'right', {
            'analysis_window': 'invalid',
            'contact_frame': contact_frame,
            'left_variance': 0.0,
            'right_variance': 0.0,
            'samples_left': 0,
            'samples_right': 0,
        }

    def compute_variance(xdata, ydata, kp_idx) -> Tuple[float, float, int]:
        xs = []
        ys = []
        for i in range(start_f, end_f + 1):
            x = xdata[kp_idx][i]
            y = ydata[kp_idx][i]
            if not np.isnan(x) and not np.isnan(y):
                xs.append(x)
                ys.append(y)
        if len(xs) < 2:
            return 0.0, 0.0, len(xs)
        return float(np.var(xs, ddof=1)), float(np.var(ys, ddof=1)), len(xs)

    left_var_x, left_var_y, left_samples = compute_variance(left_xdata, left_ydata, 10)
    right_var_x, right_var_y, right_samples = compute_variance(right_xdata, right_ydata, 9)

    left_total = left_var_x + left_var_y
    right_total = right_var_x + right_var_y

    winner = 'right' # Default
    
    if current_right_of_way == 'none':
        if left_samples == 0 and right_samples == 0:
            winner = 'right'
        elif left_total > right_total:
            winner = 'left'
        elif right_total > left_total:
            winner = 'right'
        else:
            winner = 'right'
    else:
        if current_right_of_way == 'left':
            if left_total > attack_variance_threshold:
                winner = 'left'
            else:
                winner = 'right'
        elif current_right_of_way == 'right':
            if right_total > attack_variance_threshold:
                winner = 'right'
            else:
                winner = 'left'

    details = {
        'analysis_window': f'frames {start_f}-{end_f}',
        'contact_frame': contact_frame,
        'left_variance_x': left_var_x,
        'left_variance_y': left_var_y,
        'left_variance_total': left_total,
        'right_variance_x': right_var_x,
        'right_variance_y': right_var_y,
        'right_variance_total': right_total,
        'samples_left': left_samples,
        'samples_right': right_samples,
        'current_right_of_way': current_right_of_way,
    }

    return winner, details

def calculate_speed_acceleration(xdata: Dict, ydata: Dict) -> Tuple[float, float]:
    """
    Calculate average speed and acceleration over entire dataset range.
    """
    start_frame = 0
    end_frame = len(xdata[16]) - 1
    
    if end_frame - start_frame < 2:
        return 0.0, 0.0
    
    speeds = []
    accelerations = []
    
    prev_x, prev_y = None, None
    prev_speed = None
    
    for i in range(start_frame, end_frame + 1):
        x = xdata[16][i]
        y = ydata[16][i]
        
        if np.isnan(x) or np.isnan(y):
            continue
        
        if prev_x is not None:
            dist = math.hypot(x - prev_x, y - prev_y)
            speeds.append(dist)
            
            if prev_speed is not None:
                accel = abs(dist - prev_speed)
                accelerations.append(accel)
            
            prev_speed = dist
        
        prev_x, prev_y = x, y
    
    avg_speed = np.mean(speeds) if speeds else 0.0
    avg_accel = np.mean(accelerations) if accelerations else 0.0
    
    return float(avg_speed), float(avg_accel)

def referee_decision(phrase: FencingPhrase, left_xdata: Dict, left_ydata: Dict,
                    right_xdata: Dict, right_ydata: Dict,
                    normalisation_constant: Optional[float] = None) -> Dict:
    """
    Main referee decision logic implementing FIE right-of-way rules.
    """
    
    # Detect pause/retreat intervals for both fencers
    left_pauses = detect_pause_retreat_intervals(
        left_xdata, left_ydata, is_left_fencer=True, fps=phrase.fps
    )
    right_pauses = detect_pause_retreat_intervals(
        right_xdata, right_ydata, is_left_fencer=False, fps=phrase.fps
    )
    
    # Determine right-of-way based on pauses
    current_right_of_way = 'none'
    
    if left_pauses and not right_pauses:
        current_right_of_way = 'right'
    elif right_pauses and not left_pauses:
        current_right_of_way = 'left'
    elif left_pauses and right_pauses:
        # Both paused - compare timing
        left_pause_time = left_pauses[0].start_time
        right_pause_time = right_pauses[0].start_time
        if left_pause_time < right_pause_time:
            current_right_of_way = 'right'
        else:
            current_right_of_way = 'left'
    
    # Detect lunges
    left_lunge = detect_lunge(
        left_xdata, left_ydata, normalisation_constant
    ) if normalisation_constant else False
    
    right_lunge = detect_lunge(
        right_xdata, right_ydata, normalisation_constant
    ) if normalisation_constant else False
    
    # Calculate speed and acceleration
    left_speed, left_accel = calculate_speed_acceleration(
        left_xdata, left_ydata
    )
    right_speed, right_accel = calculate_speed_acceleration(
        right_xdata, right_ydata
    )
    
    # Analyze blade contacts
    blade_winner = 'none'
    blade_details = {}
    
    if phrase.blade_contacts:
        first_contact = phrase.blade_contacts[0]
        blade_winner, blade_details = analyze_blade_contact(
            left_xdata, left_ydata, right_xdata, right_ydata,
            first_contact.frame, current_right_of_way
        )
    
    # Make final decision
    winner = 'right'  # Default
    reason = ""
    
    if current_right_of_way == 'left':
        winner = 'left'
        reason = "Left has right-of-way"
        if right_pauses:
            reason += f" (only right paused at {right_pauses[0].start_time:.2f}s)"
    elif current_right_of_way == 'right':
        winner = 'right'
        reason = "Right has right-of-way"
        if left_pauses:
            reason += f" (only left paused at {left_pauses[0].start_time:.2f}s)"
    else:
        # No clear right-of-way from pauses
        if blade_winner != 'none':
            winner = blade_winner
            reason = f"Blade analysis favors {blade_winner}"
        elif left_speed > right_speed * 1.2:
            winner = 'left'
            reason = "Left had significantly higher speed"
        elif right_speed > left_speed * 1.2:
            winner = 'right'
            reason = "Right had significantly higher speed"
        else:
            winner = 'right'
            reason = "Simultaneous action, default to right"
    
    return {
        'winner': winner,
        'reason': reason,
        'left_pauses': [asdict(p) for p in left_pauses],
        'right_pauses': [asdict(p) for p in right_pauses],
        'blade_analysis': blade_winner,
        'blade_details': blade_details,
        'speed_comparison': {
            'left_speed': left_speed,
            'right_speed': right_speed,
            'left_accel': left_accel,
            'right_accel': right_accel,
        },
        'lunge_detected': {
            'left': left_lunge,
            'right': right_lunge,
        }
    }

def process_video(
    video_path: str,
    txt_path: str,
    model: Optional[YOLO] = None,
    model_path: str = "yolo11x-pose.pt",
    return_keypoints: bool = False,
    output_dir: Optional[Path] = None,
    save_excel: bool = False,
) -> Dict[str, Any]:
    """
    Process a single video and return analysis results.
    """
    start_time = time.time()
    
    if model is None:
        model = YOLO(model_path)
    
    # Extract tracks
    tracks_per_frame = extract_tracks_from_video(video_path, model)
    
    # Process data
    left_xdata, left_ydata, right_xdata, right_ydata, c, video_angle = \
        process_video_and_extract_data(tracks_per_frame)
    
    # Save Excel if requested
    excel_path = None
    if save_excel and output_dir:
        excel_filename = Path(video_path).stem + "_keypoints.xlsx"
        excel_path = output_dir / excel_filename
        save_keypoints_to_excel(left_xdata, left_ydata, right_xdata, right_ydata, str(excel_path))
    
    # Parse phrase
    phrase = parse_txt_file(txt_path)
    
    # Make decision
    decision = referee_decision(phrase, left_xdata, left_ydata, right_xdata, right_ydata, c)
    
    processing_time = time.time() - start_time
    
    result = {
        'phrase': asdict(phrase),
        'decision': decision,
        'frames_processed': len(tracks_per_frame),
        'processing_time': processing_time,
        'video_angle': video_angle,
        'normalisation_constant': c,
        'video_path': video_path,
        'txt_path': txt_path,
        'excel_path': str(excel_path) if excel_path else None,
    }
    
    if return_keypoints:
        result['left_xdata'] = left_xdata
        result['left_ydata'] = left_ydata
        result['right_xdata'] = right_xdata
        result['right_ydata'] = right_ydata
    
    return sanitize_for_json(result)

def main():
    parser = argparse.ArgumentParser(description="AI Fencing Referee")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("txt", help="Path to txt file")
    parser.add_argument(
        "--model",
        default="yolo11x-pose.pt",
        help="Path to the YOLO pose model weights (default: yolo11x-pose.pt)",
    )
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--save-excel", action="store_true", help="Save keypoints to Excel")
    parser.add_argument("--save-overlay", action="store_true", help="Save overlay video")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else Path(args.video).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = YOLO(args.model)
    
    # Process video
    result = process_video(
        args.video,
        args.txt,
        model=model,
        return_keypoints=args.save_overlay,
        output_dir=output_dir,
        save_excel=args.save_excel,
    )
    
    # Save JSON result
    json_path = output_dir / "analysis_result.json"
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Analysis complete. Results saved to {json_path}")
    print(f"Winner: {result['decision']['winner']}")
    print(f"Reason: {result['decision']['reason']}")
    
    # Generate overlay if requested
    if args.save_overlay and 'left_xdata' in result:
        overlay_path = output_dir / (Path(args.video).stem + "_overlay.mp4")
        render_overlay_video(
            Path(args.video),
            overlay_path,
            result['left_xdata'],
            result['left_ydata'],
            result['right_xdata'],
            result['right_ydata'],
            result['normalisation_constant'],
            draw_skeleton=True,
            draw_labels=True,
            show_progress=True,
        )
        print(f"Overlay video saved to {overlay_path}")

if __name__ == "__main__":
    main()

@dataclass
class AnalysisResult:
    input_video_path: str
    input_signal_path: str
    decision: Dict[str, Any]
    artifacts: Dict[str, str]
    metadata: Dict[str, Any]
    
    def to_dict(self, include_keypoints: bool = False) -> Dict[str, Any]:
        d = {
            "input_video": self.input_video_path,
            "input_signal": self.input_signal_path,
            "decision": self.decision,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }
        return sanitize_for_json(d)

def analyze_video_signal(
    video_path: str,
    signal_path: str,
    model: Optional[YOLO] = None,
    model_path: str = "yolo11x-pose.pt",
    return_keypoints: bool = False,
    output_dir: Optional[Path] = None,
    save_excel: bool = True,
    save_overlay: bool = True,
    overlay_draw_skeleton: bool = True,
    overlay_draw_labels: bool = False,
    overlay_show_progress: bool = False,
    phrase: Optional[FencingPhrase] = None,
    fisheye_enabled: bool = False,
    fisheye_strength: float = -0.18,
    fisheye_balance: float = 0.0,
    fisheye_keep_audio: bool = True,
    fisheye_progress: bool = False,
) -> AnalysisResult:
    """
    Comprehensive analysis pipeline compatible with referee_service.py
    """
    if output_dir is None:
        output_dir = Path(video_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {}
    
    # 1. Fisheye Correction
    processing_video_path = video_path
    if fisheye_enabled:
        corrected_path = output_dir / (Path(video_path).stem + "_corrected.mp4")
        processing_video_path = correct_fisheye_video(
            video_path,
            str(corrected_path),
            strength=fisheye_strength,
            balance=fisheye_balance,
            keep_audio=fisheye_keep_audio,
            progress=fisheye_progress
        )
        artifacts["corrected_video"] = str(processing_video_path)

    # 2. Run Analysis (reuse process_video logic but adapted)
    if model is None:
        model = YOLO(model_path)
        
    # Extract tracks
    tracks_per_frame = extract_tracks_from_video(processing_video_path, model)
    
    # Process data
    left_xdata, left_ydata, right_xdata, right_ydata, c, video_angle = \
        process_video_and_extract_data(tracks_per_frame)
        
    # Save Excel
    if save_excel:
        excel_path = output_dir / (Path(video_path).stem + "_keypoints.xlsx")
        save_keypoints_to_excel(left_xdata, left_ydata, right_xdata, right_ydata, str(excel_path))
        artifacts["excel"] = str(excel_path)

    # Parse phrase if not provided
    if phrase is None:
        phrase = parse_txt_file(signal_path)

    # Make decision
    decision = referee_decision(phrase, left_xdata, left_ydata, right_xdata, right_ydata, c)
    
    # Generate Overlay
    if save_overlay:
        overlay_path = output_dir / (Path(video_path).stem + "_overlay.mp4")
        render_overlay_video(
            Path(processing_video_path),
            overlay_path,
            left_xdata,
            left_ydata,
            right_xdata,
            right_ydata,
            c,
            draw_skeleton=overlay_draw_skeleton,
            draw_labels=overlay_draw_labels,
            show_progress=overlay_show_progress
        )
        artifacts["analysis_video"] = str(overlay_path)

    metadata = {
        "video_angle": video_angle,
        "normalisation_constant": c,
        "frames_processed": len(tracks_per_frame),
    }

    return AnalysisResult(
        input_video_path=video_path,
        input_signal_path=signal_path,
        decision=decision,
        artifacts=artifacts,
        metadata=metadata
    )
