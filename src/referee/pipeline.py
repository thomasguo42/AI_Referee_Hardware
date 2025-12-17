"""Rule-based and ML referee pipeline for saber phrases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

from src.ml.data_loader import FencingPhraseLoader
from src.ml.feature_extraction import (
    bucket_events,
    extract_phrase_feature_bundle,
)

ATTACK_VELOCITY_THRESHOLD = 0.15
ATTACK_HAND_THRESHOLD = 0.2
FRAME_MARGIN = 2  # ~130 ms at 15 fps
PARRY_EXTENSION_THRESHOLD = 0.12
PARRY_WINDOW_FRAMES = 4


@dataclass
class RuleBasedDecision:
    predicted: Optional[str]
    attack_side: Optional[str]
    parry_detected: bool
    reasoning: Dict[str, float]


def _first_above_threshold(series: np.ndarray, threshold: float) -> Optional[int]:
    idx = np.where(series > threshold)[0]
    return int(idx[0]) if idx.size else None


def _clamp_frame(idx: int, total_frames: int) -> int:
    return int(max(0, min(total_frames - 1, idx)))


def _first_valid_index(series: np.ndarray) -> Optional[int]:
    idx = np.where(~np.isnan(series))[0]
    if idx.size == 0:
        return None
    return int(idx[0])


def _detect_attack_side(feature_bundle: Dict, first_hit_frame: int) -> Tuple[Optional[str], Dict[str, float]]:
    times = feature_bundle["timeseries"]
    reasoning = {}
    attack_frames = {}
    displacements = {}
    extension_delta = {}
    contact_scores = {}

    contact_scores = {}

    for side in ("left", "right"):
        velocity = times[side]["front_foot_velocity"]
        sword_velocity = times[side]["sword_velocity"]
        front_positions = times[side]["front_foot_x"]
        sword_extension = times[side]["sword_extension"]
        start_idx = _first_valid_index(front_positions)
        hit_idx = _clamp_frame(first_hit_frame, len(front_positions))
        front_idx = _first_above_threshold(velocity[: first_hit_frame + 1], ATTACK_VELOCITY_THRESHOLD)
        sword_idx = _first_above_threshold(sword_velocity[: first_hit_frame + 1], ATTACK_HAND_THRESHOLD)
        idxs = [i for i in (front_idx, sword_idx) if i is not None]
        attack_frames[side] = min(idxs) if idxs else None
        reasoning[f"{side}_front_idx"] = front_idx if front_idx is not None else -1
        reasoning[f"{side}_sword_idx"] = sword_idx if sword_idx is not None else -1

        if start_idx is not None and not np.isnan(front_positions[hit_idx]) and not np.isnan(front_positions[start_idx]):
            displacements[side] = front_positions[hit_idx] - front_positions[start_idx]
        else:
            displacements[side] = None
        if start_idx is not None and not np.isnan(sword_extension[hit_idx]) and not np.isnan(sword_extension[start_idx]):
            extension_delta[side] = sword_extension[hit_idx] - sword_extension[start_idx]
        else:
            extension_delta[side] = None
        reasoning[f"{side}_disp_hit"] = displacements[side] if displacements[side] is not None else 0.0
        reasoning[f"{side}_ext_delta"] = extension_delta[side] if extension_delta[side] is not None else 0.0

    left_idx = attack_frames["left"]
    right_idx = attack_frames["right"]
    if left_idx is None and right_idx is None:
        return None, reasoning
    if left_idx is None:
        reasoning["attack_choice"] = 1
        return "right", reasoning
    if right_idx is None:
        reasoning["attack_choice"] = -1
        return "left", reasoning

    diff = left_idx - right_idx
    reasoning["attack_diff"] = diff
    if diff <= -FRAME_MARGIN:
        return "left", reasoning
    if diff >= FRAME_MARGIN:
        return "right", reasoning

    # Tie-breaker cascade: larger forward displacement at hit, then extension delta, then average extension
    if all(v is not None for v in displacements.values()) and displacements["left"] != displacements["right"]:
        reasoning["attack_tiebreak"] = 1
        return ("left" if displacements["left"] > displacements["right"] else "right"), reasoning

    if all(v is not None for v in extension_delta.values()) and extension_delta["left"] != extension_delta["right"]:
        reasoning["attack_tiebreak"] = 2
        return ("left" if extension_delta["left"] > extension_delta["right"] else "right"), reasoning

    left_extension = np.nanmean(times["left"]["sword_extension"][: first_hit_frame])
    right_extension = np.nanmean(times["right"]["sword_extension"][: first_hit_frame])
    reasoning["left_extension_mean"] = float(left_extension) if not np.isnan(left_extension) else -1
    reasoning["right_extension_mean"] = float(right_extension) if not np.isnan(right_extension) else -1
    if left_extension > right_extension:
        return "left", reasoning
    if right_extension > left_extension:
        return "right", reasoning
    return None, reasoning


def _detect_parry(feature_bundle: Dict, attack_side: Optional[str], first_hit_frame: int, contact_frames: List[int]) -> bool:
    if attack_side is None or not contact_frames:
        return False
    defender = "right" if attack_side == "left" else "left"
    attacker = attack_side

    times = feature_bundle["timeseries"]
    defender_extension = times[defender]["sword_extension"]
    attacker_velocity = times[attacker]["sword_velocity"]
    defender_velocity = times[defender]["front_foot_velocity"]
    attacker_front_velocity = times[attacker]["front_foot_velocity"]

    relevant_contacts = [f for f in contact_frames if f < first_hit_frame]
    if not relevant_contacts:
        return False

    last_contact = max(relevant_contacts)
    start = _clamp_frame(last_contact, len(defender_extension))
    end = _clamp_frame(last_contact + PARRY_WINDOW_FRAMES, len(defender_extension))
    if end <= start:
        return False

    before_window = defender_extension[max(0, start - 2):start + 1]
    after_window = defender_extension[start:end]
    if before_window.size == 0 or after_window.size == 0:
        return False

    gain = np.nanmax(after_window) - np.nanmin(before_window)
    attacker_spike = np.nanmax(attacker_velocity[start:end]) if np.any(~np.isnan(attacker_velocity[start:end])) else 0.0
    defender_push = np.nanmax(defender_velocity[start:end]) if np.any(~np.isnan(defender_velocity[start:end])) else 0.0
    attacker_drop = np.nanmin(attacker_front_velocity[start:end]) if np.any(~np.isnan(attacker_front_velocity[start:end])) else 0.0

    return (
        gain > PARRY_EXTENSION_THRESHOLD
        and attacker_spike < ATTACK_HAND_THRESHOLD
        and defender_push > 0.05
        and attacker_drop < 0.02
    )


def rule_based_referee(phrase: Dict, feature_bundle: Dict, fps: float) -> RuleBasedDecision:
    events = bucket_events(phrase["events"])
    total_frames = phrase["left_x"].shape[0]

    # Determine earliest scoring frame
    scoring_frames = []
    for bucket in ("hit_left", "hit_right", "double_hit"):
        scoring_frames.extend(ev["frame"] for ev in events[bucket])
    first_hit_frame = min(scoring_frames) if scoring_frames else total_frames - 1

    attack_side, reasoning = _detect_attack_side(feature_bundle, first_hit_frame)

    # If only one fencer recorded a hit, honor it directly
    if events["hit_left"] and not events["hit_right"] and not events["double_hit"]:
        return RuleBasedDecision("left", attack_side, False, reasoning)
    if events["hit_right"] and not events["hit_left"] and not events["double_hit"]:
        return RuleBasedDecision("right", attack_side, False, reasoning)

    contact_frames = [ev["frame"] for ev in events["blade_contact"]]
    parry = _detect_parry(feature_bundle, attack_side, first_hit_frame, contact_frames)

    if parry and attack_side is not None:
        predicted = "right" if attack_side == "left" else "left"
        return RuleBasedDecision(predicted, attack_side, True, reasoning)

    return RuleBasedDecision(attack_side, attack_side, False, reasoning)


def _metadata_features(phrase: Dict) -> Dict[str, float]:
    meta = phrase.get("metadata", {})
    lockouts = meta.get("lockout_windows", []) or []
    feat = {
        "lockout_count": float(len(lockouts)),
        "phrase_duration": float(meta.get("end_time", 0.0) - meta.get("start_time", 0.0)
                                 if meta.get("end_time") is not None and meta.get("start_time") is not None else 0.0),
    }
    if lockouts:
        feat["lockout_first_window"] = float(lockouts[0].get("window", 0.0))
        feat["lockout_first_frame"] = float(lockouts[0].get("frame", -1))
        feat["lockout_last_window"] = float(lockouts[-1].get("window", 0.0))
        feat["lockout_last_frame"] = float(lockouts[-1].get("frame", -1))
    else:
        feat["lockout_first_window"] = 0.0
        feat["lockout_first_frame"] = -1.0
        feat["lockout_last_window"] = 0.0
        feat["lockout_last_frame"] = -1.0
    return feat


def _event_relative_features(bundle: Dict, phrase: Dict, fps: float) -> Dict[str, float]:
    events = bucket_events(phrase["events"])
    times = bundle["timeseries"]
    total_frames = phrase["left_x"].shape[0]

    scoring_events: List[Tuple[int, str]] = []
    for label in ("hit_left", "hit_right", "double_hit"):
        for ev in events[label]:
            scoring_events.append((ev["frame"], label))
    if scoring_events:
        scoring_events.sort(key=lambda x: x[0])
        first_hit_frame = scoring_events[0][0]
        first_hit_label = scoring_events[0][1]
    else:
        first_hit_frame = total_frames - 1
        first_hit_label = "none"

    feature_dict: Dict[str, float] = {
        "first_hit_frame": float(first_hit_frame),
        "first_hit_type": {
            "hit_left": 1.0,
            "hit_right": 2.0,
            "double_hit": 3.0,
        }.get(first_hit_label, 0.0),
        "double_hit_flag": 1.0 if events["double_hit"] else 0.0,
        "blade_contact_count": float(len(events["blade_contact"])),
    }

    left_hit_frame = events["hit_left"][0]["frame"] if events["hit_left"] else -1
    right_hit_frame = events["hit_right"][0]["frame"] if events["hit_right"] else -1
    double_frame = events["double_hit"][0]["frame"] if events["double_hit"] else -1

    feature_dict.update({
        "left_first_hit_frame": float(left_hit_frame),
        "right_first_hit_frame": float(right_hit_frame),
        "double_first_frame": float(double_frame),
    })

    if left_hit_frame >= 0 and (right_hit_frame < 0 or left_hit_frame < right_hit_frame):
        feature_dict["hit_order_indicator"] = -1.0
    elif right_hit_frame >= 0 and (left_hit_frame < 0 or right_hit_frame < left_hit_frame):
        feature_dict["hit_order_indicator"] = 1.0
    elif double_frame >= 0:
        feature_dict["hit_order_indicator"] = 0.0
    else:
        feature_dict["hit_order_indicator"] = 0.5  # unresolved

    contact_before = [ev["frame"] for ev in events["blade_contact"] if ev["frame"] < first_hit_frame]
    last_contact = None
    if contact_before:
        last_contact = max(contact_before)
        feature_dict["contact_last_before_hit"] = float(last_contact)
        frame_gap = first_hit_frame - last_contact
        feature_dict["contact_to_hit_gap"] = float(frame_gap)
        feature_dict["contact_to_hit_time"] = float(frame_gap / fps)
    else:
        feature_dict["contact_last_before_hit"] = -1.0
        feature_dict["contact_to_hit_gap"] = float(total_frames)
        feature_dict["contact_to_hit_time"] = float(total_frames / fps)

    pair_gap = times["pair"]["front_foot_gap"]
    gap_start_idx = _first_valid_index(pair_gap)
    hit_idx = _clamp_frame(first_hit_frame, len(pair_gap))
    if gap_start_idx is not None and not np.isnan(pair_gap[hit_idx]) and not np.isnan(pair_gap[gap_start_idx]):
        feature_dict["gap_delta_hit"] = float(pair_gap[hit_idx] - pair_gap[gap_start_idx])
        feature_dict["gap_at_hit"] = float(pair_gap[hit_idx])
    else:
        feature_dict["gap_delta_hit"] = 0.0
        feature_dict["gap_at_hit"] = 0.0

    CONTACT_WINDOW = 4
    HIT_WINDOW = 4

    def _window_stats(values: np.ndarray, center: int, pre: int, post: int) -> Tuple[float, float]:
        pre_start = max(0, center - pre)
        pre_end = max(pre_start, center)
        post_start = center
        post_end = min(len(values), center + post)
        pre_window = values[pre_start:pre_end]
        post_window = values[post_start:post_end]
        pre_mean = float(np.nanmean(pre_window)) if pre_window.size else 0.0
        post_mean = float(np.nanmean(post_window)) if post_window.size else 0.0
        return pre_mean, post_mean

    contact_scores = {}

    for side in ("left", "right"):
        front = times[side]["front_foot_x"]
        opp_front = times["right" if side == "left" else "left"]["front_foot_x"]
        sword_ext = times[side]["sword_extension"]
        sword_vel = times[side]["sword_velocity"]
        foot_vel = times[side]["front_foot_velocity"]
        sword_x = times[side]["sword_x"]

        start_idx = _first_valid_index(front)
        hit_idx = _clamp_frame(first_hit_frame, len(front))

        if start_idx is not None and not np.isnan(front[hit_idx]):
            displacement = front[hit_idx] - front[start_idx]
        else:
            displacement = np.nan

        if not np.isnan(sword_ext[hit_idx]) and start_idx is not None and not np.isnan(sword_ext[start_idx]):
            extension_delta = sword_ext[hit_idx] - sword_ext[start_idx]
        else:
            extension_delta = np.nan

        valid_vel = foot_vel[:hit_idx]
        valid_sword_vel = sword_vel[:hit_idx]
        positive_ratio = float(np.mean(valid_vel > 0)) if valid_vel.size else np.nan
        sword_speed_peak = float(np.nanmax(valid_sword_vel)) if valid_sword_vel.size else np.nan

        window_start = max(0, hit_idx - 5)
        window_vel = foot_vel[window_start:hit_idx]
        window_sword = sword_vel[window_start:hit_idx]
        window_vel_mean = float(np.nanmean(window_vel)) if window_vel.size else 0.0
        window_sword_mean = float(np.nanmean(window_sword)) if window_sword.size else 0.0

        feature_dict[f"{side}_front_displacement_hit"] = float(displacement) if not np.isnan(displacement) else 0.0
        feature_dict[f"{side}_sword_extension_delta"] = float(extension_delta) if not np.isnan(extension_delta) else 0.0
        feature_dict[f"{side}_positive_velocity_ratio"] = positive_ratio if not np.isnan(positive_ratio) else 0.0
        feature_dict[f"{side}_sword_speed_peak"] = sword_speed_peak if not np.isnan(sword_speed_peak) else 0.0
        feature_dict[f"{side}_window_velocity_mean"] = window_vel_mean if not np.isnan(window_vel_mean) else 0.0
        feature_dict[f"{side}_window_sword_mean"] = window_sword_mean if not np.isnan(window_sword_mean) else 0.0

        if last_contact is not None:
            c_idx = _clamp_frame(last_contact, len(sword_ext))
            pre_vel, post_vel = _window_stats(foot_vel, c_idx, CONTACT_WINDOW, CONTACT_WINDOW)
            pre_sword, post_sword = _window_stats(sword_vel, c_idx, CONTACT_WINDOW, CONTACT_WINDOW)
            pre_ext, post_ext = _window_stats(sword_ext, c_idx, CONTACT_WINDOW, CONTACT_WINDOW)
            feature_dict[f"{side}_contact_pre_foot_vel"] = pre_vel
            feature_dict[f"{side}_contact_post_foot_vel"] = post_vel
            feature_dict[f"{side}_contact_pre_sword_vel"] = pre_sword
            feature_dict[f"{side}_contact_post_sword_vel"] = post_sword
            feature_dict[f"{side}_contact_pre_sword_ext"] = pre_ext
            feature_dict[f"{side}_contact_post_sword_ext"] = post_ext
            contact_to_hit_ext_delta = (
                float(sword_ext[hit_idx] - sword_ext[c_idx])
                if not np.isnan(sword_ext[hit_idx]) and not np.isnan(sword_ext[c_idx])
                else 0.0
            )
            feature_dict[f"{side}_contact_to_hit_ext_delta"] = contact_to_hit_ext_delta
            if not np.isnan(sword_x[c_idx]) and not np.isnan(opp_front[c_idx]):
                feature_dict[f"{side}_contact_sword_vs_opp_front"] = float(opp_front[c_idx] - sword_x[c_idx])
            else:
                feature_dict[f"{side}_contact_sword_vs_opp_front"] = 0.0
        else:
            for suffix in [
                "contact_pre_foot_vel",
                "contact_post_foot_vel",
                "contact_pre_sword_vel",
                "contact_post_sword_vel",
                "contact_pre_sword_ext",
                "contact_post_sword_ext",
                "contact_to_hit_ext_delta",
            ]:
                feature_dict[f"{side}_{suffix}"] = 0.0
            feature_dict[f"{side}_contact_sword_vs_opp_front"] = 0.0

        hit_pre_vel, hit_post_vel = _window_stats(foot_vel, hit_idx, HIT_WINDOW, HIT_WINDOW)
        hit_pre_sword, hit_post_sword = _window_stats(sword_vel, hit_idx, HIT_WINDOW, HIT_WINDOW)
        hit_pre_ext, hit_post_ext = _window_stats(sword_ext, hit_idx, HIT_WINDOW, HIT_WINDOW)
        feature_dict[f"{side}_hit_pre_foot_vel"] = hit_pre_vel
        feature_dict[f"{side}_hit_post_foot_vel"] = hit_post_vel
        feature_dict[f"{side}_hit_pre_sword_vel"] = hit_pre_sword
        feature_dict[f"{side}_hit_post_sword_vel"] = hit_post_sword
        feature_dict[f"{side}_hit_pre_sword_ext"] = hit_pre_ext
        feature_dict[f"{side}_hit_post_sword_ext"] = hit_post_ext
        if not np.isnan(sword_x[hit_idx]) and not np.isnan(opp_front[hit_idx]):
            feature_dict[f"{side}_hit_sword_vs_opp_front"] = float(opp_front[hit_idx] - sword_x[hit_idx])
        else:
            feature_dict[f"{side}_hit_sword_vs_opp_front"] = 0.0

        contact_scores[side] = (
            feature_dict.get(f"{side}_contact_post_foot_vel", 0.0)
            + 0.5 * feature_dict.get(f"{side}_contact_post_sword_vel", 0.0)
            + feature_dict.get(f"{side}_contact_to_hit_ext_delta", 0.0)
            - 0.3 * abs(feature_dict.get(f"{side}_contact_pre_sword_vel", 0.0))
        )

    if all(side in contact_scores for side in ("left", "right")):
        feature_dict["contact_attack_score_left"] = contact_scores["left"]
        feature_dict["contact_attack_score_right"] = contact_scores["right"]
        feature_dict["contact_attack_score_diff"] = contact_scores["right"] - contact_scores["left"]
    else:
        feature_dict["contact_attack_score_left"] = 0.0
        feature_dict["contact_attack_score_right"] = 0.0
        feature_dict["contact_attack_score_diff"] = 0.0

    return feature_dict


def _side_delta_features(bundle: Dict) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    left_summary = bundle["summary"]["left"]
    right_summary = bundle["summary"]["right"]
    for key, left_val in left_summary.items():
        right_val = right_summary.get(key)
        if right_val is None:
            continue
        deltas[f"delta_{key}"] = float(right_val - left_val)
    return deltas


def flatten_summary_features(bundle: Dict, phrase: Dict, fps: float, decision: Optional[RuleBasedDecision] = None) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for side in ("left", "right", "pair"):
        for key, value in bundle["summary"][side].items():
            flat[f"{side}_{key}"] = value
    flat.update(_side_delta_features(bundle))
    events = bucket_events(phrase["events"])
    for key, evs in events.items():
        flat[f"events_{key}_count"] = float(len(evs))
        if evs:
            flat[f"events_{key}_first_frame"] = float(evs[0]["frame"])
            flat[f"events_{key}_last_frame"] = float(evs[-1]["frame"])
        else:
            flat[f"events_{key}_first_frame"] = -1.0
            flat[f"events_{key}_last_frame"] = -1.0
    flat["phrase_frames"] = float(phrase["left_x"].shape[0])
    flat.update(_event_relative_features(bundle, phrase, fps))
    flat.update(_metadata_features(phrase))
    if decision is not None:
        mapping = {"left": -1.0, "right": 1.0, None: 0.0}
        flat["rule_predicted_side"] = mapping.get(decision.predicted, 0.0)
        flat["rule_attack_side"] = mapping.get(decision.attack_side, 0.0)
        flat["rule_parry_flag"] = 1.0 if decision.parry_detected else 0.0
        for key, value in decision.reasoning.items():
            flat[f"rule_reason_{key}"] = float(value)
    return flat


def build_feature_table(root_dir: str) -> Tuple[pd.DataFrame, List[Dict]]:
    loader = FencingPhraseLoader(root_dir)
    phrases = loader.load_all()
    records = []
    metadata: List[Dict] = []
    for phrase in phrases:
        bundle = extract_phrase_feature_bundle(phrase, loader.fps)
        decision = rule_based_referee(phrase, bundle, loader.fps)
        features = flatten_summary_features(bundle, phrase, loader.fps, decision)
        features["label"] = phrase["winner"]
        features["folder"] = phrase["folder"]
        features["session"] = phrase["folder"].split('_')[0]
        records.append(features)
        metadata.append({
            "phrase": phrase,
            "bundle": bundle,
            "decision": decision,
        })
    df = pd.DataFrame(records)
    return df, metadata


def evaluate_rule_based(metadata: List[Dict], fps: float) -> pd.DataFrame:
    rows = []
    for meta in metadata:
        phrase = meta["phrase"]
        bundle = meta["bundle"]
        decision = rule_based_referee(phrase, bundle, fps)
        rows.append({
            "folder": phrase["folder"],
            "winner": phrase["winner"],
            "predicted": decision.predicted,
            "attack_side": decision.attack_side,
            "parry": decision.parry_detected,
        })
    return pd.DataFrame(rows)


def _build_model(name: str):
    if name == "gb":
        return GradientBoostingClassifier(random_state=42)
    if name == "hist":
        return HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=300, random_state=42)
    if name == "rf":
        return RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                                      class_weight='balanced', random_state=42)
    if name == "logit":
        return LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced')
    if name == "cat":
        return CatBoostClassifier(
            iterations=1800,
            depth=8,
            learning_rate=0.03,
            loss_function='Logloss',
            eval_metric='Accuracy',
            l2_leaf_reg=5.0,
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"Unknown model {name}")


def train_classical_model(df: pd.DataFrame, model_name: str = "gb", groups: Optional[np.ndarray] = None):
    features = df.drop(columns=["label", "folder", "session"], errors="ignore")
    labels = df["label"].map({"left": 0, "right": 1})
    mask = labels.notna()
    features = features[mask]
    labels = labels[mask]
    groups = df.loc[mask, "session"].to_numpy() if groups is None else groups[mask]

    X = features.to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = labels.to_numpy(dtype=int)

    unique_sessions = np.unique(groups)
    n_splits = min(5, len(unique_sessions)) if len(unique_sessions) > 1 else 1

    gkf = GroupKFold(n_splits=n_splits) if n_splits > 1 else None
    fold_results = []
    models = []
    class_counts = np.bincount(y)
    total = y.shape[0]
    class_weights = {cls: total / (2 * count) for cls, count in enumerate(class_counts) if count > 0}
    sample_weights = np.array([class_weights[label] for label in y])

    if gkf:
        for train_idx, test_idx in gkf.split(X, y, groups):
            model = _build_model(model_name)
            model.fit(X[train_idx], y[train_idx], sample_weight=sample_weights[train_idx])
            preds = model.predict(X[test_idx])
            acc = accuracy_score(y[test_idx], preds)
            cm = confusion_matrix(y[test_idx], preds)
            fold_results.append({
                "accuracy": acc,
                "confusion_matrix": cm,
                "test_sessions": np.unique(groups[test_idx]),
            })
            models.append(model)
    else:
        model = _build_model(model_name)
        model.fit(X, y, sample_weight=sample_weights)
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        cm = confusion_matrix(y, preds)
        fold_results.append({"accuracy": acc, "confusion_matrix": cm, "test_sessions": unique_sessions})
        models.append(model)

    return {
        "models": models,
        "folds": fold_results,
        "feature_columns": features.columns.tolist(),
        "model_name": model_name,
    }


def train_ensemble_model(df: pd.DataFrame, base_models: Tuple[str, ...] = ("gb", "hist", "rf", "cat")):
    features = df.drop(columns=["label", "folder", "session"], errors="ignore")
    labels = df["label"].map({"left": 0, "right": 1})
    mask = labels.notna()
    features = features[mask]
    labels = labels[mask].to_numpy(dtype=int)
    groups = df.loc[mask, "session"].to_numpy()
    folders = df.loc[mask, "folder"].to_numpy()

    X = features.to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0)
    n_samples = len(labels)

    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    splits = list(gkf.split(X, labels, groups))

    base_oof = {name: np.zeros(n_samples) for name in base_models}

    for train_idx, test_idx in splits:
        class_counts = np.bincount(labels[train_idx], minlength=2)
        total = train_idx.size
        weights = {cls: total / (2 * count) for cls, count in enumerate(class_counts) if count > 0}
        sample_weights = np.array([weights[y] for y in labels[train_idx]])

        for name in base_models:
            model = _build_model(name)
            model.fit(X[train_idx], labels[train_idx], sample_weight=sample_weights)
            prob = model.predict_proba(X[test_idx])[:, 1]
            base_oof[name][test_idx] = prob

    stack_features = np.column_stack([base_oof[name] for name in base_models])

    extra_cols = []
    for col in ["rule_predicted_side", "rule_attack_side", "rule_parry_flag", "hit_order_indicator", "double_hit_flag"]:
        if col in df.columns:
            extra_cols.append(df.loc[mask, col].to_numpy(dtype=float))
    if extra_cols:
        stack_features = np.column_stack([stack_features] + extra_cols)

    stack_features = np.nan_to_num(stack_features, nan=0.0)

    oof_probs = np.zeros(n_samples)
    fold_rows = []
    for train_idx, test_idx in splits:
        class_counts = np.bincount(labels[train_idx], minlength=2)
        total = train_idx.size
        weights = {cls: total / (2 * count) for cls, count in enumerate(class_counts) if count > 0}
        sample_weights = np.array([weights[y] for y in labels[train_idx]])

        meta_model = LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced')
        meta_model.fit(stack_features[train_idx], labels[train_idx], sample_weight=sample_weights)
        probs = meta_model.predict_proba(stack_features[test_idx])[:, 1]
        oof_probs[test_idx] = probs
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(labels[test_idx], preds)
        cm = confusion_matrix(labels[test_idx], preds)
        fold_rows.append({
            "sessions": np.unique(groups[test_idx]),
            "accuracy": acc,
            "confusion": cm,
        })

    thresholds = np.linspace(0.3, 0.7, 41)
    best_thresh = 0.5
    best_acc = 0.0
    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    oof_preds = (oof_probs >= best_thresh).astype(int)
    double_flags = features["double_hit_flag"].to_numpy() if "double_hit_flag" in features.columns else np.zeros(n_samples)
    if "events_hit_left_count" in features.columns and "events_hit_right_count" in features.columns:
        left_counts = features["events_hit_left_count"].to_numpy()
        right_counts = features["events_hit_right_count"].to_numpy()
        single_mask = double_flags == 0
        single_override = np.where(left_counts > right_counts, 0, 1)
        ambiguous = (left_counts == right_counts)
        single_override[ambiguous] = oof_preds[ambiguous]
        oof_preds[single_mask] = single_override[single_mask]

    overall_acc = accuracy_score(labels, oof_preds)
    return {
        "folds": fold_rows,
        "overall_accuracy": overall_acc,
        "folders": folders,
        "predictions": oof_preds,
        "labels": labels,
        "double_flags": double_flags,
        "threshold": best_thresh,
    }


def main():
    df, metadata = build_feature_table("/workspace/data/training_data")
    print(f"Dataset size: {len(df)} phrases")
    rule_df = evaluate_rule_based(metadata, fps=15.0)
    valid = rule_df.dropna(subset=["predicted"])
    accuracy = (valid["predicted"] == valid["winner"]).mean()
    print(f"Rule-based accuracy (where prediction available): {accuracy:.3f} ({len(valid)} evaluated)")
    for model_name in ("gb", "hist", "rf", "cat", "logit"):
        report = train_classical_model(df, model_name=model_name)
        print(f"Model {model_name} results:")
        for fold in report["folds"]:
            sessions = ','.join(fold["test_sessions"])
            print(f"  Fold sessions [{sessions}] accuracy={fold['accuracy']:.3f}")
    ensemble = train_ensemble_model(df)
    print("Ensemble (gb+hist+rf+cat) results:")
    for fold in ensemble["folds"]:
        sessions = ','.join(fold["sessions"])
        print(f"  Fold sessions [{sessions}] accuracy={fold['accuracy']:.3f}")
    print(f"OOF accuracy={ensemble['overall_accuracy']:.3f} (meta threshold={ensemble['threshold']:.3f})")


if __name__ == "__main__":
    main()
