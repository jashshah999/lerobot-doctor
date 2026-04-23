"""Check 11: Per-Episode Summary -- flags specific bad episodes with reasons."""

from __future__ import annotations

import numpy as np

from lerobot_doctor.dataset_loader import LoadedDataset
from lerobot_doctor.runner import CheckResult, Severity


SKIP_COLUMNS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _is_numeric(arr) -> bool:
    if isinstance(arr, np.ndarray) and arr.dtype.kind in ("f", "i", "u"):
        return True
    return False


def _to_numeric(vals) -> np.ndarray | None:
    if _is_numeric(vals):
        return vals.astype(np.float64)
    try:
        return np.array(vals, dtype=np.float64)
    except (ValueError, TypeError):
        return None


def _max_consecutive_true(arr: np.ndarray) -> int:
    if len(arr) == 0:
        return 0
    max_run = current = 0
    for v in arr:
        if v:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _action_columns(dataset: LoadedDataset) -> list[str]:
    if not dataset.episodes_data:
        return []
    cols = dataset.episodes_data[0].columns.keys()
    action_cols = [c for c in cols if c.startswith("action")]
    return action_cols if action_cols else [c for c in cols if c not in SKIP_COLUMNS]


def check_per_episode(dataset: LoadedDataset) -> CheckResult:
    result = CheckResult(name="Per-Episode Summary", severity=Severity.PASS)

    if not dataset.episodes_data:
        result.warn("No episode data loaded")
        return result

    fps = dataset.info.fps if dataset.info and dataset.info.fps else 1
    action_cols = _action_columns(dataset)

    # Precompute global action diff stats for jump detection
    global_diff_std: dict[str, np.ndarray] = {}
    for col_name in action_cols:
        all_diffs = []
        for ep in dataset.episodes_data:
            if col_name not in ep.columns:
                continue
            vals = _to_numeric(ep.columns[col_name])
            if vals is None or len(vals) < 3:
                continue
            diffs = np.diff(vals, axis=0)
            if diffs.ndim == 1:
                diffs = diffs.reshape(-1, 1)
            all_diffs.append(diffs)
        if all_diffs:
            combined = np.concatenate(all_diffs, axis=0)
            std = np.std(combined, axis=0)
            std[std == 0] = 1.0
            global_diff_std[col_name] = std

    flagged: dict[int, list[str]] = {}

    for ep in dataset.episodes_data:
        ep_idx = ep.episode_index
        reasons: list[str] = []

        # Short episode
        short_threshold = max(5, fps)
        if ep.length < short_threshold:
            reasons.append(f"too short ({ep.length} frames, <{short_threshold/fps:.1f}s)")

        if ep.length <= 1:
            reasons.append("single frame (unusable)")

        # Timestamp issues
        if "timestamp" in ep.columns:
            ts = _to_numeric(ep.columns["timestamp"])
            if ts is not None and len(ts) > 1:
                diffs = np.diff(ts)
                if np.any(diffs <= 0):
                    reasons.append("non-monotonic timestamps")
                if fps > 0:
                    expected_dt = 1.0 / fps
                    gaps = diffs[diffs > expected_dt * 2.5]
                    if len(gaps) > 0:
                        reasons.append(f"{len(gaps)} dropped frame(s)")

        # Action issues per column
        for col_name in action_cols:
            if col_name not in ep.columns:
                continue
            vals = _to_numeric(ep.columns[col_name])
            if vals is None:
                continue

            if np.any(np.isnan(vals)):
                reasons.append(f"NaN in {col_name}")
            if np.any(np.isinf(vals)):
                reasons.append(f"Inf in {col_name}")

            if len(vals) < 2:
                continue

            if vals.ndim == 1:
                vals_2d = vals.reshape(-1, 1)
            else:
                vals_2d = vals

            # Frozen actions
            same = np.all(vals_2d[1:] == vals_2d[:-1], axis=-1)
            max_run = _max_consecutive_true(same)
            pct = max_run / len(vals) * 100 if len(vals) > 0 else 0
            if pct >= 5:
                reasons.append(f"{pct:.0f}% of {col_name} frozen (consecutive identical)")

            # Action jumps
            if col_name in global_diff_std and len(vals) >= 3:
                diffs = np.diff(vals, axis=0)
                if diffs.ndim == 1:
                    diffs = diffs.reshape(-1, 1)
                abs_z = np.abs(diffs / global_diff_std[col_name])
                mean_z = np.mean(abs_z, axis=-1)
                n_jumps = int(np.sum(mean_z > 8))
                if n_jumps > 0:
                    reasons.append(f"{n_jumps} action jump(s) in {col_name}")

        if reasons:
            flagged[ep_idx] = reasons

    if not flagged:
        result.pass_(f"All {len(dataset.episodes_data)} episodes look clean")
        return result

    result.warn(
        f"{len(flagged)}/{len(dataset.episodes_data)} episode(s) flagged"
    )
    for ep_idx in sorted(flagged.keys())[:20]:
        reasons = flagged[ep_idx]
        result.warn(f"Episode {ep_idx}: {'; '.join(reasons)}")
    if len(flagged) > 20:
        result.warn(f"...and {len(flagged) - 20} more flagged episodes")

    return result
