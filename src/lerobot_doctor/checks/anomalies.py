"""Check 9: Anomaly Detection -- stuck actuators, repeated episodes, distribution shifts."""

from __future__ import annotations

import numpy as np

from lerobot_doctor.dataset_loader import LoadedDataset
from lerobot_doctor.runner import CheckResult, Severity

SKIP_COLUMNS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def check_anomalies(dataset: LoadedDataset) -> CheckResult:
    result = CheckResult(name="Anomaly Detection", severity=Severity.PASS)

    if not dataset.episodes_data:
        result.warn("No episode data loaded")
        return result

    _check_stuck_actuators(dataset, result)
    _check_near_duplicate_episodes(dataset, result)
    _check_distribution_shift(dataset, result)
    _check_constant_observations(dataset, result)

    return result


def _check_stuck_actuators(dataset: LoadedDataset, result: CheckResult):
    """Detect dimensions that never change across entire episodes (stuck joints/grippers)."""
    stuck_episodes = []

    for ep in dataset.episodes_data:
        for col_name, vals in ep.columns.items():
            if col_name in SKIP_COLUMNS:
                continue
            if not (col_name.startswith("action") or "state" in col_name):
                continue
            try:
                arr = np.array(vals, dtype=np.float64)
            except (ValueError, TypeError):
                continue
            if arr.ndim < 2 or len(arr) < 5:
                continue

            # Check each dimension
            for dim in range(arr.shape[1]):
                dim_vals = arr[:, dim]
                if np.std(dim_vals) == 0:
                    continue  # already caught by statistics check
                # Check if dimension is stuck for >80% of the episode
                diffs = np.diff(dim_vals)
                pct_static = (diffs == 0).sum() / len(diffs)
                if pct_static > 0.8 and len(diffs) > 10:
                    stuck_episodes.append((ep.episode_index, col_name, dim, pct_static))

    if stuck_episodes:
        # Group by column and dimension
        by_col_dim: dict[tuple[str, int], list] = {}
        for ep_idx, col, dim, pct in stuck_episodes:
            key = (col, dim)
            if key not in by_col_dim:
                by_col_dim[key] = []
            by_col_dim[key].append(ep_idx)

        for (col, dim), episodes in list(by_col_dim.items())[:5]:
            # Only warn if stuck in >80% of ALL episodes (not just 30%)
            # because gripper DOFs being mostly static in some episodes is normal
            if len(episodes) > len(dataset.episodes_data) * 0.8:
                result.warn(
                    f"{col}[{dim}]: stuck/static in {len(episodes)}/{len(dataset.episodes_data)} "
                    f"episodes (>80% unchanged each) -- possible stuck actuator or unused DOF"
                )


def _check_near_duplicate_episodes(dataset: LoadedDataset, result: CheckResult):
    """Detect episodes that are nearly identical (copy-paste data, recording bugs)."""
    if len(dataset.episodes_data) < 2:
        return

    # Use action fingerprints for comparison
    fingerprints = []
    for ep in dataset.episodes_data:
        for col_name, vals in ep.columns.items():
            if not col_name.startswith("action"):
                continue
            try:
                arr = np.array(vals, dtype=np.float64)
            except (ValueError, TypeError):
                continue
            if arr.size == 0:
                continue
            # Fingerprint: mean of first 10 frames + last 10 frames + length
            n = min(10, len(arr))
            fp = np.concatenate([
                arr[:n].flatten()[:20],  # first few values
                arr[-n:].flatten()[:20],  # last few values
                [float(len(arr))],
            ])
            fingerprints.append((ep.episode_index, fp))
            break

    if len(fingerprints) < 2:
        return

    # Compare all pairs (O(n^2) but N is typically small with max_episodes)
    duplicates = []
    for i in range(len(fingerprints)):
        for j in range(i + 1, len(fingerprints)):
            idx_i, fp_i = fingerprints[i]
            idx_j, fp_j = fingerprints[j]
            if len(fp_i) != len(fp_j):
                continue
            if np.allclose(fp_i, fp_j, atol=1e-6):
                duplicates.append((idx_i, idx_j))

    if duplicates:
        result.warn(
            f"{len(duplicates)} near-duplicate episode pair(s) detected: "
            f"{duplicates[:5]}{'...' if len(duplicates) > 5 else ''}"
        )


def _check_distribution_shift(dataset: LoadedDataset, result: CheckResult):
    """Detect if data distribution changes significantly across the dataset."""
    if len(dataset.episodes_data) < 10:
        return

    # Compare first 25% vs last 25% of episodes
    n = len(dataset.episodes_data)
    first_quarter = dataset.episodes_data[:n // 4]
    last_quarter = dataset.episodes_data[-(n // 4):]

    for col_name in dataset.episodes_data[0].columns:
        if col_name in SKIP_COLUMNS:
            continue
        if not (col_name.startswith("action") or "state" in col_name):
            continue

        try:
            first_vals = np.concatenate([
                np.array(ep.columns[col_name], dtype=np.float64)
                for ep in first_quarter
                if col_name in ep.columns
            ], axis=0)
            last_vals = np.concatenate([
                np.array(ep.columns[col_name], dtype=np.float64)
                for ep in last_quarter
                if col_name in ep.columns
            ], axis=0)
        except (ValueError, TypeError):
            continue

        if first_vals.size == 0 or last_vals.size == 0:
            continue

        # Compare means
        first_mean = np.mean(first_vals, axis=0)
        last_mean = np.mean(last_vals, axis=0)
        first_std = np.std(first_vals, axis=0)

        # Avoid div by zero
        first_std_safe = np.where(first_std > 0, first_std, 1.0)
        shift = np.abs(last_mean - first_mean) / first_std_safe
        max_shift = np.max(shift)

        if max_shift > 3.0:
            dim = int(np.argmax(shift))
            result.warn(
                f"{col_name}: distribution shift detected between first and last quarter "
                f"of episodes (dim {dim}: {max_shift:.1f} std shift). "
                f"Could indicate environment changes or operator drift."
            )


def _check_constant_observations(dataset: LoadedDataset, result: CheckResult):
    """Detect observation features that are constant across ALL episodes (broken sensor)."""
    if not dataset.episodes_data:
        return

    # Collect all values per column, then check per-dimension
    col_all_vals: dict[str, list[np.ndarray]] = {}

    for ep in dataset.episodes_data:
        for col_name, vals in ep.columns.items():
            if col_name in SKIP_COLUMNS or col_name.startswith("action") or col_name.startswith("next."):
                continue
            try:
                arr = np.array(vals, dtype=np.float64)
            except (ValueError, TypeError):
                continue
            if arr.size == 0:
                continue
            if col_name not in col_all_vals:
                col_all_vals[col_name] = []
            col_all_vals[col_name].append(arr)

    for col_name, arrays in col_all_vals.items():
        try:
            combined = np.concatenate(arrays, axis=0)
        except ValueError:
            continue
        if combined.ndim == 1:
            combined = combined.reshape(-1, 1)

        constant_dims = []
        for dim in range(combined.shape[1]):
            dim_vals = combined[:, dim]
            if np.nanstd(dim_vals) == 0:
                constant_dims.append((dim, float(dim_vals[0])))

        if constant_dims:
            if len(constant_dims) == combined.shape[1]:
                result.warn(
                    f"{col_name}: ALL {len(constant_dims)} dimensions constant across "
                    f"ALL episodes -- possible broken sensor or unused feature"
                )
            elif len(constant_dims) > 0:
                dims_str = ", ".join(f"dim {d} (={v:.4f})" for d, v in constant_dims[:5])
                result.warn(
                    f"{col_name}: {len(constant_dims)} constant dimension(s) across "
                    f"ALL episodes: {dims_str} -- possible stuck sensor"
                )
