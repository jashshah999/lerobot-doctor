"""Check 3: Action Quality."""

from __future__ import annotations

import numpy as np

from lerobot_doctor.dataset_loader import LoadedDataset
from lerobot_doctor.runner import CheckResult, Severity

# Default features that are not actions/observations
SKIP_COLUMNS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _find_action_columns(dataset: LoadedDataset) -> list[str]:
    """Find columns that look like actions."""
    if not dataset.episodes_data:
        return []
    cols = dataset.episodes_data[0].columns.keys()
    # Explicitly named action columns
    action_cols = [c for c in cols if c.startswith("action")]
    if action_cols:
        return action_cols
    # If no explicit action columns, skip known non-action columns
    return [c for c in cols if c not in SKIP_COLUMNS]


def _is_numeric_array(arr) -> bool:
    if isinstance(arr, np.ndarray) and arr.dtype.kind in ('f', 'i', 'u'):
        return True
    return False


def check_actions(dataset: LoadedDataset) -> CheckResult:
    result = CheckResult(name="Action Quality", severity=Severity.PASS)

    if not dataset.episodes_data:
        result.warn("No episode data loaded, skipping action checks")
        return result

    action_cols = _find_action_columns(dataset)
    if not action_cols:
        result.warn("No action columns found")
        return result

    result.pass_(f"Found action columns: {action_cols}")

    for col_name in action_cols:
        _check_action_column(dataset, col_name, result)

    return result


def _check_action_column(dataset: LoadedDataset, col_name: str, result: CheckResult):
    # Collect all values for this column across episodes
    all_values = []
    for ep in dataset.episodes_data:
        if col_name not in ep.columns:
            continue
        vals = ep.columns[col_name]
        if not _is_numeric_array(vals):
            # Try converting lists of lists to 2D array
            try:
                vals = np.array(vals, dtype=np.float64)
            except (ValueError, TypeError):
                continue
        all_values.append(vals)

    if not all_values:
        return

    try:
        combined = np.concatenate(all_values, axis=0)
    except ValueError:
        return

    if combined.dtype.kind not in ('f', 'i', 'u'):
        return

    combined = combined.astype(np.float64)

    # NaN / Inf check
    nan_count = np.isnan(combined).sum()
    inf_count = np.isinf(combined).sum()
    if nan_count > 0:
        result.fail(f"{col_name}: {nan_count} NaN values detected")
    if inf_count > 0:
        result.fail(f"{col_name}: {inf_count} Inf values detected")

    if nan_count > 0 or inf_count > 0:
        # Can't do further analysis with bad values
        clean = combined[np.isfinite(combined)]
        if len(clean) == 0:
            return
        combined_clean = clean
    else:
        combined_clean = combined

    # Per-dimension analysis
    if combined_clean.ndim == 1:
        combined_clean = combined_clean.reshape(-1, 1)

    n_dims = combined_clean.shape[1] if combined_clean.ndim > 1 else 1

    for dim in range(n_dims):
        dim_vals = combined_clean[:, dim] if combined_clean.ndim > 1 else combined_clean
        dim_label = f"{col_name}[{dim}]" if n_dims > 1 else col_name

        # Clipping detection: >99% at min or max
        if len(dim_vals) > 10:
            vmin, vmax = dim_vals.min(), dim_vals.max()
            if vmin != vmax:
                at_min = (dim_vals == vmin).sum() / len(dim_vals)
                at_max = (dim_vals == vmax).sum() / len(dim_vals)
                if at_min > 0.99:
                    result.fail(f"{dim_label}: {at_min:.1%} of values at minimum ({vmin:.4f}) -- clipping detected")
                elif at_min > 0.5:
                    result.warn(f"{dim_label}: {at_min:.1%} of values at minimum ({vmin:.4f})")
                if at_max > 0.99:
                    result.fail(f"{dim_label}: {at_max:.1%} of values at maximum ({vmax:.4f}) -- clipping detected")
                elif at_max > 0.5:
                    result.warn(f"{dim_label}: {at_max:.1%} of values at maximum ({vmax:.4f})")

    # Frozen actions: check per episode
    frozen_episodes = []
    for ep in dataset.episodes_data:
        if col_name not in ep.columns:
            continue
        vals = ep.columns[col_name]
        try:
            vals = np.array(vals, dtype=np.float64)
        except (ValueError, TypeError):
            continue
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        if len(vals) < 2:
            continue

        # Check for consecutive identical rows
        same_as_prev = np.all(vals[1:] == vals[:-1], axis=-1) if vals.ndim > 1 else (vals[1:] == vals[:-1]).flatten()
        max_run = _max_consecutive_true(same_as_prev)
        ep_len = len(vals)
        if max_run >= 10:
            pct = max_run / ep_len * 100 if ep_len > 0 else 0
            frozen_episodes.append((ep.episode_index, pct))

    if frozen_episodes:
        for ep_idx, pct in frozen_episodes[:5]:
            result.warn(f"{col_name}: {pct:.0f}% of episode {ep_idx} is consecutive identical actions (frozen)")
        if len(frozen_episodes) > 5:
            result.warn(f"{col_name}: ...and {len(frozen_episodes) - 5} more episodes with frozen actions")

    # Action jumps: sudden large changes
    # Use GLOBAL std of diffs (not per-episode) for more stable threshold
    all_diffs = []
    for ep in dataset.episodes_data:
        if col_name not in ep.columns:
            continue
        vals = ep.columns[col_name]
        try:
            vals = np.array(vals, dtype=np.float64)
        except (ValueError, TypeError):
            continue
        if len(vals) < 3:
            continue
        diffs = np.diff(vals, axis=0)
        if diffs.ndim == 1:
            diffs = diffs.reshape(-1, 1)
        all_diffs.append(diffs)

    if all_diffs:
        combined_diffs = np.concatenate(all_diffs, axis=0)
        global_std = np.std(combined_diffs, axis=0)
        global_std[global_std == 0] = 1.0

        jump_episodes = []
        for ep, diffs in zip(
            [ep for ep in dataset.episodes_data if col_name in ep.columns],
            all_diffs,
        ):
            abs_z = np.abs(diffs / global_std)
            # Use MEAN z-score across dims (not max) to avoid flagging single-dim noise
            mean_z = np.mean(abs_z, axis=-1) if abs_z.ndim > 1 else abs_z.flatten()
            big_jumps = np.where(mean_z > 8)[0]
            if len(big_jumps) > 0:
                jump_episodes.append((ep.episode_index, len(big_jumps)))

        if jump_episodes:
            for ep_idx, n_jumps in jump_episodes[:5]:
                result.warn(f"{col_name}: Episode {ep_idx} has {n_jumps} sudden large action jumps (>8 std mean across dims)")
            if len(jump_episodes) > 5:
                result.warn(f"{col_name}: ...and {len(jump_episodes) - 5} more episodes with large action jumps")


def _max_consecutive_true(arr: np.ndarray) -> int:
    """Return the length of the longest consecutive True run."""
    if len(arr) == 0:
        return 0
    max_run = 0
    current = 0
    for v in arr:
        if v:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run
