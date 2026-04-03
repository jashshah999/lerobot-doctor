"""Check 5: Data Distribution Sanity."""

from __future__ import annotations

import json

import numpy as np

from lerobot_doctor.dataset_loader import LoadedDataset
from lerobot_doctor.runner import CheckResult, Severity

SKIP_COLUMNS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
# Binary/boolean columns where outlier detection doesn't make sense
BINARY_COLUMNS = {"next.done", "next.success", "next.reward"}


def check_statistics(dataset: LoadedDataset) -> CheckResult:
    result = CheckResult(name="Data Distribution", severity=Severity.PASS)

    if not dataset.episodes_data:
        result.warn("No episode data loaded, skipping statistics checks")
        return result

    # Collect all numeric columns
    numeric_cols = _find_numeric_columns(dataset)
    if not numeric_cols:
        result.warn("No numeric columns found for statistics checks")
        return result

    for col_name in numeric_cols:
        _check_column_stats(dataset, col_name, result)

    # Compare against stored stats.json if available
    if dataset.info is not None:
        _check_stats_json(dataset, result)

    return result


def _find_numeric_columns(dataset: LoadedDataset) -> list[str]:
    if not dataset.episodes_data:
        return []
    cols = []
    sample_ep = dataset.episodes_data[0]
    for col_name, vals in sample_ep.columns.items():
        if col_name in SKIP_COLUMNS:
            continue
        if isinstance(vals, np.ndarray) and vals.dtype.kind in ('f', 'i', 'u'):
            cols.append(col_name)
        elif isinstance(vals, (list, np.ndarray)):
            try:
                arr = np.array(vals, dtype=np.float64)
                if arr.size > 0:
                    cols.append(col_name)
            except (ValueError, TypeError):
                pass
    return cols


def _check_column_stats(dataset: LoadedDataset, col_name: str, result: CheckResult):
    # Skip outlier checks for binary signal columns
    is_binary = col_name in BINARY_COLUMNS or col_name.startswith("next.")

    all_vals = []
    for ep in dataset.episodes_data:
        if col_name not in ep.columns:
            continue
        vals = ep.columns[col_name]
        try:
            vals = np.array(vals, dtype=np.float64)
        except (ValueError, TypeError):
            continue
        all_vals.append(vals)

    if not all_vals:
        return

    try:
        combined = np.concatenate(all_vals, axis=0)
    except ValueError:
        return

    if combined.size == 0:
        return

    # NaN / Inf
    nan_count = np.isnan(combined).sum()
    inf_count = np.isinf(combined).sum()
    if nan_count > 0:
        result.fail(f"{col_name}: {nan_count} NaN values")
    if inf_count > 0:
        result.fail(f"{col_name}: {inf_count} Inf values")

    clean = combined[np.isfinite(combined)] if combined.ndim == 1 else combined[np.all(np.isfinite(combined), axis=-1)]
    if clean.size == 0:
        return

    if clean.ndim == 1:
        clean = clean.reshape(-1, 1)

    n_dims = clean.shape[1]
    for dim in range(n_dims):
        dim_vals = clean[:, dim]
        dim_label = f"{col_name}[{dim}]" if n_dims > 1 else col_name

        std = np.std(dim_vals)
        if std == 0:
            result.warn(f"{dim_label}: zero variance (constant value {dim_vals[0]:.4f})")
            continue

        if is_binary:
            continue  # binary signals have extreme z-scores by nature

        mean = np.mean(dim_vals)
        z_scores = np.abs((dim_vals - mean) / std)
        extreme = (z_scores > 10).sum()
        if extreme > 0:
            result.warn(f"{dim_label}: {extreme} extreme outlier(s) (>10 std from mean)")


def _check_stats_json(dataset: LoadedDataset, result: CheckResult):
    stats_path = dataset.root / "meta" / "stats.json"
    if not stats_path.exists():
        result.warn("stats.json not found -- cannot compare computed vs stored statistics")
        return

    try:
        stored = json.loads(stats_path.read_text())
    except (json.JSONDecodeError, OSError):
        result.warn("stats.json could not be parsed")
        return

    result.pass_("stats.json found and loaded for comparison")

    # Basic check: are the stored feature names present?
    if dataset.episodes_data:
        data_cols = set(dataset.episodes_data[0].columns.keys()) - SKIP_COLUMNS
        stats_features = set(stored.keys())
        missing_in_stats = data_cols - stats_features
        if missing_in_stats:
            result.warn(f"Features in data but not in stats.json: {sorted(missing_in_stats)[:5]}")
