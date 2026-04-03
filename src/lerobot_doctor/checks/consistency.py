"""Check 7: Feature Consistency -- cross-episode dtype/shape validation."""

from __future__ import annotations

import numpy as np

from lerobot_doctor.dataset_loader import LoadedDataset
from lerobot_doctor.runner import CheckResult, Severity

SKIP_COLUMNS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def check_consistency(dataset: LoadedDataset) -> CheckResult:
    result = CheckResult(name="Feature Consistency", severity=Severity.PASS)

    if not dataset.episodes_data:
        result.warn("No episode data loaded")
        return result

    if len(dataset.episodes_data) < 2:
        result.pass_("Only 1 episode loaded, cross-episode consistency check skipped")
        return result

    # Get reference from first episode
    ref_ep = dataset.episodes_data[0]
    ref_cols = set(ref_ep.columns.keys())

    # Track issues
    missing_cols_episodes = []  # (ep_idx, missing_cols)
    extra_cols_episodes = []    # (ep_idx, extra_cols)
    shape_mismatches = []       # (ep_idx, col, expected_shape, actual_shape)
    dtype_mismatches = []       # (ep_idx, col, expected_dtype, actual_dtype)

    # Get reference shapes and dtypes
    ref_shapes = {}
    ref_dtypes = {}
    for col, vals in ref_ep.columns.items():
        if col in SKIP_COLUMNS:
            continue
        if isinstance(vals, np.ndarray):
            if vals.ndim > 1:
                ref_shapes[col] = vals.shape[1:]
            ref_dtypes[col] = vals.dtype
        elif isinstance(vals, list) and len(vals) > 0:
            try:
                arr = np.array(vals[0])
                ref_shapes[col] = arr.shape
                ref_dtypes[col] = arr.dtype
            except (ValueError, TypeError):
                pass

    # Check each episode against reference
    for ep in dataset.episodes_data[1:]:
        ep_cols = set(ep.columns.keys())

        # Missing columns
        missing = ref_cols - ep_cols
        if missing:
            meaningful_missing = missing - SKIP_COLUMNS
            if meaningful_missing:
                missing_cols_episodes.append((ep.episode_index, meaningful_missing))

        # Extra columns
        extra = ep_cols - ref_cols
        if extra:
            meaningful_extra = extra - SKIP_COLUMNS
            if meaningful_extra:
                extra_cols_episodes.append((ep.episode_index, meaningful_extra))

        # Shape and dtype consistency
        for col in ref_cols & ep_cols:
            if col in SKIP_COLUMNS:
                continue
            vals = ep.columns[col]

            if isinstance(vals, np.ndarray) and col in ref_shapes:
                if vals.ndim > 1 and vals.shape[1:] != ref_shapes[col]:
                    shape_mismatches.append((ep.episode_index, col, ref_shapes[col], vals.shape[1:]))
                if col in ref_dtypes and vals.dtype != ref_dtypes[col]:
                    dtype_mismatches.append((ep.episode_index, col, ref_dtypes[col], vals.dtype))
            elif isinstance(vals, list) and len(vals) > 0 and col in ref_shapes:
                try:
                    sample = np.array(vals[0])
                    if sample.shape != ref_shapes[col]:
                        shape_mismatches.append((ep.episode_index, col, ref_shapes[col], sample.shape))
                except (ValueError, TypeError):
                    pass

    # Report
    if missing_cols_episodes:
        result.fail(
            f"{len(missing_cols_episodes)} episode(s) missing features present in episode 0: "
            f"{[(idx, sorted(cols)) for idx, cols in missing_cols_episodes[:5]]}"
        )

    if extra_cols_episodes:
        result.warn(
            f"{len(extra_cols_episodes)} episode(s) have extra features not in episode 0: "
            f"{[(idx, sorted(cols)) for idx, cols in extra_cols_episodes[:5]]}"
        )

    if shape_mismatches:
        result.fail(
            f"{len(shape_mismatches)} shape mismatch(es) across episodes: "
            f"{[(idx, col, f'{exp}->{act}') for idx, col, exp, act in shape_mismatches[:5]]}"
        )

    if dtype_mismatches:
        result.warn(
            f"{len(dtype_mismatches)} dtype mismatch(es) across episodes: "
            f"{[(idx, col, f'{exp}->{act}') for idx, col, exp, act in dtype_mismatches[:5]]}"
        )

    if not (missing_cols_episodes or extra_cols_episodes or shape_mismatches or dtype_mismatches):
        result.pass_(
            f"All {len(dataset.episodes_data)} episodes have consistent features "
            f"({len(ref_cols - SKIP_COLUMNS)} data columns)"
        )

    # Check within-episode consistency: do all frames have same shape?
    for ep in dataset.episodes_data[:10]:  # sample first 10 episodes
        for col, vals in ep.columns.items():
            if col in SKIP_COLUMNS:
                continue
            if isinstance(vals, list) and len(vals) > 1:
                try:
                    shapes = [np.array(v).shape for v in vals[:5]]
                    if len(set(shapes)) > 1:
                        result.fail(
                            f"Episode {ep.episode_index}, feature '{col}': "
                            f"inconsistent shapes within episode: {set(shapes)}"
                        )
                except (ValueError, TypeError):
                    pass

    return result
