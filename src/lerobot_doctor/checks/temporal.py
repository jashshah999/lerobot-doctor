"""Check 2: Temporal & Frame Consistency."""

from __future__ import annotations

import numpy as np

from lerobot_doctor.dataset_loader import LoadedDataset
from lerobot_doctor.runner import CheckResult, Severity


def check_temporal(dataset: LoadedDataset) -> CheckResult:
    result = CheckResult(name="Temporal Consistency", severity=Severity.PASS)

    if dataset.info is None:
        result.fail("Cannot check temporal consistency: info.json not loaded")
        return result

    if not dataset.episodes_data:
        result.warn("No episode data loaded, skipping temporal checks")
        return result

    fps = dataset.info.fps
    if fps is None or fps <= 0:
        result.fail(f"Invalid fps={fps}, cannot check temporal consistency")
        return result

    expected_interval = 1.0 / fps
    total_dropped = 0
    total_duplicates = 0
    episodes_with_issues = []

    for ep in dataset.episodes_data:
        ep_issues = []

        # Check timestamp monotonicity and intervals
        if "timestamp" in ep.columns:
            ts = np.array(ep.columns["timestamp"], dtype=np.float64)
            if len(ts) > 1:
                diffs = np.diff(ts)

                # Non-monotonic
                non_mono = np.where(diffs <= 0)[0]
                if len(non_mono) > 0:
                    total_duplicates += len(non_mono)
                    ep_issues.append(
                        f"{len(non_mono)} non-monotonic timestamp(s) at frame indices "
                        f"{non_mono[:5].tolist()}{'...' if len(non_mono) > 5 else ''}"
                    )

                # Dropped frames (gap > 1.5x expected)
                gaps = np.where(diffs > expected_interval * 1.5)[0]
                if len(gaps) > 0:
                    total_dropped += len(gaps)
                    ep_issues.append(
                        f"{len(gaps)} dropped frame gap(s) at frame indices "
                        f"{gaps[:5].tolist()}{'...' if len(gaps) > 5 else ''}"
                    )

                # FPS consistency (intervals within 10% tolerance)
                positive_diffs = diffs[diffs > 0]
                if len(positive_diffs) > 0:
                    mean_interval = np.mean(positive_diffs)
                    if abs(mean_interval - expected_interval) > expected_interval * 0.1:
                        ep_issues.append(
                            f"Mean interval {mean_interval:.4f}s differs from expected "
                            f"{expected_interval:.4f}s (fps={fps})"
                        )

        # Check frame_index sequential
        if "frame_index" in ep.columns:
            fi = np.array(ep.columns["frame_index"], dtype=np.int64)
            expected_fi = np.arange(len(fi))
            if not np.array_equal(fi, expected_fi):
                mismatches = np.where(fi != expected_fi)[0]
                ep_issues.append(
                    f"frame_index not sequential: {len(mismatches)} mismatches, "
                    f"first at position {mismatches[0]} (got {fi[mismatches[0]]}, expected {expected_fi[mismatches[0]]})"
                )

        if ep_issues:
            episodes_with_issues.append((ep.episode_index, ep_issues))

    # Check episode_index contiguity
    ep_indices = [ep.episode_index for ep in dataset.episodes_data]
    if ep_indices:
        expected_ep = list(range(min(ep_indices), max(ep_indices) + 1))
        missing_eps = set(expected_ep) - set(ep_indices)
        if missing_eps:
            result.warn(f"Missing episode indices: {sorted(missing_eps)[:10]}{'...' if len(missing_eps) > 10 else ''}")

    # Check global index sequential
    all_indices = []
    for ep in dataset.episodes_data:
        if "index" in ep.columns:
            all_indices.extend(ep.columns["index"].tolist() if hasattr(ep.columns["index"], 'tolist') else list(ep.columns["index"]))
    if all_indices:
        expected_global = list(range(len(all_indices)))
        if all_indices != expected_global:
            result.warn("Global index column is not sequential (0, 1, 2, ...)")

    # Report
    if not episodes_with_issues and total_dropped == 0 and total_duplicates == 0:
        result.pass_(f"All {len(dataset.episodes_data)} episodes have consistent timestamps and frame indices")
    else:
        for ep_idx, issues in episodes_with_issues[:10]:
            for issue in issues:
                result.warn(f"Episode {ep_idx}: {issue}")
        if len(episodes_with_issues) > 10:
            result.warn(f"...and {len(episodes_with_issues) - 10} more episodes with issues")
        if total_dropped > 0:
            result.warn(f"Total dropped frame gaps across all episodes: {total_dropped}")
        if total_duplicates > 0:
            result.warn(f"Total non-monotonic timestamps across all episodes: {total_duplicates}")

    return result
