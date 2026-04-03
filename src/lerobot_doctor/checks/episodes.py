"""Check 6: Episode Health -- per-episode quality scoring."""

from __future__ import annotations

import numpy as np

from lerobot_doctor.dataset_loader import LoadedDataset
from lerobot_doctor.runner import CheckResult, Severity

# Common policy chunk sizes (action prediction horizons)
COMMON_CHUNK_SIZES = [10, 16, 20, 50, 100]


def check_episodes(dataset: LoadedDataset) -> CheckResult:
    result = CheckResult(name="Episode Health", severity=Severity.PASS)

    if dataset.info is None:
        result.fail("Cannot check episodes: info.json not loaded")
        return result

    if not dataset.episodes_data:
        result.warn("No episode data loaded")
        return result

    fps = dataset.info.fps or 1
    lengths = [ep.length for ep in dataset.episodes_data]
    lengths_arr = np.array(lengths)

    # Basic length stats
    min_len = int(lengths_arr.min())
    max_len = int(lengths_arr.max())
    mean_len = float(lengths_arr.mean())
    std_len = float(lengths_arr.std())
    median_len = float(np.median(lengths_arr))

    result.pass_(
        f"Episode lengths: min={min_len}, max={max_len}, "
        f"mean={mean_len:.0f}, median={median_len:.0f}, std={std_len:.1f}"
    )
    result.pass_(
        f"Episode durations: min={min_len/fps:.1f}s, max={max_len/fps:.1f}s, "
        f"mean={mean_len/fps:.1f}s"
    )

    # Very short episodes (< 1 second or < 5 frames)
    short_threshold = max(5, fps)  # at least 1 second
    short_eps = [(ep.episode_index, ep.length) for ep in dataset.episodes_data if ep.length < short_threshold]
    if short_eps:
        result.warn(
            f"{len(short_eps)} episode(s) shorter than {short_threshold} frames "
            f"(<{short_threshold/fps:.1f}s): {[e[0] for e in short_eps[:10]]}"
            f"{'...' if len(short_eps) > 10 else ''}"
        )

    # Single-frame episodes (can't compute stats)
    single_frame = [ep.episode_index for ep in dataset.episodes_data if ep.length <= 1]
    if single_frame:
        result.fail(
            f"{len(single_frame)} episode(s) with <=1 frame (can't compute statistics): "
            f"{single_frame[:10]}"
        )

    # Empty episodes
    empty = [ep.episode_index for ep in dataset.episodes_data if ep.length == 0]
    if empty:
        result.fail(f"{len(empty)} empty episode(s): {empty[:10]}")

    # High variance in lengths (>2x std/mean ratio = very inconsistent)
    if mean_len > 0 and std_len / mean_len > 1.0:
        result.warn(
            f"High episode length variance: std/mean ratio = {std_len/mean_len:.2f}. "
            f"This can hurt training stability."
        )

    # Outlier episodes (>3 std from mean)
    if std_len > 0:
        outlier_short = [
            ep.episode_index for ep in dataset.episodes_data
            if ep.length < mean_len - 3 * std_len
        ]
        outlier_long = [
            ep.episode_index for ep in dataset.episodes_data
            if ep.length > mean_len + 3 * std_len
        ]
        if outlier_short:
            result.warn(f"{len(outlier_short)} abnormally short episode(s) (>3 std below mean): {outlier_short[:5]}")
        if outlier_long:
            result.warn(f"{len(outlier_long)} abnormally long episode(s) (>3 std above mean): {outlier_long[:5]}")

    # Policy window compatibility
    for chunk_size in COMMON_CHUNK_SIZES:
        too_short = [ep.episode_index for ep in dataset.episodes_data if ep.length < chunk_size]
        if too_short and len(too_short) > len(dataset.episodes_data) * 0.1:
            result.warn(
                f"{len(too_short)}/{len(dataset.episodes_data)} episodes shorter than "
                f"chunk_size={chunk_size} (used by ACT/Diffusion policies)"
            )
            break  # only warn about the smallest problematic chunk size

    # Episode length consistency with metadata
    if dataset.episodes_meta:
        mismatches = []
        for ep_data in dataset.episodes_data:
            matching_meta = [m for m in dataset.episodes_meta if m.episode_index == ep_data.episode_index]
            if matching_meta:
                meta_len = matching_meta[0].length
                if meta_len != ep_data.length:
                    mismatches.append((ep_data.episode_index, ep_data.length, meta_len))
        if mismatches:
            result.fail(
                f"{len(mismatches)} episode(s) have data/metadata length mismatch: "
                f"{[(idx, f'data={d} meta={m}') for idx, d, m in mismatches[:5]]}"
            )
        else:
            result.pass_("All episode lengths match metadata")

    # Task distribution
    if "task_index" in dataset.episodes_data[0].columns:
        task_counts: dict[int, int] = {}
        for ep in dataset.episodes_data:
            if "task_index" in ep.columns:
                tasks = ep.columns["task_index"]
                unique_tasks = set(tasks.tolist() if hasattr(tasks, 'tolist') else tasks)
                for t in unique_tasks:
                    task_counts[t] = task_counts.get(t, 0) + 1
        if len(task_counts) > 1:
            result.pass_(f"Task distribution across episodes: {dict(sorted(task_counts.items()))}")
            # Check for severe imbalance
            counts = list(task_counts.values())
            if max(counts) > 10 * min(counts):
                result.warn(
                    f"Severe task imbalance: most common task has {max(counts)}x more episodes "
                    f"than least common ({max(counts)} vs {min(counts)})"
                )

    return result
