"""Explore subcommand -- data diagnostics beyond pass/fail checks.

Usage:
    lerobot-doctor explore outliers <dataset> --actions [options]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


@dataclass
class OutlierRecord:
    """A single outlier point."""
    dim: int                    # dimension index in the feature vector
    episode_index: int
    frame_index: int
    value: float
    mean: float
    std: float
    z_score: float

    def as_dict(self) -> dict:
        return {
            "dim": self.dim,
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "value": round(self.value, 6),
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "z_score": round(self.z_score, 3),
        }


@dataclass
class OutlierResult:
    """Summary of outlier exploration."""
    feature_name: str
    n_dims: int
    total_frames: int
    mean_per_dim: list[float]
    std_per_dim: list[float]
    outliers: list[OutlierRecord] = field(default_factory=list)
    threshold: float = 10.0

    def group_by_dim(self) -> dict[int, list[OutlierRecord]]:
        groups: dict[int, list[OutlierRecord]] = {}
        for o in self.outliers:
            groups.setdefault(o.dim, []).append(o)
        return groups


def _load_feature_with_location(
    root: Path,
    feature_name: str,
    max_episodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a feature column alongside its episode_index and frame_index.

    Returns:
        feature_vals: (N, D) array of feature values
        ep_indices: (N,) int array of episode_index
        frame_indices: (N,) int array of frame_index
    """
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"No data/ directory under {root}")

    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")

    all_vals: list[np.ndarray] = []
    all_ep: list[np.ndarray] = []
    all_frame: list[np.ndarray] = []
    seen_eps: set[int] = set()

    for pf in parquet_files:
        table = pq.read_table(pf, columns=[feature_name, "episode_index", "frame_index"])

        vals_col = table.column(feature_name).to_pylist()
        ep_col = table.column("episode_index").to_pylist()
        frame_col = table.column("frame_index").to_pylist()

        vals_arr = np.array(vals_col, dtype=np.float64)
        ep_arr = np.array(ep_col, dtype=np.int64)
        frame_arr = np.array(frame_col, dtype=np.int64)

        if max_episodes is not None:
            new_eps = set(ep_arr.tolist()) - seen_eps
            if not new_eps and len(seen_eps) >= max_episodes:
                break
            seen_eps.update(new_eps)
            if len(seen_eps) > max_episodes:
                # Trim to only the first max_episodes episodes encountered
                keep_mask = np.isin(ep_arr, sorted(seen_eps)[:max_episodes])
                vals_arr = vals_arr[keep_mask]
                ep_arr = ep_arr[keep_mask]
                frame_arr = frame_arr[keep_mask]

        all_vals.append(vals_arr)
        all_ep.append(ep_arr)
        all_frame.append(frame_arr)

    if not all_vals:
        raise ValueError(f"No data loaded for feature '{feature_name}'")

    feature_vals = np.concatenate(all_vals, axis=0)
    ep_indices = np.concatenate(all_ep, axis=0)
    frame_indices = np.concatenate(all_frame, axis=0)

    # Ensure 2D: (N, D)
    if feature_vals.ndim == 1:
        feature_vals = feature_vals.reshape(-1, 1)

    return feature_vals, ep_indices, frame_indices


def find_outliers(
    root: Path,
    feature: str = "action",
    threshold: float = 10.0,
    max_episodes: int | None = None,
) -> OutlierResult:
    """Find extreme outliers in a feature and locate each in episode/frame.

    Args:
        root: Dataset root directory
        feature: Column name in the parquet (e.g. 'action', 'observation.state')
        threshold: z-score threshold (default 10, matching statistics.py)
        max_episodes: Limit number of episodes to scan

    Returns:
        OutlierResult with per-dim stats and located outliers
    """
    feature_name = feature

    vals, ep_idx, frame_idx = _load_feature_with_location(root, feature_name, max_episodes)

    n_frames, n_dims = vals.shape
    mean_per_dim = np.mean(vals, axis=0)
    std_per_dim = np.std(vals, axis=0)

    outliers: list[OutlierRecord] = []

    for d in range(n_dims):
        dim_vals = vals[:, d]
        std_d = std_per_dim[d]
        mean_d = mean_per_dim[d]

        if std_d == 0:
            continue

        z_scores = np.abs((dim_vals - mean_d) / std_d)
        outlier_mask = z_scores > threshold

        outlier_positions = np.where(outlier_mask)[0]
        for pos in outlier_positions:
            outliers.append(OutlierRecord(
                dim=d,
                episode_index=int(ep_idx[pos]),
                frame_index=int(frame_idx[pos]),
                value=float(dim_vals[pos]),
                mean=float(mean_d),
                std=float(std_d),
                z_score=float(z_scores[pos]),
            ))

    outliers.sort(key=lambda o: (-o.z_score, o.dim, o.episode_index, o.frame_index))

    return OutlierResult(
        feature_name=feature_name,
        n_dims=n_dims,
        total_frames=n_frames,
        mean_per_dim=mean_per_dim.tolist(),
        std_per_dim=std_per_dim.tolist(),
        outliers=outliers,
        threshold=threshold,
    )


def format_outlier_report(result: OutlierResult, top: int | None = None) -> str:
    """Format OutlierResult as readable text."""
    lines = []
    lines.append(f"Outlier Exploration: {result.feature_name}")
    lines.append(f"  Frames scanned: {result.total_frames:,}")
    lines.append(f"  Dimensions:     {result.n_dims}")
    lines.append(f"  Threshold:      |z| > {result.threshold}")
    lines.append(f"  Total outliers: {len(result.outliers)}")
    lines.append("")

    groups = result.group_by_dim()
    for dim in sorted(groups.keys()):
        dim_outliers = groups[dim]
        lines.append(f"  dim[{dim}]: {len(dim_outliers)} outlier(s)  "
                     f"(mean={result.mean_per_dim[dim]:.4f}, "
                     f"std={result.std_per_dim[dim]:.4f})")

    if not result.outliers:
        lines.append("\n  No outliers found.")
        return "\n".join(lines)

    lines.append("")
    lines.append("-" * 80)
    lines.append(f"{'dim':>4}  {'episode':>7}  {'frame':>7}  {'value':>12}  {'mean':>10}  {'std':>10}  {'|z|':>8}")
    lines.append("-" * 80)

    display = result.outliers if top is None else result.outliers[:top]
    for o in display:
        lines.append(
            f"{o.dim:>4}  {o.episode_index:>7}  {o.frame_index:>7}  "
            f"{o.value:>12.6f}  {o.mean:>10.6f}  {o.std:>10.6f}  {o.z_score:>8.3f}"
        )

    if top is not None and len(result.outliers) > top:
        lines.append(f"\n... and {len(result.outliers) - top} more (showing top {top} by |z|)")

    return "\n".join(lines)
