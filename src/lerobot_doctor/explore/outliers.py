"""Explore subcommand -- data diagnostics beyond pass/fail checks.

Usage:
    lerobot-doctor explore outliers <dataset> --features action [options]
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lerobot_doctor.dataset_loader import LoadedDataset


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
    n_non_finite: int = 0       # NaN/inf values, excluded from mean/std and z-scores

    def group_by_dim(self) -> dict[int, list[OutlierRecord]]:
        groups: dict[int, list[OutlierRecord]] = {}
        for o in self.outliers:
            groups.setdefault(o.dim, []).append(o)
        return groups

    def as_dict(self, top: int | None = None) -> dict:
        shown = self.outliers if top is None else self.outliers[:top]
        return {
            "feature": self.feature_name,
            "n_dims": self.n_dims,
            "total_frames": self.total_frames,
            "threshold": self.threshold,
            "non_finite_values": self.n_non_finite,
            "mean_per_dim": [round(m, 6) for m in self.mean_per_dim],
            "std_per_dim": [round(s, 6) for s in self.std_per_dim],
            "total_outliers": len(self.outliers),
            "outliers": [o.as_dict() for o in shown],
        }


def numeric_features(dataset: LoadedDataset) -> list[str]:
    """Names of numeric columns present in the loaded episode data."""
    if not dataset.episodes_data:
        return []
    names = []
    for name, vals in dataset.episodes_data[0].columns.items():
        if isinstance(vals, np.ndarray) and vals.dtype.kind in "biuf":
            names.append(name)
    return sorted(names)


def _load_feature_with_location(
    dataset: LoadedDataset,
    feature_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect a feature column alongside its episode_index and frame_index.

    Returns:
        feature_vals: (N, D) array of feature values
        ep_indices: (N,) int array of episode_index
        frame_indices: (N,) int array of frame_index
    """
    all_vals: list[np.ndarray] = []
    all_ep: list[np.ndarray] = []
    all_frame: list[np.ndarray] = []

    for ep in dataset.episodes_data:
        if feature_name not in ep.columns:
            continue
        try:
            vals = np.asarray(ep.columns[feature_name], dtype=np.float64)
        except (ValueError, TypeError):
            raise ValueError(f"Feature '{feature_name}' is not numeric")
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        elif vals.ndim != 2:
            raise ValueError(f"Feature '{feature_name}' has shape {vals.shape[1:]}; only scalar or vector features are supported")
        frames = ep.columns.get("frame_index")
        frames = np.asarray(frames, dtype=np.int64) if frames is not None else np.arange(len(vals))
        all_vals.append(vals)
        all_ep.append(np.full(len(vals), ep.episode_index, dtype=np.int64))
        all_frame.append(frames)

    if not all_vals:
        available = ", ".join(numeric_features(dataset)) or "none"
        raise ValueError(f"Feature '{feature_name}' not found in dataset. Numeric features: {available}")

    return np.concatenate(all_vals), np.concatenate(all_ep), np.concatenate(all_frame)


def find_outliers(
    dataset: LoadedDataset,
    feature: str = "action",
    threshold: float = 10.0,
) -> OutlierResult:
    """Find extreme outliers in a feature and locate each in episode/frame.

    Mean and std are computed per dimension over all loaded frames, ignoring
    NaN/inf values (those are reported by ``check`` and counted here).

    Args:
        dataset: Loaded dataset (see ``dataset_loader.load_dataset``)
        feature: Column name in the parquet (e.g. 'action', 'observation.state')
        threshold: z-score threshold (default 10, matching statistics.py)

    Returns:
        OutlierResult with per-dim stats and located outliers
    """
    vals, ep_idx, frame_idx = _load_feature_with_location(dataset, feature)

    n_frames, n_dims = vals.shape
    finite = np.isfinite(vals)
    masked = np.where(finite, vals, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN dims yield NaN stats
        mean_per_dim = np.nanmean(masked, axis=0)
        std_per_dim = np.nanstd(masked, axis=0)

    outliers: list[OutlierRecord] = []

    for d in range(n_dims):
        std_d = std_per_dim[d]
        mean_d = mean_per_dim[d]
        if not np.isfinite(std_d) or std_d == 0:
            continue

        with np.errstate(invalid="ignore"):
            z_scores = np.abs((masked[:, d] - mean_d) / std_d)
        for pos in np.where(z_scores > threshold)[0]:
            outliers.append(OutlierRecord(
                dim=d,
                episode_index=int(ep_idx[pos]),
                frame_index=int(frame_idx[pos]),
                value=float(vals[pos, d]),
                mean=float(mean_d),
                std=float(std_d),
                z_score=float(z_scores[pos]),
            ))

    outliers.sort(key=lambda o: (-o.z_score, o.dim, o.episode_index, o.frame_index))

    return OutlierResult(
        feature_name=feature,
        n_dims=n_dims,
        total_frames=n_frames,
        mean_per_dim=mean_per_dim.tolist(),
        std_per_dim=std_per_dim.tolist(),
        outliers=outliers,
        threshold=threshold,
        n_non_finite=int((~finite).sum()),
    )


def format_outlier_report(result: OutlierResult, top: int | None = None) -> str:
    """Format OutlierResult as readable text."""
    lines = []
    lines.append(f"Outlier Exploration: {result.feature_name}")
    lines.append(f"  Frames scanned: {result.total_frames:,}")
    lines.append(f"  Dimensions:     {result.n_dims}")
    lines.append(f"  Threshold:      |z| > {result.threshold}")
    lines.append(f"  Total outliers: {len(result.outliers)}")
    if result.n_non_finite:
        lines.append(f"  Non-finite:     {result.n_non_finite} NaN/inf value(s) ignored")
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


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def video_features(dataset: LoadedDataset) -> list[str]:
    """Names of video features declared in info.json."""
    if dataset.info is None:
        return []
    return [name for name, spec in dataset.info.features.items() if spec.get("dtype") == "video"]


def _video_location(dataset: LoadedDataset, video_key: str, episode_index: int) -> tuple[Path, float] | None:
    """Resolve the video file holding an episode and the episode's start time in it.

    v2 stores one video per episode (start 0). v3 concatenates episodes into
    shared files; episode metadata gives the file and ``from_timestamp``.
    """
    info = dataset.info
    if info is None or not info.video_path:
        return None
    meta = next((m for m in dataset.episodes_meta if m.episode_index == episode_index), None)
    raw = meta.raw if meta is not None else {}
    chunks_size = info.chunks_size or 1000
    chunk_idx = raw.get(f"videos/{video_key}/chunk_index", episode_index // chunks_size)
    file_idx = raw.get(f"videos/{video_key}/file_index", episode_index)
    try:
        relpath = info.video_path.format(
            video_key=video_key,
            episode_chunk=chunk_idx,
            episode_index=file_idx,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
    except (KeyError, IndexError):
        return None
    path = dataset.root / relpath
    if not path.exists():
        return None
    return path, float(raw.get(f"videos/{video_key}/from_timestamp") or 0.0)


def _read_video_frames(video_path: Path, timestamps: list[float]) -> list[np.ndarray | None]:
    """Decode the frames shown at the given timestamps (seconds) as BGR arrays.

    Uses PyAV, which decodes AV1 (LeRobot's default codec); OpenCV wheels
    usually cannot.
    """
    import av

    out: list[np.ndarray | None] = [None] * len(timestamps)
    order = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        tb = float(stream.time_base)
        first = max(timestamps[order[0]], 0.0)
        container.seek(int(max(first - 1.0, 0.0) / tb), stream=stream, backward=True)
        k = 0
        prev = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * tb)
            # A target is resolved once a decoded frame reaches it; use whichever
            # of the previous/current frame is nearer.
            while k < len(order) and t >= timestamps[order[k]]:
                target = timestamps[order[k]]
                best = prev if prev is not None and target - prev[0] < t - target else (t, frame)
                out[order[k]] = best[1].to_ndarray(format="bgr24")
                k += 1
            if k >= len(order):
                break
            prev = (t, frame)
        if k < len(order) and prev is not None:
            # Targets past the last decoded frame: use the last frame.
            for i in order[k:]:
                out[i] = prev[1].to_ndarray(format="bgr24")
    return out


def visualize_outliers(
    dataset: LoadedDataset,
    result: OutlierResult,
    output_dir: Path,
    video_key: str | None = None,
    context_frames: int = 2,
    top: int | None = None,
) -> list[Path]:
    """Generate visualization images for each unique outlier frame.

    For each unique (episode_index, frame_index) that has at least one outlier,
    extracts frames [frame-context_frames, ..., frame+context_frames] from the
    video, annotates each with frame_index and the outlier dimension values,
    and saves a horizontal strip image.

    Args:
        dataset: Loaded local dataset with videos
        result: OutlierResult from find_outliers()
        output_dir: Where to save the visualization PNGs
        video_key: Camera to show (default: first video feature)
        context_frames: Number of frames before/after the outlier frame (default 2)
        top: Limit number of outlier frames to visualize (by max |z| in frame)

    Returns:
        List of saved image paths
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python-headless is required for visualization: pip install 'lerobot-doctor[viz]'")

    keys = video_features(dataset)
    if not keys:
        raise ValueError("Dataset declares no video features; nothing to visualize")
    if video_key is None:
        video_key = keys[0]
    elif video_key not in keys:
        raise ValueError(f"Unknown camera '{video_key}'. Video features: {', '.join(keys)}")

    fps = dataset.info.fps if dataset.info and dataset.info.fps else None
    episodes = {ep.episode_index: ep for ep in dataset.episodes_data}

    # Group outliers by (episode_index, frame_index)
    frame_groups: dict[tuple[int, int], list[OutlierRecord]] = {}
    for o in result.outliers:
        key = (o.episode_index, o.frame_index)
        frame_groups.setdefault(key, []).append(o)

    # Sort by max |z| in each frame
    sorted_frames = sorted(
        frame_groups.keys(),
        key=lambda k: max(o.z_score for o in frame_groups[k]),
        reverse=True,
    )
    if top is not None:
        sorted_frames = sorted_frames[:top]

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for ep_idx, frame_idx in sorted_frames:
        outliers_here = frame_groups[(ep_idx, frame_idx)]
        ep = episodes[ep_idx]

        location = _video_location(dataset, video_key, ep_idx)
        if location is None:
            print(f"  [SKIP] No {video_key} video found for episode {ep_idx}", file=sys.stderr)
            continue
        video_path, ep_start = location

        # Map frame_index -> row, and row -> feature value / timestamp
        frame_col = ep.columns.get("frame_index")
        rows = {int(f): i for i, f in enumerate(frame_col)} if frame_col is not None else {i: i for i in range(ep.length)}
        values = np.asarray(ep.columns[result.feature_name], dtype=np.float64).reshape(ep.length, -1)
        ts_col = ep.columns.get("timestamp")

        def row_time(row: int) -> float | None:
            if ts_col is not None:
                return ep_start + float(np.asarray(ts_col[row]).reshape(-1)[0])
            if fps:
                return ep_start + row / fps
            return None

        frame_indices = list(range(frame_idx - context_frames, frame_idx + context_frames + 1))
        times = [row_time(rows[f]) if f in rows else None for f in frame_indices]
        wanted = [t for t in times if t is not None]
        decoded = iter(_read_video_frames(video_path, wanted)) if wanted else iter(())
        frames = [next(decoded) if t is not None else None for t in times]

        ref = next((f for f in frames if f is not None), None)
        ph, pw = (ref.shape[:2] if ref is not None else (360, 640))
        scale = max(1, int(np.ceil(240 / ph)))  # upscale tiny frames so labels stay readable

        images: list[np.ndarray] = []
        for fi, (target_frame, frame) in enumerate(zip(frame_indices, frames)):
            is_center = fi == context_frames
            if frame is None:
                frame = np.zeros((ph, pw, 3), dtype=np.uint8)
            if scale > 1:
                frame = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale),
                                   interpolation=cv2.INTER_NEAREST)

            label_parts = [f"frame {target_frame}"]
            if target_frame in rows:
                act_vals = values[rows[target_frame]]
                if is_center:
                    dims_str = ", ".join(
                        f"d{o.dim}={act_vals[o.dim]:.3f}(|z|={o.z_score:.1f})" for o in outliers_here
                    )
                else:
                    dims_str = ", ".join(f"d{o.dim}={act_vals[o.dim]:.3f}" for o in outliers_here)
                label_parts.append(dims_str)
            else:
                label_parts.append("(outside episode)")

            images.append(_annotate_frame(frame, "\n".join(label_parts), is_center))

        # Ensure all frames have the same height before hstack
        max_h = max(img.shape[0] for img in images)
        for i in range(len(images)):
            if images[i].shape[0] != max_h:
                diff = max_h - images[i].shape[0]
                images[i] = cv2.copyMakeBorder(images[i], 0, diff, 0, 0,
                                               cv2.BORDER_CONSTANT, value=(0, 0, 0))

        strip = np.hstack(images)

        banner_text = (f"Episode {ep_idx} | Frame {frame_idx} (outlier center) | "
                       f"{len(outliers_here)} outlier dim(s) | {video_key}")
        strip = _add_top_banner(strip, banner_text)

        out_path = output_dir / f"ep{ep_idx:06d}_frame{frame_idx:06d}_outlier.png"
        cv2.imwrite(str(out_path), strip)
        saved_paths.append(out_path)
        print(f"  [SAVE] {out_path}", file=sys.stderr)

    return saved_paths


def _annotate_frame(frame: np.ndarray, label: str, is_center: bool) -> np.ndarray:
    """Add annotation text and border to a single video frame."""
    import cv2

    # Add border: thick red for center, thin gray for context
    if is_center:
        border_color = (0, 0, 255)  # BGR red
        border_px = 6
    else:
        border_color = (128, 128, 128)
        border_px = 2

    frame = cv2.copyMakeBorder(
        frame, border_px, border_px, border_px, border_px,
        cv2.BORDER_CONSTANT, value=border_color,
    )

    # Add semi-transparent overlay at bottom for label
    overlay_h = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, frame.shape[0] - overlay_h), (frame.shape[1], frame.shape[0]),
                  (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Draw label text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1 if not is_center else 2
    y_start = frame.shape[0] - overlay_h + 20

    color = (0, 255, 255) if is_center else (255, 255, 255)  # yellow for center
    for li, line in enumerate(label.split("\n")):
        cv2.putText(frame, line, (10, y_start + li * 22),
                    font, font_scale, color, thickness, cv2.LINE_AA)

    return frame


def _add_top_banner(image: np.ndarray, text: str) -> np.ndarray:
    """Add a top banner strip to a horizontal image."""
    import cv2

    banner_h = 50
    w = image.shape[1]
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    # Dark blue banner
    banner[:] = (30, 30, 80)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    cv2.putText(banner, text, (15, 32), font, font_scale, (200, 220, 255), 2, cv2.LINE_AA)

    return np.vstack([banner, image])
