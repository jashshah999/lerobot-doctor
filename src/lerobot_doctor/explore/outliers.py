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


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _find_video_path(root: Path, episode_index: int) -> Path | None:
    """Locate the video file for a given episode index.

    LeRobot v2 layout: videos/chunk-{ep//1000:03d}/{video_key}/episode_{ep:06d}.mp4
    """
    info_path = root / "meta" / "info.json"
    video_key = "observation.images.front"
    chunks_size = 1000

    if info_path.exists():
        import json
        try:
            info = json.loads(info_path.read_text())
            chunks_size = info.get("chunks_size", 1000)
            # Try to extract video_key from video_path template
            vp = info.get("video_path", "")
            # e.g. "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            if "{video_key}" in vp:
                # We don't know the exact key, search directories
                pass
        except Exception:
            pass

    chunk = episode_index // chunks_size
    # Search for any video file matching this episode
    videos_dir = root / "videos"
    if not videos_dir.exists():
        return None

    # Walk: videos/chunk-XXX/<any_key>/episode_XXXXXX.mp4
    candidates = sorted(videos_dir.rglob(f"episode_{episode_index:06d}.mp4"))
    if candidates:
        return candidates[0]
    return None


def _read_video_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    """Read a single frame from an mp4 video using OpenCV."""
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python-headless is required for visualization: pip install opencv-python-headless")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None
    return frame


def _load_episode_actions(root: Path, episode_index: int, feature_name: str) -> np.ndarray | None:
    """Load the feature values for a specific episode, indexed by frame_index."""
    info_path = root / "meta" / "info.json"
    chunks_size = 1000
    if info_path.exists():
        import json
        try:
            info = json.loads(info_path.read_text())
            chunks_size = info.get("chunks_size", 1000)
        except Exception:
            pass

    chunk = episode_index // chunks_size
    parquet_path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        return None

    try:
        table = pq.read_table(str(parquet_path), columns=[feature_name, "frame_index"])
    except Exception:
        return None

    frames = table.column("frame_index").to_pylist()
    values = table.column(feature_name).to_pylist()

    # Build a dict: frame_index -> feature_value
    # feature_value could be a list (multi-dim) or scalar
    action_map: dict[int, np.ndarray] = {}
    for f, v in zip(frames, values):
        arr = np.array(v, dtype=np.float64)
        action_map[int(f)] = arr

    return action_map


def visualize_outliers(
    root: Path,
    result: OutlierResult,
    output_dir: Path,
    context_frames: int = 2,
    top: int | None = None,
) -> list[Path]:
    """Generate visualization images for each unique outlier frame.

    For each unique (episode_index, frame_index) that has at least one outlier,
    extracts frames [frame-context_frames, ..., frame+context_frames] from the
    video, annotates each with frame_index and the outlier dimension values,
    and saves a horizontal strip image.

    Args:
        root: Dataset root directory
        result: OutlierResult from find_outliers()
        output_dir: Where to save the visualization PNGs
        context_frames: Number of frames before/after the outlier frame (default 2)
        top: Limit number of outlier frames to visualize (by max |z| in frame)

    Returns:
        List of saved image paths
    """
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)

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

    saved_paths: list[Path] = []

    for ep_idx, frame_idx in sorted_frames:
        outliers_here = frame_groups[(ep_idx, frame_idx)]

        # Load action values for this episode
        action_map = _load_episode_actions(root, ep_idx, result.feature_name)

        # Find video
        video_path = _find_video_path(root, ep_idx)
        if video_path is None:
            print(f"  [SKIP] No video found for episode {ep_idx}")
            continue

        # Extract context frames
        frame_indices = list(range(frame_idx - context_frames, frame_idx + context_frames + 1))
        n_frames = len(frame_indices)
        is_center = [i == context_frames for i in range(n_frames)]

        images: list[np.ndarray] = []
        labels: list[str] = []

        for fi, target_frame in enumerate(frame_indices):
            frame = _read_video_frame(video_path, target_frame)
            if frame is None:
                # Black placeholder if frame can't be read
                frame = np.zeros((360, 640, 3), dtype=np.uint8)

            # Build label: frame_index + outlier dim values
            label_parts = [f"frame {target_frame}"]
            if action_map is not None and target_frame in action_map:
                act_vals = action_map[target_frame]
                # Show values for dims that are outliers at the CENTER frame
                if fi == context_frames:
                    # Center: show all outlier dims with their values
                    dims_str = ", ".join(
                        f"d{o.dim}={act_vals[o.dim]:.3f}(|z|={o.z_score:.1f})"
                        for o in outliers_here
                    )
                else:
                    # Context: show the same dims but just values
                    dims_str = ", ".join(
                        f"d{o.dim}={act_vals[o.dim]:.3f}"
                        for o in outliers_here
                    )
                label_parts.append(dims_str)

            label = "\n".join(label_parts)
            labels.append(label)

            # Annotate frame
            annotated = _annotate_frame(frame, label, is_center[fi], outliers_here if fi == context_frames else None)
            images.append(annotated)

        # Ensure all frames have the same height before hstack
        max_h = max(img.shape[0] for img in images)
        for i in range(len(images)):
            if images[i].shape[0] != max_h:
                diff = max_h - images[i].shape[0]
                images[i] = cv2.copyMakeBorder(images[i], 0, diff, 0, 0,
                                               cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Stitch horizontally
        strip = np.hstack(images)

        # Top banner
        banner_text = f"Episode {ep_idx} | Frame {frame_idx} (outlier center) | {len(outliers_here)} outlier dim(s)"
        strip = _add_top_banner(strip, banner_text)

        out_path = output_dir / f"ep{ep_idx:06d}_frame{frame_idx:06d}_outlier.png"
        cv2.imwrite(str(out_path), strip)
        saved_paths.append(out_path)
        print(f"  [SAVE] {out_path}")

    return saved_paths


def _annotate_frame(
    frame: np.ndarray,
    label: str,
    is_center: bool,
    center_outliers: list[OutlierRecord] | None = None,
) -> np.ndarray:
    """Add annotation text and border to a single video frame."""
    import cv2

    h, w = frame.shape[:2]

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

    lines = label.split("\n")
    for li, line in enumerate(lines):
        color = (0, 255, 255) if is_center else (255, 255, 255)  # yellow for center
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
