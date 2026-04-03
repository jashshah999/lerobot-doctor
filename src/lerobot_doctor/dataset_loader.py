"""Load LeRobot v3 datasets from local paths or HuggingFace Hub."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


@dataclass
class DatasetInfo:
    """Parsed info.json contents."""
    raw: dict
    root: Path
    codebase_version: str | None = None
    fps: int | None = None
    total_episodes: int | None = None
    total_frames: int | None = None
    total_tasks: int | None = None
    chunks_size: int | None = None
    features: dict = field(default_factory=dict)
    data_path: str | None = None
    video_path: str | None = None
    splits: dict = field(default_factory=dict)
    robot_type: str | None = None


@dataclass
class EpisodeData:
    """Data for a single episode loaded from parquet."""
    episode_index: int
    columns: dict[str, np.ndarray]  # column_name -> values
    length: int


@dataclass
class EpisodeMeta:
    """Metadata row for a single episode from episodes parquet."""
    episode_index: int
    length: int
    raw: dict  # all columns as dict


@dataclass
class LoadedDataset:
    """Everything needed for running checks."""
    root: Path
    info: DatasetInfo | None = None
    info_error: str | None = None
    episodes_meta: list[EpisodeMeta] = field(default_factory=list)
    episodes_data: list[EpisodeData] = field(default_factory=list)
    tasks: list[dict] | None = None
    is_local: bool = True


def load_info(root: Path) -> DatasetInfo | str:
    """Load and parse info.json. Returns DatasetInfo or error string."""
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return f"info.json not found at {info_path}"
    try:
        raw = json.loads(info_path.read_text())
    except json.JSONDecodeError as e:
        return f"info.json is not valid JSON: {e}"

    return DatasetInfo(
        raw=raw,
        root=root,
        codebase_version=raw.get("codebase_version"),
        fps=raw.get("fps"),
        total_episodes=raw.get("total_episodes"),
        total_frames=raw.get("total_frames"),
        total_tasks=raw.get("total_tasks"),
        chunks_size=raw.get("chunks_size", 1000),
        features=raw.get("features", {}),
        data_path=raw.get("data_path"),
        video_path=raw.get("video_path"),
        splits=raw.get("splits", {}),
        robot_type=raw.get("robot_type"),
    )


def load_episodes_meta(root: Path, info: DatasetInfo) -> list[EpisodeMeta]:
    """Load episode metadata from meta/episodes/ parquet files."""
    episodes_dir = root / "meta" / "episodes"
    if not episodes_dir.exists():
        return []
    parquet_files = sorted(episodes_dir.rglob("*.parquet"))
    metas = []
    for pf in parquet_files:
        table = pq.read_table(pf)
        for i in range(len(table)):
            row = {col: table.column(col)[i].as_py() for col in table.column_names}
            metas.append(EpisodeMeta(
                episode_index=row.get("episode_index", -1),
                length=row.get("length", 0),
                raw=row,
            ))
    metas.sort(key=lambda m: m.episode_index)
    return metas


def load_episode_data(root: Path, info: DatasetInfo, max_episodes: int | None = None) -> list[EpisodeData]:
    """Load episode data from data/ parquet files."""
    data_dir = root / "data"
    if not data_dir.exists():
        return []
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        return []

    # Read all parquet files into one table
    tables = [pq.read_table(pf) for pf in parquet_files]
    import pyarrow as pa
    try:
        full_table = pa.concat_tables(tables)
    except pa.lib.ArrowInvalid:
        # Schema mismatch between files -- promote to union schema
        full_table = pa.concat_tables(tables, promote_options="permissive")

    # Group by episode_index
    ep_col = full_table.column("episode_index").to_pylist()
    episodes = {}
    for i, ep_idx in enumerate(ep_col):
        if ep_idx not in episodes:
            episodes[ep_idx] = []
        episodes[ep_idx].append(i)

    sorted_eps = sorted(episodes.keys())
    if max_episodes is not None:
        sorted_eps = sorted_eps[:max_episodes]

    result = []
    for ep_idx in sorted_eps:
        row_indices = episodes[ep_idx]
        columns = {}
        for col_name in full_table.column_names:
            col = full_table.column(col_name)
            values = [col[i].as_py() for i in row_indices]
            try:
                columns[col_name] = np.array(values)
            except (ValueError, TypeError):
                # Some columns (like lists of different lengths) can't be numpy arrays
                columns[col_name] = values
        result.append(EpisodeData(
            episode_index=ep_idx,
            columns=columns,
            length=len(row_indices),
        ))
    return result


def load_tasks(root: Path) -> list[dict] | None:
    """Load tasks.parquet if it exists."""
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return None
    table = pq.read_table(tasks_path)
    return [
        {col: table.column(col)[i].as_py() for col in table.column_names}
        for i in range(len(table))
    ]


def load_from_hf(repo_id: str, cache_dir: Path | None = None, max_episodes: int | None = None) -> LoadedDataset:
    """Download dataset files from HuggingFace Hub and load them."""
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=cache_dir,
        allow_patterns=["meta/**", "data/**"],
        # Skip videos for now -- too large. Video check will download on demand.
    )
    ds = load_local(Path(local_dir), max_episodes=max_episodes)
    ds.is_local = False
    return ds


def load_local(root: Path, max_episodes: int | None = None) -> LoadedDataset:
    """Load a local dataset."""
    ds = LoadedDataset(root=root, is_local=True)

    info_result = load_info(root)
    if isinstance(info_result, str):
        ds.info_error = info_result
        return ds
    ds.info = info_result

    ds.episodes_meta = load_episodes_meta(root, ds.info)
    ds.episodes_data = load_episode_data(root, ds.info, max_episodes=max_episodes)
    ds.tasks = load_tasks(root)
    return ds


def load_dataset(path_or_repo: str, max_episodes: int | None = None) -> LoadedDataset:
    """Load dataset from local path or HF repo_id."""
    local = Path(path_or_repo)
    if local.exists() and local.is_dir():
        return load_local(local, max_episodes=max_episodes)
    # Treat as HF repo_id
    return load_from_hf(path_or_repo, max_episodes=max_episodes)
