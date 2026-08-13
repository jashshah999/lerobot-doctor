"""Load LeRobot v2/v3 datasets from local paths, archives, or HuggingFace Hub."""

from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pyarrow.parquet as pq


@dataclass
class DatasetInfo:
    """Parsed info.json contents."""
    raw: dict
    root: Path
    codebase_version: str | None = None
    format_version: str | None = None  # "v2", "v3", or None when unknown
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
    display_path: str | None = None
    info: DatasetInfo | None = None
    info_error: str | None = None
    episodes_meta: list[EpisodeMeta] = field(default_factory=list)
    episodes_data: list[EpisodeData] = field(default_factory=list)
    tasks: list[dict] | None = None
    is_local: bool = True
    max_episodes_applied: int | None = None  # set when user passed --max-episodes
    robot_urdf: Path | None = None  # set when user passed --urdf, for check_kinematics
    archive_path: Path | None = None
    archive_inner_root: str | None = None
    _temp_dir: Any | None = field(default=None, repr=False, compare=False)


def detect_format_version(root: Path, raw_info: dict | None = None) -> str | None:
    """Detect LeRobot dataset major format version from info.json or layout."""
    codebase_version = (raw_info or {}).get("codebase_version")
    if isinstance(codebase_version, str):
        if codebase_version.startswith("v3"):
            return "v3"
        if codebase_version.startswith("v2"):
            return "v2"

    # Layout fallback:
    # - v3 stores episode metadata as parquet shards under meta/episodes/.
    # - v2.1 stores per-episode parquet files and meta/episodes.jsonl.
    if (root / "meta" / "episodes").is_dir():
        return "v3"
    if (root / "meta" / "episodes.jsonl").exists():
        return "v2"
    if list((root / "data").rglob("file-*.parquet")):
        return "v3"
    if list((root / "data").rglob("episode_*.parquet")):
        return "v2"
    return None


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
        format_version=detect_format_version(root, raw),
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


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSON Lines file."""
    rows: list[dict] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} line {line_no} is not valid JSON: {e}") from e
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def load_episodes_meta(root: Path, info: DatasetInfo) -> list[EpisodeMeta]:
    """Load episode metadata for both v3 parquet and v2 JSONL layouts."""
    episodes_dir = root / "meta" / "episodes"
    parquet_files = sorted(episodes_dir.rglob("*.parquet")) if episodes_dir.exists() else []
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
    if metas:
        return metas

    episodes_jsonl = root / "meta" / "episodes.jsonl"
    if episodes_jsonl.exists():
        rows = _load_jsonl(episodes_jsonl)
        for fallback_idx, row in enumerate(rows):
            ep_idx = row.get("episode_index", fallback_idx)
            length = row.get("length", row.get("num_frames", 0))
            metas.append(EpisodeMeta(
                episode_index=int(ep_idx) if ep_idx is not None else fallback_idx,
                length=int(length) if length is not None else 0,
                raw=row,
            ))
        metas.sort(key=lambda m: m.episode_index)
    return metas


def load_episode_data(root: Path, info: DatasetInfo, max_episodes: int | None = None) -> list[EpisodeData]:
    """Load episode data from data/ parquet files.

    Streams files in sorted order and stops early once max_episodes unique
    episodes are collected. Avoids loading the entire dataset into memory.
    """
    data_dir = root / "data"
    if not data_dir.exists():
        return []
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        return []

    import pyarrow as pa

    # Stream files; accumulate episodes. Stop once max_episodes is reached.
    episode_rows: dict[int, list[tuple[pa.Table, int]]] = {}
    seen_order: list[int] = []

    for pf in parquet_files:
        if max_episodes is not None and len(seen_order) >= max_episodes:
            # Already have enough episodes; but might still get rows for
            # already-seen episodes split across files. Keep scanning only
            # if any seen episode's last row could continue.
            # Safer: stop scanning once we have max_episodes distinct AND
            # the current file's first episode isn't one we're still collecting.
            # Simpler: stop scanning -- v3 episodes are contiguous within a file.
            break

        table = pq.read_table(pf)
        if "episode_index" not in table.column_names:
            continue
        ep_col = table.column("episode_index").to_pylist()
        for row_i, ep_idx in enumerate(ep_col):
            if ep_idx not in episode_rows:
                if max_episodes is not None and len(seen_order) >= max_episodes:
                    continue
                episode_rows[ep_idx] = []
                seen_order.append(ep_idx)
            episode_rows[ep_idx].append((table, row_i))

    sorted_eps = sorted(episode_rows.keys())
    if max_episodes is not None:
        sorted_eps = sorted_eps[:max_episodes]

    result = []
    for ep_idx in sorted_eps:
        entries = episode_rows[ep_idx]
        if not entries:
            continue
        # Group entries by table to extract rows efficiently
        tables_for_ep: dict[int, tuple[pa.Table, list[int]]] = {}
        for tbl, ri in entries:
            key = id(tbl)
            if key not in tables_for_ep:
                tables_for_ep[key] = (tbl, [])
            tables_for_ep[key][1].append(ri)

        columns: dict[str, np.ndarray] = {}
        col_names: list[str] | None = None
        for tbl, row_indices in tables_for_ep.values():
            if col_names is None:
                col_names = tbl.column_names
            for col_name in tbl.column_names:
                if col_name not in col_names:
                    continue
                col = tbl.column(col_name)
                values = [col[i].as_py() for i in row_indices]
                if col_name in columns:
                    # Append if episode spans multiple files (rare for v3)
                    existing = columns[col_name]
                    if isinstance(existing, np.ndarray):
                        existing = existing.tolist()
                    existing.extend(values)
                    try:
                        columns[col_name] = np.array(existing)
                    except (ValueError, TypeError):
                        columns[col_name] = existing
                else:
                    try:
                        columns[col_name] = np.array(values)
                    except (ValueError, TypeError):
                        columns[col_name] = values

        total_len = sum(len(rs) for _, rs in tables_for_ep.values())
        result.append(EpisodeData(
            episode_index=ep_idx,
            columns=columns,
            length=total_len,
        ))
    return result


def load_tasks(root: Path) -> list[dict] | None:
    """Load task metadata from v3 parquet or v2 JSON/JSONL files."""
    tasks_path = root / "meta" / "tasks.parquet"
    if tasks_path.exists():
        table = pq.read_table(tasks_path)
        return [
            {col: table.column(col)[i].as_py() for col in table.column_names}
            for i in range(len(table))
        ]

    tasks_jsonl = root / "meta" / "tasks.jsonl"
    if tasks_jsonl.exists():
        return _load_jsonl(tasks_jsonl)

    tasks_json = root / "meta" / "tasks.json"
    if tasks_json.exists():
        raw = json.loads(tasks_json.read_text())
        if isinstance(raw, list):
            return [r for r in raw if isinstance(r, dict)]
        if isinstance(raw, dict):
            if all(isinstance(v, str) for v in raw.values()):
                return [
                    {"task_index": int(k) if str(k).isdigit() else k, "task": v}
                    for k, v in raw.items()
                ]
            return [raw]
    return None


def load_from_hf(repo_id: str, cache_dir: Path | None = None, max_episodes: int | None = None) -> LoadedDataset:
    """Download dataset files from HuggingFace Hub and load them.

    If max_episodes is set, downloads meta files incrementally and only pulls
    the data parquet files that contain the first N episodes. Avoids pulling
    hundreds of GB for repos like lerobot/droid_1.0.1.
    """
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    if max_episodes is None:
        # Full download: everything under meta/ and data/.
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=cache_dir,
            allow_patterns=["meta/**", "data/**"],
        )
        ds = load_local(Path(local_dir), max_episodes=None)
        ds.is_local = False
        ds.display_path = repo_id
        return ds

    # List repo files once so we know what meta/episodes parquets exist.
    api = HfApi()
    try:
        all_files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception:
        all_files = []

    root: Path | None = None

    def _download(relpath: str) -> Path:
        nonlocal root
        p = Path(hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=relpath,
            cache_dir=cache_dir,
        ))
        if root is None:
            # hf_hub_download returns snapshot/<relpath>; strip to get root.
            root = p
            for _ in Path(relpath).parts:
                root = root.parent
        return p

    # Phase 1: download small meta files (info, tasks, stats).
    small_meta = [f for f in all_files if f.startswith("meta/") and not f.startswith("meta/episodes/")]
    for f in small_meta:
        try:
            _download(f)
        except Exception:
            pass

    if root is None:
        # Couldn't download anything; fall back to full snapshot.
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=cache_dir,
            allow_patterns=["meta/**", "data/**"],
        )
        ds = load_local(Path(local_dir), max_episodes=max_episodes)
        ds.is_local = False
        ds.display_path = repo_id
        return ds

    # Phase 2: download episodes meta files incrementally until we've covered
    # the first max_episodes episodes. Collect (chunk, file) of data parquets
    # needed along the way.
    episodes_meta_files = sorted(f for f in all_files if f.startswith("meta/episodes/") and f.endswith(".parquet"))
    collected: set[tuple[int, int]] = set()
    seen_episode_indices: set[int] = set()

    for mf_rel in episodes_meta_files:
        local_path = _download(mf_rel)
        table = pq.read_table(local_path)
        cols = table.column_names
        ep_col = (
            table.column("episode_index").to_pylist()
            if "episode_index" in cols
            else list(range(len(table)))
        )
        have_mapping = "data/chunk_index" in cols and "data/file_index" in cols
        chunk_col = table.column("data/chunk_index").to_pylist() if have_mapping else [None] * len(table)
        file_col = table.column("data/file_index").to_pylist() if have_mapping else [None] * len(table)

        for ep_idx, ci, fi in zip(ep_col, chunk_col, file_col):
            if ep_idx is None or ep_idx >= max_episodes:
                continue
            seen_episode_indices.add(int(ep_idx))
            if ci is not None and fi is not None:
                collected.add((int(ci), int(fi)))

        # Stop once we've covered episodes 0..max_episodes-1.
        if seen_episode_indices and max(seen_episode_indices) >= max_episodes - 1:
            break

    # Phase 3: download the data parquet files that actually contain our episodes.
    if collected:
        for ci, fi in sorted(collected):
            try:
                _download(f"data/chunk-{ci:03d}/file-{fi:03d}.parquet")
            except Exception:
                pass
    else:
        # No mapping columns; fall back to a bounded pattern download.
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=cache_dir,
            allow_patterns=["data/chunk-000/*.parquet"],
        )

    ds = load_local(root, max_episodes=max_episodes)
    ds.is_local = False
    ds.display_path = repo_id
    return ds


def load_local(root: Path, max_episodes: int | None = None) -> LoadedDataset:
    """Load a local dataset."""
    ds = LoadedDataset(root=root, display_path=str(root), is_local=True)

    info_result = load_info(root)
    if isinstance(info_result, str):
        ds.info_error = info_result
        return ds
    ds.info = info_result

    ds.episodes_meta = load_episodes_meta(root, ds.info)
    ds.episodes_data = load_episode_data(root, ds.info, max_episodes=max_episodes)
    ds.tasks = load_tasks(root)
    ds.max_episodes_applied = max_episodes
    return ds


def _find_zip_dataset_root(zip_file: zipfile.ZipFile) -> tuple[str, str | None]:
    """Return (info_json_member_name, dataset_root_prefix) for a dataset zip."""
    candidates: list[tuple[str, str | None]] = []
    for name in zip_file.namelist():
        path = PurePosixPath(name)
        if path.name != "info.json":
            continue
        parts = path.parts
        if len(parts) >= 2 and parts[-2] == "meta":
            root_parts = parts[:-2]
            root = "/".join(root_parts) if root_parts else None
            candidates.append((name, root))

    if not candidates:
        raise ValueError("Archive does not contain meta/info.json")

    # Prefer the shallowest dataset root if more than one is present.
    candidates.sort(key=lambda item: 0 if item[1] is None else len(PurePosixPath(item[1]).parts))
    return candidates[0]


def _safe_extract_zip_dataset(zip_path: Path, dest: Path) -> tuple[Path, str | None]:
    """Extract a dataset zip under dest/dataset and return the extracted root."""
    with zipfile.ZipFile(zip_path) as zf:
        _, root_prefix = _find_zip_dataset_root(zf)
        root_parts = PurePosixPath(root_prefix).parts if root_prefix else ()
        out_root = dest / "dataset"
        out_root.mkdir(parents=True, exist_ok=True)
        out_root_resolved = out_root.resolve()

        for member in zf.infolist():
            if member.is_dir():
                continue
            member_path = PurePosixPath(member.filename)
            parts = member_path.parts
            if member_path.is_absolute() or ".." in parts:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
            if root_parts:
                if parts[: len(root_parts)] != root_parts:
                    continue
                rel_parts = parts[len(root_parts):]
            else:
                rel_parts = parts
            if not rel_parts:
                continue

            target = out_root.joinpath(*rel_parts)
            if out_root_resolved not in target.parent.resolve().parents and target.parent.resolve() != out_root_resolved:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())

        return out_root, root_prefix


def load_archive(path: Path, max_episodes: int | None = None) -> LoadedDataset:
    """Load a dataset from a local .zip archive."""
    if path.suffix.lower() != ".zip":
        raise ValueError(f"Unsupported archive format: {path.suffix or path.name}")

    tmp = tempfile.TemporaryDirectory(prefix="lerobot-doctor-")
    try:
        root, inner_root = _safe_extract_zip_dataset(path, Path(tmp.name))
        ds = load_local(root, max_episodes=max_episodes)
        ds.display_path = str(path)
        ds.archive_path = path
        ds.archive_inner_root = inner_root
        ds._temp_dir = tmp  # keep extracted files alive while checks run
        return ds
    except Exception:
        tmp.cleanup()
        raise


def load_dataset(path_or_repo: str, max_episodes: int | None = None) -> LoadedDataset:
    """Load dataset from local directory, local archive, or HF repo_id."""
    local = Path(path_or_repo)
    if local.exists() and local.is_dir():
        return load_local(local, max_episodes=max_episodes)
    if local.exists() and local.is_file():
        return load_archive(local, max_episodes=max_episodes)
    if local.is_absolute():
        raise FileNotFoundError(f"Local path does not exist: {local}")
    # Treat as HF repo_id
    return load_from_hf(path_or_repo, max_episodes=max_episodes)
