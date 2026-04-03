"""Shared test fixtures for lerobot-doctor tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def tmp_dataset(tmp_path):
    """Create a minimal valid LeRobot v3 dataset."""
    return create_dataset(tmp_path / "dataset", n_episodes=3, n_frames_per_ep=10, fps=10)


def create_dataset(
    root: Path,
    n_episodes: int = 3,
    n_frames_per_ep: int = 10,
    fps: int = 10,
    action_dims: int = 2,
    state_dims: int = 4,
    include_videos: bool = False,
    include_stats: bool = True,
) -> Path:
    """Create a synthetic LeRobot v3 dataset directory."""
    root.mkdir(parents=True, exist_ok=True)
    meta_dir = root / "meta"
    meta_dir.mkdir(exist_ok=True)
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = meta_dir / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    features = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "action": {"dtype": "float32", "shape": [action_dims], "names": None},
        "observation.state": {"dtype": "float32", "shape": [state_dims], "names": None},
    }

    total_frames = n_episodes * n_frames_per_ep

    # info.json
    info = {
        "codebase_version": "v3.0",
        "robot_type": "test_robot",
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/file-{episode_index:03d}.parquet",
        "video_path": None,
        "features": features,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2))

    # tasks.parquet
    tasks_table = pa.table({"task_index": [0], "task": ["pick and place"]})
    pq.write_table(tasks_table, meta_dir / "tasks.parquet")

    # Episode data and metadata
    global_idx = 0
    ep_meta_rows = []
    interval = 1.0 / fps

    for ep in range(n_episodes):
        timestamps = [i * interval for i in range(n_frames_per_ep)]
        frame_indices = list(range(n_frames_per_ep))
        episode_indices = [ep] * n_frames_per_ep
        indices = list(range(global_idx, global_idx + n_frames_per_ep))
        task_indices = [0] * n_frames_per_ep
        actions = np.random.randn(n_frames_per_ep, action_dims).astype(np.float32)
        states = np.random.randn(n_frames_per_ep, state_dims).astype(np.float32)

        table = pa.table({
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "index": pa.array(indices, type=pa.int64()),
            "task_index": pa.array(task_indices, type=pa.int64()),
            "action": [actions[i].tolist() for i in range(n_frames_per_ep)],
            "observation.state": [states[i].tolist() for i in range(n_frames_per_ep)],
        })
        pq.write_table(table, data_dir / f"file-{ep:03d}.parquet")

        ep_meta_rows.append({
            "episode_index": ep,
            "length": n_frames_per_ep,
            "tasks": ["pick and place"],
            "dataset_from_index": global_idx,
            "dataset_to_index": global_idx + n_frames_per_ep,
        })
        global_idx += n_frames_per_ep

    # Episodes metadata parquet
    ep_table = pa.table({
        "episode_index": [r["episode_index"] for r in ep_meta_rows],
        "length": [r["length"] for r in ep_meta_rows],
        "dataset_from_index": [r["dataset_from_index"] for r in ep_meta_rows],
        "dataset_to_index": [r["dataset_to_index"] for r in ep_meta_rows],
    })
    pq.write_table(ep_table, episodes_dir / "file-000.parquet")

    # stats.json
    if include_stats:
        stats = {
            "action": {
                "mean": [0.0] * action_dims,
                "std": [1.0] * action_dims,
                "min": [-3.0] * action_dims,
                "max": [3.0] * action_dims,
            },
            "observation.state": {
                "mean": [0.0] * state_dims,
                "std": [1.0] * state_dims,
                "min": [-3.0] * state_dims,
                "max": [3.0] * state_dims,
            },
        }
        (meta_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    return root
