"""Tests for per-episode summary check."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_doctor.checks.per_episode import check_per_episode
from lerobot_doctor.dataset_loader import load_local
from lerobot_doctor.runner import Severity
from tests.conftest import create_dataset


def test_clean_episodes(tmp_path):
    root = create_dataset(tmp_path / "ds", n_episodes=3, n_frames_per_ep=200, fps=10)
    ds = load_local(root)
    result = check_per_episode(ds)
    assert result.severity == Severity.PASS
    assert any("look clean" in m.message for m in result.messages)


def test_flags_short_episodes(tmp_path):
    root = create_dataset(tmp_path / "ds", n_episodes=3, n_frames_per_ep=2, fps=10)
    ds = load_local(root)
    result = check_per_episode(ds)
    assert result.severity == Severity.WARN
    assert any("too short" in m.message for m in result.messages)


def test_flags_single_frame(tmp_path):
    root = create_dataset(tmp_path / "ds", n_episodes=1, n_frames_per_ep=1, fps=10)
    ds = load_local(root)
    result = check_per_episode(ds)
    assert result.severity == Severity.WARN
    assert any("single frame" in m.message for m in result.messages)


def test_flags_frozen_actions(tmp_path):
    root = create_dataset(tmp_path / "ds", n_episodes=2, n_frames_per_ep=50, fps=10)
    # Make episode 0's actions all identical
    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_file)
    n = len(table)
    frozen = [[0.5, 0.5]] * n
    new_table = table.set_column(
        table.column_names.index("action"),
        "action",
        pa.array(frozen),
    )
    pq.write_table(new_table, data_file)
    ds = load_local(root)
    result = check_per_episode(ds)
    assert any("frozen" in m.message for m in result.messages)


def test_flags_nan_actions(tmp_path):
    root = create_dataset(tmp_path / "ds", n_episodes=2, n_frames_per_ep=20, fps=10)
    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_file)
    actions = [[float("nan"), 0.0]] * len(table)
    new_table = table.set_column(
        table.column_names.index("action"),
        "action",
        pa.array(actions),
    )
    pq.write_table(new_table, data_file)
    ds = load_local(root)
    result = check_per_episode(ds)
    assert any("NaN" in m.message for m in result.messages)


def test_flags_timestamp_gaps(tmp_path):
    root = create_dataset(tmp_path / "ds", n_episodes=1, n_frames_per_ep=20, fps=10)
    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_file)
    ts = table.column("timestamp").to_pylist()
    # Insert a big gap at frame 5
    for i in range(5, len(ts)):
        ts[i] = ts[i] + 10.0
    new_table = table.set_column(
        table.column_names.index("timestamp"),
        "timestamp",
        pa.array(ts, type=pa.float32()),
    )
    pq.write_table(new_table, data_file)
    ds = load_local(root)
    result = check_per_episode(ds)
    assert any("dropped frame" in m.message for m in result.messages)


def test_no_episode_data(tmp_path):
    root = create_dataset(tmp_path / "ds", n_episodes=1, n_frames_per_ep=10, fps=10)
    ds = load_local(root)
    ds.episodes_data = []
    result = check_per_episode(ds)
    assert result.severity == Severity.WARN
    assert any("No episode data" in m.message for m in result.messages)


def test_episode_count_in_summary(tmp_path):
    root = create_dataset(tmp_path / "ds", n_episodes=5, n_frames_per_ep=2, fps=10)
    ds = load_local(root)
    result = check_per_episode(ds)
    assert any("5/5" in m.message for m in result.messages)
