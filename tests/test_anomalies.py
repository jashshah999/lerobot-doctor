"""Tests for anomaly detection check."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_doctor.checks.anomalies import check_anomalies
from lerobot_doctor.dataset_loader import load_local
from lerobot_doctor.runner import Severity
from tests.conftest import create_dataset


def test_no_anomalies(tmp_dataset):
    ds = load_local(tmp_dataset)
    result = check_anomalies(ds)
    # Random data shouldn't have anomalies (mostly)
    assert result.severity in (Severity.PASS, Severity.WARN)


def test_duplicate_episodes(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=3, n_frames_per_ep=20, fps=10)
    # Make episode 1 identical to episode 0
    src_file = root / "data" / "chunk-000" / "file-000.parquet"
    dst_file = root / "data" / "chunk-000" / "file-001.parquet"
    table = pq.read_table(src_file)
    # Change episode_index but keep same action data
    ep_indices = [1] * len(table)
    indices = list(range(20, 40))
    new_table = table.set_column(
        table.column_names.index("episode_index"),
        "episode_index",
        pa.array(ep_indices, type=pa.int64()),
    ).set_column(
        table.column_names.index("index"),
        "index",
        pa.array(indices, type=pa.int64()),
    )
    pq.write_table(new_table, dst_file)
    ds = load_local(root)
    result = check_anomalies(ds)
    assert any("duplicate" in m.message.lower() for m in result.messages)


def test_constant_observation(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=3, n_frames_per_ep=20, fps=10)
    # Make observation.state constant across all episodes
    for i in range(3):
        data_file = root / "data" / "chunk-000" / f"file-{i:03d}.parquet"
        table = pq.read_table(data_file)
        states = [[1.0, 2.0, 3.0, 4.0]] * 20
        new_table = table.set_column(
            table.column_names.index("observation.state"),
            "observation.state",
            pa.array(states),
        )
        pq.write_table(new_table, data_file)
    ds = load_local(root)
    result = check_anomalies(ds)
    assert any("constant value" in m.message.lower() or "broken sensor" in m.message.lower()
               for m in result.messages)


def test_stuck_actuator(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=3, n_frames_per_ep=30, fps=10)
    # Make action dim 0 nearly static (>80% unchanged) in all episodes
    for i in range(3):
        data_file = root / "data" / "chunk-000" / f"file-{i:03d}.parquet"
        table = pq.read_table(data_file)
        actions = [row.as_py() for row in table.column("action")]
        # Set dim 0 to same value for 90% of frames
        for j in range(27):  # 27/30 = 90%
            actions[j] = [1.0, actions[j][1]]
        new_table = table.set_column(
            table.column_names.index("action"),
            "action",
            pa.array(actions),
        )
        pq.write_table(new_table, data_file)
    ds = load_local(root)
    result = check_anomalies(ds)
    assert any("stuck" in m.message.lower() or "static" in m.message.lower()
               for m in result.messages)
