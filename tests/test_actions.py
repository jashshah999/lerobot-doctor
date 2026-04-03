"""Tests for action quality check."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_doctor.checks.actions import check_actions
from lerobot_doctor.dataset_loader import load_local
from lerobot_doctor.runner import Severity
from tests.conftest import create_dataset


def test_valid_actions(tmp_dataset):
    ds = load_local(tmp_dataset)
    result = check_actions(ds)
    assert result.severity in (Severity.PASS, Severity.WARN)  # random data may trigger jump warnings


def test_nan_actions(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=1, n_frames_per_ep=20, fps=10)
    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_file)
    actions = [row.as_py() for row in table.column("action")]
    actions[5] = [float("nan"), float("nan")]
    new_table = table.set_column(
        table.column_names.index("action"),
        "action",
        pa.array(actions),
    )
    pq.write_table(new_table, data_file)
    ds = load_local(root)
    result = check_actions(ds)
    assert result.severity == Severity.FAIL
    assert any("NaN" in m.message for m in result.messages)


def test_inf_actions(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=1, n_frames_per_ep=20, fps=10)
    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_file)
    actions = [row.as_py() for row in table.column("action")]
    actions[5] = [float("inf"), 0.0]
    new_table = table.set_column(
        table.column_names.index("action"),
        "action",
        pa.array(actions),
    )
    pq.write_table(new_table, data_file)
    ds = load_local(root)
    result = check_actions(ds)
    assert result.severity == Severity.FAIL
    assert any("Inf" in m.message for m in result.messages)


def test_frozen_actions(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=1, n_frames_per_ep=30, fps=10)
    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_file)
    actions = [row.as_py() for row in table.column("action")]
    # Set 15 consecutive identical actions
    for i in range(5, 20):
        actions[i] = [1.0, 2.0]
    new_table = table.set_column(
        table.column_names.index("action"),
        "action",
        pa.array(actions),
    )
    pq.write_table(new_table, data_file)
    ds = load_local(root)
    result = check_actions(ds)
    assert any("frozen" in m.message.lower() for m in result.messages)


def test_clipped_actions(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=1, n_frames_per_ep=100, fps=10)
    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_file)
    # Set 99%+ of dim 0 to the same max value
    actions = [[1.0, np.random.randn()] for _ in range(100)]
    actions[0] = [0.5, np.random.randn()]  # one different so min != max
    new_table = table.set_column(
        table.column_names.index("action"),
        "action",
        pa.array(actions),
    )
    pq.write_table(new_table, data_file)
    ds = load_local(root)
    result = check_actions(ds)
    assert any("clipping" in m.message.lower() or "at maximum" in m.message.lower() for m in result.messages)


def test_no_episode_data(tmp_path):
    root = create_dataset(tmp_path / "dataset")
    import shutil
    shutil.rmtree(root / "data")
    (root / "data").mkdir()
    ds = load_local(root)
    result = check_actions(ds)
    assert result.severity == Severity.WARN
