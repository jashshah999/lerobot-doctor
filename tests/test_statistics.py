"""Tests for statistics sanity check."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_doctor.checks.statistics import check_statistics
from lerobot_doctor.dataset_loader import load_local
from lerobot_doctor.runner import Severity
from tests.conftest import create_dataset


def test_valid_stats(tmp_dataset):
    ds = load_local(tmp_dataset)
    result = check_statistics(ds)
    # Might warn about outliers in random data, but shouldn't fail
    assert result.severity in (Severity.PASS, Severity.WARN)


def test_nan_in_observations(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=1, n_frames_per_ep=20, fps=10)
    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_file)
    states = [row.as_py() for row in table.column("observation.state")]
    states[5] = [float("nan"), 0.0, 0.0, 0.0]
    new_table = table.set_column(
        table.column_names.index("observation.state"),
        "observation.state",
        pa.array(states),
    )
    pq.write_table(new_table, data_file)
    ds = load_local(root)
    result = check_statistics(ds)
    assert result.severity == Severity.FAIL
    assert any("NaN" in m.message for m in result.messages)


def test_zero_variance(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=1, n_frames_per_ep=20, fps=10)
    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_file)
    # Set all observation.state values to constant
    states = [[1.0, 2.0, 3.0, 4.0]] * 20
    new_table = table.set_column(
        table.column_names.index("observation.state"),
        "observation.state",
        pa.array(states),
    )
    pq.write_table(new_table, data_file)
    ds = load_local(root)
    result = check_statistics(ds)
    assert any("zero variance" in m.message for m in result.messages)


def test_missing_stats_json(tmp_path):
    root = create_dataset(tmp_path / "dataset", include_stats=False)
    ds = load_local(root)
    result = check_statistics(ds)
    assert any("stats.json not found" in m.message for m in result.messages)


def test_no_data(tmp_path):
    root = create_dataset(tmp_path / "dataset")
    import shutil
    shutil.rmtree(root / "data")
    (root / "data").mkdir()
    ds = load_local(root)
    result = check_statistics(ds)
    assert result.severity == Severity.WARN
