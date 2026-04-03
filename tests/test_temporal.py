"""Tests for temporal consistency check."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_doctor.checks.temporal import check_temporal
from lerobot_doctor.dataset_loader import load_local
from lerobot_doctor.runner import Severity
from tests.conftest import create_dataset


def test_valid_timestamps(tmp_dataset):
    ds = load_local(tmp_dataset)
    result = check_temporal(ds)
    assert result.severity == Severity.PASS


def test_non_monotonic_timestamps(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=1, n_frames_per_ep=10, fps=10)
    # Corrupt timestamps in the data file
    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_file)
    ts = table.column("timestamp").to_pylist()
    ts[5] = ts[3]  # Make timestamp go backwards
    new_table = table.set_column(
        table.column_names.index("timestamp"),
        "timestamp",
        pa.array(ts, type=pa.float32()),
    )
    pq.write_table(new_table, data_file)
    ds = load_local(root)
    result = check_temporal(ds)
    assert result.severity == Severity.WARN
    assert any("non-monotonic" in m.message for m in result.messages)


def test_dropped_frames(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=1, n_frames_per_ep=10, fps=10)
    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_file)
    ts = table.column("timestamp").to_pylist()
    # Create a large gap (simulating dropped frames)
    ts[5] = ts[4] + 0.5  # 500ms gap when expecting 100ms
    for i in range(6, len(ts)):
        ts[i] = ts[5] + (i - 5) * 0.1
    new_table = table.set_column(
        table.column_names.index("timestamp"),
        "timestamp",
        pa.array(ts, type=pa.float32()),
    )
    pq.write_table(new_table, data_file)
    ds = load_local(root)
    result = check_temporal(ds)
    assert result.severity == Severity.WARN
    assert any("dropped frame" in m.message.lower() for m in result.messages)


def test_non_sequential_frame_index(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=1, n_frames_per_ep=10, fps=10)
    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_file)
    fi = table.column("frame_index").to_pylist()
    fi[5] = 99  # break sequence
    new_table = table.set_column(
        table.column_names.index("frame_index"),
        "frame_index",
        pa.array(fi, type=pa.int64()),
    )
    pq.write_table(new_table, data_file)
    ds = load_local(root)
    result = check_temporal(ds)
    assert result.severity == Severity.WARN
    assert any("frame_index not sequential" in m.message for m in result.messages)


def test_no_data(tmp_path):
    root = create_dataset(tmp_path / "dataset")
    import shutil
    shutil.rmtree(root / "data")
    (root / "data").mkdir()
    ds = load_local(root)
    result = check_temporal(ds)
    assert result.severity == Severity.WARN
    assert any("No episode data" in m.message for m in result.messages)
