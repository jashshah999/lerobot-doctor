"""Tests for feature consistency check."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_doctor.checks.consistency import check_consistency
from lerobot_doctor.dataset_loader import load_local
from lerobot_doctor.runner import Severity
from tests.conftest import create_dataset


def test_consistent_features(tmp_dataset):
    ds = load_local(tmp_dataset)
    result = check_consistency(ds)
    assert result.severity == Severity.PASS


def test_missing_column_in_episode(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=2, n_frames_per_ep=10, fps=10)
    # Rewrite episode 1's data without observation.state column
    data_file = root / "data" / "chunk-000" / "file-001.parquet"
    table = pq.read_table(data_file)
    cols_to_keep = [c for c in table.column_names if c != "observation.state"]
    new_table = table.select(cols_to_keep)
    pq.write_table(new_table, data_file)
    ds = load_local(root)
    result = check_consistency(ds)
    assert result.severity in (Severity.FAIL, Severity.WARN)
    # Will detect either as missing feature or dtype mismatch (null fill)
    assert any(
        "missing features" in m.message.lower() or "dtype mismatch" in m.message.lower()
        for m in result.messages
    )


def test_single_episode_skips(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=1, n_frames_per_ep=10, fps=10)
    ds = load_local(root)
    result = check_consistency(ds)
    assert result.severity == Severity.PASS
    assert any("Only 1 episode" in m.message for m in result.messages)
