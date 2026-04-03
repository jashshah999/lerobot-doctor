"""Tests for episode health check."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_doctor.checks.episodes import check_episodes
from lerobot_doctor.dataset_loader import load_local
from lerobot_doctor.runner import Severity
from tests.conftest import create_dataset


def test_healthy_episodes(tmp_path):
    # Use longer episodes so they don't trigger policy window warnings
    root = create_dataset(tmp_path / "dataset", n_episodes=3, n_frames_per_ep=200, fps=10)
    ds = load_local(root)
    result = check_episodes(ds)
    assert result.severity == Severity.PASS


def test_short_episodes(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=3, n_frames_per_ep=2, fps=10)
    ds = load_local(root)
    result = check_episodes(ds)
    assert any("shorter than" in m.message for m in result.messages)


def test_single_frame_episode(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=1, n_frames_per_ep=1, fps=10)
    ds = load_local(root)
    result = check_episodes(ds)
    assert result.severity == Severity.FAIL
    assert any("<=1 frame" in m.message for m in result.messages)


def test_length_metadata_mismatch(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=2, n_frames_per_ep=10, fps=10)
    # Corrupt episode metadata length
    ep_file = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(ep_file)
    lengths = table.column("length").to_pylist()
    lengths[0] = 999  # wrong
    new_table = table.set_column(
        table.column_names.index("length"),
        "length",
        pa.array(lengths, type=pa.int64()),
    )
    pq.write_table(new_table, ep_file)
    ds = load_local(root)
    result = check_episodes(ds)
    assert result.severity == Severity.FAIL
    assert any("mismatch" in m.message for m in result.messages)


def test_policy_window_warning(tmp_path):
    # Episodes too short for common policy chunk sizes
    root = create_dataset(tmp_path / "dataset", n_episodes=5, n_frames_per_ep=8, fps=10)
    ds = load_local(root)
    result = check_episodes(ds)
    assert any("chunk_size" in m.message for m in result.messages)
