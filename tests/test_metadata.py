"""Tests for metadata check."""

import json
from pathlib import Path

import pytest

from lerobot_doctor.checks.metadata import check_metadata
from lerobot_doctor.dataset_loader import load_local
from lerobot_doctor.runner import Severity
from tests.conftest import create_dataset


def test_valid_dataset(tmp_dataset):
    ds = load_local(tmp_dataset)
    result = check_metadata(ds)
    assert result.severity == Severity.PASS


def test_missing_info_json(tmp_path):
    root = tmp_path / "empty_dataset"
    root.mkdir()
    (root / "meta").mkdir()
    ds = load_local(root)
    result = check_metadata(ds)
    assert result.severity == Severity.FAIL
    assert any("info.json" in m.message for m in result.messages)


def test_invalid_info_json(tmp_path):
    root = create_dataset(tmp_path / "dataset")
    (root / "meta" / "info.json").write_text("not json {{{")
    ds = load_local(root)
    result = check_metadata(ds)
    assert result.severity == Severity.FAIL


def test_missing_required_fields(tmp_path):
    root = create_dataset(tmp_path / "dataset")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    del info["fps"]
    info_path.write_text(json.dumps(info))
    ds = load_local(root)
    result = check_metadata(ds)
    assert result.severity == Severity.FAIL
    assert any("missing required fields" in m.message for m in result.messages)


def test_wrong_total_frames(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=2, n_frames_per_ep=5)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] = 999  # wrong
    info_path.write_text(json.dumps(info))
    ds = load_local(root)
    result = check_metadata(ds)
    # Should fail because frame count doesn't match
    assert result.severity == Severity.FAIL
    assert any("total_frames" in m.message for m in result.messages)


def test_wrong_total_episodes(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=3)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_episodes"] = 99
    info_path.write_text(json.dumps(info))
    ds = load_local(root)
    result = check_metadata(ds)
    assert result.severity == Severity.FAIL
    assert any("total_episodes" in m.message for m in result.messages)


def test_missing_tasks_parquet(tmp_path):
    root = create_dataset(tmp_path / "dataset")
    (root / "meta" / "tasks.parquet").unlink()
    ds = load_local(root)
    result = check_metadata(ds)
    assert result.severity == Severity.FAIL
    assert any("tasks.parquet" in m.message for m in result.messages)


def test_no_data_directory(tmp_path):
    root = create_dataset(tmp_path / "dataset")
    import shutil
    shutil.rmtree(root / "data")
    ds = load_local(root)
    result = check_metadata(ds)
    assert result.severity == Severity.FAIL
    assert any("data/" in m.message for m in result.messages)


def test_non_v3_version_warns(tmp_path):
    root = create_dataset(tmp_path / "dataset")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["codebase_version"] = "v2.0"
    info_path.write_text(json.dumps(info))
    ds = load_local(root)
    result = check_metadata(ds)
    assert any(m.severity == Severity.WARN and "v2.0" in m.message for m in result.messages)
