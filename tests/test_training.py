"""Tests for training readiness check."""

import json

import pytest

from lerobot_doctor.checks.training import check_training
from lerobot_doctor.dataset_loader import load_local
from lerobot_doctor.runner import Severity
from tests.conftest import create_dataset


def test_training_ready(tmp_dataset):
    ds = load_local(tmp_dataset)
    result = check_training(ds)
    assert result.severity in (Severity.PASS, Severity.WARN)
    assert any("Action features found" in m.message for m in result.messages)


def test_no_actions(tmp_path):
    root = create_dataset(tmp_path / "dataset")
    # Remove action from features in info.json
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    del info["features"]["action"]
    info_path.write_text(json.dumps(info))
    ds = load_local(root)
    result = check_training(ds)
    assert result.severity == Severity.FAIL
    assert any("No action features" in m.message for m in result.messages)


def test_missing_stats(tmp_path):
    root = create_dataset(tmp_path / "dataset", include_stats=False)
    ds = load_local(root)
    result = check_training(ds)
    assert any("stats.json" in m.message.lower() for m in result.messages)


def test_zero_std_warning(tmp_path):
    root = create_dataset(tmp_path / "dataset")
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["action"]["std"] = [0.0, 0.0]  # zero std
    stats_path.write_text(json.dumps(stats))
    ds = load_local(root)
    result = check_training(ds)
    assert any("zero std" in m.message for m in result.messages)


def test_short_episodes_for_prediction(tmp_path):
    root = create_dataset(tmp_path / "dataset", n_episodes=3, n_frames_per_ep=5, fps=10)
    ds = load_local(root)
    result = check_training(ds)
    # Should warn about episodes too short for prediction horizons
    assert any("too short" in m.message for m in result.messages)
