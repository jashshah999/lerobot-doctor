"""Tests for video integrity check."""

import json

import pytest

from lerobot_doctor.checks.videos import check_videos
from lerobot_doctor.dataset_loader import load_local
from lerobot_doctor.runner import Severity
from tests.conftest import create_dataset


def test_no_video_features(tmp_dataset):
    """Dataset without video features should pass."""
    ds = load_local(tmp_dataset)
    result = check_videos(ds)
    assert result.severity == Severity.PASS
    assert any("No video features" in m.message for m in result.messages)


def test_video_feature_declared_but_no_path(tmp_path):
    """Video feature declared but no video_path in info.json."""
    root = create_dataset(tmp_path / "dataset")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.top"] = {"dtype": "video", "shape": [3, 480, 640], "names": None}
    info_path.write_text(json.dumps(info))
    ds = load_local(root)
    result = check_videos(ds)
    assert result.severity == Severity.FAIL
    assert any("video_path" in m.message for m in result.messages)


def test_video_files_missing(tmp_path):
    """Video feature and path declared but files don't exist."""
    root = create_dataset(tmp_path / "dataset")
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.top"] = {"dtype": "video", "shape": [3, 480, 640], "names": None}
    info["video_path"] = "videos/{video_key}/chunk-{episode_chunk:03d}/file-{episode_index:03d}.mp4"
    info_path.write_text(json.dumps(info))
    ds = load_local(root)
    result = check_videos(ds)
    assert result.severity == Severity.FAIL
    assert any("missing" in m.message.lower() for m in result.messages)


def test_no_info(tmp_path):
    """Can't check videos without info.json."""
    root = tmp_path / "empty"
    root.mkdir()
    (root / "meta").mkdir()
    ds = load_local(root)
    result = check_videos(ds)
    assert result.severity == Severity.FAIL
