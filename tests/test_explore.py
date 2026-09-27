"""Tests for `explore outliers`."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_doctor.cli import main
from lerobot_doctor.dataset_loader import load_local
from lerobot_doctor.explore.outliers import _read_video_frames, find_outliers, visualize_outliers
from tests.conftest import create_dataset

FPS = 10
N_EPS = 3
N_FRAMES = 8
VIDEO_KEY = "observation.images.front"
SPIKE = (1, 5)  # (episode, frame) carrying the injected outlier


def _set_action(root: Path, episode: int, frame: int, dim: int, value: float):
    for pf in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(pf)
        eps = table.column("episode_index").to_pylist()
        frames = table.column("frame_index").to_pylist()
        rows = [i for i, (e, f) in enumerate(zip(eps, frames)) if e == episode and f == frame]
        if not rows:
            continue
        actions = table.column("action").to_pylist()
        actions[rows[0]][dim] = value
        i = table.column_names.index("action")
        pq.write_table(table.set_column(i, "action", pa.array(actions)), pf)
        return
    raise AssertionError("row not found")


def _dataset_with_spike(tmp_path, n_episodes=5, n_frames=40):
    root = create_dataset(tmp_path / "ds", n_episodes=n_episodes, n_frames_per_ep=n_frames, fps=FPS)
    _set_action(root, *SPIKE, dim=1, value=1000.0)
    return root


def _write_gray_video(path: Path, levels: list[int], size=(32, 32)):
    """MP4 whose frame i is a flat gray image of brightness levels[i]."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=FPS)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.options = {"qscale": "1"}
        for i, level in enumerate(levels):
            img = np.full((size[1], size[0], 3), level, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _level(episode: int, frame: int) -> int:
    return 20 + 9 * (episode * N_FRAMES + frame)


def _video_dataset(tmp_path, layout: str) -> Path:
    """Dataset with a video feature whose pixels encode (episode, frame)."""
    root = create_dataset(tmp_path / layout, n_episodes=N_EPS, n_frames_per_ep=N_FRAMES, fps=FPS)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"][VIDEO_KEY] = {"dtype": "video", "shape": [3, 32, 32], "names": None}

    if layout == "v3":
        info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        levels = [_level(e, f) for e in range(N_EPS) for f in range(N_FRAMES)]
        _write_gray_video(root / "videos" / VIDEO_KEY / "chunk-000" / "file-000.mp4", levels)
        ep_meta = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pydict()
        ep_meta[f"videos/{VIDEO_KEY}/chunk_index"] = [0] * N_EPS
        ep_meta[f"videos/{VIDEO_KEY}/file_index"] = [0] * N_EPS
        ep_meta[f"videos/{VIDEO_KEY}/from_timestamp"] = [e * N_FRAMES / FPS for e in range(N_EPS)]
        ep_meta[f"videos/{VIDEO_KEY}/to_timestamp"] = [(e + 1) * N_FRAMES / FPS for e in range(N_EPS)]
        pq.write_table(pa.table(ep_meta), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    else:
        info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        for e in range(N_EPS):
            _write_gray_video(
                root / "videos" / "chunk-000" / VIDEO_KEY / f"episode_{e:06d}.mp4",
                [_level(e, f) for f in range(N_FRAMES)],
            )
    info_path.write_text(json.dumps(info, indent=2))
    _set_action(root, *SPIKE, dim=1, value=1000.0)
    return root


def test_locates_injected_outlier(tmp_path):
    ds = load_local(_dataset_with_spike(tmp_path))
    result = find_outliers(ds, feature="action", threshold=10)
    assert result.total_frames == 200
    assert result.n_dims == 2
    assert [(o.dim, o.episode_index, o.frame_index) for o in result.outliers] == [(1, *SPIKE)]
    assert result.outliers[0].value == 1000.0


def test_nan_does_not_hide_outliers(tmp_path):
    root = _dataset_with_spike(tmp_path)
    _set_action(root, 3, 7, dim=1, value=float("nan"))
    result = find_outliers(load_local(root), feature="action", threshold=10)
    assert [(o.dim, o.episode_index, o.frame_index) for o in result.outliers] == [(1, *SPIKE)]
    assert result.n_non_finite == 1
    assert np.isfinite(result.mean_per_dim).all()


def test_missing_feature_lists_available(tmp_path):
    ds = load_local(_dataset_with_spike(tmp_path))
    with pytest.raises(ValueError, match="observation.state"):
        find_outliers(ds, feature="nope")


def test_max_episodes_limits_scan(tmp_path):
    ds = load_local(_dataset_with_spike(tmp_path), max_episodes=1)
    result = find_outliers(ds, feature="action", threshold=10)
    assert result.total_frames == 40
    assert result.outliers == []


def test_cli_json_multi_feature_is_one_document(tmp_path, capsys):
    root = _dataset_with_spike(tmp_path)
    main(["explore", "outliers", str(root), "--features", "action,observation.state", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert [r["feature"] for r in data["results"]] == ["action", "observation.state"]
    action = data["results"][0]
    assert action["total_outliers"] == 1
    assert action["outliers"][0]["episode_index"] == SPIKE[0]
    assert action["outliers"][0]["frame_index"] == SPIKE[1]


def test_cli_json_top(tmp_path, capsys):
    root = _dataset_with_spike(tmp_path)
    main(["explore", "outliers", str(root), "--threshold", "1", "--top", "3", "--json"])
    action = json.loads(capsys.readouterr().out)["results"][0]
    assert action["total_outliers"] > 3
    assert len(action["outliers"]) == 3
    assert action["outliers"][0]["frame_index"] == SPIKE[1]


def test_cli_text_report(tmp_path, capsys):
    root = _dataset_with_spike(tmp_path)
    main(["explore", "outliers", str(root)])
    out = capsys.readouterr().out
    assert "Total outliers: 1" in out
    assert "1000.000000" in out


def test_cli_missing_feature_exits_1(tmp_path, capsys):
    root = _dataset_with_spike(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["explore", "outliers", str(root), "--features", "nope"])
    assert exc.value.code == 1
    assert "not found" in capsys.readouterr().err


def test_cli_visualize_without_videos_exits_1(tmp_path, capsys):
    pytest.importorskip("cv2")
    root = _dataset_with_spike(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["explore", "outliers", str(root), "--visualize", "--output-dir", str(tmp_path / "out")])
    assert exc.value.code == 1
    assert "no video features" in capsys.readouterr().err


@pytest.mark.parametrize("layout", ["v2", "v3"])
def test_read_video_frames_returns_the_right_frames(tmp_path, layout):
    from lerobot_doctor.explore.outliers import _video_location

    root = _video_dataset(tmp_path, layout)
    ds = load_local(root)
    for episode in range(N_EPS):
        path, start = _video_location(ds, VIDEO_KEY, episode)
        frames = list(range(N_FRAMES))
        decoded = _read_video_frames(path, [start + f / FPS for f in frames])
        all_levels = np.array([_level(e, f) for e in range(N_EPS) for f in range(N_FRAMES)])
        got = [int(np.argmin(np.abs(all_levels - img.mean()))) for img in decoded]
        assert got == [episode * N_FRAMES + f for f in frames]


@pytest.mark.parametrize("layout", ["v2", "v3"])
def test_visualize_strip(tmp_path, layout):
    cv2 = pytest.importorskip("cv2")
    ds = load_local(_video_dataset(tmp_path, layout))
    result = find_outliers(ds, feature="action", threshold=3)
    assert (result.outliers[0].episode_index, result.outliers[0].frame_index) == SPIKE

    saved = visualize_outliers(ds, result, tmp_path / "out", context_frames=2, top=1)
    assert [p.name for p in saved] == [f"ep{SPIKE[0]:06d}_frame{SPIKE[1]:06d}_outlier.png"]
    img = cv2.imread(str(saved[0]))
    assert img is not None
    # 5 panels; the 32px frames are upscaled 8x to 256px plus borders.
    assert img.shape[1] > 5 * 256


def test_cli_visualize_writes_images(tmp_path, capsys):
    pytest.importorskip("cv2")
    root = _video_dataset(tmp_path, "v3")
    out_dir = tmp_path / "viz"
    main(["explore", "outliers", str(root), "--threshold", "3", "--top", "1",
          "--visualize", "--output-dir", str(out_dir), "--json"])
    captured = capsys.readouterr()
    json.loads(captured.out)  # progress messages must not corrupt stdout
    assert sorted(p.name for p in (out_dir / "action").glob("*.png")) == [
        f"ep{SPIKE[0]:06d}_frame{SPIKE[1]:06d}_outlier.png"
    ]


def test_visualize_unknown_camera(tmp_path):
    pytest.importorskip("cv2")
    ds = load_local(_video_dataset(tmp_path, "v3"))
    result = find_outliers(ds, feature="action", threshold=3)
    with pytest.raises(ValueError, match=VIDEO_KEY):
        visualize_outliers(ds, result, tmp_path / "out", video_key="nope")
