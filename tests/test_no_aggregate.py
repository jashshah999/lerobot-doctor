"""Tests for --no-aggregate (list every flagged item instead of truncating)."""

import json

from lerobot_doctor.cli import main
from lerobot_doctor.runner import head, preview
from tests.conftest import create_dataset


def test_preview_truncates_by_default():
    assert preview(range(7), 5) == "[0, 1, 2, 3, 4]..."
    assert preview(range(5), 5) == "[0, 1, 2, 3, 4]"


def test_preview_full():
    assert preview(range(7), 5, full=True) == "[0, 1, 2, 3, 4, 5, 6]"


def test_head():
    assert head(range(7), 5) == ([0, 1, 2, 3, 4], 2)
    assert head(range(3), 5) == ([0, 1, 2], 0)
    assert head(range(7), 5, full=True) == ([0, 1, 2, 3, 4, 5, 6], 0)


def _check_messages(out: str, name: str) -> list[str]:
    data = json.loads(out)
    check = next(c for c in data["checks"] if c["name"] == name)
    return [m["message"] for m in check["messages"]]


def _short_dataset(tmp_path, n_episodes=30):
    # 3-frame episodes at 10 fps: every episode is flagged as too short.
    return create_dataset(tmp_path / "ds", n_episodes=n_episodes, n_frames_per_ep=3, fps=10)


def test_json_aggregates_by_default(tmp_path, capsys):
    root = _short_dataset(tmp_path)
    main([str(root), "--json", "--checks", "per_episode,episodes"])
    out = capsys.readouterr().out

    per_ep = _check_messages(out, "Per-Episode Summary")
    assert "...and 10 more flagged episodes" in per_ep
    assert sum(m.startswith("Episode ") for m in per_ep) == 20

    episodes = _check_messages(out, "Episode Health")
    short = next(m for m in episodes if "shorter than" in m)
    assert short.endswith("...")


def test_json_no_aggregate_lists_everything(tmp_path, capsys):
    root = _short_dataset(tmp_path)
    main([str(root), "--json", "--no-aggregate", "--checks", "per_episode,episodes"])
    out = capsys.readouterr().out

    per_ep = _check_messages(out, "Per-Episode Summary")
    assert not any("more flagged" in m for m in per_ep)
    flagged = sorted(int(m.split(":")[0].split()[1]) for m in per_ep if m.startswith("Episode "))
    assert flagged == list(range(30))

    episodes = _check_messages(out, "Episode Health")
    short = next(m for m in episodes if "shorter than" in m)
    assert not short.endswith("...")
    assert str(list(range(30))) in short


def test_no_aggregate_text_output(tmp_path, capsys):
    root = _short_dataset(tmp_path)
    main([str(root), "--no-aggregate", "--checks", "per_episode"])
    out = capsys.readouterr().out
    assert "more flagged" not in out
    assert "Episode 29:" in out


def test_no_aggregate_ci_output(tmp_path, capsys):
    root = _short_dataset(tmp_path)
    main([str(root), "--ci", "--no-aggregate", "--checks", "per_episode"])
    out = capsys.readouterr().out
    per_ep = _check_messages(out, "Per-Episode Summary")
    assert sum(m.startswith("Episode ") for m in per_ep) == 30
