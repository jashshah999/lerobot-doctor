"""Tests for the episode identity check."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot_doctor.checks.identity import check_identity, evidence_hashes, parse_splits
from lerobot_doctor.dataset_loader import load_local
from lerobot_doctor.runner import Severity
from tests.conftest import create_dataset


def _data_file(root, ep):
    return root / "data" / "chunk-000" / f"file-{ep:03d}.parquet"


def _copy_episode(root, src, dst, **overrides):
    """Overwrite episode `dst` with episode `src`'s data, keeping dst's bookkeeping."""
    table = pq.read_table(_data_file(root, src))
    n = len(table)
    offset = dst * n
    cols = {
        "episode_index": pa.array([dst] * n, type=pa.int64()),
        "index": pa.array(list(range(offset, offset + n)), type=pa.int64()),
    }
    cols.update(overrides)
    for name, arr in cols.items():
        if name in table.column_names:
            table = table.set_column(table.column_names.index(name), name, arr)
        else:
            table = table.append_column(name, arr)
    pq.write_table(table, _data_file(root, dst))


def _add_task(root, text):
    tasks = pq.read_table(root / "meta" / "tasks.parquet").to_pydict()
    tasks["task_index"].append(len(tasks["task_index"]))
    tasks["task"].append(text)
    pq.write_table(pa.table(tasks), root / "meta" / "tasks.parquet")
    return len(tasks["task_index"]) - 1


def _set_splits(root, splits):
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["splits"] = splits
    info_path.write_text(json.dumps(info))


def test_distinct_episodes_pass(tmp_path):
    root = create_dataset(tmp_path / "ds", n_episodes=4, n_frames_per_ep=20)
    result = check_identity(load_local(root))
    assert result.severity == Severity.PASS


def test_exact_copy_detected(tmp_path):
    root = create_dataset(tmp_path / "ds", n_episodes=4, n_frames_per_ep=20)
    _copy_episode(root, 0, 3)
    result = check_identity(load_local(root))
    assert result.severity == Severity.WARN
    assert any("exact copies" in m.message and "[[0, 3]]" in m.message for m in result.messages)
    assert not any("conflicting" in m.message for m in result.messages)


def test_same_start_and_end_is_not_a_copy(tmp_path):
    # Episodes that begin and end at the same pose but differ in the middle.
    root = create_dataset(tmp_path / "ds", n_episodes=2, n_frames_per_ep=40)
    table = pq.read_table(_data_file(root, 0))
    actions = np.array(table.column("action").to_pylist())
    actions[15:25] += 1.0
    _copy_episode(root, 0, 1, action=pa.array(actions.tolist()))
    result = check_identity(load_local(root))
    assert result.severity == Severity.PASS


def test_bookkeeping_and_dtype_do_not_matter(tmp_path):
    root = create_dataset(tmp_path / "ds", n_episodes=2, n_frames_per_ep=10)
    table = pq.read_table(_data_file(root, 0))
    as_f64 = pa.array(table.column("action").to_pylist(), type=pa.list_(pa.float64()))
    _copy_episode(root, 0, 1, action=as_f64,
                  timestamp=pa.array([5.0 + i for i in range(10)], type=pa.float32()))
    hashes = evidence_hashes(load_local(root))
    assert hashes[0] == hashes[1]


def test_conflicting_task_labels(tmp_path):
    root = create_dataset(tmp_path / "ds", n_episodes=3, n_frames_per_ep=20)
    other = _add_task(root, "stack cups in wrong order")
    _copy_episode(root, 0, 2, task_index=pa.array([other] * 20, type=pa.int64()))
    result = check_identity(load_local(root))
    msg = next(m.message for m in result.messages if "conflicting" in m.message)
    assert "ep 0: pick and place" in msg
    assert "ep 2: stack cups in wrong order" in msg


def test_conflicting_success_labels(tmp_path):
    root = create_dataset(tmp_path / "ds", n_episodes=2, n_frames_per_ep=10)
    for ep, success in [(0, True), (1, False)]:
        table = pq.read_table(_data_file(root, ep))
        pq.write_table(table.append_column("next.success", pa.array([success] * 10)), _data_file(root, ep))
    _copy_episode(root, 0, 1, **{"next.success": pa.array([False] * 10)})
    result = check_identity(load_local(root))
    assert any("conflicting" in m.message for m in result.messages)


def test_split_leakage_fails(tmp_path):
    root = create_dataset(tmp_path / "ds", n_episodes=4, n_frames_per_ep=20)
    _set_splits(root, {"train": "0:3", "test": "3:4"})
    _copy_episode(root, 0, 3)
    result = check_identity(load_local(root))
    assert result.severity == Severity.FAIL
    assert any("leakage" in m.message and "'test', 'train'" in m.message for m in result.messages)


def test_duplicate_within_one_split_is_not_leakage(tmp_path):
    root = create_dataset(tmp_path / "ds", n_episodes=4, n_frames_per_ep=20)
    _set_splits(root, {"train": "0:3", "test": "3:4"})
    _copy_episode(root, 0, 2)
    result = check_identity(load_local(root))
    assert result.severity == Severity.WARN


def test_parse_splits():
    assert parse_splits({"train": "0:2", "val": "2:3"}) == {0: "train", 1: "train", 2: "val"}
    assert parse_splits({"train": ["0:1", "3:4"], "test": [1]}) == {0: "train", 3: "train", 1: "test"}
    assert parse_splits({"train": "bogus"}) == {}
