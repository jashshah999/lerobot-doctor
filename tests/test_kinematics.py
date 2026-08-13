"""Tests for the kinematic feasibility check."""

import json
from pathlib import Path
from textwrap import dedent

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_doctor.checks.kinematics import check_kinematics
from lerobot_doctor.dataset_loader import load_local
from lerobot_doctor.robots.limits import lookup_known_robot, normalize_robot_type
from lerobot_doctor.robots.urdf import parse_urdf_limits
from lerobot_doctor.runner import Severity
from tests.conftest import create_dataset

SO101_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _so101_dataset(
    tmp_path,
    actions: list[list[float]],
    robot_type: str = "so101_follower",
    names: list[str] | None = None,
) -> Path:
    """A dataset with SO-101-style action names and a given sequence of actions."""
    n = len(actions)
    root = create_dataset(tmp_path / "dataset", n_episodes=1, n_frames_per_ep=n, fps=30, action_dims=6)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["robot_type"] = robot_type
    info["features"]["action"]["names"] = names or [f"{j}.pos" for j in SO101_JOINT_NAMES]
    info_path.write_text(json.dumps(info))

    data_file = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(data_file)
    new_table = table.set_column(table.column_names.index("action"), "action", pa.array(actions))
    pq.write_table(new_table, data_file)

    return root


def test_within_limits_passes(tmp_path):
    actions = [[0.0, 0.0, -0.5, 0.0, 0.0, 0.5] for _ in range(10)]
    ds = load_local(_so101_dataset(tmp_path, actions))

    result = check_kinematics(ds)

    assert result.severity == Severity.PASS


def test_position_beyond_urdf_limit_is_flagged(tmp_path):
    # shoulder_pan's real SO-101 limit is +-1.91986 rad; 3.0 is physically unreachable.
    actions = [[3.0, 0.0, -0.5, 0.0, 0.0, 0.5] for _ in range(10)]
    ds = load_local(_so101_dataset(tmp_path, actions))

    result = check_kinematics(ds)

    assert result.severity == Severity.FAIL
    assert any("shoulder_pan" in m.message and "outside physical joint limits" in m.message for m in result.messages)


def test_unmatched_robot_type_is_skipped_not_failed(tmp_path):
    # No registry entry and no --urdf: an informational skip, not a warning.
    # Most datasets have no registry entry; a default run must not warn on them.
    actions = [[3.0, 0.0, -0.5, 0.0, 0.0, 0.5] for _ in range(10)]
    ds = load_local(_so101_dataset(tmp_path, actions, robot_type="some_custom_arm"))

    result = check_kinematics(ds)

    assert result.severity == Severity.PASS
    assert any("not in the built-in registry" in m.message for m in result.messages)


def test_explicit_urdf_overrides_registry(tmp_path):
    urdf = tmp_path / "custom.urdf"
    urdf.write_text(dedent("""\
        <robot name="custom">
          <joint name="shoulder_pan" type="revolute">
            <limit lower="-0.1" upper="0.1" velocity="1.0" effort="1.0"/>
          </joint>
        </robot>
    """))
    actions = [[0.05, 0.0, -0.5, 0.0, 0.0, 0.5] for _ in range(5)]
    ds = load_local(_so101_dataset(tmp_path, actions))
    ds.robot_urdf = urdf  # 0.05 is within [-0.1, 0.1] here, but would also pass the SO-101 registry

    ok_result = check_kinematics(ds)
    # The URDF only declares shoulder_pan, so the other 5 dims legitimately warn as
    # unmatched -- what matters here is that shoulder_pan itself isn't flagged.
    assert not any("shoulder_pan" in m.message and "outside physical joint limits" in m.message for m in ok_result.messages)

    actions_over = [[0.5, 0.0, -0.5, 0.0, 0.0, 0.5] for _ in range(5)]  # outside custom urdf, inside SO-101 registry
    ds2 = load_local(_so101_dataset(tmp_path, actions_over))
    ds2.robot_urdf = urdf

    over_result = check_kinematics(ds2)
    assert over_result.severity == Severity.FAIL


def test_velocity_beyond_limit_is_warned(tmp_path):
    # SO-101 wrist_roll velocity limit is 10 rad/s; at 30fps a jump of 1.0 rad
    # per frame implies 30 rad/s, well above it, while staying within position limits.
    actions = [[0.0, 0.0, -0.5, 0.0, (i % 2) * 1.0, 0.5] for i in range(10)]
    ds = load_local(_so101_dataset(tmp_path, actions))

    result = check_kinematics(ds)

    assert any("velocity" in m.message for m in result.messages)


def test_normalized_percent_actions_are_flagged_as_units_mismatch_not_failed(tmp_path):
    # Real-world case: several SO-100/SO-101 LeRobot configs store actions as a
    # normalized percentage (~[-100, 100]) rather than URDF radians. Naively
    # checking these against the radian registry would flag ~100% of frames as
    # "outside physical limits" -- which is a units mismatch, not a real
    # kinematic violation, and must not be reported as a FAIL.
    actions = [[85.0, -90.0, 70.0, 60.0, -50.0, 20.0] for _ in range(10)]
    ds = load_local(_so101_dataset(tmp_path, actions))

    result = check_kinematics(ds)

    assert result.severity == Severity.WARN
    assert any("units mismatch" in m.message for m in result.messages)
    assert not any("outside physical joint limits" in m.message for m in result.messages)


def test_normalize_robot_type_strips_follower_leader_suffixes():
    assert normalize_robot_type("so100_follower") == "so100"
    assert normalize_robot_type("SO101_Leader") == "so101"
    assert normalize_robot_type("so100") == "so100"


def test_lookup_known_robot_returns_none_for_unknown():
    assert lookup_known_robot("unknown_arm") is None
    assert lookup_known_robot(None) is None


def test_parse_urdf_limits_skips_fixed_and_unlimited_joints(tmp_path):
    urdf = tmp_path / "test.urdf"
    urdf.write_text(dedent("""\
        <robot name="test">
          <joint name="fixed_joint" type="fixed"/>
          <joint name="unbounded" type="continuous"/>
          <joint name="real_joint" type="revolute">
            <limit lower="-1.0" upper="1.0" velocity="2.0" effort="5.0"/>
          </joint>
        </robot>
    """))

    limits = parse_urdf_limits(urdf)

    assert set(limits.keys()) == {"real_joint"}
    assert limits["real_joint"].lower == -1.0
    assert limits["real_joint"].upper == 1.0
    assert limits["real_joint"].velocity == 2.0


def test_vel_dim_within_velocity_limit_is_not_a_position_violation(tmp_path):
    # 3.0 rad/s commanded velocity is legal (SO-101 velocity limit is 10 rad/s) but
    # far outside shoulder_pan's position bounds; it must not be flagged.
    names = ["shoulder_pan.pos", "shoulder_pan.vel", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    actions = [[0.0, 3.0, -0.5, 0.0, 0.0, 0.5] for _ in range(10)]
    ds = load_local(_so101_dataset(tmp_path, actions, names=names))

    result = check_kinematics(ds)

    assert result.severity == Severity.PASS
    assert not any("outside physical joint limits" in m.message for m in result.messages)


def test_vel_dim_above_velocity_limit_is_flagged(tmp_path):
    # 30 rad/s commanded velocity is above the SO-101 declared limit of 10 rad/s.
    names = ["shoulder_pan.pos", "shoulder_pan.vel", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
    actions = [[0.0, 30.0, -0.5, 0.0, 0.0, 0.5] for _ in range(10)]
    ds = load_local(_so101_dataset(tmp_path, actions, names=names))

    result = check_kinematics(ds)

    assert result.severity == Severity.FAIL
    assert any("shoulder_pan.vel" in m.message and "velocity above" in m.message for m in result.messages)


def test_nan_does_not_defeat_units_mismatch_guard(tmp_path):
    # A single NaN frame must not turn the documented WARN-and-skip units mismatch
    # into a misdiagnosed physical-limit FAIL.
    actions = [[85.0, -90.0, 70.0, 60.0, -50.0, 20.0] for _ in range(10)]
    actions[0] = [float("nan"), -90.0, 70.0, 60.0, -50.0, 20.0]
    ds = load_local(_so101_dataset(tmp_path, actions))

    result = check_kinematics(ds)

    assert result.severity == Severity.WARN
    assert any("units mismatch" in m.message for m in result.messages)
    assert not any("outside physical joint limits" in m.message for m in result.messages)


def test_ragged_action_rows_do_not_crash(tmp_path):
    # An episode whose action rows have inhomogeneous lengths is stored as a raw
    # list by the loader; the kinematics check must skip it, not crash the run.
    actions = [[0.0, 0.0, -0.5, 0.0, 0.0, 0.5] for _ in range(9)]
    actions.append([0.0, 0.0, -0.5, 0.0, 0.0])  # 5 dims instead of 6
    ds = load_local(_so101_dataset(tmp_path, actions))

    result = check_kinematics(ds)

    assert result.severity in (Severity.PASS, Severity.WARN)


def test_nonexistent_explicit_urdf_fails(tmp_path):
    actions = [[0.0, 0.0, -0.5, 0.0, 0.0, 0.5] for _ in range(5)]
    ds = load_local(_so101_dataset(tmp_path, actions))
    ds.robot_urdf = tmp_path / "does_not_exist.urdf"

    result = check_kinematics(ds)

    assert result.severity == Severity.FAIL
    assert any("failed to parse" in m.message for m in result.messages)


def test_explicit_urdf_without_limited_joints_fails(tmp_path):
    urdf = tmp_path / "empty.urdf"
    urdf.write_text('<robot name="empty"><joint name="a" type="fixed"/></robot>')
    actions = [[0.0, 0.0, -0.5, 0.0, 0.0, 0.5] for _ in range(5)]
    ds = load_local(_so101_dataset(tmp_path, actions))
    ds.robot_urdf = urdf

    result = check_kinematics(ds)

    assert result.severity == Severity.FAIL
    assert any("no revolute/prismatic joints" in m.message for m in result.messages)


def test_malformed_limits_lower_greater_than_upper_skipped_with_warn(tmp_path):
    urdf = tmp_path / "malformed.urdf"
    urdf.write_text(dedent("""\
        <robot name="malformed">
          <joint name="shoulder_pan" type="revolute">
            <limit lower="1.0" upper="-1.0" velocity="1.0" effort="1.0"/>
          </joint>
        </robot>
    """))
    actions = [[0.05, 0.0, -0.5, 0.0, 0.0, 0.5] for _ in range(5)]
    ds = load_local(_so101_dataset(tmp_path, actions))
    ds.robot_urdf = urdf

    result = check_kinematics(ds)

    assert result.severity == Severity.WARN
    assert any("malformed limits" in m.message for m in result.messages)
    assert not any("outside physical joint limits" in m.message for m in result.messages)
