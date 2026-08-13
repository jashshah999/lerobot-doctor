"""Minimal URDF joint-limit parser.

Only extracts what check_kinematics needs (revolute/prismatic joint names and their
<limit> tags) via the standard library's xml.etree — no new dependency, and no need
to resolve mesh/visual assets that a full URDF parser would require.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from lerobot_doctor.robots.limits import JointLimit

_LIMITED_JOINT_TYPES = {"revolute", "prismatic"}


def parse_urdf_limits(urdf_path: Path) -> dict[str, JointLimit]:
    """Parse joint position/velocity/effort limits from a URDF file.

    Joints without a <limit> tag (fixed joints, continuous joints with no bound) are
    skipped rather than raising — a URDF describing more than just the arm's revolute
    joints is normal and not an error.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    limits: dict[str, JointLimit] = {}
    for joint in root.findall("joint"):
        joint_type = joint.get("type")
        name = joint.get("name")
        if joint_type not in _LIMITED_JOINT_TYPES or not name:
            continue

        limit_el = joint.find("limit")
        if limit_el is None:
            continue

        lower = limit_el.get("lower")
        upper = limit_el.get("upper")
        if lower is None or upper is None:
            continue

        limits[name] = JointLimit(
            lower=float(lower),
            upper=float(upper),
            velocity=float(limit_el.get("velocity")) if limit_el.get("velocity") is not None else None,
            effort=float(limit_el.get("effort")) if limit_el.get("effort") is not None else None,
        )

    return limits
