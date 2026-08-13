"""Known kinematic limits (position/velocity/effort) for common LeRobot-compatible robots.

Unlike stats.json (dataset-internal empirical min/max), these come from each robot's
real URDF, so they catch commands that are physically infeasible even if every
dataset that happens to exist for that robot never actually reached them.

Values are copied directly from the joint <limit> tags of the manufacturer/community
URDFs, not re-derived or guessed:
  - SO-100: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO100/so100.urdf
  - SO-101: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf

Contributions adding more robots (Koch, ALOHA/ViperX, WidowX, ...) are welcome —
please cite the exact URDF source, the same way the two entries below do.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointLimit:
    lower: float
    upper: float
    velocity: float | None = None  # rad/s (or m/s for prismatic joints)
    effort: float | None = None


# Joint names match the <joint name="..."> values in each URDF, which is also what
# LeRobot action feature names are typically built from (e.g. "shoulder_pan.pos").
KNOWN_ROBOTS: dict[str, dict[str, JointLimit]] = {
    "so100": {
        "shoulder_pan": JointLimit(lower=-2.0, upper=2.0, velocity=1.0, effort=35.0),
        "shoulder_lift": JointLimit(lower=0.0, upper=3.5, velocity=1.0, effort=35.0),
        "elbow_flex": JointLimit(lower=-3.14158, upper=0.0, velocity=1.0, effort=35.0),
        "wrist_flex": JointLimit(lower=-2.5, upper=1.2, velocity=1.0, effort=35.0),
        "wrist_roll": JointLimit(lower=-3.14158, upper=3.14158, velocity=1.0, effort=35.0),
        "gripper": JointLimit(lower=-0.2, upper=2.0, velocity=1.0, effort=35.0),
    },
    "so101": {
        "shoulder_pan": JointLimit(lower=-1.91986, upper=1.91986, velocity=10.0, effort=10.0),
        "shoulder_lift": JointLimit(lower=-1.74533, upper=1.74533, velocity=10.0, effort=10.0),
        "elbow_flex": JointLimit(lower=-1.69, upper=1.69, velocity=10.0, effort=10.0),
        "wrist_flex": JointLimit(lower=-1.65806, upper=1.65806, velocity=10.0, effort=10.0),
        "wrist_roll": JointLimit(lower=-2.74385, upper=2.84121, velocity=10.0, effort=10.0),
        "gripper": JointLimit(lower=-0.174533, upper=1.74533, velocity=10.0, effort=10.0),
    },
}


def normalize_robot_type(raw: str) -> str:
    """Map a dataset's raw robot_type string to a KNOWN_ROBOTS key.

    Real datasets store things like "so100_follower" or "SO101_Leader" rather than
    a bare "so100"/"so101" — LeRobot's teleop setups distinguish follower/leader arms
    even though they're kinematically identical for this purpose.
    """
    normalized = raw.strip().lower()
    for suffix in ("_follower", "_leader", "-follower", "-leader"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    return normalized


def lookup_known_robot(robot_type: str | None) -> dict[str, JointLimit] | None:
    if not robot_type:
        return None
    return KNOWN_ROBOTS.get(normalize_robot_type(robot_type))
