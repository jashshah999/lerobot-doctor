"""Check: Kinematic Feasibility.

Unlike the existing Action Quality check (clipping/frozen/jump detection against the
dataset's *own* statistics), this checks commanded actions against a robot's *real*
physical joint limits (position, and velocity where declared) pulled from a URDF.
A dataset can look statistically clean -- no clipping, no jumps, no NaNs -- and still
command a joint angle the robot physically cannot reach; that's what this check catches.

Limits come from one of two sources:
  1. --urdf PATH, if the user passed one (works for any robot)
  2. a small built-in registry keyed by the dataset's robot_type (see robots/limits.py)
If an explicit --urdf is unreadable or has no usable joints, the check FAILs (the user
asked for it and got nothing). If neither source is available, the check is skipped
with an informational message, not a warning: most datasets have no registry entry
and a default run should not warn on them.
"""

from __future__ import annotations

import numpy as np

from lerobot_doctor.dataset_loader import LoadedDataset
from lerobot_doctor.robots.limits import JointLimit, lookup_known_robot
from lerobot_doctor.robots.urdf import parse_urdf_limits
from lerobot_doctor.runner import CheckResult, Severity

# Position violations beyond this fraction of frames fail the check outright;
# below it, but still present, they warn. A handful of frames can be noise
# (e.g. teleop overshoot near a joint's soft limit); a sustained pattern can't.
_FAIL_FRACTION = 0.01


def _resolve_limits(dataset: LoadedDataset) -> tuple[dict[str, JointLimit] | None, str, bool]:
    """Returns (limits, source_description, from_explicit_urdf).

    limits is None if nothing resolved; from_explicit_urdf tells the caller whether
    the user explicitly asked for a URDF (failures should then FAIL, not skip).
    """
    urdf_path = getattr(dataset, "robot_urdf", None)
    if urdf_path is not None:
        try:
            limits = parse_urdf_limits(urdf_path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as a check message
            return None, f"failed to parse --urdf {urdf_path}: {exc}", True
        if not limits:
            return None, f"--urdf {urdf_path} contained no revolute/prismatic joints with <limit> tags", True
        return limits, f"--urdf {urdf_path}", True

    robot_type = dataset.info.robot_type if dataset.info else None
    known = lookup_known_robot(robot_type)
    if known is not None:
        return known, f"built-in registry for robot_type={robot_type!r}", False

    return None, f"no --urdf given and robot_type={robot_type!r} is not in the built-in registry", False


def _action_names(dataset: LoadedDataset) -> list[str] | None:
    if dataset.info is None:
        return None
    action_feature = dataset.info.features.get("action")
    if not action_feature:
        return None
    names = action_feature.get("names")
    if not names or not isinstance(names, list):
        return None
    return [str(n) for n in names]


def _strip_suffix(name: str) -> str:
    """"shoulder_pan.pos" -> "shoulder_pan" -- LeRobot action names are often
    "<joint>.<kind>" (.pos, .vel); URDF joint names are bare."""
    return name.split(".")[0]


def _dim_kind(name: str) -> str:
    """"shoulder_pan.pos" -> "pos"; bare joint names count as position dims."""
    parts = name.split(".", 1)
    return parts[1] if len(parts) > 1 else "pos"


def check_kinematics(dataset: LoadedDataset) -> CheckResult:
    result = CheckResult(name="Kinematic Feasibility", severity=Severity.PASS)

    if not dataset.episodes_data:
        result.warn("No episode data loaded, skipping kinematics check")
        return result

    limits, source, explicit_urdf = _resolve_limits(dataset)
    if limits is None:
        if explicit_urdf:
            result.fail(source)
        else:
            result.pass_(f"Skipped: {source}")
        return result

    action_names = _action_names(dataset)
    if action_names is None:
        result.warn("No named action features in this dataset; cannot map actions to joints, skipping")
        return result

    position_dims: dict[int, tuple[str, JointLimit]] = {}
    velocity_dims: dict[int, tuple[str, JointLimit]] = {}
    unmatched: list[str] = []
    for i, name in enumerate(action_names):
        joint = _strip_suffix(name)
        if joint not in limits:
            unmatched.append(name)
            continue
        limit = limits[joint]
        kind = _dim_kind(name)
        if kind == "pos":
            if limit.lower > limit.upper:
                result.warn(
                    f"{name}: malformed limits in {source} (lower {limit.lower:.3f} > upper "
                    f"{limit.upper:.3f}); skipping this dim"
                )
            else:
                position_dims[i] = (name, limit)
        elif kind == "vel":
            velocity_dims[i] = (name, limit)
        else:
            # .effort or other suffixes: no limit semantics implemented for them
            unmatched.append(name)

    if not position_dims and not velocity_dims:
        result.warn(f"None of the action names {action_names} matched joints in {source}; skipping")
        return result

    n_checked = len(position_dims) + len(velocity_dims)
    result.pass_(f"Checking {n_checked}/{len(action_names)} action dims against kinematic limits from {source}")

    fps = dataset.info.fps if dataset.info and dataset.info.fps else None

    for dim, (name, limit) in position_dims.items():
        _check_position_dim(dataset, dim, name, limit, fps, result)

    for dim, (name, limit) in velocity_dims.items():
        _check_velocity_dim(dataset, dim, name, limit, result)

    if unmatched:
        result.warn(f"No kinematic limit info for action dims (not in {source}): {unmatched}")

    return result


def _collect_dim_values(dataset: LoadedDataset, dim: int) -> list[tuple[int, np.ndarray]]:
    """Per-episode values for one action dim. Episodes whose action column cannot be
    converted to a float array (ragged/non-numeric rows) are skipped here; the
    consistency and actions checks report that corruption."""
    per_episode_values: list[tuple[int, np.ndarray]] = []
    for ep in dataset.episodes_data:
        if "action" not in ep.columns:
            continue
        try:
            actions = np.asarray(ep.columns["action"], dtype=np.float64)
        except (ValueError, TypeError):
            continue
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        if dim >= actions.shape[1]:
            continue
        per_episode_values.append((ep.episode_index, actions[:, dim]))
    return per_episode_values


def _check_position_dim(
    dataset: LoadedDataset,
    dim: int,
    name: str,
    limit: JointLimit,
    fps: float | None,
    result: CheckResult,
) -> None:
    per_episode_values = _collect_dim_values(dataset, dim)
    if not per_episode_values:
        return

    all_values = np.concatenate([v for _, v in per_episode_values])
    finite = np.isfinite(all_values)
    n_finite = int(finite.sum())
    if n_finite == 0:
        result.warn(f"{name}: no finite action values, skipping kinematics for this dim")
        return

    # NaN/Inf are excluded from the kinematics math entirely (the actions check
    # reports non-finite values); they must neither count as violations nor
    # poison the units-mismatch median below.
    out_of_bounds = finite & ((all_values < limit.lower) | (all_values > limit.upper))
    fraction = float(out_of_bounds.sum() / n_finite)

    # Some LeRobot robot configs (notably several SO-100/SO-101 setups) store
    # actions as a normalized percentage (~[-100, 100]) instead of radians. That
    # produces near-100% "violations" against a URDF's radian limits -- not a
    # real kinematic problem, just a unit mismatch. Distinguish the two using the
    # *typical* magnitude of commanded values (median |value|, robust to a
    # constant-action dataset or a handful of outliers) rather than the raw
    # range: a real violation still commands values on the same order of
    # magnitude as the limit; a unit mismatch is off by an order of magnitude
    # or more.
    limit_scale = max(abs(limit.lower), abs(limit.upper))
    observed_scale = float(np.median(np.abs(all_values[finite])))
    if fraction > 0.5 and limit_scale > 0 and observed_scale > 10 * limit_scale:
        finite_values = all_values[finite]
        result.warn(
            f"{name}: {fraction:.0%} of values fall outside [{limit.lower:.3f}, {limit.upper:.3f}], but the "
            f"observed range ({finite_values.min():.2f} to {finite_values.max():.2f}) is much wider than the limit's "
            f"range -- this looks like a units mismatch (e.g. normalized/percent actions vs. URDF radians), "
            f"not a real kinematic violation. Skipping this dim; verify the dataset's action units before trusting "
            f"a kinematics check against this URDF."
        )
        return

    if out_of_bounds.any():
        bad_idx = int(np.argmax(out_of_bounds))
        worst_episode, worst_value = None, None
        seen = 0
        for ep_idx, values in per_episode_values:
            if seen <= bad_idx < seen + len(values):
                worst_episode, worst_value = ep_idx, float(values[bad_idx - seen])
                break
            seen += len(values)
        example = f" (e.g. episode {worst_episode}: commanded {worst_value:.3f}, limit [{limit.lower:.3f}, {limit.upper:.3f}])" if worst_episode is not None else ""
        n_bad = int(out_of_bounds.sum())
        message = f"{name}: {n_bad}/{n_finite} frames ({fraction:.1%}) outside physical joint limits [{limit.lower:.3f}, {limit.upper:.3f}]{example}"
        if fraction > _FAIL_FRACTION:
            result.fail(message)
        else:
            result.warn(message)

    if limit.velocity is not None and fps:
        velocity_violations = 0
        for _, values in per_episode_values:
            if len(values) < 2:
                continue
            implied_velocity = np.abs(np.diff(values)) * fps
            implied_velocity = implied_velocity[np.isfinite(implied_velocity)]
            velocity_violations += int((implied_velocity > limit.velocity).sum())
        if velocity_violations:
            result.warn(
                f"{name}: {velocity_violations} frame-to-frame transitions imply a velocity above the joint's "
                f"declared limit ({limit.velocity:.2f} rad/s) -- may be unreachable, or fps/units may not match the URDF"
            )


def _check_velocity_dim(
    dataset: LoadedDataset,
    dim: int,
    name: str,
    limit: JointLimit,
    result: CheckResult,
) -> None:
    """A ".vel" action dim commands velocities directly; compare against the joint's
    declared velocity limit, never its position bounds."""
    if limit.velocity is None:
        result.warn(f"{name}: no velocity limit declared for this joint, skipping this dim")
        return

    per_episode_values = _collect_dim_values(dataset, dim)
    if not per_episode_values:
        return

    all_values = np.concatenate([v for _, v in per_episode_values])
    finite = np.isfinite(all_values)
    n_finite = int(finite.sum())
    if n_finite == 0:
        result.warn(f"{name}: no finite action values, skipping kinematics for this dim")
        return

    violations = finite & (np.abs(all_values) > limit.velocity)
    if violations.any():
        n_bad = int(violations.sum())
        fraction = float(n_bad / n_finite)
        message = (
            f"{name}: {n_bad}/{n_finite} frames ({fraction:.1%}) command a velocity above the joint's "
            f"declared limit ({limit.velocity:.2f} rad/s)"
        )
        if fraction > _FAIL_FRACTION:
            result.fail(message)
        else:
            result.warn(message)
