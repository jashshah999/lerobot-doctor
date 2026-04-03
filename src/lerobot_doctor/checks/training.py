"""Check 8: Training Readiness -- will this dataset work with common policies?"""

from __future__ import annotations

import json

import numpy as np

from lerobot_doctor.dataset_loader import LoadedDataset
from lerobot_doctor.runner import CheckResult, Severity


# Common delta_timestamps configs for popular policies
POLICY_CONFIGS = {
    "ACT": {
        "chunk_size": 100,
        "description": "Action Chunking Transformer",
        "needs_actions": True,
        "needs_state": True,
    },
    "Diffusion": {
        "chunk_size": 16,
        "description": "Diffusion Policy",
        "needs_actions": True,
        "needs_state": True,
    },
    "VLA": {
        "chunk_size": 1,
        "description": "Vision-Language-Action",
        "needs_actions": True,
        "needs_images": True,
    },
}


def check_training(dataset: LoadedDataset) -> CheckResult:
    result = CheckResult(name="Training Readiness", severity=Severity.PASS)

    if dataset.info is None:
        result.fail("Cannot check training readiness: info.json not loaded")
        return result

    fps = dataset.info.fps
    features = dataset.info.features

    # Identify what's available
    has_actions = any(k.startswith("action") for k in features)
    has_state = any("state" in k for k in features)
    has_images = any(features[k].get("dtype") in ("image", "video") for k in features)

    if has_actions:
        result.pass_("Action features found")
    else:
        result.fail("No action features found -- cannot train any policy")
        return result

    if has_state:
        result.pass_("State observation features found")
    if has_images:
        result.pass_("Image/video features found")

    # Check policy compatibility
    for policy_name, config in POLICY_CONFIGS.items():
        issues = []
        if config["needs_actions"] and not has_actions:
            issues.append("no action features")
        if config.get("needs_state") and not has_state:
            issues.append("no state features")
        if config.get("needs_images") and not has_images:
            issues.append("no image/video features")

        if issues:
            result.warn(f"{policy_name} ({config['description']}): {', '.join(issues)}")

    # Check normalization readiness
    _check_normalization(dataset, result)

    # Check action space properties
    _check_action_space(dataset, result)

    # Check delta_timestamps compatibility
    if fps:
        _check_delta_timestamps(dataset, fps, result)

    return result


def _check_normalization(dataset: LoadedDataset, result: CheckResult):
    """Check if dataset has valid statistics for normalization."""
    stats_path = dataset.root / "meta" / "stats.json"
    if not stats_path.exists():
        result.warn("No stats.json -- normalization will need to be computed before training")
        return

    try:
        stats = json.loads(stats_path.read_text())
    except (json.JSONDecodeError, OSError):
        result.fail("stats.json exists but is invalid -- training will fail on normalization")
        return

    # Check action stats specifically
    action_keys = [k for k in stats if k.startswith("action")]
    if not action_keys:
        result.warn("stats.json has no action statistics -- normalization may fail")
        return

    for key in action_keys:
        stat = stats[key]
        required = ["mean", "std"]
        missing = [r for r in required if r not in stat]
        if missing:
            result.warn(f"stats.json[{key}] missing {missing} -- some normalizers may fail")
            continue

        # Check for zero std (will cause div-by-zero in normalization)
        std = np.array(stat["std"])
        zero_dims = (std == 0).sum()
        if zero_dims > 0:
            result.warn(
                f"stats.json[{key}]: {zero_dims} dimension(s) have zero std -- "
                f"normalization will produce NaN/Inf"
            )

    result.pass_("Normalization statistics available for actions")


def _check_action_space(dataset: LoadedDataset, result: CheckResult):
    """Check action space properties that affect training."""
    if not dataset.episodes_data:
        return

    for ep in dataset.episodes_data[:5]:  # sample
        for col_name, vals in ep.columns.items():
            if not col_name.startswith("action"):
                continue
            try:
                arr = np.array(vals, dtype=np.float64)
            except (ValueError, TypeError):
                continue
            if arr.ndim < 2:
                continue

            n_dims = arr.shape[1]
            if n_dims > 20:
                result.warn(
                    f"{col_name}: {n_dims} dimensions is unusually large -- "
                    f"verify this is correct"
                )
            break  # only check first episode
        break


def _check_delta_timestamps(dataset: LoadedDataset, fps: int, result: CheckResult):
    """Check if common delta_timestamps would work."""
    if not dataset.episodes_data:
        return

    min_ep_len = min(ep.length for ep in dataset.episodes_data)
    interval = 1.0 / fps

    # Common delta_timestamps: looking back 1 frame, forward N frames
    for n_future in [1, 10, 16, 50, 100]:
        required_frames = n_future + 1  # current + future
        if min_ep_len < required_frames:
            if n_future <= 16:  # only warn for common small horizons
                result.warn(
                    f"Shortest episode ({min_ep_len} frames) is too short for "
                    f"delta_timestamps with {n_future}-step prediction horizon"
                )
            break
