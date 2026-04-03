"""Check 1: Metadata & Format Compliance."""

from __future__ import annotations

from pathlib import Path

from lerobot_doctor.dataset_loader import LoadedDataset
from lerobot_doctor.runner import CheckResult, Severity

REQUIRED_INFO_FIELDS = [
    "codebase_version", "fps", "total_episodes", "total_frames",
    "features", "data_path",
]


def check_metadata(dataset: LoadedDataset) -> CheckResult:
    result = CheckResult(name="Metadata & Format Compliance", severity=Severity.PASS)

    # Check info.json loaded
    if dataset.info is None:
        result.fail(dataset.info_error or "info.json could not be loaded")
        return result

    info = dataset.info
    result.pass_("info.json loaded successfully")

    # Check required fields
    missing = [f for f in REQUIRED_INFO_FIELDS if f not in info.raw]
    if missing:
        result.fail(f"info.json missing required fields: {missing}")
    else:
        result.pass_("All required fields present in info.json")

    # Check codebase version
    if info.codebase_version and not info.codebase_version.startswith("v3"):
        result.warn(f"Codebase version is {info.codebase_version}, expected v3.x")

    # Check fps is positive
    if info.fps is not None and info.fps <= 0:
        result.fail(f"fps must be positive, got {info.fps}")

    # Check data files exist
    _check_data_files(dataset, result)

    # Check episode metadata files
    _check_episode_meta(dataset, result)

    # Check tasks.parquet
    if info.total_tasks and info.total_tasks > 0:
        if dataset.tasks is None:
            result.fail(f"total_tasks={info.total_tasks} but tasks.parquet not found")
        elif len(dataset.tasks) != info.total_tasks:
            result.warn(
                f"total_tasks={info.total_tasks} but tasks.parquet has {len(dataset.tasks)} rows"
            )

    # Check total_frames vs actual
    if dataset.episodes_data:
        actual_frames = sum(ep.length for ep in dataset.episodes_data)
        if info.total_frames is not None and actual_frames != info.total_frames:
            # Only fail if we loaded all episodes
            if len(dataset.episodes_data) == (info.total_episodes or 0):
                result.fail(
                    f"total_frames={info.total_frames} but actual frame count is {actual_frames}"
                )

    # Check total_episodes vs actual episode meta
    if dataset.episodes_meta:
        if info.total_episodes is not None and len(dataset.episodes_meta) != info.total_episodes:
            result.fail(
                f"total_episodes={info.total_episodes} but found {len(dataset.episodes_meta)} episode metadata entries"
            )
        else:
            result.pass_(f"Episode count matches: {info.total_episodes}")

    # Check feature columns exist in data
    if dataset.episodes_data:
        data_cols = set(dataset.episodes_data[0].columns.keys())
        for feat_name in info.features:
            if feat_name not in data_cols:
                # Video features won't be in parquet
                feat_dtype = info.features[feat_name].get("dtype", "")
                if feat_dtype != "video":
                    result.warn(f"Feature '{feat_name}' declared in info.json but not in data parquet")

    return result


def _check_data_files(dataset: LoadedDataset, result: CheckResult):
    data_dir = dataset.root / "data"
    if not data_dir.exists():
        result.fail("data/ directory not found")
        return
    parquet_files = list(data_dir.rglob("*.parquet"))
    if not parquet_files:
        result.fail("No parquet files found in data/")
    else:
        result.pass_(f"Found {len(parquet_files)} data parquet file(s)")


def _check_episode_meta(dataset: LoadedDataset, result: CheckResult):
    episodes_dir = dataset.root / "meta" / "episodes"
    if not episodes_dir.exists():
        result.warn("meta/episodes/ directory not found")
        return
    parquet_files = list(episodes_dir.rglob("*.parquet"))
    if not parquet_files:
        result.warn("No parquet files found in meta/episodes/")
    else:
        result.pass_(f"Found {len(parquet_files)} episode metadata file(s)")
