"""Check 10: Portability -- will this dataset work on other machines / in CI?"""

from __future__ import annotations

import os

from lerobot_doctor.dataset_loader import LoadedDataset
from lerobot_doctor.runner import CheckResult, Severity


def check_portability(dataset: LoadedDataset) -> CheckResult:
    result = CheckResult(name="Portability", severity=Severity.PASS)

    if dataset.info is None:
        result.fail("Cannot check portability: info.json not loaded")
        return result

    root = dataset.root

    # Check total dataset size
    total_size = 0
    for f in root.rglob("*"):
        if f.is_file():
            total_size += f.stat().st_size
    size_gb = total_size / (1024 ** 3)
    size_mb = total_size / (1024 ** 2)

    if size_gb > 100:
        result.warn(f"Dataset is {size_gb:.1f} GB -- very large, consider using --max-episodes for checks")
    elif size_gb > 10:
        result.warn(f"Dataset is {size_gb:.1f} GB -- large dataset")
    else:
        result.pass_(f"Dataset size: {size_mb:.0f} MB")

    # Check for absolute paths in data_path / video_path templates
    if dataset.info.data_path and os.path.isabs(dataset.info.data_path):
        result.fail(
            f"data_path uses absolute path: {dataset.info.data_path} -- "
            f"dataset won't be portable to other machines"
        )

    if dataset.info.video_path and os.path.isabs(dataset.info.video_path):
        result.fail(
            f"video_path uses absolute path: {dataset.info.video_path} -- "
            f"dataset won't be portable to other machines"
        )

    # Check for symlinks (can break on different OS/machines)
    # Skip this check for HF cache dirs which always use symlinks
    if dataset.is_local:
        symlinks = [f for f in root.rglob("*") if f.is_symlink()]
        if symlinks:
            result.warn(
                f"{len(symlinks)} symlink(s) found -- may break on different machines: "
                f"{[str(s.relative_to(root)) for s in symlinks[:5]]}"
            )

    # Check file permissions (warn if not readable)
    unreadable = []
    for f in root.rglob("*"):
        if f.is_file() and not os.access(f, os.R_OK):
            unreadable.append(str(f.relative_to(root)))
    if unreadable:
        result.fail(f"{len(unreadable)} unreadable file(s): {unreadable[:5]}")

    # Check for non-standard file extensions in data/
    data_dir = root / "data"
    if data_dir.exists():
        non_parquet = [
            f.name for f in data_dir.rglob("*")
            if f.is_file() and f.suffix != ".parquet"
        ]
        if non_parquet:
            result.warn(f"Non-parquet files in data/: {non_parquet[:5]}")

    # Check for HF Hub compatibility
    _check_hf_compatibility(dataset, result)

    return result


def _check_hf_compatibility(dataset: LoadedDataset, result: CheckResult):
    """Check if dataset structure is compatible with HuggingFace Hub hosting."""
    root = dataset.root

    # Large individual files (HF Hub limit is 50GB per file, but recommends <5GB)
    large_files = []
    for f in root.rglob("*"):
        if f.is_file():
            size_gb = f.stat().st_size / (1024 ** 3)
            if size_gb > 5:
                large_files.append((str(f.relative_to(root)), f"{size_gb:.1f} GB"))
    if large_files:
        result.warn(
            f"{len(large_files)} file(s) over 5GB (may need git-lfs for HF Hub): "
            f"{large_files[:3]}"
        )

    # Check that data_path template uses standard LeRobot format
    if dataset.info.data_path:
        expected_pattern = "data/chunk-{episode_chunk:03d}/file-{episode_index:03d}.parquet"
        # Don't enforce exact format, just check it's a relative template
        if "{" not in dataset.info.data_path:
            result.warn(
                f"data_path '{dataset.info.data_path}' is not a template -- "
                f"expected format like '{expected_pattern}'"
            )
