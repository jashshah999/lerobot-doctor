"""Check 13: Episode Identity -- exact duplicates, conflicting labels, split leakage."""

from __future__ import annotations

import hashlib

import numpy as np

from lerobot_doctor.dataset_loader import LoadedDataset
from lerobot_doctor.runner import CheckResult, Severity, preview

# Bookkeeping columns: differ between copies of the same recording.
BOOKKEEPING_COLUMNS = {"timestamp", "frame_index", "episode_index", "index"}
# Label columns: compared across identical evidence rather than hashed into it.
LABEL_COLUMNS = {"task_index", "next.reward", "next.success"}


def _column_bytes(vals) -> bytes:
    arr = np.asarray(vals)
    if arr.dtype.kind in "biuf":
        # Normalize dtype so float32/float64 copies of the same values hash equal.
        return np.ascontiguousarray(arr, dtype=np.float64).tobytes() + repr(arr.shape).encode()
    return repr(list(vals) if not isinstance(vals, list) else vals).encode()


def evidence_hashes(dataset: LoadedDataset) -> dict[int, str]:
    """SHA-256 of each episode's model-visible parquet data, keyed by episode_index.

    Covers every column except bookkeeping (timestamps, indices) and labels
    (task, reward, success). Video frames are not part of the hash.
    """
    hashes = {}
    for ep in dataset.episodes_data:
        h = hashlib.sha256()
        h.update(str(ep.length).encode())
        for name in sorted(ep.columns):
            if name in BOOKKEEPING_COLUMNS or name in LABEL_COLUMNS:
                continue
            h.update(name.encode())
            h.update(_column_bytes(ep.columns[name]))
        hashes[ep.episode_index] = h.hexdigest()
    return hashes


def _labels(dataset: LoadedDataset, ep) -> tuple:
    task_names = {}
    for t in dataset.tasks or []:
        if "task_index" in t:
            task_names[t["task_index"]] = t.get("task", t["task_index"])
    parts = []
    if "task_index" in ep.columns:
        idx = sorted({int(v) for v in np.asarray(ep.columns["task_index"]).reshape(-1)})
        parts.append(("task", tuple(str(task_names.get(i, i)) for i in idx)))
    for name in ("next.reward", "next.success"):
        if name in ep.columns:
            parts.append((name, _column_bytes(ep.columns[name])))
    return tuple(parts)


def _describe_labels(labels: tuple) -> str:
    for name, value in labels:
        if name == "task":
            return "/".join(value) if value else "(none)"
    return "(no task)"


def parse_splits(splits: dict) -> dict[int, str]:
    """Map episode_index -> split name from info.json ``splits`` ("start:end" ranges)."""
    membership = {}
    for name, spec in (splits or {}).items():
        ranges = spec if isinstance(spec, list) else [spec]
        for r in ranges:
            if isinstance(r, int):
                membership[r] = name
                continue
            try:
                start, end = (int(x) for x in str(r).split(":"))
            except ValueError:
                continue
            for i in range(start, end):
                membership[i] = name
    return membership


def check_identity(dataset: LoadedDataset) -> CheckResult:
    result = CheckResult(name="Episode Identity", severity=Severity.PASS)

    if len(dataset.episodes_data) < 2:
        result.pass_("Fewer than 2 episodes loaded -- nothing to compare")
        return result

    groups: dict[str, list[int]] = {}
    for ep_idx, digest in evidence_hashes(dataset).items():
        groups.setdefault(digest, []).append(ep_idx)
    dup_groups = sorted(sorted(g) for g in groups.values() if len(g) > 1)

    if not dup_groups:
        result.pass_(f"All {len(dataset.episodes_data)} episodes have distinct content")
        return result

    n_extra = sum(len(g) - 1 for g in dup_groups)
    result.warn(
        f"{n_extra} episode(s) are exact copies of another episode's observations/actions: "
        f"{preview(dup_groups, 5, dataset.no_aggregate)}"
    )

    episodes = {ep.episode_index: ep for ep in dataset.episodes_data}
    conflicts = []
    for group in dup_groups:
        labels = {i: _labels(dataset, episodes[i]) for i in group}
        if len(set(labels.values())) > 1:
            conflicts.append((group, labels))
    if conflicts:
        shown = [
            "{" + ", ".join(f"ep {i}: {_describe_labels(lab)}" for i, lab in labels.items()) + "}"
            for _, labels in conflicts
        ]
        result.warn(
            f"{len(conflicts)} duplicate group(s) carry conflicting labels (task/reward/success) "
            f"for identical evidence: {preview(shown, 5, dataset.no_aggregate)}"
        )

    membership = parse_splits(dataset.info.splits if dataset.info else {})
    if len(set(membership.values())) > 1:
        leaks = []
        for group in dup_groups:
            names = {membership.get(i) for i in group} - {None}
            if len(names) > 1:
                leaks.append((group, sorted(names)))
        if leaks:
            result.fail(
                f"{len(leaks)} duplicate group(s) appear in more than one split "
                f"(train/eval leakage): {preview(leaks, 5, dataset.no_aggregate)}"
            )

    return result
