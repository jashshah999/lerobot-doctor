"""Orchestrates diagnostic checks and collects results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from lerobot_doctor.dataset_loader import LoadedDataset


def _get_all_checks():
    from lerobot_doctor.checks.metadata import check_metadata
    from lerobot_doctor.checks.temporal import check_temporal
    from lerobot_doctor.checks.actions import check_actions
    from lerobot_doctor.checks.videos import check_videos
    from lerobot_doctor.checks.statistics import check_statistics
    from lerobot_doctor.checks.episodes import check_episodes
    from lerobot_doctor.checks.consistency import check_consistency
    from lerobot_doctor.checks.training import check_training
    from lerobot_doctor.checks.anomalies import check_anomalies
    from lerobot_doctor.checks.portability import check_portability
    from lerobot_doctor.checks.per_episode import check_per_episode
    return {
        "metadata": check_metadata,
        "temporal": check_temporal,
        "actions": check_actions,
        "videos": check_videos,
        "statistics": check_statistics,
        "episodes": check_episodes,
        "consistency": check_consistency,
        "training": check_training,
        "anomalies": check_anomalies,
        "portability": check_portability,
        "per_episode": check_per_episode,
    }


class Severity(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class CheckMessage:
    severity: Severity
    message: str


@dataclass
class CheckResult:
    name: str
    severity: Severity  # Overall severity (worst of all messages)
    messages: list[CheckMessage] = field(default_factory=list)

    def pass_(self, msg: str):
        self.messages.append(CheckMessage(Severity.PASS, msg))

    def warn(self, msg: str):
        self.messages.append(CheckMessage(Severity.WARN, msg))
        if self.severity == Severity.PASS:
            self.severity = Severity.WARN

    def fail(self, msg: str):
        self.messages.append(CheckMessage(Severity.FAIL, msg))
        self.severity = Severity.FAIL


@dataclass
class DiagnosticReport:
    dataset_path: str
    dataset_name: str | None = None
    codebase_version: str | None = None
    total_episodes: int | None = None
    total_frames: int | None = None
    fps: int | None = None
    results: list[CheckResult] = field(default_factory=list)

    @property
    def overall_severity(self) -> Severity:
        if any(r.severity == Severity.FAIL for r in self.results):
            return Severity.FAIL
        if any(r.severity == Severity.WARN for r in self.results):
            return Severity.WARN
        return Severity.PASS

    @property
    def summary_counts(self) -> dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for r in self.results:
            counts[r.severity.value] += 1
        return counts


def run_checks(
    dataset: LoadedDataset,
    checks: list[str] | None = None,
    verbose: bool = False,
) -> DiagnosticReport:
    """Run selected checks on a loaded dataset."""
    report = DiagnosticReport(dataset_path=str(dataset.root))

    if dataset.info is not None:
        report.codebase_version = dataset.info.codebase_version
        report.total_episodes = dataset.info.total_episodes
        report.total_frames = dataset.info.total_frames
        report.fps = dataset.info.fps
        report.dataset_name = dataset.root.name

    all_checks = _get_all_checks()
    check_names = checks if checks else list(all_checks.keys())

    for name in check_names:
        if name not in all_checks:
            result = CheckResult(name=name, severity=Severity.FAIL)
            result.fail(f"Unknown check: {name}")
            report.results.append(result)
            continue
        check_fn = all_checks[name]
        result = check_fn(dataset)
        report.results.append(result)

    return report
