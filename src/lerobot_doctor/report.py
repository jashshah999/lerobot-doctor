"""Format diagnostic reports for terminal and JSON output."""

from __future__ import annotations

import json

from lerobot_doctor import __version__
from lerobot_doctor.runner import CheckResult, DiagnosticReport, Severity


SEVERITY_COLORS = {
    Severity.PASS: "green",
    Severity.WARN: "yellow",
    Severity.FAIL: "red",
}

SEVERITY_SYMBOLS = {
    Severity.PASS: "PASS",
    Severity.WARN: "WARN",
    Severity.FAIL: "FAIL",
}


def print_report(report: DiagnosticReport, verbose: bool = False):
    """Print a rich-formatted report to terminal."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        console = Console()
    except ImportError:
        _print_plain(report, verbose)
        return

    # Header
    header = Text()
    header.append(f"lerobot-doctor v{__version__}", style="bold")
    header.append(" -- Dataset Quality Report\n")
    header.append(f"Dataset: {report.dataset_path}")
    if report.codebase_version:
        header.append(f" ({report.codebase_version})")
    header.append("\n")
    parts = []
    if report.total_episodes is not None:
        parts.append(f"Episodes: {report.total_episodes}")
    if report.total_frames is not None:
        parts.append(f"Frames: {report.total_frames:,}")
    if report.fps is not None:
        parts.append(f"FPS: {report.fps}")
    if parts:
        header.append(" | ".join(parts))

    console.print(Panel(header, border_style="blue"))
    console.print()

    # Results
    for check_result in report.results:
        sev = check_result.severity
        color = SEVERITY_COLORS[sev]
        symbol = SEVERITY_SYMBOLS[sev]
        console.print(f"[bold {color}][{symbol}][/bold {color}] {check_result.name}")

        for msg in check_result.messages:
            msg_color = SEVERITY_COLORS[msg.severity]
            prefix = "  "
            if msg.severity == Severity.PASS:
                if verbose:
                    console.print(f"{prefix}[{msg_color}]+ {msg.message}[/{msg_color}]")
            else:
                console.print(f"{prefix}[{msg_color}]- {msg.message}[/{msg_color}]")
        console.print()

    # Summary
    counts = report.summary_counts
    summary_parts = []
    for sev in [Severity.PASS, Severity.WARN, Severity.FAIL]:
        count = counts[sev.value]
        if count > 0:
            color = SEVERITY_COLORS[sev]
            summary_parts.append(f"[{color}]{count} {sev.value}[/{color}]")
    console.print(f"[bold]Summary:[/bold] {' | '.join(summary_parts)}")

    # Actionable fix suggestions for failures
    fixes = _get_fix_suggestions(report)
    if fixes:
        console.print()
        console.print("[bold]Suggested fixes:[/bold]")
        for fix in fixes:
            console.print(f"  [cyan]{fix}[/cyan]")


def _print_plain(report: DiagnosticReport, verbose: bool = False):
    """Fallback plain text output without rich."""
    print(f"lerobot-doctor v{__version__} -- Dataset Quality Report")
    print(f"Dataset: {report.dataset_path}")
    parts = []
    if report.total_episodes is not None:
        parts.append(f"Episodes: {report.total_episodes}")
    if report.total_frames is not None:
        parts.append(f"Frames: {report.total_frames:,}")
    if report.fps is not None:
        parts.append(f"FPS: {report.fps}")
    if parts:
        print(" | ".join(parts))
    print()

    for check_result in report.results:
        sev = SEVERITY_SYMBOLS[check_result.severity]
        print(f"[{sev}] {check_result.name}")
        for msg in check_result.messages:
            if msg.severity == Severity.PASS and not verbose:
                continue
            print(f"  - {msg.message}")
        print()

    counts = report.summary_counts
    parts = [f"{counts[s.value]} {s.value}" for s in Severity if counts[s.value] > 0]
    print(f"Summary: {' | '.join(parts)}")


FIX_PATTERNS = [
    ("NaN values", "Filter or interpolate NaN values: df.interpolate() or drop affected episodes"),
    ("Inf values", "Clip infinite values: np.clip(actions, -1e6, 1e6)"),
    ("clipping detected", "Check robot joint limits or rescale actions to avoid saturation"),
    ("frozen", "Remove episodes with stuck actuators using lerobot-edit-dataset"),
    ("non-monotonic", "Re-sort frames by timestamp or re-record affected episodes"),
    ("dropped frame", "Lower camera resolution or encoding quality during recording"),
    ("total_frames", "Regenerate metadata: update info.json total_frames to match actual data"),
    ("total_episodes", "Regenerate metadata: update info.json total_episodes count"),
    ("zero std", "Remove constant features or check sensor connections before re-recording"),
    ("zero variance", "Check sensor connections -- constant readings indicate hardware issues"),
    ("missing features", "Re-record affected episodes or fill missing features with defaults"),
    ("too short for delta_timestamps", "Use shorter prediction horizons or filter short episodes"),
    ("chunk_size", "Filter episodes shorter than your policy's chunk_size before training"),
    ("stats.json not found", "Compute stats: python -c 'from lerobot.datasets import compute_stats; ...'"),
    ("broken sensor", "Check hardware connections and re-record dataset"),
    ("stuck", "Verify robot actuator connections and re-record affected episodes"),
    ("distribution shift", "Consider splitting dataset by recording session for separate training"),
    ("duplicate", "Remove duplicate episodes to avoid overfitting"),
    ("video files missing", "Re-download dataset or re-encode videos"),
    ("absolute path", "Use relative path templates in info.json for portability"),
]


def _get_fix_suggestions(report: DiagnosticReport) -> list[str]:
    """Generate actionable fix suggestions based on failures and warnings."""
    suggestions = []
    seen = set()

    for check_result in report.results:
        if check_result.severity == Severity.PASS:
            continue
        for msg in check_result.messages:
            if msg.severity == Severity.PASS:
                continue
            msg_lower = msg.message.lower()
            for pattern, fix in FIX_PATTERNS:
                if pattern.lower() in msg_lower and fix not in seen:
                    suggestions.append(fix)
                    seen.add(fix)
                    break

    return suggestions


def report_to_json(report: DiagnosticReport) -> str:
    """Convert report to JSON string."""
    data = {
        "version": __version__,
        "dataset_path": report.dataset_path,
        "dataset_name": report.dataset_name,
        "codebase_version": report.codebase_version,
        "total_episodes": report.total_episodes,
        "total_frames": report.total_frames,
        "fps": report.fps,
        "overall_severity": report.overall_severity.value,
        "checks": [
            {
                "name": r.name,
                "severity": r.severity.value,
                "messages": [
                    {"severity": m.severity.value, "message": m.message}
                    for m in r.messages
                ],
            }
            for r in report.results
        ],
        "summary": report.summary_counts,
    }
    return json.dumps(data, indent=2)
