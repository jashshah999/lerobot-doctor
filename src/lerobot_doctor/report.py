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
        for title, code in fixes:
            console.print(f"  [cyan]# {title}[/cyan]")
            for line in code.splitlines():
                console.print(f"    [dim]{line}[/dim]")


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

    fixes = _get_fix_suggestions(report)
    if fixes:
        print()
        print("Suggested fixes:")
        for title, code in fixes:
            print(f"  # {title}")
            for line in code.splitlines():
                print(f"    {line}")


FIX_PATTERNS: list[tuple[str, str, str]] = [
    (
        "nan values",
        "Drop frames containing NaN in observations/actions",
        "import pandas as pd\ndf = pd.read_parquet('data/chunk-000/file-000.parquet')\ndf = df.dropna(subset=['action', 'observation.state'])\ndf.to_parquet('data/chunk-000/file-000.parquet')",
    ),
    (
        "inf values",
        "Clip infinite values in actions",
        "import numpy as np\nactions = np.clip(actions, -1e6, 1e6)",
    ),
    (
        "non-monotonic",
        "Re-sort frames by timestamp within each episode",
        "df = df.sort_values(['episode_index', 'timestamp']).reset_index(drop=True)",
    ),
    (
        "total_frames",
        "Regenerate info.json total_frames from data",
        "import json, pyarrow.parquet as pq, glob\ninfo = json.load(open('meta/info.json'))\ninfo['total_frames'] = sum(pq.read_metadata(p).num_rows for p in glob.glob('data/**/*.parquet', recursive=True))\njson.dump(info, open('meta/info.json', 'w'), indent=2)",
    ),
    (
        "total_episodes",
        "Regenerate info.json total_episodes from episodes meta",
        "import json, pyarrow.parquet as pq, glob\ninfo = json.load(open('meta/info.json'))\ninfo['total_episodes'] = sum(pq.read_metadata(p).num_rows for p in glob.glob('meta/episodes/**/*.parquet', recursive=True))\njson.dump(info, open('meta/info.json', 'w'), indent=2)",
    ),
    (
        "chunk_size",
        "Filter episodes shorter than your policy's chunk_size before training",
        "MIN_LEN = 100  # chunk_size for ACT/Diffusion\nkeep = [m.episode_index for m in ds.episodes_meta if m.length >= MIN_LEN]",
    ),
    (
        "zero variance",
        "Drop zero-variance features from training inputs",
        "# Exclude constant features in your policy config\ninput_features = [f for f in features if dataset_stats[f]['std'].max() > 0]",
    ),
    (
        "zero std",
        "Drop zero-std dims from normalization",
        "std = np.where(std < 1e-6, 1.0, std)  # avoid divide-by-zero",
    ),
    (
        "duplicate",
        "Drop near-duplicate episodes",
        "keep_ids = [m.episode_index for m in ds.episodes_meta if m.episode_index not in DUPLICATE_IDS]",
    ),
    (
        "absolute path",
        "Make info.json paths relative",
        'info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"',
    ),
    (
        "stats.json not found",
        "Recompute stats for the dataset",
        "# If using lerobot:\n# lerobot-compute-stats --repo-id your/dataset",
    ),
]


def _get_fix_suggestions(report: DiagnosticReport) -> list[tuple[str, str]]:
    """Return [(title, code_snippet)] for failures/warnings, deduped."""
    suggestions: list[tuple[str, str]] = []
    seen: set[str] = set()

    for check_result in report.results:
        if check_result.severity == Severity.PASS:
            continue
        for msg in check_result.messages:
            if msg.severity == Severity.PASS:
                continue
            msg_lower = msg.message.lower()
            for pattern, title, code in FIX_PATTERNS:
                if pattern in msg_lower and title not in seen:
                    suggestions.append((title, code))
                    seen.add(title)
                    break

    return suggestions


def report_to_markdown(report: DiagnosticReport) -> str:
    """Render a shareable markdown report (for PRs, dataset cards)."""
    lines: list[str] = []
    lines.append(f"# lerobot-doctor report")
    lines.append("")
    lines.append(f"- **Version:** {__version__}")
    lines.append(f"- **Dataset:** `{report.dataset_path}`")
    if report.codebase_version:
        lines.append(f"- **Codebase:** {report.codebase_version}")
    if report.total_episodes is not None:
        lines.append(f"- **Episodes:** {report.total_episodes}")
    if report.total_frames is not None:
        lines.append(f"- **Frames:** {report.total_frames:,}")
    if report.fps is not None:
        lines.append(f"- **FPS:** {report.fps}")
    lines.append(f"- **Overall:** **{report.overall_severity.value}**")
    lines.append("")

    counts = report.summary_counts
    summary_bits = [f"{counts[s.value]} {s.value}" for s in Severity if counts[s.value] > 0]
    lines.append(f"**Summary:** {' | '.join(summary_bits)}")
    lines.append("")

    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Severity | Messages |")
    lines.append("| --- | --- | --- |")
    for r in report.results:
        msgs = [m.message for m in r.messages if m.severity != Severity.PASS]
        cell = "<br>".join(f"- {m}" for m in msgs) if msgs else "_clean_"
        lines.append(f"| {r.name} | **{r.severity.value}** | {cell} |")
    lines.append("")

    fixes = _get_fix_suggestions(report)
    if fixes:
        lines.append("## Suggested fixes")
        lines.append("")
        for title, code in fixes:
            lines.append(f"### {title}")
            lines.append("")
            lines.append("```python")
            lines.append(code)
            lines.append("```")
            lines.append("")

    lines.append("_Generated by [lerobot-doctor](https://github.com/jashshah999/lerobot-doctor)._")
    return "\n".join(lines)


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
