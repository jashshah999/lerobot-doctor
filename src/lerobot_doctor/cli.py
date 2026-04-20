"""CLI entry point for lerobot-doctor."""

from __future__ import annotations

import argparse
import sys

from lerobot_doctor import __version__


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="lerobot-doctor",
        description="Dataset quality diagnostics for LeRobot v3 datasets",
    )
    parser.add_argument(
        "dataset",
        help="Path to local dataset directory or HuggingFace repo_id (e.g. lerobot/pusht)",
    )
    parser.add_argument(
        "--checks",
        type=str,
        default=None,
        help="Comma-separated list of checks to run (default: all). "
             "Available: metadata,temporal,actions,videos,statistics,"
             "episodes,consistency,training,anomalies,portability,per_episode",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to load for checking (default: all)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output report as JSON",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show all messages including PASS details",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: JSON output, one-line summary to stderr, exit code based on --fail-on",
    )
    parser.add_argument(
        "--fail-on",
        choices=["warn", "fail"],
        default="fail",
        help="Minimum severity that triggers exit code 1 (default: fail). Only used with --ci",
    )
    parser.add_argument(
        "--markdown",
        type=str,
        default=None,
        metavar="PATH",
        help="Also write a markdown report to PATH (good for PRs and dataset cards)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"lerobot-doctor {__version__}",
    )

    args = parser.parse_args(argv)

    # Parse checks
    check_names = None
    if args.checks:
        check_names = [c.strip() for c in args.checks.split(",")]

    # Load dataset
    from lerobot_doctor.dataset_loader import load_dataset
    from lerobot_doctor.runner import run_checks
    from lerobot_doctor.report import print_report, report_to_json, report_to_markdown

    try:
        dataset = load_dataset(args.dataset, max_episodes=args.max_episodes)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    # Run checks
    report = run_checks(dataset, checks=check_names, verbose=args.verbose)

    # Output
    if args.ci:
        print(report_to_json(report))
        counts = report.summary_counts
        summary = f"lerobot-doctor: {counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail"
        print(summary, file=sys.stderr)
        threshold = args.fail_on.upper()
        if threshold == "WARN" and report.overall_severity.value in ("WARN", "FAIL"):
            sys.exit(1)
        elif threshold == "FAIL" and report.overall_severity.value == "FAIL":
            sys.exit(1)
    elif args.json_output:
        print(report_to_json(report))
    else:
        print_report(report, verbose=args.verbose)

    if args.markdown:
        from pathlib import Path
        Path(args.markdown).write_text(report_to_markdown(report))
        print(f"Wrote markdown report to {args.markdown}", file=sys.stderr)

    # Exit code: 0 for PASS/WARN, 1 for FAIL (non-CI mode)
    if not args.ci and report.overall_severity.value == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
