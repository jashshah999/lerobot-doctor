"""CLI entry point for lerobot-doctor."""

from __future__ import annotations

import argparse
import sys
import json as json_module

from lerobot_doctor import __version__


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="lerobot-doctor",
        description="Dataset quality diagnostics, repair, and curation for LeRobot v2/v3 datasets",
    )
    parser.add_argument("--version", action="version", version=f"lerobot-doctor {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    # === CHECK (default) ===
    check_p = subparsers.add_parser("check", help="Run diagnostic checks (default)")
    check_p.add_argument("dataset", help="Path to local dataset or HF repo_id")
    check_p.add_argument("--checks", type=str, default=None,
                         help="Comma-separated list of checks to run (default: all)")
    check_p.add_argument("--max-episodes", type=int, default=None)
    check_p.add_argument("--urdf", type=str, default=None, metavar="PATH",
                         help="URDF file to check actions against for the kinematics check "
                              "(otherwise looked up from robot_type in a small built-in registry)")
    check_p.add_argument("--json", action="store_true", dest="json_output")
    check_p.add_argument("-v", "--verbose", action="store_true")
    check_p.add_argument("--ci", action="store_true")
    check_p.add_argument("--fail-on", choices=["warn", "fail"], default="fail")
    check_p.add_argument("--markdown", type=str, default=None, metavar="PATH")

    # === FIX ===
    fix_p = subparsers.add_parser("fix", help="Auto-fix common dataset issues")
    fix_p.add_argument("dataset", help="Path to local dataset")
    fix_p.add_argument("--fixes", type=str, default=None,
                       help="Comma-separated: reindex,timestamps,nan,metadata,episodes (default: all)")
    fix_p.add_argument("--no-backup", action="store_true")
    fix_p.add_argument("--dry-run", action="store_true")

    # === TRIM ===
    trim_p = subparsers.add_parser("trim", help="Remove idle/static frames from episodes")
    trim_p.add_argument("dataset", help="Path to local dataset")
    trim_p.add_argument("--threshold", type=float, default=0.01)
    trim_p.add_argument("--min-active", type=int, default=10)
    trim_p.add_argument("--keep-static", action="store_true")
    trim_p.add_argument("--dry-run", action="store_true")

    # === SCORE ===
    score_p = subparsers.add_parser("score", help="Score episodes on quality")
    score_p.add_argument("dataset", help="Path to dataset")
    score_p.add_argument("--max-episodes", type=int, default=None)
    score_p.add_argument("--drop-threshold", type=float, default=30.0)
    score_p.add_argument("--json", action="store_true", dest="json_output")

    # === GATE ===
    gate_p = subparsers.add_parser("gate", help="Pre-training policy compatibility check")
    gate_p.add_argument("dataset", help="Path to dataset")
    gate_p.add_argument("--policy", default="act", choices=["act", "diffusion", "smolvla", "pi0"])
    gate_p.add_argument("--chunk-size", type=int, default=None)
    gate_p.add_argument("--json", action="store_true", dest="json_output")

    # === MERGE-CHECK ===
    merge_p = subparsers.add_parser("merge-check", help="Validate merge compatibility")
    merge_p.add_argument("datasets", nargs="+", help="Paths to datasets")
    merge_p.add_argument("--post-merge", action="store_true")

    # Parse -- handle legacy (no subcommand = "check")
    if argv is None:
        argv = sys.argv[1:]

    known_commands = {"check", "fix", "trim", "score", "gate", "merge-check", "--version", "-h", "--help"}
    if argv and argv[0] not in known_commands and not argv[0].startswith("-"):
        argv = ["check"] + argv

    args = parser.parse_args(argv)

    if args.command == "check" or args.command is None:
        _run_check(args)
    elif args.command == "fix":
        _run_fix(args)
    elif args.command == "trim":
        _run_trim(args)
    elif args.command == "score":
        _run_score(args)
    elif args.command == "gate":
        _run_gate(args)
    elif args.command == "merge-check":
        _run_merge_check(args)
    else:
        parser.print_help()


def _load_dataset_or_exit(path: str, max_episodes: int | None = None):
    """Load a local dir, zip archive, or HF repo_id; exit 1 on failure."""
    from lerobot_doctor.dataset_loader import load_dataset

    try:
        ds = load_dataset(path, max_episodes=max_episodes)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)
    if ds.info is None:
        print(f"Error loading dataset: {ds.info_error}", file=sys.stderr)
        sys.exit(1)
    return ds


def _run_check(args):
    from lerobot_doctor.runner import run_checks
    from lerobot_doctor.report import print_report, report_to_json, report_to_markdown

    check_names = [c.strip() for c in args.checks.split(",")] if args.checks else None
    dataset = _load_dataset_or_exit(args.dataset, max_episodes=args.max_episodes)
    if args.urdf:
        from pathlib import Path
        dataset.robot_urdf = Path(args.urdf)

    report = run_checks(dataset, checks=check_names, verbose=args.verbose)

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

    if not args.ci and report.overall_severity.value == "FAIL":
        sys.exit(1)


def _run_fix(args):
    from pathlib import Path
    from lerobot_doctor.fix import fix_dataset

    fixes = [f.strip() for f in args.fixes.split(",")] if args.fixes else None
    result = fix_dataset(Path(args.dataset), fixes=fixes, backup=not args.no_backup, dry_run=args.dry_run)

    prefix = "[DRY RUN] " if args.dry_run else ""
    for msg in result.fixed:
        print(f"  {prefix}FIXED: {msg}")
    for msg in result.skipped:
        print(f"  SKIP: {msg}")
    for msg in result.errors:
        print(f"  ERROR: {msg}", file=sys.stderr)
    print(f"\n{prefix}{result.n_fixed} issues fixed")


def _run_trim(args):
    from pathlib import Path
    from lerobot_doctor.trim import trim_dataset

    result = trim_dataset(
        Path(args.dataset), action_threshold=args.threshold,
        min_active_frames=args.min_active, remove_fully_static=not args.keep_static,
        dry_run=args.dry_run,
    )
    prefix = "[DRY RUN] " if args.dry_run else ""
    for d in result.details:
        print(f"  {prefix}{d}")
    print(f"\n{prefix}{result.episodes_trimmed} trimmed, {result.frames_removed} frames removed, "
          f"{result.episodes_removed} episodes dropped")


def _run_score(args):
    from lerobot_doctor.score import score_episodes

    ds = _load_dataset_or_exit(args.dataset, max_episodes=args.max_episodes)
    result = score_episodes(ds.root, max_episodes=args.max_episodes, drop_threshold=args.drop_threshold)
    if args.json_output:
        print(json_module.dumps({
            "mean_score": result.mean_score,
            "recommended_drops": result.recommended_drops,
            "scores": [{"ep": s.episode_index, "score": round(s.overall_score, 1), "outlier": s.is_outlier}
                       for s in result.get_sorted()],
        }, indent=2))
    else:
        print(f"Episode Scores (mean: {result.mean_score:.1f}/100)\n{'='*50}")
        for s in result.get_sorted()[:20]:
            flag = " [DROP]" if s.is_outlier else ""
            print(f"  Ep {s.episode_index:4d}: {s.overall_score:5.1f}/100{flag}")
        if result.recommended_drops:
            print(f"\nRecommend dropping: {result.recommended_drops}")


def _run_gate(args):
    from lerobot_doctor.gate import gate_check

    ds = _load_dataset_or_exit(args.dataset)
    result = gate_check(ds.root, policy=args.policy, custom_chunk_size=args.chunk_size)
    if args.json_output:
        print(json_module.dumps({"passed": result.passed, "policy": result.policy,
                                 "blockers": result.blockers, "warnings": result.warnings}, indent=2))
    else:
        status = "PASS" if result.passed else "FAIL"
        print(f"Training Gate [{status}] policy={result.policy}\n{'='*50}")
        for b in result.blockers:
            print(f"  [BLOCK] {b}")
        for w in result.warnings:
            print(f"  [WARN] {w}")
        for i in result.info:
            print(f"  [OK] {i}")
    if not result.passed:
        sys.exit(1)


def _run_merge_check(args):
    from pathlib import Path
    from lerobot_doctor.merge_check import check_merge_compatibility, validate_merged_dataset

    if args.post_merge:
        result = validate_merged_dataset(Path(args.datasets[0]))
    else:
        result = check_merge_compatibility([Path(p) for p in args.datasets])

    status = "PASS" if result.compatible else "FAIL"
    print(f"Merge Check [{status}]\n{'='*50}")
    for i in result.issues:
        print(f"  [ISSUE] {i}")
    for w in result.warnings:
        print(f"  [WARN] {w}")
    for s in result.suggestions:
        print(f"  [FIX] {s}")
    if not result.compatible:
        sys.exit(1)


if __name__ == "__main__":
    main()
