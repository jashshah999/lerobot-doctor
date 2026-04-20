"""Tests for markdown report and fix snippets."""

from lerobot_doctor.report import _get_fix_suggestions, report_to_markdown
from lerobot_doctor.runner import CheckResult, DiagnosticReport, Severity


def _report_with_messages(messages: list[tuple[Severity, str]]) -> DiagnosticReport:
    report = DiagnosticReport(dataset_path="/test")
    r = CheckResult(name="test", severity=Severity.PASS)
    for sev, msg in messages:
        if sev == Severity.WARN:
            r.warn(msg)
        elif sev == Severity.FAIL:
            r.fail(msg)
        else:
            r.pass_(msg)
    report.results = [r]
    return report


def test_fix_suggestions_return_tuples():
    report = _report_with_messages([(Severity.WARN, "action contains NaN values")])
    fixes = _get_fix_suggestions(report)
    assert len(fixes) == 1
    title, code = fixes[0]
    assert "NaN" in title
    assert "dropna" in code


def test_fix_suggestions_dedup():
    report = _report_with_messages([
        (Severity.WARN, "action contains NaN values in ep 1"),
        (Severity.WARN, "action contains NaN values in ep 2"),
    ])
    fixes = _get_fix_suggestions(report)
    assert len(fixes) == 1


def test_fix_suggestions_skip_pass():
    report = _report_with_messages([(Severity.PASS, "action contains NaN values")])
    assert _get_fix_suggestions(report) == []


def test_markdown_report_structure():
    report = _report_with_messages([(Severity.FAIL, "total_frames mismatch")])
    report.total_episodes = 10
    report.fps = 30
    md = report_to_markdown(report)
    assert "# lerobot-doctor report" in md
    assert "**Overall:**" in md
    assert "| Check | Severity | Messages |" in md
    assert "Suggested fixes" in md
    assert "```python" in md


def test_markdown_no_fixes_when_clean():
    report = DiagnosticReport(dataset_path="/test")
    r = CheckResult(name="test", severity=Severity.PASS)
    r.pass_("ok")
    report.results = [r]
    md = report_to_markdown(report)
    assert "Suggested fixes" not in md
    assert "_clean_" in md
