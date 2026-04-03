"""Tests for the runner module."""

import pytest

from lerobot_doctor.dataset_loader import load_local
from lerobot_doctor.runner import CheckResult, DiagnosticReport, Severity, run_checks
from tests.conftest import create_dataset


def test_run_all_checks(tmp_dataset):
    ds = load_local(tmp_dataset)
    report = run_checks(ds)
    assert len(report.results) == 10
    assert all(isinstance(r, CheckResult) for r in report.results)


def test_run_subset_checks(tmp_dataset):
    ds = load_local(tmp_dataset)
    report = run_checks(ds, checks=["metadata", "temporal"])
    assert len(report.results) == 2
    assert report.results[0].name == "Metadata & Format Compliance"
    assert report.results[1].name == "Temporal Consistency"


def test_unknown_check(tmp_dataset):
    ds = load_local(tmp_dataset)
    report = run_checks(ds, checks=["nonexistent"])
    assert len(report.results) == 1
    assert report.results[0].severity == Severity.FAIL


def test_overall_severity_pass(tmp_dataset):
    ds = load_local(tmp_dataset)
    report = run_checks(ds, checks=["metadata"])
    assert report.overall_severity == Severity.PASS


def test_summary_counts():
    report = DiagnosticReport(dataset_path="/test")
    r1 = CheckResult(name="test1", severity=Severity.PASS)
    r2 = CheckResult(name="test2", severity=Severity.WARN)
    r3 = CheckResult(name="test3", severity=Severity.FAIL)
    report.results = [r1, r2, r3]
    counts = report.summary_counts
    assert counts == {"PASS": 1, "WARN": 1, "FAIL": 1}
    assert report.overall_severity == Severity.FAIL


def test_check_result_escalation():
    r = CheckResult(name="test", severity=Severity.PASS)
    r.pass_("ok")
    assert r.severity == Severity.PASS
    r.warn("something")
    assert r.severity == Severity.WARN
    r.fail("bad")
    assert r.severity == Severity.FAIL
