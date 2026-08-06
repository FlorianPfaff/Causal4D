from __future__ import annotations

import json
from pathlib import Path

import pytest

from causal4d.acquisition_campaign import (
    CAMPAIGN_SUMMARY_KIND,
    build_acquisition_campaign_summary,
    render_acquisition_campaign_html,
    render_acquisition_campaign_markdown,
    validate_acquisition_doctor_report,
)
from causal4d.acquisition_flight_common import (
    DOCTOR_REPORT_KIND,
    _canonical_sha256,
)
from causal4d.cli.acquisition_operations import main


def _doctor_report(
    *,
    state: str = "ready",
    completed: int = 1,
    total: int = 2,
    checks: list[dict[str, object]] | None = None,
    next_execution: dict[str, object] | None = None,
) -> dict[str, object]:
    if checks is None:
        checks = [
            {
                "check_id": "frozen_checkout",
                "status": "pass",
                "message": "Frozen checkout validates.",
            },
            {
                "check_id": "session_journal",
                "status": "pass",
                "message": "No prior journal bytes exist.",
            },
        ]
    if next_execution is None and state != "complete":
        next_execution = {
            "acquisition_execution_index": completed,
            "execution_id": "execution-2",
            "session_id": "session-1",
            "command_profile_id": "command-2",
        }
    ready = state == "ready"
    complete = state == "complete"
    valid = state != "invalid"
    report: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": DOCTOR_REPORT_KIND,
        "generated_at_utc": "2026-08-06T04:00:00+00:00",
        "protocol_id": "protocol-v1",
        "protocol_design_sha256": "a" * 64,
        "repository_root": "/opt/causal4d-frozen",
        "dataset_root": "/data/causal4d",
        "thresholds": {
            "minimum_free_bytes": 0,
            "write_probe_bytes": 0,
            "minimum_write_mib_s": 0.0,
        },
        "resume_acknowledged": False,
        "checks": checks,
        "failure_count": sum(check["status"] == "fail" for check in checks),
        "warning_count": sum(check["status"] == "warn" for check in checks),
        "completed_executions": total if complete else completed,
        "total_executions": total,
        "next_execution": None if complete else next_execution,
        "ready_to_record": ready,
        "collection_complete": complete,
        "valid": valid,
        "passed": ready or complete,
        "target_outcomes_used": False,
    }
    report["report_sha256"] = _canonical_sha256(
        report,
        omitted="report_sha256",
    )
    return report


def test_campaign_summary_reports_ready_progress_and_next_execution() -> None:
    summary = build_acquisition_campaign_summary(_doctor_report())

    assert summary["artifact_kind"] == CAMPAIGN_SUMMARY_KIND
    assert summary["state"] == "ready"
    assert summary["completed_executions"] == 1
    assert summary["remaining_executions"] == 1
    assert summary["progress_fraction"] == 0.5
    assert summary["next_execution"]["execution_id"] == "execution-2"
    assert summary["blocking_checks"] == []
    assert summary["target_outcomes_used"] is False


def test_campaign_summary_surfaces_blocking_warning_without_failure() -> None:
    report = _doctor_report(
        state="blocked",
        checks=[
            {
                "check_id": "frozen_checkout",
                "status": "pass",
                "message": "Frozen checkout validates.",
            },
            {
                "check_id": "session_journal",
                "status": "warn",
                "message": "Review the unsealed journal before resuming.",
            },
        ],
    )

    summary = build_acquisition_campaign_summary(report)

    assert summary["state"] == "blocked"
    assert summary["blocking_checks"] == summary["warnings"]
    assert summary["blocking_checks"][0]["check_id"] == "session_journal"


def test_campaign_renderers_preserve_boundary_and_escape_html() -> None:
    report = _doctor_report(
        next_execution={
            "execution_id": "execution-<2>",
            "session_id": "session-&1",
        }
    )

    markdown = render_acquisition_campaign_markdown(report)
    html = render_acquisition_campaign_html(report)

    assert "1/2 executions (50.0%)" in markdown
    assert "Target outcomes used:** `false`" in markdown
    assert "execution-<2>" in markdown
    assert "execution-&lt;2&gt;" in html
    assert "session-&amp;1" in html
    assert "<code>false</code>" in html


def test_campaign_rejects_doctor_report_tampering() -> None:
    report = _doctor_report()
    report["completed_executions"] = 0

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_acquisition_doctor_report(report)


def test_campaign_status_and_report_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "doctor.json"
    summary_path = tmp_path / "campaign.json"
    report_path = tmp_path / "campaign.md"
    source.write_text(json.dumps(_doctor_report()), encoding="utf-8")

    assert (
        main(
            [
                "campaign",
                "status",
                str(source),
                "--output-json",
                str(summary_path),
                "--require-ready",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["state"] == "ready"
    assert json.loads(summary_path.read_text(encoding="utf-8"))["state"] == "ready"

    assert (
        main(
            [
                "campaign",
                "report",
                str(source),
                str(report_path),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["format"] == "markdown"
    assert result["state"] == "ready"
    assert "# Causal4D acquisition campaign" in report_path.read_text(encoding="utf-8")


def test_campaign_next_cli_blocks_when_not_ready(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "doctor.json"
    source.write_text(
        json.dumps(
            _doctor_report(
                state="blocked",
                checks=[
                    {
                        "check_id": "session_journal",
                        "status": "warn",
                        "message": "Resume acknowledgement is required.",
                    }
                ],
            )
        ),
        encoding="utf-8",
    )

    assert main(["campaign", "next", str(source)]) == 3
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "blocked"
    assert result["next_execution"]["execution_id"] == "execution-2"
