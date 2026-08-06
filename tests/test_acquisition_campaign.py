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


_CHECK_ORDER = (
    "protocol_schedule",
    "frozen_checkout",
    "sealed_readiness",
    "storage_capacity",
    "storage_write_probe",
    "real_evidence_status",
    "execution_manifests",
    "next_execution",
    "session_journal",
)
_INVALID_CHECK_IDS = frozenset(
    {
        "frozen_checkout",
        "sealed_readiness",
        "real_evidence_status",
        "execution_manifests",
        "session_journal",
    }
)
_UNSET = object()


def _checks_with(*overrides: dict[str, object]) -> list[dict[str, object]]:
    checks = {
        check_id: {
            "check_id": check_id,
            "status": "pass",
            "message": f"{check_id} validates.",
        }
        for check_id in _CHECK_ORDER
    }
    for override in overrides:
        checks[str(override["check_id"])] = dict(override)
    return [checks[check_id] for check_id in _CHECK_ORDER]


def _doctor_report(
    *,
    completed: int = 1,
    total: int = 2,
    complete: bool = False,
    checks: list[dict[str, object]] | None = None,
    next_execution: dict[str, object] | None | object = _UNSET,
    resume_acknowledged: bool = False,
    protocol_id: str = "protocol-v1",
) -> dict[str, object]:
    if checks is None:
        checks = _checks_with()
    if complete:
        completed = total
    if next_execution is _UNSET:
        next_execution = (
            None
            if complete
            else {
                "acquisition_execution_index": completed,
                "execution_id": "execution-2",
                "session_id": "session-1",
                "pair_order": 1,
                "contact_region_id": "upper_torso",
                "command_profile_id": "command-2",
                "realization_condition_id": "nominal",
                "replicate_block": 0,
            }
        )
    failures = [check for check in checks if check["status"] == "fail"]
    warnings = [check for check in checks if check["status"] == "warn"]
    by_id = {str(check["check_id"]): check for check in checks}
    valid = not any(
        check["status"] == "fail" and check_id in _INVALID_CHECK_IDS
        for check_id, check in by_id.items()
    )
    session_journal = by_id["session_journal"]
    journal_requires_review = (
        session_journal["status"] == "warn" and not resume_acknowledged
    )
    ready = next_execution is not None and not failures and not journal_requires_review
    collection_complete = next_execution is None and not failures
    report: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": DOCTOR_REPORT_KIND,
        "generated_at_utc": "2026-08-06T04:00:00+00:00",
        "protocol_id": protocol_id,
        "protocol_design_sha256": "a" * 64,
        "repository_root": "/opt/causal4d-frozen",
        "dataset_root": "/data/causal4d",
        "thresholds": {
            "minimum_free_bytes": 0,
            "write_probe_bytes": 0,
            "minimum_write_mib_s": 0.0,
        },
        "resume_acknowledged": resume_acknowledged,
        "checks": checks,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "completed_executions": completed,
        "total_executions": total,
        "next_execution": next_execution,
        "ready_to_record": ready,
        "collection_complete": collection_complete,
        "valid": valid,
        "passed": ready or collection_complete,
        "target_outcomes_used": False,
    }
    return _rehash(report)


def _rehash(report: dict[str, object]) -> dict[str, object]:
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
        checks=_checks_with(
            {
                "check_id": "session_journal",
                "status": "warn",
                "message": "Review the unsealed journal before resuming.",
                "resume_acknowledged": False,
            }
        )
    )

    summary = build_acquisition_campaign_summary(report)

    assert summary["state"] == "blocked"
    assert summary["blocking_checks"] == summary["warnings"]
    assert summary["blocking_checks"][0]["check_id"] == "session_journal"


def test_campaign_renderers_escape_operator_controlled_markdown_and_html() -> None:
    next_execution = {
        "acquisition_execution_index": 1,
        "execution_id": "execution-|<2>\n## injected",
        "session_id": "session-&1`",
        "pair_order": 1,
        "contact_region_id": "upper_torso",
        "command_profile_id": "command-2",
        "realization_condition_id": "nominal",
        "replicate_block": 0,
    }
    report = _doctor_report(
        protocol_id="protocol|<v1>\n## protocol-injected",
        next_execution=next_execution,
        checks=_checks_with(
            {
                "check_id": "sealed_readiness",
                "status": "warn",
                "message": "Relocated | archive\n## message-injected",
            }
        ),
    )

    markdown = render_acquisition_campaign_markdown(report)
    html = render_acquisition_campaign_html(report)

    assert "1/2 executions (50.0%)" in markdown
    assert "Target outcomes used:** <code>false</code>" in markdown
    assert "&#124;" in markdown
    assert "&lt;2&gt;<br>## injected" in markdown
    assert "\n## injected" not in markdown
    assert "\n## protocol-injected" not in markdown
    assert "\n## message-injected" not in markdown
    assert "execution-&#124;&lt;2&gt;" in markdown
    assert "execution-|&lt;2&gt;\n## injected" in html
    assert "session-&amp;1`" in html
    assert "<code>false</code>" in html


def test_campaign_rejects_doctor_report_tampering() -> None:
    report = _doctor_report()
    report["completed_executions"] = 0

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_acquisition_doctor_report(report)


def test_campaign_rejects_boolean_schema_version_after_rehash() -> None:
    report = _doctor_report()
    report["schema_version"] = True
    _rehash(report)

    with pytest.raises(ValueError, match="schema version"):
        validate_acquisition_doctor_report(report)


def test_campaign_rejects_nested_target_outcomes_after_rehash() -> None:
    report = _doctor_report()
    next_execution = dict(report["next_execution"])
    next_execution["target_metrics"] = {"track_error_m": 0.0}
    report["next_execution"] = next_execution
    _rehash(report)

    with pytest.raises(ValueError, match="target-outcome"):
        validate_acquisition_doctor_report(report)


def test_campaign_rejects_contradictory_next_execution_index() -> None:
    report = _doctor_report()
    next_execution = dict(report["next_execution"])
    next_execution["acquisition_execution_index"] = 0
    report["next_execution"] = next_execution
    _rehash(report)

    with pytest.raises(ValueError, match="index differs"):
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


def test_campaign_outputs_cannot_replace_the_source_doctor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "doctor.json"
    original = json.dumps(_doctor_report()).encode("utf-8")
    source.write_bytes(original)

    assert (
        main(
            [
                "campaign",
                "status",
                str(source),
                "--output-json",
                str(source),
                "--overwrite",
            ]
        )
        == 2
    )
    status_error = json.loads(capsys.readouterr().out)
    assert "must differ" in status_error["error"]
    assert source.read_bytes() == original

    assert (
        main(
            [
                "campaign",
                "report",
                str(source),
                str(source),
                "--overwrite",
            ]
        )
        == 2
    )
    report_error = json.loads(capsys.readouterr().out)
    assert "must differ" in report_error["error"]
    assert source.read_bytes() == original


def test_campaign_next_cli_blocks_when_not_ready(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "doctor.json"
    source.write_text(
        json.dumps(
            _doctor_report(
                checks=_checks_with(
                    {
                        "check_id": "session_journal",
                        "status": "warn",
                        "message": "Resume acknowledgement is required.",
                        "resume_acknowledged": False,
                    }
                )
            )
        ),
        encoding="utf-8",
    )

    assert main(["campaign", "next", str(source)]) == 3
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "blocked"
    assert result["next_execution"]["execution_id"] == "execution-2"
