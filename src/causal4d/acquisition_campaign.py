"""Operator-facing summaries for hash-verified acquisition-doctor reports."""

from __future__ import annotations

import math
from collections.abc import Mapping
from html import escape
from typing import Any, Literal

from causal4d.acquisition_flight_common import (
    DOCTOR_REPORT_KIND,
    _canonical_sha256,
    _is_sha256,
    _parse_utc,
    _reject_target_outcomes,
    _require,
)

CAMPAIGN_SCHEMA_VERSION = 1
CAMPAIGN_SUMMARY_KIND = "Causal4DAcquisitionCampaignSummary"
CampaignState = Literal["invalid", "blocked", "ready", "complete"]
_CHECK_STATUSES = frozenset({"pass", "warn", "fail", "skipped"})
_DOCTOR_CHECK_IDS = frozenset(
    {
        "protocol_schedule",
        "frozen_checkout",
        "sealed_readiness",
        "storage_capacity",
        "storage_write_probe",
        "real_evidence_status",
        "execution_manifests",
        "next_execution",
        "session_journal",
    }
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
_NEXT_EXECUTION_FIELDS = (
    "acquisition_execution_index",
    "execution_id",
    "session_id",
    "pair_order",
    "contact_region_id",
    "command_profile_id",
    "realization_condition_id",
    "replicate_block",
)
_NEXT_EXECUTION_INTEGER_FIELDS = frozenset(
    {
        "acquisition_execution_index",
        "pair_order",
        "replicate_block",
    }
)


def _nonnegative_integer(value: Any, *, name: str) -> int:
    _require(type(value) is int and value >= 0, f"{name} must be a nonnegative integer")
    return value


def _nonnegative_number(value: Any, *, name: str) -> float:
    _require(
        type(value) in {int, float}
        and math.isfinite(float(value))
        and float(value) >= 0.0,
        f"{name} must be a finite nonnegative number",
    )
    return float(value)


def _boolean(value: Any, *, name: str) -> bool:
    _require(type(value) is bool, f"{name} must be Boolean")
    return value


def _nonempty_string(value: Any, *, name: str) -> str:
    _require(type(value) is str and bool(value), f"{name} must be a nonempty string")
    return value


def _validated_checks(
    value: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    _require(isinstance(value, list), "doctor report checks must be a list")
    checks: list[dict[str, Any]] = []
    by_identifier: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        _require(isinstance(raw, Mapping), f"doctor check {index} must be an object")
        check = dict(raw)
        identifier = check.get("check_id")
        status = check.get("status")
        message = check.get("message")
        _require(
            type(identifier) is str and bool(identifier),
            f"doctor check {index} lacks check_id",
        )
        _require(
            identifier not in by_identifier,
            f"duplicate doctor check: {identifier}",
        )
        _require(
            type(status) is str and status in _CHECK_STATUSES,
            f"doctor check {identifier} has invalid status",
        )
        _require(
            type(message) is str and bool(message),
            f"doctor check {identifier} lacks message",
        )
        by_identifier[identifier] = check
        checks.append(check)
    _require(
        set(by_identifier) == _DOCTOR_CHECK_IDS,
        "doctor check inventory differs from schema version 1",
    )
    return checks, by_identifier


def _validated_next_execution(value: Any) -> dict[str, Any] | None:
    _require(
        value is None or isinstance(value, Mapping),
        "doctor next_execution must be an object or null",
    )
    if value is None:
        return None
    result = dict(value)
    _require(
        set(result) == set(_NEXT_EXECUTION_FIELDS),
        "doctor next_execution inventory differs from the registered summary contract",
    )
    for field in _NEXT_EXECUTION_FIELDS:
        if field in _NEXT_EXECUTION_INTEGER_FIELDS:
            result[field] = _nonnegative_integer(
                result[field],
                name=f"doctor next_execution {field}",
            )
        else:
            result[field] = _nonempty_string(
                result[field],
                name=f"doctor next_execution {field}",
            )
    return result


def _validate_thresholds(value: Any) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "doctor thresholds must be an object")
    thresholds = dict(value)
    _require(
        set(thresholds)
        == {
            "minimum_free_bytes",
            "write_probe_bytes",
            "minimum_write_mib_s",
        },
        "doctor threshold inventory differs from schema version 1",
    )
    thresholds["minimum_free_bytes"] = _nonnegative_integer(
        thresholds["minimum_free_bytes"],
        name="doctor minimum_free_bytes",
    )
    thresholds["write_probe_bytes"] = _nonnegative_integer(
        thresholds["write_probe_bytes"],
        name="doctor write_probe_bytes",
    )
    thresholds["minimum_write_mib_s"] = _nonnegative_number(
        thresholds["minimum_write_mib_s"],
        name="doctor minimum_write_mib_s",
    )
    return thresholds


def validate_acquisition_doctor_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one complete doctor artifact before deriving operator summaries."""

    values = dict(report)
    schema_version = values.get("schema_version")
    _require(
        type(schema_version) is int and schema_version == 1,
        "unsupported acquisition-doctor schema version",
    )
    _require(
        type(values.get("artifact_kind")) is str
        and values["artifact_kind"] == DOCTOR_REPORT_KIND,
        "wrong acquisition-doctor artifact kind",
    )
    _parse_utc(
        values.get("generated_at_utc"),
        name="acquisition-doctor generated_at_utc",
    )
    protocol_id = _nonempty_string(
        values.get("protocol_id"),
        name="doctor report protocol_id",
    )
    _require(
        _is_sha256(values.get("protocol_design_sha256")),
        "doctor protocol_design_sha256 is invalid",
    )
    _nonempty_string(values.get("repository_root"), name="doctor repository_root")
    _nonempty_string(values.get("dataset_root"), name="doctor dataset_root")
    thresholds = _validate_thresholds(values.get("thresholds"))
    resume_acknowledged = _boolean(
        values.get("resume_acknowledged"),
        name="resume_acknowledged",
    )
    source_sha = values.get("report_sha256")
    _require(_is_sha256(source_sha), "doctor report SHA-256 is invalid")
    _require(
        source_sha == _canonical_sha256(values, omitted="report_sha256"),
        "doctor report checksum mismatch",
    )
    _reject_target_outcomes(values)
    _require(
        values.get("target_outcomes_used") is False,
        "doctor report must preserve the target-outcome boundary",
    )

    completed = _nonnegative_integer(
        values.get("completed_executions"),
        name="completed_executions",
    )
    total = _nonnegative_integer(
        values.get("total_executions"),
        name="total_executions",
    )
    _require(total > 0, "total_executions must be positive")
    _require(completed <= total, "completed executions exceed the protocol total")
    failure_count = _nonnegative_integer(
        values.get("failure_count"),
        name="failure_count",
    )
    warning_count = _nonnegative_integer(
        values.get("warning_count"),
        name="warning_count",
    )
    checks, check_by_identifier = _validated_checks(values.get("checks"))
    actual_failure_count = sum(check["status"] == "fail" for check in checks)
    actual_warning_count = sum(check["status"] == "warn" for check in checks)
    _require(
        failure_count == actual_failure_count,
        "doctor failure_count differs from its checks",
    )
    _require(
        warning_count == actual_warning_count,
        "doctor warning_count differs from its checks",
    )

    valid = _boolean(values.get("valid"), name="valid")
    ready = _boolean(values.get("ready_to_record"), name="ready_to_record")
    complete = _boolean(values.get("collection_complete"), name="collection_complete")
    passed = _boolean(values.get("passed"), name="passed")
    expected_valid = not any(
        check["status"] == "fail" and identifier in _INVALID_CHECK_IDS
        for identifier, check in check_by_identifier.items()
    )
    _require(valid == expected_valid, "doctor valid flag is contradictory")

    session_journal = check_by_identifier["session_journal"]
    if session_journal["status"] == "warn":
        journal_resume = _boolean(
            session_journal.get("resume_acknowledged"),
            name="session_journal resume_acknowledged",
        )
        _require(
            journal_resume == resume_acknowledged,
            "session-journal resume acknowledgement differs from the doctor report",
        )
        journal_requires_review = not resume_acknowledged
    else:
        _require(
            "resume_acknowledged" not in session_journal,
            "non-warning session_journal check must not contain resume acknowledgement",
        )
        journal_requires_review = False

    next_execution = _validated_next_execution(values.get("next_execution"))
    if next_execution is None:
        _require(
            completed == total,
            "doctor without a next execution must report all executions completed",
        )
    else:
        _require(
            completed < total,
            "doctor with a next execution must have remaining executions",
        )
        _require(
            next_execution["acquisition_execution_index"] == completed,
            "doctor next-execution index differs from completed execution count",
        )

    expected_complete = next_execution is None and failure_count == 0
    expected_ready = (
        next_execution is not None
        and failure_count == 0
        and not journal_requires_review
    )
    _require(
        complete == expected_complete,
        "doctor collection_complete flag is contradictory",
    )
    _require(ready == expected_ready, "doctor ready_to_record flag is contradictory")
    _require(not (ready and complete), "campaign cannot be ready and complete")
    _require(passed == (ready or complete), "doctor passed flag is contradictory")

    values["protocol_id"] = protocol_id
    values["thresholds"] = thresholds
    values["checks"] = checks
    values["next_execution"] = next_execution
    values["resume_acknowledged"] = resume_acknowledged
    values["valid"] = valid
    values["ready_to_record"] = ready
    values["collection_complete"] = complete
    values["passed"] = passed
    return values


def _campaign_state(report: Mapping[str, Any]) -> CampaignState:
    if report["valid"] is not True:
        return "invalid"
    if report["collection_complete"] is True:
        return "complete"
    if report["ready_to_record"] is True:
        return "ready"
    return "blocked"


def build_acquisition_campaign_summary(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive a compact status artifact from a verified doctor report."""

    values = validate_acquisition_doctor_report(report)
    state = _campaign_state(values)
    checks = values["checks"]
    failures = [
        {
            "check_id": check["check_id"],
            "status": check["status"],
            "message": check["message"],
        }
        for check in checks
        if check["status"] == "fail"
    ]
    warnings = [
        {
            "check_id": check["check_id"],
            "status": check["status"],
            "message": check["message"],
        }
        for check in checks
        if check["status"] == "warn"
    ]
    blocking = failures if failures else (warnings if state == "blocked" else [])
    completed = values["completed_executions"]
    total = values["total_executions"]
    summary: dict[str, Any] = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "artifact_kind": CAMPAIGN_SUMMARY_KIND,
        "source_doctor_report_sha256": values["report_sha256"],
        "protocol_id": values["protocol_id"],
        "state": state,
        "completed_executions": completed,
        "total_executions": total,
        "remaining_executions": total - completed,
        "progress_fraction": completed / total,
        "next_execution": values["next_execution"],
        "blocking_checks": blocking,
        "warnings": warnings,
        "target_outcomes_used": False,
    }
    summary["summary_sha256"] = _canonical_sha256(summary, omitted="summary_sha256")
    return summary


def _markdown_text(value: Any) -> str:
    return (
        escape(str(value))
        .replace("|", "&#124;")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "<br>")
    )


def _markdown_code(value: Any) -> str:
    return f"<code>{_markdown_text(value)}</code>"


def _next_execution_markdown(next_execution: Mapping[str, Any] | None) -> list[str]:
    if next_execution is None:
        return ["No registered execution remains."]
    rows = ["| Field | Value |", "| --- | --- |"]
    for key, value in next_execution.items():
        rows.append(f"| {_markdown_code(key)} | {_markdown_code(value)} |")
    return rows


def render_acquisition_campaign_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render a deterministic human-readable campaign report."""

    values = validate_acquisition_doctor_report(report)
    summary = build_acquisition_campaign_summary(values)
    progress = 100.0 * summary["progress_fraction"]
    lines = [
        "# Causal4D acquisition campaign",
        "",
        f"- **Protocol:** {_markdown_code(summary['protocol_id'])}",
        f"- **State:** {_markdown_code(summary['state'])}",
        (
            "- **Progress:** "
            f"{summary['completed_executions']}/{summary['total_executions']} "
            f"executions ({progress:.1f}%)"
        ),
        (
            "- **Source doctor report:** "
            f"{_markdown_code(summary['source_doctor_report_sha256'])}"
        ),
        "- **Target outcomes used:** <code>false</code>",
        "",
        "## Next registered execution",
        "",
        *_next_execution_markdown(summary["next_execution"]),
        "",
        "## Blocking checks",
        "",
    ]
    if summary["blocking_checks"]:
        lines.extend(
            f"- {_markdown_code(check['check_id'])}: {_markdown_text(check['message'])}"
            for check in summary["blocking_checks"]
        )
    else:
        lines.append("None.")
    lines.extend(["", "## Warnings", ""])
    if summary["warnings"]:
        lines.extend(
            f"- {_markdown_code(check['check_id'])}: {_markdown_text(check['message'])}"
            for check in summary["warnings"]
        )
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Complete doctor checks",
            "",
            "| Status | Check | Message |",
            "| --- | --- | --- |",
        ]
    )
    for check in values["checks"]:
        lines.append(
            f"| {_markdown_code(check['status'])} | "
            f"{_markdown_code(check['check_id'])} | "
            f"{_markdown_text(check['message'])} |"
        )
    lines.extend(
        [
            "",
            "This is operational provenance only. It does not increment the "
            "confirmatory evidence count or authorize target-informed changes.",
            "",
        ]
    )
    return "\n".join(lines)


def render_acquisition_campaign_html(
    report: Mapping[str, Any],
) -> str:
    """Render a standalone deterministic HTML campaign report."""

    values = validate_acquisition_doctor_report(report)
    summary = build_acquisition_campaign_summary(values)
    progress = 100.0 * summary["progress_fraction"]
    next_execution = summary["next_execution"]
    if next_execution is None:
        next_html = "<p>No registered execution remains.</p>"
    else:
        rows = "".join(
            "<tr><th><code>"
            + escape(str(key))
            + "</code></th><td><code>"
            + escape(str(value))
            + "</code></td></tr>"
            for key, value in next_execution.items()
        )
        next_html = f"<table><tbody>{rows}</tbody></table>"

    def list_html(items: list[dict[str, Any]]) -> str:
        if not items:
            return "<p>None.</p>"
        rows = "".join(
            "<li><code>"
            + escape(str(item["check_id"]))
            + "</code>: "
            + escape(str(item["message"]))
            + "</li>"
            for item in items
        )
        return f"<ul>{rows}</ul>"

    check_rows = "".join(
        "<tr><td><code>"
        + escape(str(check["status"]))
        + "</code></td><td><code>"
        + escape(str(check["check_id"]))
        + "</code></td><td>"
        + escape(str(check["message"]))
        + "</td></tr>"
        for check in values["checks"]
    )
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>Causal4D acquisition campaign</title>",
            "</head>",
            "<body>",
            "<main>",
            "<h1>Causal4D acquisition campaign</h1>",
            "<dl>",
            f"<dt>Protocol</dt><dd><code>{escape(summary['protocol_id'])}</code></dd>",
            f"<dt>State</dt><dd><code>{escape(summary['state'])}</code></dd>",
            (
                "<dt>Progress</dt><dd>"
                f"{summary['completed_executions']}/{summary['total_executions']} "
                f"executions ({progress:.1f}%)</dd>"
            ),
            (
                "<dt>Source doctor report</dt><dd><code>"
                f"{escape(summary['source_doctor_report_sha256'])}"
                "</code></dd>"
            ),
            "<dt>Target outcomes used</dt><dd><code>false</code></dd>",
            "</dl>",
            "<h2>Next registered execution</h2>",
            next_html,
            "<h2>Blocking checks</h2>",
            list_html(summary["blocking_checks"]),
            "<h2>Warnings</h2>",
            list_html(summary["warnings"]),
            "<h2>Complete doctor checks</h2>",
            "<table>",
            "<thead><tr><th>Status</th><th>Check</th><th>Message</th></tr></thead>",
            f"<tbody>{check_rows}</tbody>",
            "</table>",
            (
                "<p>This is operational provenance only. It does not increment "
                "the confirmatory evidence count or authorize target-informed "
                "changes.</p>"
            ),
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


__all__ = [
    "CAMPAIGN_SCHEMA_VERSION",
    "CAMPAIGN_SUMMARY_KIND",
    "build_acquisition_campaign_summary",
    "render_acquisition_campaign_html",
    "render_acquisition_campaign_markdown",
    "validate_acquisition_doctor_report",
]
