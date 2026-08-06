"""Operator-facing summaries for hash-verified acquisition-doctor reports."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from typing import Any, Literal

from causal4d.acquisition_flight_common import (
    DOCTOR_REPORT_KIND,
    _canonical_sha256,
    _is_sha256,
    _require,
)

CAMPAIGN_SCHEMA_VERSION = 1
CAMPAIGN_SUMMARY_KIND = "Causal4DAcquisitionCampaignSummary"
CampaignState = Literal["invalid", "blocked", "ready", "complete"]
_CHECK_STATUSES = frozenset({"pass", "warn", "fail", "skipped"})


def _nonnegative_integer(value: Any, *, name: str) -> int:
    _require(type(value) is int and value >= 0, f"{name} must be a nonnegative integer")
    return value


def _boolean(value: Any, *, name: str) -> bool:
    _require(type(value) is bool, f"{name} must be Boolean")
    return value


def _validated_checks(value: Any) -> list[dict[str, Any]]:
    _require(isinstance(value, list), "doctor report checks must be a list")
    checks: list[dict[str, Any]] = []
    identifiers: set[str] = set()
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
        _require(identifier not in identifiers, f"duplicate doctor check: {identifier}")
        _require(
            type(status) is str and status in _CHECK_STATUSES,
            f"doctor check {identifier} has invalid status",
        )
        _require(
            type(message) is str and bool(message),
            f"doctor check {identifier} lacks message",
        )
        identifiers.add(identifier)
        checks.append(check)
    _require(bool(checks), "doctor report must contain checks")
    return checks


def validate_acquisition_doctor_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one complete doctor artifact before deriving operator summaries."""

    values = dict(report)
    _require(
        values.get("schema_version") == 1,
        "unsupported acquisition-doctor schema version",
    )
    _require(
        values.get("artifact_kind") == DOCTOR_REPORT_KIND,
        "wrong acquisition-doctor artifact kind",
    )
    protocol_id = values.get("protocol_id")
    _require(
        type(protocol_id) is str and bool(protocol_id),
        "doctor report protocol_id is missing",
    )
    source_sha = values.get("report_sha256")
    _require(_is_sha256(source_sha), "doctor report SHA-256 is invalid")
    _require(
        source_sha == _canonical_sha256(values, omitted="report_sha256"),
        "doctor report checksum mismatch",
    )
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
    checks = _validated_checks(values.get("checks"))
    _require(
        failure_count == sum(check["status"] == "fail" for check in checks),
        "doctor failure_count differs from its checks",
    )
    _require(
        warning_count == sum(check["status"] == "warn" for check in checks),
        "doctor warning_count differs from its checks",
    )
    valid = _boolean(values.get("valid"), name="valid")
    ready = _boolean(values.get("ready_to_record"), name="ready_to_record")
    complete = _boolean(values.get("collection_complete"), name="collection_complete")
    passed = _boolean(values.get("passed"), name="passed")
    _require(not (ready and complete), "campaign cannot be ready and complete")
    _require(passed == (ready or complete), "doctor passed flag is contradictory")
    next_execution = values.get("next_execution")
    _require(
        next_execution is None or isinstance(next_execution, Mapping),
        "doctor next_execution must be an object or null",
    )
    if next_execution is not None:
        identifier = next_execution.get("execution_id")
        _require(
            type(identifier) is str and bool(identifier),
            "doctor next_execution lacks execution_id",
        )
    _require(
        complete
        == (
            completed == total
            and next_execution is None
            and failure_count == 0
        ),
        "doctor completion fields are contradictory",
    )
    values["checks"] = checks
    values["next_execution"] = (
        None if next_execution is None else dict(next_execution)
    )
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


def _next_execution_markdown(next_execution: Mapping[str, Any] | None) -> list[str]:
    if next_execution is None:
        return ["No registered execution remains."]
    rows = ["| Field | Value |", "| --- | --- |"]
    for key, value in next_execution.items():
        rows.append(f"| `{key}` | `{value}` |")
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
        f"- **Protocol:** `{summary['protocol_id']}`",
        f"- **State:** `{summary['state']}`",
        (
            "- **Progress:** "
            f"{summary['completed_executions']}/{summary['total_executions']} "
            f"executions ({progress:.1f}%)"
        ),
        (
            "- **Source doctor report:** "
            f"`{summary['source_doctor_report_sha256']}`"
        ),
        "- **Target outcomes used:** `false`",
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
            f"- `{check['check_id']}`: {check['message']}"
            for check in summary["blocking_checks"]
        )
    else:
        lines.append("None.")
    lines.extend(["", "## Warnings", ""])
    if summary["warnings"]:
        lines.extend(
            f"- `{check['check_id']}`: {check['message']}"
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
        message = check["message"].replace("|", "\\|")
        lines.append(f"| `{check['status']}` | `{check['check_id']}` | {message} |")
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
