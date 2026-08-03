"""Fail-closed pre-session acquisition doctor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from causal4d.acquisition_flight_common import (
    DOCTOR_REPORT_KIND,
    _assert_no_symlink_components,
    _canonical_sha256,
    _require,
    _utc_now,
)
from causal4d.acquisition_doctor_environment import (
    frozen_checkout_check,
    protocol_schedule,
    readiness_check,
    storage_checks,
)
from causal4d.acquisition_doctor_progress import (
    evidence_check,
    manifest_check,
    next_execution_checks,
)
from causal4d.acquisition_doctor_support import DoctorThresholds


def build_acquisition_doctor_report(
    protocol: Mapping[str, Any],
    repository_root: str | Path,
    dataset_root: str | Path,
    *,
    readiness_path: str | Path | None = None,
    method_freeze_path: str | Path | None = None,
    thresholds: DoctorThresholds | None = None,
    perform_write_probe: bool = True,
    allow_resume: bool = False,
) -> dict[str, Any]:
    """Run method-neutral pre-session checks and identify the next execution."""

    settings = thresholds or DoctorThresholds()
    repository = Path(repository_root)
    dataset = Path(dataset_root)
    _assert_no_symlink_components(repository, name="repository root")
    _assert_no_symlink_components(dataset, name="dataset root")
    _require(repository.is_dir(), "repository root must exist")
    _require(dataset.is_dir(), "dataset root must exist")

    protocol_id, design_sha, executions, schedule_check = protocol_schedule(protocol)
    checks = [
        schedule_check,
        frozen_checkout_check(repository, dataset, method_freeze_path),
        readiness_check(protocol, dataset, readiness_path),
        *storage_checks(
            dataset,
            settings,
            perform_write_probe=perform_write_probe,
        ),
    ]
    evidence, completed, malformed, present = evidence_check(
        protocol,
        repository,
        dataset,
    )
    checks.append(evidence)
    manifests, next_execution = manifest_check(
        executions,
        completed,
        malformed,
        present,
    )
    checks.append(manifests)
    next_checks, next_summary, journal_requires_review = next_execution_checks(
        protocol_id,
        executions,
        completed,
        next_execution,
        dataset,
        allow_resume=allow_resume,
    )
    checks.extend(next_checks)

    failures = [check for check in checks if check["status"] == "fail"]
    warnings = [check for check in checks if check["status"] == "warn"]
    invalid_check_ids = {
        "frozen_checkout",
        "sealed_readiness",
        "real_evidence_status",
        "execution_manifests",
        "session_journal",
    }
    valid = not any(
        check["status"] == "fail" and check["check_id"] in invalid_check_ids
        for check in checks
    )
    collection_complete = next_execution is None and not failures
    ready_to_record = (
        next_execution is not None and not failures and not journal_requires_review
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": DOCTOR_REPORT_KIND,
        "generated_at_utc": _utc_now(),
        "protocol_id": protocol_id,
        "protocol_design_sha256": design_sha,
        "repository_root": str(repository.resolve()),
        "dataset_root": str(dataset.resolve()),
        "thresholds": asdict(settings),
        "resume_acknowledged": allow_resume,
        "checks": checks,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "completed_executions": len(completed),
        "total_executions": len(executions),
        "next_execution": next_summary,
        "ready_to_record": ready_to_record,
        "collection_complete": collection_complete,
        "valid": valid,
        "passed": ready_to_record or collection_complete,
        "target_outcomes_used": False,
    }
    report["report_sha256"] = _canonical_sha256(report, omitted="report_sha256")
    return report


__all__ = [
    "DOCTOR_REPORT_KIND",
    "DoctorThresholds",
    "build_acquisition_doctor_report",
]
