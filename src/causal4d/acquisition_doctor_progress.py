"""Evidence, execution-order, and recovery checks for acquisition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causal4d.acquisition_flight_common import _require, journal_seal_path
from causal4d.acquisition_journal import validate_acquisition_journal
from causal4d.acquisition_doctor_support import (
    _doctor_check,
    _execution_progress_from_status,
)


def evidence_check(
    protocol: Mapping[str, Any],
    repository: Path,
    dataset: Path,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], list[str]]:
    try:
        from causal4d.real_evidence_contract_v2 import build_real_evidence_status

        evidence_status = build_real_evidence_status(
            protocol,
            dataset,
            repository_root=repository,
            verify_file_hashes=True,
        )
        completed, malformed, present = _execution_progress_from_status(
            protocol,
            evidence_status,
        )
        prerequisites = evidence_status.get("prerequisites")
        _require(
            isinstance(prerequisites, Mapping),
            "real evidence status lacks prerequisites",
        )
        invalid_prerequisites = [
            str(name)
            for name, result in prerequisites.items()
            if not isinstance(result, Mapping) or result.get("valid") is not True
        ]
        structural_errors = {
            "invalid_prerequisites": invalid_prerequisites,
            "invalid_session_ids": list(
                evidence_status.get("invalid_session_ids", [])
            ),
            "unexpected_execution_directories": list(
                evidence_status.get("unexpected_execution_directories", [])
            ),
            "unexpected_session_directories": list(
                evidence_status.get("unexpected_session_directories", [])
            ),
        }
        structural_errors = {
            key: value for key, value in structural_errors.items() if value
        }
        _require(
            evidence_status.get("file_hashes_requested") is True,
            "real evidence status omitted file hash verification",
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        return (
            _doctor_check(
                "real_evidence_status",
                "fail",
                f"Registered evidence validation failed: {error}",
            ),
            [],
            [],
            [],
        )
    return (
        _doctor_check(
            "real_evidence_status",
            "fail" if structural_errors else "pass",
            (
                "Registered evidence has invalid prerequisites or directories."
                if structural_errors
                else "Registered evidence and referenced file hashes validate."
            ),
            specified_executions=evidence_status["specified_executions"],
            manifest_executions=evidence_status["manifest_executions"],
            acquired_executions=evidence_status["acquired_executions"],
            validated_executions=evidence_status["validated_executions"],
            structural_errors=structural_errors,
        ),
        completed,
        malformed,
        present,
    )


def manifest_check(
    executions: Sequence[Mapping[str, Any]],
    completed: list[str],
    malformed: list[dict[str, Any]],
    present: list[str],
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    completed_set = set(completed)
    next_execution = next(
        (
            execution
            for execution in executions
            if execution["execution_id"] not in completed_set
        ),
        None,
    )
    if malformed:
        return (
            _doctor_check(
                "execution_manifests",
                "fail",
                "One or more execution manifests are present but not explicitly "
                "complete.",
                malformed=malformed,
            ),
            next_execution,
        )
    expected_prefix = [
        str(execution["execution_id"]) for execution in executions[: len(completed)]
    ]
    if completed != expected_prefix:
        return (
            _doctor_check(
                "execution_manifests",
                "fail",
                "Completed execution manifests do not form the locked acquisition "
                "prefix.",
                completed=completed,
                expected_prefix=expected_prefix,
                present=present,
            ),
            next_execution,
        )
    return (
        _doctor_check(
            "execution_manifests",
            "pass",
            "Completed execution manifests form the locked acquisition prefix.",
            completed_executions=len(completed),
            total_executions=len(executions),
        ),
        next_execution,
    )


def next_execution_checks(
    protocol_id: str,
    executions: Sequence[Mapping[str, Any]],
    completed: list[str],
    next_execution: Mapping[str, Any] | None,
    dataset: Path,
    *,
    allow_resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, bool]:
    checks: list[dict[str, Any]] = []
    if next_execution is None:
        checks.append(
            _doctor_check(
                "next_execution",
                "pass",
                "All registered executions have complete manifests.",
            )
        )
        return checks, None, False

    next_summary = {
        key: next_execution[key]
        for key in (
            "acquisition_execution_index",
            "execution_id",
            "session_id",
            "pair_order",
            "contact_region_id",
            "command_profile_id",
            "realization_condition_id",
            "replicate_block",
        )
        if key in next_execution
    }
    execution_root = dataset / "executions" / str(next_execution["execution_id"])
    session_root = dataset / "sessions" / str(next_execution["session_id"])
    if execution_root.is_dir() and session_root.is_dir():
        checks.append(
            _doctor_check(
                "next_execution",
                "pass",
                "Next execution and session directories are scaffolded.",
                next_execution=next_summary,
                execution_root=str(execution_root),
                session_root=str(session_root),
            )
        )
    else:
        checks.append(
            _doctor_check(
                "next_execution",
                "fail",
                "Next execution or session directory is missing.",
                next_execution=next_summary,
                execution_root=str(execution_root),
                execution_root_present=execution_root.is_dir(),
                session_root=str(session_root),
                session_root_present=session_root.is_dir(),
            )
        )

    journal_requires_review = False
    journal = session_root / "acquisition.jsonl"
    seal = journal_seal_path(journal)
    if seal.exists():
        checks.append(
            _doctor_check(
                "session_journal",
                "fail",
                "The next session journal is already sealed although an execution "
                "is incomplete.",
                journal_path=str(journal),
                seal_path=str(seal),
            )
        )
    elif journal.exists():
        try:
            validation = validate_acquisition_journal(journal)
            expected_session_id = str(next_execution["session_id"])
            expected_execution_ids = {
                str(execution["execution_id"])
                for execution in executions
                if str(execution["session_id"]) == expected_session_id
            }
            completed_session_ids = {
                identifier
                for identifier in completed
                if identifier in expected_execution_ids
            }
            terminal_journal_ids = set(
                validation["completed_execution_ids"]
            ) | set(validation["aborted_execution_ids"])
            _require(
                validation["protocol_id"] == protocol_id,
                "session journal protocol differs from the registered protocol",
            )
            _require(
                validation["session_id"] == expected_session_id,
                "session journal identifies the wrong session",
            )
            _require(
                set(validation["seen_execution_ids"]) <= expected_execution_ids,
                "session journal names an execution outside this session",
            )
            _require(
                terminal_journal_ids == completed_session_ids,
                "session journal terminal executions differ from validated manifests",
            )
            active_execution_id = validation["active_execution_id"]
            _require(
                active_execution_id is None
                or active_execution_id == next_execution["execution_id"],
                "session journal has the wrong active execution",
            )
        except (OSError, TypeError, ValueError) as error:
            checks.append(
                _doctor_check(
                    "session_journal",
                    "fail",
                    f"Existing session journal is invalid: {error}",
                    journal_path=str(journal),
                )
            )
        else:
            journal_requires_review = not allow_resume
            checks.append(
                _doctor_check(
                    "session_journal",
                    "warn",
                    (
                        "An unsealed session journal exists and resume was explicitly "
                        "acknowledged."
                        if allow_resume
                        else "An unsealed session journal exists; review it and rerun "
                        "with --allow-resume before recording."
                    ),
                    journal_path=str(journal),
                    resume_acknowledged=allow_resume,
                    validation=validation,
                )
            )
    else:
        checks.append(
            _doctor_check(
                "session_journal",
                "pass",
                "No prior journal bytes exist for the next session.",
                journal_path=str(journal),
            )
        )
    return checks, next_summary, journal_requires_review
