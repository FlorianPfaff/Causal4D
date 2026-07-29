"""Same-grasp session continuity and analysis-readiness checks."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from causal4d.real_evidence_common import (
    SESSION_MANIFEST_SCHEMA_VERSION,
    _error_text,
    _load_json_mapping,
    _parse_utc_timestamp,
    _require,
    _sha256_file,
)


def _validate_session_manifest(
    protocol: Mapping[str, Any],
    session: Mapping[str, Any],
    path: Path,
    *,
    execution_results: Mapping[str, Mapping[str, Any]],
    contact_registration_sha256: str | None,
    timebase_calibration_sha256: str | None,
    clock_domain_id: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "session_id": str(session["session_id"]),
        "manifest_path": str(path),
        "manifest_present": path.is_file(),
        "validated": False,
        "error": None,
    }
    if not path.is_file():
        result["error"] = "session.json is missing"
        return result
    try:
        payload = _load_json_mapping(path)
        result["manifest_sha256"], result["manifest_bytes"] = _sha256_file(path)
        _require(
            payload.get("schema_version") == SESSION_MANIFEST_SCHEMA_VERSION,
            "unsupported session manifest schema",
        )
        _require(
            payload.get("artifact_kind") == "SameGraspSessionManifest",
            "unexpected session manifest kind",
        )
        _require(
            payload.get("protocol_id") == protocol["protocol_id"],
            "session protocol mismatch",
        )
        _require(
            payload.get("protocol_design_sha256") == protocol["design_sha256"],
            "session protocol digest mismatch",
        )
        _require(
            payload.get("session_id") == session["session_id"], "session id changed"
        )
        _require(
            payload.get("acquisition_session_index")
            == session["acquisition_session_index"],
            "session acquisition index changed",
        )
        _require(
            payload.get("acquisition_status") == "complete", "session is incomplete"
        )
        grasp_instance_id = payload.get("grasp_instance_id")
        _require(
            isinstance(grasp_instance_id, str) and bool(grasp_instance_id),
            "session grasp_instance_id is missing",
        )
        _require(clock_domain_id is not None, "approved timebase is unavailable")
        _require(
            payload.get("clock_domain_id") == clock_domain_id,
            "session clock domain changed",
        )
        _require(
            payload.get("contact_registration_sha256") == contact_registration_sha256,
            "session binds a different contact registration",
        )
        _require(
            payload.get("timebase_calibration_sha256") == timebase_calibration_sha256,
            "session binds a different timebase calibration",
        )
        _require(
            isinstance(payload.get("operator_id"), str)
            and bool(payload["operator_id"]),
            "session operator id is missing",
        )
        started_at = _parse_utc_timestamp(
            payload.get("started_at_utc"), name="session started_at_utc"
        )
        ended_at = _parse_utc_timestamp(
            payload.get("ended_at_utc"), name="session ended_at_utc"
        )
        _require(ended_at > started_at, "session end must follow its start")

        execution_by_id = {
            execution["execution_id"]: execution for execution in protocol["executions"]
        }
        expected_order = sorted(
            session["execution_ids"],
            key=lambda identifier: execution_by_id[identifier]["pair_order"],
        )
        _require(
            payload.get("execution_order") == expected_order,
            "session execution order changed",
        )
        _require(
            payload.get("same_grasp_confirmed") is True, "same grasp was not confirmed"
        )
        _require(
            payload.get("release_between_executions") is False,
            "object release occurred between paired executions",
        )
        _require(
            payload.get("neutral_state_checks")
            == {
                "before_first": True,
                "between_executions": True,
                "after_second": True,
            },
            "session neutral-state checks are incomplete",
        )
        hashes = payload.get("execution_manifest_sha256")
        _require(
            isinstance(hashes, Mapping) and set(hashes) == set(expected_order),
            "session execution hash inventory changed",
        )
        ordered_results = []
        for identifier in expected_order:
            execution_result = execution_results[identifier]
            _require(
                execution_result["validated"],
                f"session execution is invalid: {identifier}",
            )
            _require(
                hashes[identifier] == execution_result.get("manifest_sha256"),
                f"session execution manifest checksum mismatch: {identifier}",
            )
            _require(
                execution_result.get("grasp_instance_id") == grasp_instance_id,
                f"execution grasp identity differs within session: {identifier}",
            )
            _require(
                execution_result.get("clock_domain_id") == clock_domain_id,
                f"execution clock domain differs within session: {identifier}",
            )
            ordered_results.append(execution_result)
        first, second = ordered_results
        _require(
            started_at
            <= first["_started_at"]
            < first["_ended_at"]
            <= second["_started_at"]
            < second["_ended_at"]
            <= ended_at,
            "paired executions are not chronological and non-overlapping",
        )
        approval = payload.get("approval", {})
        _require(approval.get("approved") is True, "session approval is missing")
        _require(
            isinstance(approval.get("approver_id"), str)
            and bool(approval["approver_id"]),
            "session approver id is missing",
        )
        approved_at = _parse_utc_timestamp(
            approval.get("approved_at_utc"), name="session approved_at_utc"
        )
        _require(
            approved_at >= ended_at, "session approval predates acquisition completion"
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        result["error"] = _error_text(error)
        return result
    result["validated"] = True
    result["grasp_instance_id"] = grasp_instance_id
    result["clock_domain_id"] = clock_domain_id
    return result


def _unexpected_directories(root: Path, *, expected_ids: set[str]) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and path.name not in expected_ids
    )


def _analysis_readiness(
    protocol: Mapping[str, Any],
    execution_results: list[Mapping[str, Any]],
) -> dict[str, Any]:
    included_ids = {
        str(result["execution_id"])
        for result in execution_results
        if result.get("included") is True
    }
    folds = []
    for fold in protocol["splits"]["cross_action_contact_calibration_folds"]:
        fit = set(fold["fit_execution_ids"])
        calibration = set(fold["calibration_execution_ids"])
        target = set(fold["target_execution_ids"])
        row = {
            "outer_fold_id": fold.get(
                "outer_fold_id",
                f"hold-{fold['held_out_contact_region_id']}-{fold['held_out_command_profile_id']}",
            ),
            "included_fit": len(fit & included_ids),
            "registered_fit": len(fit),
            "included_calibration": len(calibration & included_ids),
            "registered_calibration": len(calibration),
            "included_target": len(target & included_ids),
            "registered_target": len(target),
        }
        row["analysable"] = bool(
            row["included_fit"] >= 1
            and row["included_calibration"] >= 1
            and row["included_target"] >= 1
        )
        folds.append(row)
    same_grasp_pairs = protocol["splits"]["same_grasp_intervention_prediction"]
    included_pairs = sum(
        pair["source_execution_id"] in included_ids
        and pair["target_execution_id"] in included_ids
        for pair in same_grasp_pairs
    )
    blockers = [str(row["outer_fold_id"]) for row in folds if not row["analysable"]]
    if included_pairs == 0:
        blockers.append("same_grasp_intervention_prediction")
    return {
        "analysis_ready": not blockers,
        "full_registered_power": len(included_ids) == len(execution_results),
        "included_same_grasp_pairs": included_pairs,
        "registered_same_grasp_pairs": len(same_grasp_pairs),
        "folds": folds,
        "blockers": blockers,
    }


def _claim_blockers(
    prerequisites: Mapping[str, Mapping[str, Any]],
    execution_results: list[Mapping[str, Any]],
    session_results: list[Mapping[str, Any]],
    *,
    unexpected_execution_directories: list[str],
    unexpected_session_directories: list[str],
    verify_file_hashes: bool,
) -> list[str]:
    blockers = [
        f"prerequisite:{name}"
        for name, result in prerequisites.items()
        if not result.get("valid")
    ]
    missing = [
        str(result["execution_id"])
        for result in execution_results
        if not result["manifest_present"]
    ]
    incomplete = [
        str(result["execution_id"])
        for result in execution_results
        if result["manifest_parsed"] and not result["acquired"]
    ]
    invalid = [
        str(result["execution_id"])
        for result in execution_results
        if result["manifest_present"]
        and (
            not result["manifest_parsed"]
            or (result["acquired"] and not result["validated"])
        )
    ]
    missing_sessions = [
        str(result["session_id"])
        for result in session_results
        if not result["manifest_present"]
    ]
    invalid_sessions = [
        str(result["session_id"])
        for result in session_results
        if result["manifest_present"] and not result["validated"]
    ]
    if missing:
        blockers.append(f"missing_execution_manifests:{len(missing)}")
    if incomplete:
        blockers.append(f"incomplete_execution_manifests:{len(incomplete)}")
    if invalid:
        blockers.append(f"invalid_execution_manifests:{len(invalid)}")
    if missing_sessions:
        blockers.append(f"missing_session_manifests:{len(missing_sessions)}")
    if invalid_sessions:
        blockers.append(f"invalid_session_manifests:{len(invalid_sessions)}")
    if unexpected_execution_directories:
        blockers.append(
            f"unexpected_execution_directories:{len(unexpected_execution_directories)}"
        )
    if unexpected_session_directories:
        blockers.append(
            f"unexpected_session_directories:{len(unexpected_session_directories)}"
        )
    if not verify_file_hashes:
        blockers.append("file_hashes_not_verified")
    return blockers


__all__ = [
    "_analysis_readiness",
    "_claim_blockers",
    "_unexpected_directories",
    "_validate_session_manifest",
]
