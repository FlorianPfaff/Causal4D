"""Validate a persisted pre-acquisition action immediately before use."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from causal4d.acquisition_flight_common import (
    _assert_no_symlink_components,
    _require,
)
from causal4d.atomic_io import atomic_write_json
from causal4d.preacquisition_next_action import (
    NEXT_ACTION_ARTIFACT_KIND,
    next_action_evidence_sha256,
    next_action_status_sha256,
)
from causal4d.preacquisition_operator_flow import (
    NEXT_ACTION_SCHEMA_VERSION,
    build_preacquisition_operator_next_action,
)
from causal4d.preacquisition_readiness_contracts import (
    _canonical_sha256,
    _read_json_mapping,
    _sha256_file,
)

NEXT_ACTION_VALIDATION_SCHEMA_VERSION = 1
NEXT_ACTION_VALIDATION_ARTIFACT_KIND = "Causal4DPreacquisitionNextActionValidation"


def _action_identity(decision: Mapping[str, Any]) -> dict[str, Any]:
    action = decision.get("action")
    _require(isinstance(action, Mapping), "next-action decision has no action object")
    registered = action.get("registered_execution")
    execution_id = None
    session_id = None
    if isinstance(registered, Mapping):
        execution_id = registered.get("execution_id")
        session_id = registered.get("session_id")
    return {
        "action_id": action.get("action_id"),
        "category": action.get("category"),
        "execution_id": execution_id,
        "session_id": session_id,
    }


def _validate_decision_envelope(decision: Mapping[str, Any]) -> None:
    _require(
        decision.get("schema_version") == NEXT_ACTION_SCHEMA_VERSION,
        "unsupported next-action decision schema",
    )
    _require(
        decision.get("artifact_kind") == NEXT_ACTION_ARTIFACT_KIND,
        "unexpected next-action artifact kind",
    )
    _require(
        decision.get("target_outcomes_used") is False,
        "target outcomes entered the next-action decision",
    )
    action = decision.get("action")
    _require(isinstance(action, Mapping), "next-action decision has no action object")
    _require(
        action.get("target_outcomes_permitted") is False,
        "next-action decision permits target outcomes",
    )
    _require(
        action.get("changes_registered_method") is False,
        "next-action decision permits a registered-method change",
    )
    _require(
        decision.get("evidence_sha256") == next_action_evidence_sha256(decision),
        "next-action evidence SHA-256 mismatch",
    )
    _require(
        decision.get("status_sha256") == next_action_status_sha256(decision),
        "next-action status SHA-256 mismatch",
    )


def next_action_validation_evidence_sha256(values: Mapping[str, Any]) -> str:
    """Return a mount-independent digest for one freshness validation."""

    payload = deepcopy(dict(values))
    for field in (
        "repository_root",
        "dataset_root",
        "decision_json",
        "decision_file_sha256",
        "decision_file_bytes",
        "decision_status_sha256",
        "current_status_sha256",
        "evidence_sha256",
        "status_sha256",
    ):
        payload.pop(field, None)
    return _canonical_sha256(payload, omitted_field="evidence_sha256")


def next_action_validation_status_sha256(values: Mapping[str, Any]) -> str:
    """Return the digest of the exact host-local freshness validation."""

    return _canonical_sha256(values, omitted_field="status_sha256")


def validate_preacquisition_next_action_report(
    repository_root: str | Path,
    dataset_root: str | Path,
    decision_json: str | Path,
) -> dict[str, Any]:
    """Require a persisted action to equal the current hash-verified decision."""

    repository = Path(repository_root).resolve()
    dataset = Path(dataset_root).resolve()
    source = Path(decision_json)
    _assert_no_symlink_components(source, name="next-action decision")
    _require(source.is_file(), "next-action decision is missing")
    source = source.resolve(strict=True)
    decision_digest_before, decision_bytes_before = _sha256_file(source)
    decision = _read_json_mapping(source, name="next-action decision")
    _validate_decision_envelope(decision)

    _require(
        decision.get("repository_root") == str(repository),
        "next-action repository root differs from the current checkout",
    )
    _require(
        decision.get("dataset_root") == str(dataset),
        "next-action dataset root differs from the current evidence tree",
    )

    current = build_preacquisition_operator_next_action(
        repository,
        dataset,
        verify_file_hashes=True,
    )
    _validate_decision_envelope(current)
    decision_digest_after, decision_bytes_after = _sha256_file(source)
    _require(
        (decision_digest_after, decision_bytes_after)
        == (decision_digest_before, decision_bytes_before),
        "next-action decision changed during validation",
    )

    supplied_identity = _action_identity(decision)
    current_identity = _action_identity(current)
    _require(
        decision.get("evidence_sha256") == current.get("evidence_sha256"),
        "next-action decision is stale",
    )
    _require(
        supplied_identity == current_identity,
        "next-action identity differs from the current decision",
    )

    report: dict[str, Any] = {
        "schema_version": NEXT_ACTION_VALIDATION_SCHEMA_VERSION,
        "artifact_kind": NEXT_ACTION_VALIDATION_ARTIFACT_KIND,
        "protocol_id": current["protocol_id"],
        "protocol_design_sha256": current["protocol_design_sha256"],
        "preacquisition_plan_id": current["preacquisition_plan_id"],
        "preacquisition_amendment_sha256": current["preacquisition_amendment_sha256"],
        "repository_root": str(repository),
        "dataset_root": str(dataset),
        "decision_json": str(source),
        "decision_file_sha256": decision_digest_after,
        "decision_file_bytes": decision_bytes_after,
        "decision_evidence_sha256": decision["evidence_sha256"],
        "decision_status_sha256": decision["status_sha256"],
        "current_evidence_sha256": current["evidence_sha256"],
        "current_status_sha256": current["status_sha256"],
        "action_identity": current_identity,
        "readiness_evidence_sha256": current.get("readiness_evidence_sha256"),
        "source_panel_evidence_sha256": current.get("source_panel_evidence_sha256"),
        "current": True,
        "safe_to_execute": True,
        "file_hashes_verified": True,
        "changes_registered_method": False,
        "target_outcomes_used": False,
        "valid": True,
        "complete": True,
        "passed": True,
    }
    report["evidence_sha256"] = next_action_validation_evidence_sha256(report)
    report["status_sha256"] = next_action_validation_status_sha256(report)
    return report


def write_preacquisition_next_action_validation(
    path: str | Path,
    report: Mapping[str, Any],
) -> Path:
    """Atomically write one derived freshness-validation report."""

    output = Path(path)
    atomic_write_json(output, dict(report))
    return output


__all__ = [
    "NEXT_ACTION_VALIDATION_ARTIFACT_KIND",
    "NEXT_ACTION_VALIDATION_SCHEMA_VERSION",
    "next_action_validation_evidence_sha256",
    "next_action_validation_status_sha256",
    "validate_preacquisition_next_action_report",
    "write_preacquisition_next_action_validation",
]
