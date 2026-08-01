"""Identity-policy integration for acquisition approvals and freeze evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causal4d.operator_registry import (
    load_registered_operator_registry,
    validate_attestation_operator_identities,
    validate_gate_approver_identity,
    validate_method_freeze_operator_identity,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    _require(path.is_file(), f"{name} is missing")
    _require(not path.is_symlink(), f"{name} must not be a symlink")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), f"{name} must be a JSON object")
    return dict(payload)


def validate_gate_file_operator_identity(
    gate_id: str,
    path: str | Path,
    registry: Mapping[str, Any],
    prerequisites: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the identity behind one otherwise-valid operational gate."""

    gate = _read_json_mapping(Path(path), name=f"{gate_id} gate")
    approval = gate.get("approval")
    _require(isinstance(approval, Mapping), f"gate approval is invalid: {gate_id}")
    freezer_digest = prerequisites.get("method_freeze", {}).get(
        "freezer_person_identity_sha256"
    )
    approver = validate_gate_approver_identity(
        gate_id,
        approval.get("approver_id"),
        approval.get("approved_at_utc"),
        registry,
        freezer_person_identity_sha256=(
            str(freezer_digest) if freezer_digest is not None else None
        ),
    )
    return {
        "approver_operator_id": str(approver["operator_id"]),
        "approver_person_identity_sha256": str(
            approver["person_identity_sha256"]
        ),
    }


def validate_method_freeze_identity_evidence(
    method_freeze: Mapping[str, Any],
    attestation: Mapping[str, Any] | None,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the freezer and, when present, the independent verifier."""

    freezer = validate_method_freeze_operator_identity(method_freeze, registry)
    result = {
        "freezer_operator_id": str(freezer["operator_id"]),
        "freezer_person_identity_sha256": str(
            freezer["person_identity_sha256"]
        ),
    }
    if attestation is not None:
        _, verifier = validate_attestation_operator_identities(
            method_freeze,
            attestation,
            registry,
        )
        result.update(
            {
                "verifier_operator_id": str(verifier["operator_id"]),
                "verifier_person_identity_sha256": str(
                    verifier["person_identity_sha256"]
                ),
            }
        )
    return result


def seal_registered_preacquisition_gate(
    repository_root: str | Path,
    dataset_root: str | Path,
    gate_id: str,
    *,
    approved_by: str,
    approved_at_utc: str | None = None,
) -> dict[str, Any]:
    """Require a registered, role-compatible approver before sealing a gate."""

    from causal4d.preacquisition_gate_validation import seal_preacquisition_gate

    registry_result, registry = load_registered_operator_registry(
        repository_root,
        dataset_root,
    )
    _require(
        registry_result.get("valid") is True and registry is not None,
        str(registry_result.get("error") or "operator registry is invalid"),
    )
    approved_at = approved_at_utc or datetime.now(timezone.utc).isoformat()
    freezer_digest: str | None = None
    if gate_id == "software_environment_locked":
        method_freeze = _read_json_mapping(
            Path(dataset_root) / "method_freeze.json",
            name="method freeze",
        )
        freezer = validate_method_freeze_operator_identity(method_freeze, registry)
        freezer_digest = str(freezer["person_identity_sha256"])
    validate_gate_approver_identity(
        gate_id,
        approved_by,
        approved_at,
        registry,
        freezer_person_identity_sha256=freezer_digest,
    )
    result = seal_preacquisition_gate(
        repository_root,
        dataset_root,
        gate_id,
        approved_by=approved_by,
        approved_at_utc=approved_at,
    )
    result["operator_registry_artifact_sha256"] = registry_result[
        "artifact_sha256"
    ]
    return result


__all__ = [
    "seal_registered_preacquisition_gate",
    "validate_gate_file_operator_identity",
    "validate_method_freeze_identity_evidence",
]
