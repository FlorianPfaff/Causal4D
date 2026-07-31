"""Validation and atomic sealing for pre-acquisition gate evidence."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causal4d.preacquisition_readiness_contracts import (
    GATE_EVIDENCE_ARTIFACT_KIND,
    GATE_EVIDENCE_SCHEMA_VERSION,
    GATE_PATHS,
    OPERATIONAL_GATES_BEFORE_FREEZE,
    REQUIRED_DRY_RUN_STAGES,
    _finite_number,
    _is_hex_digest,
    _parse_utc_timestamp,
    _read_json_mapping,
    _require,
    _sha256_file,
    _validate_descriptor,
    _validate_descriptor_list,
    gate_evidence_sha256,
    load_registered_preacquisition_chain,
)
from causal4d.preacquisition_source_validation import (
    _validate_actuator_sync,
    _validate_signature_panel,
)
from causal4d.real_evidence_contract_v2 import build_real_evidence_status


def _validate_common_gate(
    protocol: Mapping[str, Any],
    v4: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    gate_id: str,
    dataset_root: Path,
    verify_file_hashes: bool,
) -> tuple[Mapping[str, Any], set[str], datetime]:
    _require(
        gate.get("schema_version") == GATE_EVIDENCE_SCHEMA_VERSION,
        f"unsupported gate schema: {gate_id}",
    )
    _require(
        gate.get("artifact_kind") == GATE_EVIDENCE_ARTIFACT_KIND,
        f"unexpected gate artifact kind: {gate_id}",
    )
    _require(gate.get("gate_id") == gate_id, f"gate id mismatch: {gate_id}")
    _require(gate.get("status") == "passed", f"gate is not passed: {gate_id}")
    _require(
        gate.get("protocol_id") == protocol["protocol_id"],
        f"gate protocol mismatch: {gate_id}",
    )
    _require(
        gate.get("protocol_design_sha256") == protocol["design_sha256"],
        f"gate protocol digest mismatch: {gate_id}",
    )
    _require(
        gate.get("preacquisition_plan_id") == v4["plan_id"],
        f"gate v4 plan mismatch: {gate_id}",
    )
    _require(
        gate.get("preacquisition_amendment_sha256") == v4["amendment_sha256"],
        f"gate v4 digest mismatch: {gate_id}",
    )
    _require(
        gate.get("locked_before_confirmatory_collection") is True,
        f"gate was not locked before collection: {gate_id}",
    )
    _require(
        gate.get("target_outcomes_used") is False,
        f"target outcomes entered gate evidence: {gate_id}",
    )
    completed_at = _parse_utc_timestamp(
        gate.get("completed_at_utc"),
        name=f"{gate_id} completed_at_utc",
    )
    approval = gate.get("approval")
    _require(isinstance(approval, Mapping), f"gate approval is invalid: {gate_id}")
    _require(approval.get("approved") is True, f"gate approval is missing: {gate_id}")
    _require(
        isinstance(approval.get("approver_id"), str)
        and bool(str(approval["approver_id"]).strip()),
        f"gate approver is missing: {gate_id}",
    )
    approved_at = _parse_utc_timestamp(
        approval.get("approved_at_utc"),
        name=f"{gate_id} approved_at_utc",
    )
    _require(
        approved_at >= completed_at, f"gate approval predates completion: {gate_id}"
    )
    _require(
        gate.get("artifact_sha256") == gate_evidence_sha256(gate),
        f"gate digest mismatch: {gate_id}",
    )
    checks = gate.get("checks")
    _require(isinstance(checks, Mapping), f"gate checks are invalid: {gate_id}")
    evidence_paths = _validate_descriptor_list(
        dataset_root,
        gate.get("evidence"),
        name=f"{gate_id}.evidence",
        verify_file_hashes=verify_file_hashes,
    )
    return checks, evidence_paths, approved_at


def _validate_support_registration(
    checks: Mapping[str, Any],
    evidence_paths: set[str],
) -> None:
    for field in (
        "support_geometry_registered",
        "gravity_registered",
        "quality_gate_passed",
    ):
        _require(checks.get(field) is True, f"support registration failed: {field}")
    _require(
        isinstance(checks.get("world_frame_id"), str)
        and bool(str(checks["world_frame_id"]).strip()),
        "support world frame is missing",
    )
    vector = checks.get("gravity_vector_mps2")
    _require(isinstance(vector, list) and len(vector) == 3, "gravity vector is invalid")
    gravity = [_finite_number(value, name="gravity component") for value in vector]
    magnitude = math.sqrt(sum(value * value for value in gravity))
    _require(
        8.0 <= magnitude <= 11.5, "gravity magnitude fails the physical sanity gate"
    )
    closure = _finite_number(
        checks.get("registration_closure_error_m"),
        name="support registration closure error",
    )
    threshold = _finite_number(
        checks.get("maximum_registration_closure_error_m"),
        name="support registration closure threshold",
    )
    _require(closure >= 0.0 and threshold > 0.0, "support closure values are invalid")
    _require(
        closure <= threshold, "support registration closure error exceeds its gate"
    )
    relative = checks.get("registration_file")
    _require(
        isinstance(relative, str) and relative in evidence_paths,
        "support registration file is not bound as evidence",
    )


def _validate_dry_run(
    protocol: Mapping[str, Any],
    checks: Mapping[str, Any],
    evidence_paths: set[str],
) -> None:
    _require(checks.get("nonconfirmatory") is True, "dry run is not nonconfirmatory")
    _require(
        checks.get("target_outcomes_used") is False,
        "target outcomes entered the dry run",
    )
    _require(
        checks.get("frozen_entrypoints_exercised") is True,
        "dry run did not exercise the frozen entrypoints",
    )
    execution_id = checks.get("execution_id")
    _require(
        isinstance(execution_id, str) and bool(execution_id), "dry-run id is missing"
    )
    confirmatory_ids = {
        str(execution["execution_id"]) for execution in protocol["executions"]
    }
    _require(execution_id not in confirmatory_ids, "dry run reuses a confirmatory id")
    stages = checks.get("pipeline_stages")
    _require(isinstance(stages, Mapping), "dry-run stage map is invalid")
    _require(
        set(stages) == set(REQUIRED_DRY_RUN_STAGES),
        "dry-run stage set differs from the registered readiness contract",
    )
    _require(
        all(stages[name] is True for name in REQUIRED_DRY_RUN_STAGES), "dry run failed"
    )
    relative = checks.get("output_manifest")
    _require(
        isinstance(relative, str) and relative in evidence_paths,
        "dry-run output manifest is not bound as evidence",
    )


def _validate_distribution_descriptor(
    dataset_root: Path,
    value: Any,
    *,
    name: str,
    evidence_paths: set[str],
    verify_file_hashes: bool,
) -> None:
    _require(isinstance(value, Mapping), f"{name} distribution descriptor is invalid")
    path = _validate_descriptor(
        dataset_root,
        value,
        name=f"{name} distribution",
        verify_file_hashes=verify_file_hashes,
    )
    _require(path in evidence_paths, f"{name} distribution is not bound as evidence")


def _validate_software_environment(
    checks: Mapping[str, Any],
    evidence_paths: set[str],
    *,
    dataset_root: Path,
    prerequisites: Mapping[str, Mapping[str, Any]],
    verify_file_hashes: bool,
) -> None:
    freeze = prerequisites["method_freeze"]
    _require(
        freeze.get("valid") is True, "software lock requires a valid method freeze"
    )
    _require(
        checks.get("method_freeze_sha256") == freeze.get("sha256"),
        "software lock binds a different method freeze",
    )
    attestation = prerequisites["method_freeze_validation"]
    _require(
        attestation.get("valid") is True,
        "software lock requires an independently attested method freeze",
    )
    _require(
        checks.get("method_freeze_validation_sha256") == attestation.get("sha256"),
        "software lock binds a different freeze attestation",
    )
    for name, commit_field in (
        ("causal4d", "causal4d_commit_sha"),
        ("bayesian_phystwin", "bayesian_phystwin_commit_sha"),
    ):
        package = checks.get(name)
        _require(isinstance(package, Mapping), f"{name} software lock is missing")
        _require(
            package.get("commit_sha") == freeze.get(commit_field),
            f"{name} commit differs from the method freeze",
        )
        _require(
            isinstance(package.get("version"), str) and bool(package["version"]),
            f"{name} package version is missing",
        )
        _validate_distribution_descriptor(
            dataset_root,
            package.get("distribution"),
            name=name,
            evidence_paths=evidence_paths,
            verify_file_hashes=verify_file_hashes,
        )
    prob4d = checks.get("prob4d")
    _require(isinstance(prob4d, Mapping), "Prob4D software declaration is missing")
    _require(isinstance(prob4d.get("used"), bool), "Prob4D used flag is invalid")
    if prob4d["used"]:
        _require(
            _is_hex_digest(prob4d.get("commit_sha"), 40), "Prob4D commit is invalid"
        )
        _require(
            isinstance(prob4d.get("version"), str) and bool(prob4d["version"]),
            "Prob4D version is missing",
        )
        _require(
            isinstance(prob4d.get("observation_contract_version"), str)
            and bool(prob4d["observation_contract_version"]),
            "Prob4D observation contract is missing",
        )
        _validate_distribution_descriptor(
            dataset_root,
            prob4d.get("distribution"),
            name="prob4d",
            evidence_paths=evidence_paths,
            verify_file_hashes=verify_file_hashes,
        )
    else:
        _require(
            isinstance(prob4d.get("reason"), str) and bool(prob4d["reason"].strip()),
            "the unused Prob4D declaration needs a reason",
        )
    producer = checks.get("observation_producer")
    _require(
        isinstance(producer, Mapping), "observation producer declaration is missing"
    )
    for field in ("name", "version", "artifact_contract"):
        _require(
            isinstance(producer.get(field), str) and bool(producer[field].strip()),
            f"observation producer {field} is missing",
        )
    python = checks.get("python")
    _require(isinstance(python, Mapping), "Python environment declaration is missing")
    for field in ("version", "implementation", "platform"):
        _require(
            isinstance(python.get(field), str) and bool(python[field].strip()),
            f"Python environment {field} is missing",
        )


def _validate_gate_file(
    gate_id: str,
    path: Path,
    *,
    protocol: Mapping[str, Any],
    v2: Mapping[str, Any],
    v4: Mapping[str, Any],
    dataset_root: Path,
    prerequisites: Mapping[str, Mapping[str, Any]],
    verify_file_hashes: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "gate_id": gate_id,
        "path": str(path.resolve()),
        "present": path.is_file(),
        "template": False,
        "valid": False,
        "approved_at_utc": None,
        "error": None,
    }
    if not result["present"]:
        result["error"] = "gate evidence is missing"
        return result
    try:
        gate = _read_json_mapping(path, name=f"{gate_id} gate")
        if gate.get("status") == "template":
            _require(
                gate.get("schema_version") == GATE_EVIDENCE_SCHEMA_VERSION,
                "unsupported gate template schema",
            )
            _require(
                gate.get("artifact_kind") == GATE_EVIDENCE_ARTIFACT_KIND,
                "unexpected gate template kind",
            )
            _require(gate.get("gate_id") == gate_id, "gate template id mismatch")
            _require(
                gate.get("protocol_id") == protocol["protocol_id"],
                "gate template protocol id mismatch",
            )
            _require(
                gate.get("protocol_design_sha256") == protocol["design_sha256"],
                "gate template protocol mismatch",
            )
            _require(
                gate.get("preacquisition_plan_id") == v4["plan_id"],
                "gate template v4 plan mismatch",
            )
            _require(
                gate.get("preacquisition_amendment_sha256") == v4["amendment_sha256"],
                "gate template v4 mismatch",
            )
            _require(
                gate.get("artifact_sha256") is None,
                "gate template already contains an artifact digest",
            )
            _require(
                isinstance(gate.get("checks"), Mapping),
                "gate template checks are invalid",
            )
            _require(
                isinstance(gate.get("evidence"), list),
                "gate template evidence is invalid",
            )
            approval = gate.get("approval")
            _require(
                isinstance(approval, Mapping) and approval.get("approved") is False,
                "gate template approval is invalid",
            )
            result["template"] = True
            result["error"] = "gate evidence is still a template"
            return result
        checks, evidence_paths, approved_at = _validate_common_gate(
            protocol,
            v4,
            gate,
            gate_id=gate_id,
            dataset_root=dataset_root,
            verify_file_hashes=verify_file_hashes,
        )
        if gate_id == "signature_panel_complete":
            _validate_signature_panel(
                protocol,
                v2,
                v4,
                checks,
                evidence_paths,
                dataset_root=dataset_root,
                verify_file_hashes=verify_file_hashes,
            )
        elif gate_id == "actuator_sync_passed":
            _validate_actuator_sync(
                protocol,
                v2,
                checks,
                evidence_paths,
                dataset_root=dataset_root,
            )
        elif gate_id == "support_registration_passed":
            _validate_support_registration(checks, evidence_paths)
        elif gate_id == "end_to_end_dry_run_passed":
            _validate_dry_run(protocol, checks, evidence_paths)
        elif gate_id == "software_environment_locked":
            _validate_software_environment(
                checks,
                evidence_paths,
                dataset_root=dataset_root,
                prerequisites=prerequisites,
                verify_file_hashes=verify_file_hashes,
            )
        else:
            raise KeyError(gate_id)
        result["valid"] = True
        result["approved_at_utc"] = gate["approval"]["approved_at_utc"]
        result["artifact_sha256"] = gate["artifact_sha256"]
        result["sha256"], result["bytes"] = _sha256_file(path)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        message = str(error).strip()
        result["error"] = (
            f"{type(error).__name__}: {message}" if message else type(error).__name__
        )
    return result


def seal_preacquisition_gate(
    repository_root: str | Path,
    dataset_root: str | Path,
    gate_id: str,
    *,
    approved_by: str,
    approved_at_utc: str | None = None,
) -> dict[str, Any]:
    """Seal one completed gate after validating all bound evidence hashes."""

    _require(gate_id in GATE_PATHS, f"unknown pre-acquisition gate: {gate_id}")
    protocol, v2, _, v4 = load_registered_preacquisition_chain(repository_root)
    root = Path(dataset_root)
    path = root / GATE_PATHS[gate_id]
    gate = _read_json_mapping(path, name=f"{gate_id} gate")
    _require(gate.get("status") == "template", "gate evidence is already sealed")
    _require(
        gate.get("artifact_sha256") is None,
        "gate template already contains an artifact digest",
    )
    _require(
        gate.get("target_outcomes_used") is False,
        "target_outcomes_used must be explicitly false before sealing",
    )
    approver = str(approved_by).strip()
    _require(bool(approver), "approved_by is required")
    _parse_utc_timestamp(gate.get("completed_at_utc"), name="gate completed_at_utc")
    approved_at = approved_at_utc or datetime.now(timezone.utc).isoformat()
    _parse_utc_timestamp(approved_at, name="gate approved_at_utc")
    gate["status"] = "passed"
    gate["locked_before_confirmatory_collection"] = True
    gate["approval"] = {
        "approved": True,
        "approver_id": approver,
        "approved_at_utc": approved_at,
    }
    gate["artifact_sha256"] = gate_evidence_sha256(gate)

    real_status = build_real_evidence_status(
        protocol,
        root,
        repository_root=repository_root,
        verify_file_hashes=True,
    )
    _require(
        int(real_status.get("manifest_executions", 0)) == 0
        and int(real_status.get("acquired_executions", 0)) == 0
        and int(real_status.get("validated_executions", 0)) == 0,
        "confirmatory collection has already started",
    )
    freeze = real_status["prerequisites"]["method_freeze"]
    if gate_id in OPERATIONAL_GATES_BEFORE_FREEZE and freeze.get("valid"):
        _require(
            _parse_utc_timestamp(approved_at, name="gate approved_at_utc")
            <= _parse_utc_timestamp(
                freeze.get("frozen_at_utc"),
                name="method freeze frozen_at_utc",
            ),
            "operational gate approval postdates the method freeze",
        )
    if gate_id == "software_environment_locked":
        attestation = real_status["prerequisites"]["method_freeze_validation"]
        _require(
            freeze.get("valid") is True and attestation.get("valid") is True,
            "software environment requires a valid, independently attested freeze",
        )
        approved = _parse_utc_timestamp(approved_at, name="gate approved_at_utc")
        _require(
            approved
            >= _parse_utc_timestamp(
                freeze.get("frozen_at_utc"),
                name="method freeze frozen_at_utc",
            ),
            "software environment approval predates the method freeze",
        )
        _require(
            approved
            >= _parse_utc_timestamp(
                attestation.get("verified_at_utc"),
                name="method freeze verified_at_utc",
            ),
            "software environment approval predates the freeze attestation",
        )
    # Validate the in-memory candidate before replacing the operator template.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".candidate",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(gate, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        candidate = _validate_gate_file(
            gate_id,
            temporary,
            protocol=protocol,
            v2=v2,
            v4=v4,
            dataset_root=root,
            prerequisites=real_status["prerequisites"],
            verify_file_hashes=True,
        )
        _require(candidate["valid"] is True, str(candidate["error"]))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        **candidate,
        "path": str(path.resolve()),
        "passed": True,
    }


__all__ = [
    "_validate_gate_file",
    "seal_preacquisition_gate",
]
