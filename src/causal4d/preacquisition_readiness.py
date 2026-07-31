"""Fail-closed readiness decision for the registered physical experiment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from causal4d.atomic_io import atomic_write_json
from causal4d.preacquisition_gate_validation import (
    _validate_gate_file,
    seal_preacquisition_gate as seal_preacquisition_gate,
)
from causal4d.preacquisition_readiness_contracts import (
    GATE_EVIDENCE_ARTIFACT_KIND as GATE_EVIDENCE_ARTIFACT_KIND,
    GATE_EVIDENCE_SCHEMA_VERSION as GATE_EVIDENCE_SCHEMA_VERSION,
    GATE_PATHS,
    OPERATIONAL_GATES_BEFORE_FREEZE,
    READINESS_ARTIFACT_KIND,
    READINESS_SCHEMA_VERSION,
    SOURCE_PANEL_MANIFEST_PATH as SOURCE_PANEL_MANIFEST_PATH,
    SOURCE_PANEL_MANIFEST_TEMPLATE_PATH,
    _parse_utc_timestamp,
    gate_evidence_sha256 as gate_evidence_sha256,
    gate_evidence_template,
    load_registered_preacquisition_chain,
    readiness_evidence_sha256,
    readiness_status_sha256,
    source_panel_execution_manifest_template,
)
from causal4d.real_evidence_contract_v2 import build_real_evidence_status


def _publish_template(
    path: Path,
    payload: Mapping[str, Any],
    relative: str,
    *,
    created: list[str],
    existing: list[str],
) -> None:
    try:
        atomic_write_json(path, dict(payload), overwrite=False)
    except FileExistsError:
        existing.append(relative)
    else:
        created.append(relative)


def scaffold_preacquisition_readiness(
    repository_root: str | Path,
    dataset_root: str | Path,
) -> dict[str, Any]:
    """Write incomplete gate templates without overwriting operator evidence."""

    protocol, v2, _, v4 = load_registered_preacquisition_chain(repository_root)
    root = Path(dataset_root)
    created: list[str] = []
    existing: list[str] = []
    for gate_id, relative in GATE_PATHS.items():
        path = root / relative
        _publish_template(
            path,
            gate_evidence_template(gate_id, protocol, v2, v4),
            relative,
            created=created,
            existing=existing,
        )
    source_executions = v2["preacquisition_signature_panel"]["executions"]
    for execution in source_executions:
        relative = SOURCE_PANEL_MANIFEST_TEMPLATE_PATH.format(
            execution_id=execution["execution_id"]
        )
        path = root / relative
        _publish_template(
            path,
            source_panel_execution_manifest_template(execution, protocol, v4),
            relative,
            created=created,
            existing=existing,
        )
    return {
        "passed": True,
        "dataset_root": str(root.resolve()),
        "created": created,
        "existing": existing,
    }


def evaluate_preacquisition_readiness(
    protocol: Mapping[str, Any],
    v2: Mapping[str, Any],
    v4: Mapping[str, Any],
    dataset_root: str | Path,
    real_status: Mapping[str, Any],
    *,
    verify_file_hashes: bool,
) -> dict[str, Any]:
    """Derive all collection gates from validated artifacts and chronology."""

    root = Path(dataset_root)
    prerequisites = real_status["prerequisites"]
    gate_results = {
        gate_id: _validate_gate_file(
            gate_id,
            root / relative,
            protocol=protocol,
            v2=v2,
            v4=v4,
            dataset_root=root,
            prerequisites=prerequisites,
            verify_file_hashes=verify_file_hashes,
        )
        for gate_id, relative in GATE_PATHS.items()
    }

    prerequisite_names = (
        "dataset_protocol",
        "acquisition_schedule",
        "object_registration",
        "slip_pilot",
        "timebase_calibration",
        "contact_registration",
        "method_freeze",
        "method_freeze_validation",
    )
    missing_prerequisites = [
        name for name in prerequisite_names if not prerequisites[name].get("present")
    ]
    malformed_prerequisites = [
        name
        for name in prerequisite_names
        if prerequisites[name].get("present") and not prerequisites[name].get("valid")
    ]
    missing_or_template_gates = [
        gate_id
        for gate_id, result in gate_results.items()
        if not result["present"] or result["template"]
    ]
    malformed_gates = [
        gate_id
        for gate_id, result in gate_results.items()
        if result["present"] and not result["template"] and not result["valid"]
    ]

    manifest_count = int(real_status.get("manifest_executions", 0))
    acquired_count = int(real_status.get("acquired_executions", 0))
    validated_count = int(real_status.get("validated_executions", 0))
    collection_not_started = manifest_count == acquired_count == validated_count == 0

    chronology_blockers: list[str] = []
    freeze = prerequisites["method_freeze"]
    if freeze.get("valid"):
        frozen_at = _parse_utc_timestamp(
            freeze.get("frozen_at_utc"), name="method freeze frozen_at_utc"
        )
        for gate_id in OPERATIONAL_GATES_BEFORE_FREEZE:
            approved_at = gate_results[gate_id].get("approved_at_utc")
            if approved_at is not None:
                approved = _parse_utc_timestamp(
                    approved_at, name=f"{gate_id} approved_at_utc"
                )
                if approved > frozen_at:
                    chronology_blockers.append(
                        f"method_freeze_precedes_operational_gate:{gate_id}"
                    )
        software_approved_at = gate_results["software_environment_locked"].get(
            "approved_at_utc"
        )
        attestation = prerequisites["method_freeze_validation"]
        if software_approved_at is not None:
            approved = _parse_utc_timestamp(
                software_approved_at,
                name="software_environment_locked approved_at_utc",
            )
            if approved < frozen_at:
                chronology_blockers.append(
                    "software_environment_predates_method_freeze"
                )
            if attestation.get("valid"):
                verified_at = _parse_utc_timestamp(
                    attestation.get("verified_at_utc"),
                    name="method freeze verified_at_utc",
                )
                if approved < verified_at:
                    chronology_blockers.append(
                        "software_environment_predates_freeze_attestation"
                    )

    blockers: list[str] = []
    blockers.extend(f"prerequisite:{name}" for name in missing_prerequisites)
    blockers.extend(f"prerequisite_invalid:{name}" for name in malformed_prerequisites)
    blockers.extend(f"gate:{name}" for name in missing_or_template_gates)
    blockers.extend(f"gate_invalid:{name}" for name in malformed_gates)
    blockers.extend(chronology_blockers)
    if not verify_file_hashes:
        blockers.append("file_hashes_not_verified")
    if not collection_not_started:
        blockers.append("confirmatory_collection_already_started")

    flags = {
        "signature_panel_complete": gate_results["signature_panel_complete"]["valid"],
        "contact_registration_approved": prerequisites["contact_registration"].get(
            "valid", False
        ),
        "slip_pilot_passed_or_versioned_out": prerequisites["slip_pilot"].get(
            "valid", False
        ),
        "actuator_sync_passed": gate_results["actuator_sync_passed"]["valid"],
        "support_registration_passed": gate_results["support_registration_passed"][
            "valid"
        ],
        "end_to_end_dry_run_passed": gate_results["end_to_end_dry_run_passed"]["valid"],
        "analysis_code_frozen": bool(
            prerequisites["method_freeze"].get("valid")
            and prerequisites["method_freeze_validation"].get("valid")
        ),
        "software_environment_locked": gate_results["software_environment_locked"][
            "valid"
        ],
    }
    ready = not blockers and all(flags.values())
    flags["first_confirmatory_execution_allowed"] = ready
    valid = (
        not malformed_prerequisites and not malformed_gates and not chronology_blockers
    )
    valid = bool(valid and collection_not_started)

    status: dict[str, Any] = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "artifact_kind": READINESS_ARTIFACT_KIND,
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "preacquisition_plan_id": v4["plan_id"],
        "preacquisition_amendment_sha256": v4["amendment_sha256"],
        "dataset_root": str(root.resolve()),
        "verify_file_hashes": verify_file_hashes,
        "prerequisites": {
            name: dict(prerequisites[name]) for name in prerequisite_names
        },
        "operational_gates": gate_results,
        "collection_gate": flags,
        "confirmatory_collection": {
            "manifest_executions": manifest_count,
            "acquired_executions": acquired_count,
            "validated_executions": validated_count,
            "not_started": collection_not_started,
        },
        "missing_prerequisites": missing_prerequisites,
        "malformed_prerequisites": malformed_prerequisites,
        "missing_or_template_gates": missing_or_template_gates,
        "malformed_gates": malformed_gates,
        "chronology_blockers": chronology_blockers,
        "blockers": blockers,
        "valid": valid,
        "ready": ready,
        "passed": ready,
    }
    status["evidence_sha256"] = readiness_evidence_sha256(status)
    status["status_sha256"] = readiness_status_sha256(status)
    return status


def build_preacquisition_readiness(
    repository_root: str | Path,
    dataset_root: str | Path,
    *,
    verify_file_hashes: bool = False,
) -> dict[str, Any]:
    """Load the registered chain and derive a final collection decision."""

    protocol, v2, _, v4 = load_registered_preacquisition_chain(repository_root)
    real_status = build_real_evidence_status(
        protocol,
        dataset_root,
        repository_root=repository_root,
        verify_file_hashes=verify_file_hashes,
    )
    return evaluate_preacquisition_readiness(
        protocol,
        v2,
        v4,
        dataset_root,
        real_status,
        verify_file_hashes=verify_file_hashes,
    )


def write_preacquisition_readiness(
    path: str | Path,
    status: Mapping[str, Any],
) -> Path:
    """Atomically write one deterministic readiness snapshot."""

    output = Path(path)
    atomic_write_json(output, dict(status))
    return output


__all__ = [
    "GATE_EVIDENCE_ARTIFACT_KIND",
    "GATE_EVIDENCE_SCHEMA_VERSION",
    "GATE_PATHS",
    "READINESS_ARTIFACT_KIND",
    "READINESS_SCHEMA_VERSION",
    "SOURCE_PANEL_MANIFEST_PATH",
    "SOURCE_PANEL_MANIFEST_TEMPLATE_PATH",
    "build_preacquisition_readiness",
    "evaluate_preacquisition_readiness",
    "gate_evidence_sha256",
    "gate_evidence_template",
    "load_registered_preacquisition_chain",
    "readiness_evidence_sha256",
    "readiness_status_sha256",
    "scaffold_preacquisition_readiness",
    "seal_preacquisition_gate",
    "source_panel_execution_manifest_template",
    "write_preacquisition_readiness",
]
