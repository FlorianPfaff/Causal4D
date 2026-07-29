"""Execution-level and scaffold validation for real evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from causal4d.real_evidence_common import (
    EVIDENCE_STATUS_SCHEMA_VERSION,
    SESSION_MANIFEST_SCHEMA_VERSION,
    _error_text,
    _finite_nonnegative,
    _load_json_mapping,
    _nonnegative_integer,
    _parse_utc_timestamp,
    _require,
    _sha256_file,
    timebase_calibration_template,
)
from causal4d.real_freeze_evidence import (
    method_freeze_validation_attestation_template,
)
from causal4d.real_protocol import validate_execution_manifest, validate_protocol


def session_manifest_template(
    protocol: Mapping[str, Any],
    session_id: str,
) -> dict[str, Any]:
    """Build an explicitly incomplete same-grasp session manifest template."""

    validate_protocol(protocol)
    session_by_id = {session["session_id"]: session for session in protocol["sessions"]}
    if session_id not in session_by_id:
        raise KeyError(session_id)
    session = session_by_id[session_id]
    execution_by_id = {
        execution["execution_id"]: execution for execution in protocol["executions"]
    }
    ordered = sorted(
        session["execution_ids"],
        key=lambda identifier: execution_by_id[identifier]["pair_order"],
    )
    return {
        "schema_version": SESSION_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": "SameGraspSessionManifest",
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "session_id": session_id,
        "acquisition_session_index": session["acquisition_session_index"],
        "acquisition_status": "template",
        "grasp_instance_id": None,
        "clock_domain_id": None,
        "contact_registration_sha256": None,
        "timebase_calibration_sha256": None,
        "operator_id": None,
        "started_at_utc": None,
        "ended_at_utc": None,
        "execution_order": ordered,
        "same_grasp_confirmed": None,
        "release_between_executions": None,
        "neutral_state_checks": {
            "before_first": None,
            "between_executions": None,
            "after_second": None,
        },
        "execution_manifest_sha256": {identifier: None for identifier in ordered},
        "approval": {
            "approved": False,
            "approver_id": None,
            "approved_at_utc": None,
        },
    }


def scaffold_real_evidence_v2_templates(
    protocol: Mapping[str, Any],
    dataset_root: str | Path,
) -> dict[str, Any]:
    """Add non-evidence v2 templates to a freshly scaffolded dataset."""

    validate_protocol(protocol)
    root = Path(dataset_root)
    _require(root.is_dir(), "dataset root must exist before v2 scaffolding")
    templates = {
        root / "timebase_calibration.template.json": timebase_calibration_template(
            protocol
        ),
        root
        / "method_freeze_validation.template.json": method_freeze_validation_attestation_template(
            protocol
        ),
    }
    for path, payload in templates.items():
        _require(not path.exists(), f"refusing to overwrite evidence template: {path}")
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    for session in protocol["sessions"]:
        path = root / "sessions" / session["session_id"] / "session.template.json"
        _require(
            path.is_file(), f"base session template is missing: {session['session_id']}"
        )
        path.write_text(
            json.dumps(
                session_manifest_template(protocol, session["session_id"]),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "schema_version": EVIDENCE_STATUS_SCHEMA_VERSION,
        "session_templates": len(protocol["sessions"]),
        "prerequisite_templates": len(templates),
    }


def _validate_execution_contract_v2(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    clock_domain_id: str | None,
) -> dict[str, Any]:
    acquisition = manifest.get("acquisition", {})
    _require(
        acquisition.get("acquisition_execution_index")
        == expected["acquisition_execution_index"],
        "acquisition execution index differs from the locked schedule",
    )
    grasp_instance_id = acquisition.get("grasp_instance_id")
    _require(
        isinstance(grasp_instance_id, str) and bool(grasp_instance_id),
        "execution grasp_instance_id is missing",
    )
    execution_clock_domain_id = acquisition.get("clock_domain_id")
    _require(
        isinstance(execution_clock_domain_id, str) and bool(execution_clock_domain_id),
        "execution clock_domain_id is missing",
    )
    if clock_domain_id is not None:
        _require(
            execution_clock_domain_id == clock_domain_id,
            "execution clock domain differs from the approved timebase",
        )
    effective_clock_domain_id = (
        clock_domain_id if clock_domain_id is not None else execution_clock_domain_id
    )
    started_at = _parse_utc_timestamp(
        acquisition.get("started_at_utc"),
        name="execution started_at_utc",
    )
    ended_at = _parse_utc_timestamp(
        acquisition.get("ended_at_utc"),
        name="execution ended_at_utc",
    )
    _require(ended_at > started_at, "execution end must follow its start")

    timestamped = set(protocol["recording_contract"]["timestamped_artifacts"])
    for name, descriptor in manifest.get("artifacts", {}).items():
        if name in timestamped and descriptor.get("path"):
            _require(
                descriptor.get("clock_id") == effective_clock_domain_id,
                f"{name} does not use the execution clock domain",
            )

    quality = manifest.get("quality", {})
    for metric in (
        "rgbd_actuator_sync_error_ms",
        "initial_state_chamfer_m",
        "end_effector_reset_error_m",
        "contact_centroid_error_m",
    ):
        _finite_nonnegative(quality.get(metric), name=f"quality.{metric}")
    _nonnegative_integer(
        quality.get("dropped_rgbd_frames"),
        name="quality.dropped_rgbd_frames",
    )
    slip = quality.get("slip_displacement_m")
    if slip is not None:
        _finite_nonnegative(slip, name="quality.slip_displacement_m")

    drift = manifest.get("drift_indicators", {})
    _nonnegative_integer(drift.get("wear_cycle_count"), name="drift.wear_cycle_count")
    _finite_nonnegative(
        drift.get("minutes_since_first_execution"),
        name="drift.minutes_since_first_execution",
    )
    for metric in ("object_temperature_c", "room_temperature_c"):
        value = drift.get(metric)
        if value is not None:
            _require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value)),
                f"drift.{metric} must be finite",
            )
    return {
        "grasp_instance_id": grasp_instance_id,
        "clock_domain_id": effective_clock_domain_id,
        "timebase_bound": clock_domain_id is not None,
        "started_at": started_at,
        "ended_at": ended_at,
    }


def _execution_status(
    protocol: Mapping[str, Any],
    dataset_root: Path,
    execution: Mapping[str, Any],
    *,
    clock_domain_id: str | None,
    verify_file_hashes: bool,
) -> dict[str, Any]:
    execution_id = str(execution["execution_id"])
    execution_root = dataset_root / "executions" / execution_id
    manifest_path = execution_root / "manifest.json"
    template_path = execution_root / "manifest.template.json"
    result: dict[str, Any] = {
        "execution_id": execution_id,
        "session_id": str(execution["session_id"]),
        "acquisition_execution_index": int(execution["acquisition_execution_index"]),
        "manifest_path": str(manifest_path),
        "manifest_present": manifest_path.is_file(),
        "manifest_parsed": False,
        "template_present": template_path.is_file(),
        "acquisition_status": None,
        "acquired": False,
        "validated": False,
        "included": None,
        "quality_gate_failures": [],
        "file_hashes_verified": None if not verify_file_hashes else False,
        "error": None,
    }
    if not result["manifest_present"]:
        result["error"] = "manifest.json is missing"
        return result
    try:
        manifest = _load_json_mapping(manifest_path)
        result["manifest"] = manifest
        result["manifest_parsed"] = True
        result["acquisition_status"] = manifest.get("acquisition_status")
        result["acquired"] = result["acquisition_status"] == "complete"
        result["manifest_sha256"], result["manifest_bytes"] = _sha256_file(
            manifest_path
        )
        _require(result["acquired"], "execution manifest is not explicitly complete")
        validation = validate_execution_manifest(
            protocol,
            manifest,
            execution_root=execution_root,
            verify_files=verify_file_hashes,
        )
        strict = _validate_execution_contract_v2(
            protocol,
            manifest,
            execution,
            clock_domain_id=clock_domain_id,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        result["error"] = _error_text(error)
        return result
    result["validated"] = True
    result["included"] = bool(validation["included"])
    result["quality_gate_failures"] = list(validation["quality_gate_failures"])
    result["file_hashes_verified"] = True if verify_file_hashes else None
    result["grasp_instance_id"] = strict["grasp_instance_id"]
    result["clock_domain_id"] = strict["clock_domain_id"]
    result["timebase_bound"] = strict["timebase_bound"]
    result["started_at_utc"] = manifest["acquisition"]["started_at_utc"]
    result["ended_at_utc"] = manifest["acquisition"]["ended_at_utc"]
    result["_started_at"] = strict["started_at"]
    result["_ended_at"] = strict["ended_at"]
    return result


__all__ = [
    "_execution_status",
    "_validate_execution_contract_v2",
    "scaffold_real_evidence_v2_templates",
    "session_manifest_template",
]
