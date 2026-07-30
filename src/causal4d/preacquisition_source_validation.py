"""Validation for source-panel and actuator readiness evidence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from causal4d.preacquisition_readiness_contracts import (
    _canonical_sha256,
    _expected_source_panel,
    _finite_number,
    _parse_utc_timestamp,
    _read_json_mapping,
    _require,
    _safe_relative_path,
    _validate_descriptor_list,
)

def _validate_source_execution_manifest(
    dataset_root: Path,
    relative: str,
    *,
    protocol: Mapping[str, Any],
    v4: Mapping[str, Any],
    execution_id: str,
    session_id: str,
    verify_file_hashes: bool,
) -> None:
    path = dataset_root / _safe_relative_path(relative, name="source execution manifest")
    manifest = _read_json_mapping(path, name="source execution manifest")
    _require(manifest.get("schema_version") == 1, "unsupported source execution schema")
    _require(
        manifest.get("artifact_kind") == "SourcePanelExecutionManifest",
        "unexpected source execution artifact kind",
    )
    _require(manifest.get("status") == "complete", "source execution is not complete")
    _require(
        manifest.get("protocol_id") == protocol["protocol_id"],
        "source execution protocol mismatch",
    )
    _require(
        manifest.get("protocol_design_sha256") == protocol["design_sha256"],
        "source execution protocol digest mismatch",
    )
    _require(
        manifest.get("preacquisition_plan_id") == v4["plan_id"],
        "source execution v4 plan mismatch",
    )
    _require(
        manifest.get("preacquisition_amendment_sha256") == v4["amendment_sha256"],
        "source execution v4 digest mismatch",
    )
    _require(manifest.get("execution_id") == execution_id, "source execution id mismatch")
    _require(manifest.get("session_id") == session_id, "source session id mismatch")
    _require(
        manifest.get("fresh_reset_and_fresh_grasp") is True,
        "source execution did not use a fresh reset and grasp",
    )
    _require(
        manifest.get("confirmatory_fold_member") is False,
        "source execution entered a confirmatory fold",
    )
    _require(
        manifest.get("target_outcomes_used") is False,
        "target outcomes entered a source execution",
    )
    _require(manifest.get("included") is True, "source execution is not included")
    _require(
        manifest.get("quality_gate_failures") == [],
        "source execution has quality-gate failures",
    )
    started = _parse_utc_timestamp(
        manifest.get("started_at_utc"), name="source execution started_at_utc"
    )
    ended = _parse_utc_timestamp(
        manifest.get("ended_at_utc"), name="source execution ended_at_utc"
    )
    _require(ended >= started, "source execution ends before it starts")
    _validate_descriptor_list(
        dataset_root,
        manifest.get("artifacts"),
        name=f"source execution {execution_id} artifacts",
        verify_file_hashes=verify_file_hashes,
    )


def _validate_signature_panel(
    protocol: Mapping[str, Any],
    v2: Mapping[str, Any],
    v4: Mapping[str, Any],
    checks: Mapping[str, Any],
    evidence_paths: set[str],
    *,
    dataset_root: Path,
    verify_file_hashes: bool,
) -> None:
    expected_execution_ids, expected_session_ids = _expected_source_panel(v2)
    manifest_files = checks.get("manifest_files")
    _require(isinstance(manifest_files, Mapping), "source-panel manifest map is invalid")
    _require(
        set(manifest_files) == set(expected_execution_ids)
        and len(manifest_files) == len(expected_execution_ids),
        "source-panel manifest map differs from the registered execution set",
    )
    _require(
        checks.get("execution_ids") == expected_execution_ids,
        "source-panel execution ids differ from the registered panel",
    )
    _require(
        checks.get("session_ids") == expected_session_ids,
        "source-panel session ids differ from the registered panel",
    )
    _require(
        checks.get("independent_session_count") == 12,
        "source panel must contain 12 independent sessions",
    )
    _require(checks.get("source_only") is True, "source panel is not source-only")
    for execution_id, session_id in zip(
        expected_execution_ids, expected_session_ids, strict=True
    ):
        relative = manifest_files.get(execution_id)
        _require(
            isinstance(relative, str) and relative in evidence_paths,
            f"source execution manifest is not bound as evidence: {execution_id}",
        )
        _validate_source_execution_manifest(
            dataset_root,
            relative,
            protocol=protocol,
            v4=v4,
            execution_id=execution_id,
            session_id=session_id,
            verify_file_hashes=verify_file_hashes,
        )


def _validate_actuator_calibration(
    dataset_root: Path,
    relative: str,
    *,
    execution_id: str,
) -> str:
    path = dataset_root / _safe_relative_path(relative, name="actuator calibration")
    artifact = _read_json_mapping(path, name="actuator calibration")
    _require(artifact.get("schema_version") == 1, "unsupported actuator schema")
    _require(
        artifact.get("artifact_kind") == "ActuatorRealizationCalibration",
        "unexpected actuator artifact kind",
    )
    _require(artifact.get("execution_id") == execution_id, "actuator execution id mismatch")
    _require(artifact.get("pyrecest_version") == "2.4.1", "wrong PyRecEst version")
    boundary = artifact.get("information_boundary")
    _require(isinstance(boundary, Mapping), "actuator information boundary is invalid")
    for field in (
        "source_or_dry_run_only",
        "hardware_timestamps_authoritative",
    ):
        _require(boundary.get(field) is True, f"actuator boundary failed: {field}")
    _require(
        boundary.get("target_outcomes_used") is False,
        "target outcomes entered actuator calibration",
    )
    _require(
        artifact.get("artifact_id")
        == _canonical_sha256(artifact, omitted_field="artifact_id"),
        "actuator artifact digest mismatch",
    )
    return str(artifact["artifact_id"])


def _validate_actuator_sync(
    protocol: Mapping[str, Any],
    v2: Mapping[str, Any],
    checks: Mapping[str, Any],
    evidence_paths: set[str],
    *,
    dataset_root: Path,
) -> None:
    expected_execution_ids, _ = _expected_source_panel(v2)
    calibration_files = checks.get("calibration_files")
    _require(isinstance(calibration_files, Mapping), "actuator calibration map is invalid")
    _require(
        set(calibration_files) == set(expected_execution_ids)
        and len(calibration_files) == len(expected_execution_ids),
        "actuator calibration map differs from the source-panel execution set",
    )
    _require(
        checks.get("commanded_vs_measured_validated") is True,
        "commanded-versus-measured actuation was not validated",
    )
    _require(
        checks.get("hardware_timestamps_authoritative") is True,
        "hardware timestamps are not authoritative",
    )
    measured = _finite_number(
        checks.get("maximum_measured_rgbd_actuator_sync_error_ms"),
        name="maximum measured RGB-D/actuator synchronization error",
    )
    maximum = float(protocol["quality_gates"]["maximum_rgbd_actuator_sync_error_ms"])
    declared_maximum = _finite_number(
        checks.get("maximum_allowed_rgbd_actuator_sync_error_ms"),
        name="declared maximum RGB-D/actuator synchronization error",
    )
    _require(
        declared_maximum == maximum,
        "declared actuator synchronization gate differs from the protocol",
    )
    _require(measured >= 0.0, "measured synchronization error must be nonnegative")
    _require(measured <= maximum, "actuator synchronization exceeds the locked gate")
    artifact_ids: dict[str, str] = {}
    for execution_id in expected_execution_ids:
        relative = calibration_files.get(execution_id)
        _require(
            isinstance(relative, str) and relative in evidence_paths,
            f"actuator calibration is not bound as evidence: {execution_id}",
        )
        artifact_ids[execution_id] = _validate_actuator_calibration(
            dataset_root,
            relative,
            execution_id=execution_id,
        )
    _require(
        checks.get("calibration_artifact_ids") == artifact_ids,
        "actuator artifact-id map differs from the calibrated files",
    )


__all__ = [
    "_validate_actuator_sync",
    "_validate_signature_panel",
]
