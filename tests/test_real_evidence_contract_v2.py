import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import causal4d.real_evidence_contract_v2 as evidence
import causal4d.real_freeze_evidence as freeze_evidence
from causal4d.real_protocol import (
    build_same_object_real_protocol,
    execution_manifest_template,
    scaffold_dataset,
)


def _approved_timebase(protocol: dict, tmp_path: Path) -> dict:
    artifact = tmp_path / "timebase.bin"
    artifact.write_bytes(b"timebase\n")
    import hashlib

    calibration = evidence.timebase_calibration_template(protocol)
    calibration.update(
        {
            "status": "approved",
            "clock_domain_id": "ptp-clock-0",
            "measured_max_sync_error_ms": 1.0,
            "calibrated_at_utc": "2026-07-27T06:00:00Z",
            "locked_before_confirmatory_collection": True,
        }
    )
    calibration["calibration_artifact"] = {
        "path": artifact.name,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "bytes": artifact.stat().st_size,
    }
    calibration["approval"] = {
        "approved": True,
        "approver_id": "timebase-reviewer",
        "approved_at_utc": "2026-07-27T06:05:00Z",
    }
    return calibration


def _strict_execution_manifest(protocol: dict, execution: dict) -> dict:
    manifest = execution_manifest_template(protocol, execution["execution_id"])
    manifest["acquisition"] = {
        "operator_id": "operator-1",
        "hardware_run_id": f"run-{execution['execution_id']}",
        "started_at_utc": "2026-07-27T08:00:00Z",
        "ended_at_utc": "2026-07-27T08:01:00Z",
        "acquisition_execution_index": execution["acquisition_execution_index"],
        "grasp_instance_id": f"grasp-{execution['session_id']}",
        "clock_domain_id": "ptp-clock-0",
    }
    manifest["quality"] = {
        "reset_passed": True,
        "rgbd_actuator_sync_error_ms": 1.0,
        "initial_state_chamfer_m": 0.001,
        "end_effector_reset_error_m": 0.001,
        "contact_centroid_error_m": 0.002,
        "dropped_rgbd_frames": 0,
        "slip_displacement_m": None,
        "complete_release_observed": None,
    }
    manifest["drift_indicators"] = {
        "wear_cycle_count": 4,
        "minutes_since_first_execution": 12.5,
        "object_temperature_c": 22.4,
        "room_temperature_c": 21.8,
        "notes": "none",
    }
    return manifest


def test_v2_scaffold_replaces_session_specs_with_completion_templates(
    tmp_path: Path,
) -> None:
    protocol = build_same_object_real_protocol()
    root = tmp_path / "dataset"
    scaffold_dataset(protocol, root)

    summary = evidence.scaffold_real_evidence_v2_templates(protocol, root)

    assert summary["schema_version"] == 2
    assert summary["session_templates"] == 18
    assert (root / "timebase_calibration.template.json").is_file()
    assert (root / "method_freeze_validation.template.json").is_file()
    template = json.loads(
        next(root.glob("sessions/*/session.template.json")).read_text(encoding="utf-8")
    )
    assert template["artifact_kind"] == "SameGraspSessionManifest"
    assert template["acquisition_status"] == "template"
    assert len(template["execution_order"]) == 2


def test_timebase_requires_exact_stream_coverage_and_verified_artifact(
    tmp_path: Path,
) -> None:
    protocol = build_same_object_real_protocol()
    calibration = _approved_timebase(protocol, tmp_path)

    result = evidence.validate_timebase_calibration(
        protocol,
        calibration,
        dataset_root=tmp_path,
        verify_file_hashes=True,
    )

    assert result["passed"]
    assert result["clock_domain_id"] == "ptp-clock-0"
    missing_stream = deepcopy(calibration)
    missing_stream["calibrated_streams"].pop()
    with pytest.raises(ValueError, match="exact timestamped stream set"):
        evidence.validate_timebase_calibration(protocol, missing_stream)


def test_timebase_approval_cannot_predate_calibration(tmp_path: Path) -> None:
    protocol = build_same_object_real_protocol()
    calibration = _approved_timebase(protocol, tmp_path)
    calibration["approval"]["approved_at_utc"] = "2026-07-27T05:59:59Z"

    with pytest.raises(ValueError, match="approval predates calibration"):
        evidence.validate_timebase_calibration(protocol, calibration)


def test_timebase_rejects_nonfinite_and_negative_sync_error(tmp_path: Path) -> None:
    protocol = build_same_object_real_protocol()
    calibration = _approved_timebase(protocol, tmp_path)
    for value in (float("nan"), float("inf"), -0.1):
        invalid = deepcopy(calibration)
        invalid["measured_max_sync_error_ms"] = value
        with pytest.raises(ValueError):
            evidence.validate_timebase_calibration(protocol, invalid)


def test_execution_v2_requires_one_clock_and_physical_quality_domains() -> None:
    protocol = build_same_object_real_protocol()
    execution = min(
        protocol["executions"],
        key=lambda value: value["acquisition_execution_index"],
    )
    manifest = _strict_execution_manifest(protocol, execution)
    timestamped = protocol["recording_contract"]["timestamped_artifacts"][0]
    manifest["artifacts"][timestamped] = {
        "path": "data.bin",
        "sha256": "1" * 64,
        "bytes": 1,
        "clock_id": "ptp-clock-0",
    }

    result = evidence._validate_execution_contract_v2(
        protocol,
        manifest,
        execution,
        clock_domain_id="ptp-clock-0",
    )
    assert result["grasp_instance_id"] == f"grasp-{execution['session_id']}"

    mixed_clock = deepcopy(manifest)
    mixed_clock["artifacts"][timestamped]["clock_id"] = "camera-clock"
    with pytest.raises(ValueError, match="clock domain"):
        evidence._validate_execution_contract_v2(
            protocol,
            mixed_clock,
            execution,
            clock_domain_id="ptp-clock-0",
        )

    for metric, value in (
        ("initial_state_chamfer_m", -0.001),
        ("contact_centroid_error_m", float("nan")),
        ("rgbd_actuator_sync_error_ms", float("inf")),
    ):
        invalid = deepcopy(manifest)
        invalid["quality"][metric] = value
        with pytest.raises(ValueError):
            evidence._validate_execution_contract_v2(
                protocol,
                invalid,
                execution,
                clock_domain_id="ptp-clock-0",
            )

    noninteger_drop_count = deepcopy(manifest)
    noninteger_drop_count["quality"]["dropped_rgbd_frames"] = 0.0
    with pytest.raises(ValueError, match="nonnegative integer"):
        evidence._validate_execution_contract_v2(
            protocol,
            noninteger_drop_count,
            execution,
            clock_domain_id="ptp-clock-0",
        )


def test_strict_json_loader_rejects_nan_and_infinity(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    for constant in ("NaN", "Infinity", "-Infinity"):
        path.write_text(f'{{"metric": {constant}}}', encoding="utf-8")
        with pytest.raises(ValueError, match="non-finite JSON constant"):
            evidence._load_json_mapping(path)


def test_session_manifest_binds_order_grasp_clock_and_execution_hashes(
    tmp_path: Path,
) -> None:
    protocol = build_same_object_real_protocol()
    session = min(
        protocol["sessions"],
        key=lambda value: value["acquisition_session_index"],
    )
    execution_by_id = {
        execution["execution_id"]: execution for execution in protocol["executions"]
    }
    ordered = sorted(
        session["execution_ids"],
        key=lambda identifier: execution_by_id[identifier]["pair_order"],
    )
    base = datetime(2026, 7, 27, 8, 0, tzinfo=timezone.utc)
    results = {
        ordered[0]: {
            "execution_id": ordered[0],
            "validated": True,
            "manifest_sha256": "1" * 64,
            "grasp_instance_id": "grasp-001",
            "clock_domain_id": "ptp-clock-0",
            "_started_at": base + timedelta(seconds=10),
            "_ended_at": base + timedelta(seconds=20),
        },
        ordered[1]: {
            "execution_id": ordered[1],
            "validated": True,
            "manifest_sha256": "2" * 64,
            "grasp_instance_id": "grasp-001",
            "clock_domain_id": "ptp-clock-0",
            "_started_at": base + timedelta(seconds=30),
            "_ended_at": base + timedelta(seconds=40),
        },
    }
    payload = evidence.session_manifest_template(protocol, session["session_id"])
    payload.update(
        {
            "acquisition_status": "complete",
            "grasp_instance_id": "grasp-001",
            "clock_domain_id": "ptp-clock-0",
            "contact_registration_sha256": "3" * 64,
            "timebase_calibration_sha256": "4" * 64,
            "operator_id": "operator-1",
            "started_at_utc": base.isoformat().replace("+00:00", "Z"),
            "ended_at_utc": (base + timedelta(seconds=50))
            .isoformat()
            .replace("+00:00", "Z"),
            "same_grasp_confirmed": True,
            "release_between_executions": False,
            "neutral_state_checks": {
                "before_first": True,
                "between_executions": True,
                "after_second": True,
            },
            "execution_manifest_sha256": {
                ordered[0]: "1" * 64,
                ordered[1]: "2" * 64,
            },
            "approval": {
                "approved": True,
                "approver_id": "reviewer-1",
                "approved_at_utc": (base + timedelta(seconds=60))
                .isoformat()
                .replace("+00:00", "Z"),
            },
        }
    )
    path = tmp_path / "session.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = evidence._validate_session_manifest(
        protocol,
        session,
        path,
        execution_results=results,
        contact_registration_sha256="3" * 64,
        timebase_calibration_sha256="4" * 64,
        clock_domain_id="ptp-clock-0",
    )

    assert result["validated"]
    released = deepcopy(payload)
    released["release_between_executions"] = True
    path.write_text(json.dumps(released), encoding="utf-8")
    result = evidence._validate_session_manifest(
        protocol,
        session,
        path,
        execution_results=results,
        contact_registration_sha256="3" * 64,
        timebase_calibration_sha256="4" * 64,
        clock_domain_id="ptp-clock-0",
    )
    assert not result["validated"]
    assert "release occurred" in result["error"]


def test_method_freeze_attestation_must_be_independent() -> None:
    protocol = build_same_object_real_protocol()
    method_freeze = {
        "frozen_by": "principal-investigator",
        "frozen_at_utc": "2026-07-27T06:00:00Z",
        "causal4d": {"commit_sha": "1" * 40},
        "bayesian_phystwin": {"commit_sha": "2" * 40},
    }
    attestation = evidence.method_freeze_validation_attestation_template(protocol)
    attestation.update(
        {
            "method_freeze_sha256": "3" * 64,
            "causal4d_commit_sha": "1" * 40,
            "bayesian_phystwin_commit_sha": "2" * 40,
            "verifier_id": "independent-reviewer",
            "verified_at_utc": "2026-07-27T07:00:00Z",
            "independent_of_freezer": True,
            "repository_checkout_verified": True,
            "locked_file_hashes_verified": True,
            "validation_passed": True,
        }
    )
    assert evidence._validate_method_freeze_attestation(
        protocol,
        attestation,
        method_freeze=method_freeze,
        method_freeze_sha256="3" * 64,
    )["passed"]
    attestation["verifier_id"] = "principal-investigator"
    with pytest.raises(ValueError, match="independently verified"):
        evidence._validate_method_freeze_attestation(
            protocol,
            attestation,
            method_freeze=method_freeze,
            method_freeze_sha256="3" * 64,
        )


def test_preacquisition_chronology_rejects_postdated_prerequisites() -> None:
    start = datetime(2026, 7, 27, 8, 0, tzinfo=timezone.utc)
    prerequisites = {
        "timebase_calibration": {
            "valid": True,
            "calibrated_at_utc": "2026-07-27T06:00:00Z",
            "approved_at_utc": "2026-07-27T06:05:00Z",
        },
        "contact_registration": {
            "valid": True,
            "approved_at_utc": "2026-07-27T06:10:00Z",
        },
        "method_freeze": {
            "valid": True,
            "frozen_at_utc": "2026-07-27T06:15:00Z",
        },
        "method_freeze_validation": {
            "valid": True,
            "verified_at_utc": "2026-07-27T08:00:01Z",
        },
    }
    result = evidence._preacquisition_chronology(
        prerequisites,
        [{"validated": True, "_started_at": start}],
    )

    assert not result["passed"]
    assert result["blockers"] == ["preacquisition_chronology:method_freeze_verified"]
    assert result["earliest_execution_started_at_utc"] == "2026-07-27T08:00:00Z"


def test_analysis_readiness_is_separate_from_evidence_accounting() -> None:
    protocol = build_same_object_real_protocol()
    results = [
        {"execution_id": execution["execution_id"], "included": True}
        for execution in protocol["executions"]
    ]
    complete = evidence._analysis_readiness(protocol, results)
    assert complete["analysis_ready"]
    assert complete["full_registered_power"]

    first_fold = protocol["splits"]["cross_action_contact_calibration_folds"][0]
    omitted_targets = set(first_fold["target_execution_ids"])
    for result in results:
        if result["execution_id"] in omitted_targets:
            result["included"] = False
    limited = evidence._analysis_readiness(protocol, results)
    assert not limited["analysis_ready"]
    assert not limited["full_registered_power"]
    assert limited["blockers"]


def test_method_freeze_attestation_builder_binds_exact_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = build_same_object_real_protocol()
    freeze = {
        "frozen_by": "principal-investigator",
        "frozen_at_utc": "2026-07-27T06:00:00Z",
        "causal4d": {"commit_sha": "1" * 40},
        "bayesian_phystwin": {"commit_sha": "2" * 40},
    }
    freeze_path = tmp_path / "method_freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    monkeypatch.setattr(
        freeze_evidence,
        "validate_method_freeze_manifest",
        lambda *args, **kwargs: {
            "causal4d_commit_sha": "1" * 40,
            "bayesian_phystwin_commit_sha": "2" * 40,
            "file_hashes_verified": True,
            "passed": True,
        },
    )
    monkeypatch.setattr(
        freeze_evidence,
        "validate_repository_checkout",
        lambda *args, **kwargs: {
            "commit_sha": "1" * 40,
            "dirty_worktree": False,
        },
    )

    attestation = evidence.build_method_freeze_validation_attestation(
        protocol,
        freeze_path,
        tmp_path,
        verified_by="independent-reviewer",
        verified_at_utc="2026-07-27T07:00:00Z",
    )

    import hashlib

    assert (
        attestation["method_freeze_sha256"]
        == hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    )
    assert attestation["repository_checkout_verified"]
    assert attestation["locked_file_hashes_verified"]


def test_method_freeze_attestation_publication_is_once_only(tmp_path: Path) -> None:
    target = tmp_path / "method_freeze_validation.json"
    first = {"schema_version": 1, "verifier_id": "reviewer-1"}
    second = {"schema_version": 1, "verifier_id": "reviewer-2"}

    freeze_evidence.write_method_freeze_validation_attestation(target, first)
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        freeze_evidence.write_method_freeze_validation_attestation(target, second)

    assert target.read_bytes() == original
    assert json.loads(target.read_text(encoding="utf-8")) == first
