import hashlib
import json
from pathlib import Path

from causal4d.cli.real_protocol import INCOMPLETE_EVIDENCE_EXIT_CODE, main
from causal4d.real_evidence_status import (
    build_real_evidence_status,
    write_real_evidence_status,
)
from causal4d.real_protocol import (
    build_same_object_real_protocol,
    execution_manifest_template,
    object_registration_template,
    scaffold_dataset,
    slip_pilot_template,
)


def _complete_manifest(protocol: dict, execution_id: str) -> dict:
    manifest = execution_manifest_template(protocol, execution_id)
    manifest["acquisition_status"] = "complete"
    manifest["acquisition"] = {
        "operator_id": "operator-1",
        "hardware_run_id": f"run-{execution_id}",
        "started_at_utc": "2026-07-27T08:00:00Z",
    }
    manifest["timing"] = {
        "frame_count": 120,
        "intervention_frame": 30,
        "o_plus_prefix_frames": 6,
    }
    for name in protocol["recording_contract"]["required_artifacts"]:
        descriptor = {
            "path": f"data/{name}.bin",
            "sha256": "0" * 64,
            "bytes": 1,
        }
        if name in protocol["recording_contract"]["timestamped_artifacts"]:
            descriptor["clock_id"] = "ptp-clock-0"
        manifest["artifacts"][name] = descriptor
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
    if manifest["realization_condition_id"] == "slip_low_force":
        manifest["artifacts"]["gripper_normal_force"] = {
            "path": "data/gripper_normal_force.bin",
            "sha256": "1" * 64,
            "bytes": 1,
            "clock_id": "ptp-clock-0",
        }
        manifest["quality"]["slip_displacement_m"] = 0.01
        manifest["quality"]["complete_release_observed"] = False
    manifest["drift_indicators"] = {
        "wear_cycle_count": 4,
        "minutes_since_first_execution": 12.5,
        "object_temperature_c": 22.4,
        "room_temperature_c": 21.8,
        "notes": "none",
    }
    manifest["exclusion"] = {
        "status": "included",
        "reason": None,
        "decided_before_target_evaluation": True,
    }
    return manifest


def _complete_registration(protocol: dict, root: Path) -> dict:
    registration = object_registration_template(protocol)
    registration["object_instance_serial"] = "sloth-001"
    registration["phystwin_model_id"] = "sloth-twin-v1"
    registration["phystwin_model_sha256"] = "2" * 64
    for index, (region_id, descriptor) in enumerate(
        registration["contact_regions"].items()
    ):
        relative = Path(f"contact_{index}.npz")
        content = f"canonical-node-set:{region_id}\n".encode()
        (root / relative).write_bytes(content)
        descriptor["canonical_node_set_path"] = relative.as_posix()
        descriptor["canonical_node_set_sha256"] = hashlib.sha256(content).hexdigest()
        descriptor["node_count"] = 24 + index
    return registration


def _passing_slip_pilot(protocol: dict) -> dict:
    pilot = slip_pilot_template(protocol)
    pilot.update(
        {
            "pilot_execution_ids": [f"pilot-{index}" for index in range(5)],
            "contact_region_ids": ["left_forepaw", "right_forepaw"],
            "bounded_slip_successes": 5,
            "slip_displacement_mean_m": 0.009,
            "slip_displacement_coefficient_of_variation": 0.2,
            "complete_release_count": 0,
            "passed": True,
            "decided_before_confirmatory_collection": True,
        }
    )
    return pilot


def _materialize_manifest_artifacts(manifest: dict, execution_root: Path) -> None:
    for name, descriptor in manifest["artifacts"].items():
        if not descriptor.get("path"):
            continue
        artifact_path = execution_root / descriptor["path"]
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        content = f"{manifest['execution_id']}:{name}\n".encode()
        artifact_path.write_bytes(content)
        descriptor["bytes"] = len(content)
        descriptor["sha256"] = hashlib.sha256(content).hexdigest()


def _complete_dataset(tmp_path: Path) -> tuple[dict, Path]:
    protocol = build_same_object_real_protocol()
    root = tmp_path / "dataset"
    scaffold_dataset(protocol, root)
    (root / "object_registration.json").write_text(
        json.dumps(_complete_registration(protocol, root)),
        encoding="utf-8",
    )
    (root / "slip_pilot.json").write_text(
        json.dumps(_passing_slip_pilot(protocol)),
        encoding="utf-8",
    )
    for execution in protocol["executions"]:
        execution_root = root / "executions" / execution["execution_id"]
        manifest = _complete_manifest(protocol, execution["execution_id"])
        _materialize_manifest_artifacts(manifest, execution_root)
        (execution_root / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
    return protocol, root


def test_scaffold_status_does_not_count_templates_as_acquired(tmp_path: Path) -> None:
    protocol = build_same_object_real_protocol()
    root = tmp_path / "dataset"
    scaffold_dataset(protocol, root)

    status = build_real_evidence_status(protocol, root)

    assert status["specified_executions"] == 36
    assert status["manifest_executions"] == 0
    assert status["acquired_executions"] == 0
    assert status["validated_executions"] == 0
    assert len(status["missing_execution_ids"]) == 36
    assert status["next_pending_execution"]["acquisition_execution_index"] == 0
    assert not status["complete"]
    assert not status["claim_ready"]
    assert "prerequisite:object_registration" in status["blockers"]
    assert "prerequisite:slip_pilot" in status["blockers"]


def test_complete_dataset_requires_explicit_hash_verification(tmp_path: Path) -> None:
    protocol, root = _complete_dataset(tmp_path)

    unchecked = build_real_evidence_status(protocol, root)
    assert unchecked["acquired_executions"] == 36
    assert unchecked["validated_executions"] == 36
    assert unchecked["accounting_complete"]
    assert unchecked["complete"]
    assert not unchecked["claim_ready"]
    assert unchecked["blockers"] == ["file_hashes_not_verified"]

    checked = build_real_evidence_status(
        protocol,
        root,
        verify_file_hashes=True,
    )
    assert checked["file_hashes_verified"]
    assert checked["complete"]
    assert checked["claim_ready"]
    assert checked["passed"]
    assert checked["blockers"] == []
    assert checked["next_pending_execution"] is None


def test_tampered_artifact_blocks_claim_readiness(tmp_path: Path) -> None:
    protocol, root = _complete_dataset(tmp_path)
    first = min(
        protocol["executions"],
        key=lambda value: value["acquisition_execution_index"],
    )
    manifest_path = root / "executions" / first["execution_id"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(
        descriptor
        for descriptor in manifest["artifacts"].values()
        if descriptor.get("path")
    )
    artifact_path = manifest_path.parent / artifact["path"]
    artifact_path.write_bytes(b"tampered\n")

    status = build_real_evidence_status(
        protocol,
        root,
        verify_file_hashes=True,
    )

    assert status["acquired_executions"] == 36
    assert status["validated_executions"] == 35
    assert status["invalid_execution_ids"] == [first["execution_id"]]
    assert not status["complete"]
    assert not status["claim_ready"]
    assert "invalid_execution_manifests:1" in status["blockers"]


def test_cli_status_writes_report_and_fails_closed(tmp_path: Path) -> None:
    protocol = build_same_object_real_protocol()
    root = tmp_path / "dataset"
    scaffold_dataset(protocol, root)
    output = root / "evidence-status.json"

    code = main(
        [
            "status",
            str(root / "protocol.json"),
            str(root),
            "--output-json",
            str(output),
            "--require-complete",
        ]
    )

    assert code == INCOMPLETE_EVIDENCE_EXIT_CODE
    status = json.loads(output.read_text(encoding="utf-8"))
    assert status["specified_executions"] == 36
    assert status["acquired_executions"] == 0
    assert not status["claim_ready"]


def test_status_report_write_is_atomic_and_round_trips(tmp_path: Path) -> None:
    output = tmp_path / "status.json"
    status = {"schema_version": 1, "claim_ready": False, "blockers": ["missing"]}

    written = write_real_evidence_status(output, status)

    assert written == output
    assert json.loads(output.read_text(encoding="utf-8")) == status
    assert not list(tmp_path.glob(".status.json.*.tmp"))


def test_malformed_manifest_is_invalid_not_incomplete(tmp_path: Path) -> None:
    protocol = build_same_object_real_protocol()
    root = tmp_path / "dataset"
    scaffold_dataset(protocol, root)
    first = min(
        protocol["executions"],
        key=lambda value: value["acquisition_execution_index"],
    )
    manifest_path = root / "executions" / first["execution_id"] / "manifest.json"
    manifest_path.write_text("{not-json\n", encoding="utf-8")

    status = build_real_evidence_status(protocol, root)

    assert status["manifest_executions"] == 1
    assert status["acquired_executions"] == 0
    assert status["incomplete_execution_ids"] == []
    assert status["invalid_execution_ids"] == [first["execution_id"]]
    assert "invalid_execution_manifests:1" in status["blockers"]
    assert "incomplete_execution_manifests:1" not in status["blockers"]
    assert not status["claim_ready"]
