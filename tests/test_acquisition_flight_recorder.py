from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from causal4d.acquisition_flight_recorder import (
    HEALTH_SNAPSHOT_KIND,
    DoctorThresholds,
    HealthThresholds,
    append_journal_event,
    build_acquisition_doctor_report,
    evaluate_health_snapshot,
    journal_seal_path,
    seal_acquisition_journal,
    validate_acquisition_journal,
    validate_acquisition_journal_seal,
)


UTC = "2026-08-03T08:00:00+00:00"


def _append(
    path: Path,
    event_type: str,
    monotonic_ns: int,
    *,
    execution_id: str | None = None,
    payload: dict[str, object] | None = None,
):
    return append_journal_event(
        path,
        protocol_id="protocol-v1",
        session_id="session-1",
        execution_id=execution_id,
        event_type=event_type,
        source="test",
        payload=payload,
        recorded_at_utc=UTC,
        monotonic_ns=monotonic_ns,
    )


def test_journal_round_trip_seal_and_non_overwrite(tmp_path: Path) -> None:
    journal = tmp_path / "acquisition.jsonl"
    first = _append(journal, "session_started", 10)
    second = _append(journal, "execution_started", 20, execution_id="execution-1")
    _append(journal, "execution_completed", 30, execution_id="execution-1")
    _append(journal, "session_completed", 40)

    assert first["sequence"] == 0
    assert second["previous_event_sha256"] == first["event_sha256"]
    validation = validate_acquisition_journal(journal)
    assert validation["event_count"] == 4
    assert validation["terminal"] is True
    assert validation["execution_ids"] == ["execution-1"]

    seal = seal_acquisition_journal(
        journal,
        sealed_by="operator.primary",
        sealed_at_utc=UTC,
    )
    assert seal["session_outcome"] == "completed"
    assert journal_seal_path(journal).is_file()
    assert validate_acquisition_journal_seal(journal)["valid"] is True

    with pytest.raises(ValueError, match="sealed"):
        _append(journal, "operator_note", 50)
    with pytest.raises(ValueError, match="already exists"):
        seal_acquisition_journal(journal, sealed_by="operator.primary")


def test_journal_rejects_invalid_execution_state_transitions(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "acquisition.jsonl"
    _append(journal, "session_started", 10)

    with pytest.raises(ValueError, match="requires an active execution"):
        _append(
            journal,
            "execution_completed",
            20,
            execution_id="execution-1",
        )
    with pytest.raises(ValueError, match="was not started"):
        _append(
            journal,
            "stream_heartbeat",
            25,
            execution_id="execution-1",
        )

    _append(journal, "execution_started", 30, execution_id="execution-1")
    with pytest.raises(ValueError, match="while an execution is active"):
        _append(journal, "session_aborted", 40)
    _append(journal, "execution_aborted", 50, execution_id="execution-1")

    with pytest.raises(ValueError, match="may not be restarted"):
        _append(journal, "execution_started", 60, execution_id="execution-1")
    _append(journal, "session_aborted", 70)
    validation = validate_acquisition_journal(journal)
    assert validation["aborted_execution_ids"] == ["execution-1"]
    assert validation["active_execution_id"] is None


def test_journal_append_handles_events_larger_than_tail_read_block(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "acquisition.jsonl"
    _append(
        journal,
        "session_started",
        10,
        payload={"operator_note": "x" * 12_000},
    )
    appended = _append(journal, "operator_note", 20, payload={"note": "continued"})

    assert appended["sequence"] == 1
    validation = validate_acquisition_journal(journal)
    assert validation["event_count"] == 2
    assert validation["last_event_type"] == "operator_note"


def test_journal_rejects_broken_symlink_target(tmp_path: Path) -> None:
    journal = tmp_path / "acquisition.jsonl"
    try:
        journal.symlink_to(tmp_path / "missing.jsonl")
    except OSError as error:  # pragma: no cover - symlinks unavailable
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="symlink component"):
        _append(journal, "session_started", 10)


def test_journal_detects_tampering_and_target_outcomes(tmp_path: Path) -> None:
    journal = tmp_path / "acquisition.jsonl"
    _append(journal, "session_started", 10)
    _append(journal, "operator_note", 20, payload={"note": "camera restarted"})
    lines = journal.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[1])
    tampered["payload"]["note"] = "edited after collection"
    lines[1] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    journal.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_acquisition_journal(journal)

    clean = tmp_path / "clean.jsonl"
    _append(clean, "session_started", 10)
    with pytest.raises(ValueError, match="target-outcome"):
        _append(
            clean,
            "operator_note",
            20,
            payload={"target_metrics": {"track_error_m": 0.01}},
        )


def test_health_snapshot_fails_closed_on_stream_and_storage_faults() -> None:
    snapshot = {
        "schema_version": 1,
        "artifact_kind": HEALTH_SNAPSHOT_KIND,
        "protocol_id": "protocol-v1",
        "session_id": "session-1",
        "execution_id": "execution-1",
        "captured_at_utc": UTC,
        "target_outcomes_used": False,
        "streams": {
            "rgbd": {
                "required": True,
                "alive": True,
                "heartbeat_age_s": 0.2,
                "dropped_frames": 0,
                "clock_offset_ms": 1.5,
            },
            "force_torque": {
                "required": False,
                "alive": False,
                "heartbeat_age_s": 4.0,
                "dropped_frames": 0,
            },
        },
        "storage": {"free_bytes": 1000, "write_mib_s": 100.0},
    }
    thresholds = HealthThresholds(
        minimum_free_bytes=500,
        minimum_write_mib_s=50.0,
    )
    healthy = evaluate_health_snapshot(snapshot, thresholds=thresholds)
    assert healthy["passed"] is True
    assert healthy["warnings"] == ["optional_stream:force_torque:unhealthy"]

    snapshot["streams"]["rgbd"]["dropped_frames"] = 1
    snapshot["storage"]["free_bytes"] = 100
    failed = evaluate_health_snapshot(snapshot, thresholds=thresholds)
    assert failed["passed"] is False
    assert "stream:rgbd:dropped_frame_limit_exceeded" in failed["failures"]
    assert "storage:free_space_below_threshold" in failed["failures"]


def _protocol() -> dict[str, object]:
    digest = "a" * 64
    return {
        "protocol_id": "protocol-v1",
        "design_sha256": digest,
        "executions": [
            {
                "acquisition_execution_index": 0,
                "execution_id": "execution-1",
                "session_id": "session-1",
                "pair_order": 0,
                "contact_region_id": "contact-1",
                "command_profile_id": "command-1",
                "realization_condition_id": "nominal",
                "replicate_block": 0,
            },
            {
                "acquisition_execution_index": 1,
                "execution_id": "execution-2",
                "session_id": "session-1",
                "pair_order": 1,
                "contact_region_id": "contact-1",
                "command_profile_id": "command-2",
                "realization_condition_id": "nominal",
                "replicate_block": 0,
            },
        ],
    }


def _install_doctor_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    freeze = SimpleNamespace(
        load_method_freeze_manifest=lambda path: {"status": "sealed"},
        validate_method_freeze_manifest=lambda manifest, root, verify_files: {
            "passed": True
        },
        validate_repository_checkout=lambda manifest, root: {
            "commit_sha": "b" * 40,
            "dirty_worktree": False,
        },
    )
    readiness = SimpleNamespace(
        READINESS_ARTIFACT_KIND="Readiness",
        READINESS_SCHEMA_VERSION=1,
        readiness_evidence_sha256=lambda values: "evidence",
        readiness_status_sha256=lambda values: "status",
    )
    monkeypatch.setitem(sys.modules, "causal4d.real_experiment_freeze", freeze)
    monkeypatch.setitem(
        sys.modules,
        "causal4d.preacquisition_readiness_contracts",
        readiness,
    )

    def build_status(
        protocol,
        dataset_root,
        *,
        repository_root,
        verify_file_hashes,
    ):
        results = []
        for execution in protocol["executions"]:
            identifier = execution["execution_id"]
            path = Path(dataset_root) / "executions" / identifier / "manifest.json"
            present = path.is_file()
            payload = json.loads(path.read_text()) if present else {}
            acquired = payload.get("acquisition_status") == "complete"
            results.append(
                {
                    "execution_id": identifier,
                    "manifest_present": present,
                    "acquisition_status": payload.get("acquisition_status"),
                    "acquired": acquired,
                    "validated": acquired,
                    "error": None if acquired or not present else "incomplete",
                }
            )
        return {
            "specified_executions": len(results),
            "manifest_executions": sum(item["manifest_present"] for item in results),
            "acquired_executions": sum(item["acquired"] for item in results),
            "validated_executions": sum(item["validated"] for item in results),
            "file_hashes_requested": verify_file_hashes,
            "prerequisites": {"dataset_protocol": {"valid": True}},
            "invalid_session_ids": [],
            "unexpected_execution_directories": [],
            "unexpected_session_directories": [],
            "executions": results,
        }

    evidence = SimpleNamespace(build_real_evidence_status=build_status)
    monkeypatch.setitem(sys.modules, "causal4d.real_evidence_contract_v2", evidence)


def _doctor_tree(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    dataset = tmp_path / "dataset"
    repository.mkdir()
    dataset.mkdir()
    (dataset / "method_freeze.json").write_text("{}\n", encoding="utf-8")
    readiness = {
        "schema_version": 1,
        "artifact_kind": "Readiness",
        "protocol_id": "protocol-v1",
        "protocol_design_sha256": "a" * 64,
        "verify_file_hashes": True,
        "valid": True,
        "ready": True,
        "collection_gate": {"first_confirmatory_execution_allowed": True},
        "dataset_root": str(dataset.resolve()),
        "evidence_sha256": "evidence",
        "status_sha256": "status",
    }
    (dataset / "preacquisition-readiness.json").write_text(
        json.dumps(readiness), encoding="utf-8"
    )
    for execution in ("execution-1", "execution-2"):
        (dataset / "executions" / execution).mkdir(parents=True)
    (dataset / "sessions" / "session-1").mkdir(parents=True)
    return repository, dataset


def test_doctor_identifies_next_execution_and_rejects_out_of_order_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_doctor_stubs(monkeypatch)
    repository, dataset = _doctor_tree(tmp_path)
    settings = DoctorThresholds(
        minimum_free_bytes=0,
        write_probe_bytes=0,
        minimum_write_mib_s=0.0,
    )
    report = build_acquisition_doctor_report(
        _protocol(),
        repository,
        dataset,
        thresholds=settings,
        perform_write_probe=False,
    )
    assert report["passed"] is True
    assert report["next_execution"]["execution_id"] == "execution-1"

    (dataset / "executions" / "execution-2" / "manifest.json").write_text(
        json.dumps({"acquisition_status": "complete"}),
        encoding="utf-8",
    )
    invalid = build_acquisition_doctor_report(
        _protocol(),
        repository,
        dataset,
        thresholds=settings,
        perform_write_probe=False,
    )
    assert invalid["passed"] is False
    assert any(
        check["check_id"] == "execution_manifests" and check["status"] == "fail"
        for check in invalid["checks"]
    )


def test_doctor_requires_explicit_resume_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_doctor_stubs(monkeypatch)
    repository, dataset = _doctor_tree(tmp_path)
    journal = dataset / "sessions" / "session-1" / "acquisition.jsonl"
    _append(journal, "session_started", 10)
    settings = DoctorThresholds(
        minimum_free_bytes=0,
        write_probe_bytes=0,
        minimum_write_mib_s=0.0,
    )

    blocked = build_acquisition_doctor_report(
        _protocol(),
        repository,
        dataset,
        thresholds=settings,
        perform_write_probe=False,
    )
    assert blocked["valid"] is True
    assert blocked["passed"] is False
    assert blocked["resume_acknowledged"] is False

    acknowledged = build_acquisition_doctor_report(
        _protocol(),
        repository,
        dataset,
        thresholds=settings,
        perform_write_probe=False,
        allow_resume=True,
    )
    assert acknowledged["passed"] is True
    assert acknowledged["resume_acknowledged"] is True
