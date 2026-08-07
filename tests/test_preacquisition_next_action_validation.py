from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

import causal4d.preacquisition_next_action_validation as validation
from causal4d.cli import preacquisition_readiness as readiness_cli
from causal4d.preacquisition_next_action import (
    NEXT_ACTION_ARTIFACT_KIND,
    next_action_evidence_sha256,
    next_action_status_sha256,
)
from causal4d.preacquisition_operator_flow import NEXT_ACTION_SCHEMA_VERSION


def _decision(repository: Path, dataset: Path) -> dict:
    result = {
        "schema_version": NEXT_ACTION_SCHEMA_VERSION,
        "artifact_kind": NEXT_ACTION_ARTIFACT_KIND,
        "operator_flow_schema_version": 1,
        "protocol_id": "protocol",
        "protocol_design_sha256": "a" * 64,
        "preacquisition_plan_id": "plan",
        "preacquisition_amendment_sha256": "b" * 64,
        "repository_root": str(repository.resolve()),
        "dataset_root": str(dataset.resolve()),
        "readiness_evidence_sha256": "c" * 64,
        "readiness_status_sha256": "d" * 64,
        "source_panel_evidence_sha256": "e" * 64,
        "source_panel_status_sha256": "f" * 64,
        "readiness_valid": True,
        "source_panel_valid": True,
        "ready": False,
        "complete": False,
        "passed": False,
        "valid": True,
        "target_outcomes_used": False,
        "action": {
            "action_id": "acquire_next_source_panel_execution",
            "category": "physical_source_execution",
            "title": "Acquire source-01",
            "operator_role": "acquisition_operator",
            "physical_acquisition_required": True,
            "automatable": False,
            "changes_registered_method": False,
            "target_outcomes_permitted": False,
            "command_argv": ["causal4d", "protocol", "readiness", "status"],
            "command_text": "causal4d protocol readiness status",
            "completion_check_argv": [
                "causal4d",
                "protocol",
                "readiness",
                "next-action",
            ],
            "completion_check_text": ("causal4d protocol readiness next-action"),
            "after_completion_argv": None,
            "after_completion_text": None,
            "post_acquisition_verification_argv": [
                "causal4d",
                "protocol",
                "readiness",
                "source-panel-verify-staged",
            ],
            "post_acquisition_verification_text": (
                "causal4d protocol readiness source-panel-verify-staged"
            ),
            "preflight_report_path": str(dataset / "operator/preflight.json"),
            "independent_review_required_before_publication": True,
            "claim_bearing_publication_argv": [
                "causal4d",
                "protocol",
                "readiness",
                "source-panel-publish",
            ],
            "claim_bearing_publication_text": (
                "causal4d protocol readiness source-panel-publish"
            ),
            "operator_sequence": [
                "acquire_registered_source_execution",
                "verify_staged_manifest_and_artifacts",
                "independent_review_of_preflight_report",
                "publish_exactly_once",
                "recompute_next_action",
            ],
            "input_paths": [str(dataset / "template.json")],
            "output_paths": [str(dataset / "staging/source-01.json")],
            "blocking_items": [],
            "registered_execution": {
                "execution_id": "source-01",
                "session_id": "session-01",
                "command_profile_id": "lift_high",
            },
        },
    }
    result["evidence_sha256"] = next_action_evidence_sha256(result)
    result["status_sha256"] = next_action_status_sha256(result)
    return result


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_current_decision_passes_with_exact_action_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    dataset = tmp_path / "dataset"
    repository.mkdir()
    dataset.mkdir()
    decision = _decision(repository, dataset)
    source = tmp_path / "operator" / "next-action.json"
    _write(source, decision)
    monkeypatch.setattr(
        validation,
        "build_preacquisition_operator_next_action",
        lambda *args, **kwargs: deepcopy(decision),
    )

    report = validation.validate_preacquisition_next_action_report(
        repository,
        dataset,
        source,
    )

    assert report["current"] is True
    assert report["safe_to_execute"] is True
    assert report["file_hashes_verified"] is True
    assert report["action_identity"] == {
        "action_id": "acquire_next_source_panel_execution",
        "category": "physical_source_execution",
        "execution_id": "source-01",
        "session_id": "session-01",
    }
    assert report["decision_evidence_sha256"] == decision["evidence_sha256"]
    assert report["current_evidence_sha256"] == decision["evidence_sha256"]
    assert report["evidence_sha256"] == (
        validation.next_action_validation_evidence_sha256(report)
    )
    assert report["status_sha256"] == (
        validation.next_action_validation_status_sha256(report)
    )


def test_stale_decision_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    dataset = tmp_path / "dataset"
    repository.mkdir()
    dataset.mkdir()
    decision = _decision(repository, dataset)
    source = tmp_path / "next-action.json"
    _write(source, decision)
    current = deepcopy(decision)
    current["source_panel_evidence_sha256"] = "0" * 64
    current.pop("evidence_sha256")
    current.pop("status_sha256")
    current["evidence_sha256"] = next_action_evidence_sha256(current)
    current["status_sha256"] = next_action_status_sha256(current)
    monkeypatch.setattr(
        validation,
        "build_preacquisition_operator_next_action",
        lambda *args, **kwargs: current,
    )

    with pytest.raises(ValueError, match="decision is stale"):
        validation.validate_preacquisition_next_action_report(
            repository,
            dataset,
            source,
        )


def test_tampered_decision_hash_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    dataset = tmp_path / "dataset"
    repository.mkdir()
    dataset.mkdir()
    decision = _decision(repository, dataset)
    decision["action"]["action_id"] = "changed-after-hashing"
    source = tmp_path / "next-action.json"
    _write(source, decision)
    monkeypatch.setattr(
        validation,
        "build_preacquisition_operator_next_action",
        lambda *args, **kwargs: decision,
    )

    with pytest.raises(ValueError, match="evidence SHA-256 mismatch"):
        validation.validate_preacquisition_next_action_report(
            repository,
            dataset,
            source,
        )


def test_mount_mismatch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    dataset = tmp_path / "dataset"
    other = tmp_path / "other"
    repository.mkdir()
    dataset.mkdir()
    other.mkdir()
    decision = _decision(repository, dataset)
    source = tmp_path / "next-action.json"
    _write(source, decision)
    monkeypatch.setattr(
        validation,
        "build_preacquisition_operator_next_action",
        lambda *args, **kwargs: decision,
    )

    with pytest.raises(ValueError, match="dataset root differs"):
        validation.validate_preacquisition_next_action_report(
            repository,
            other,
            source,
        )


def test_symlinked_decision_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    dataset = tmp_path / "dataset"
    repository.mkdir()
    dataset.mkdir()
    decision = _decision(repository, dataset)
    actual = tmp_path / "actual.json"
    source = tmp_path / "next-action.json"
    _write(actual, decision)
    source.symlink_to(actual)
    monkeypatch.setattr(
        validation,
        "build_preacquisition_operator_next_action",
        lambda *args, **kwargs: decision,
    )

    with pytest.raises(ValueError, match="symlink component"):
        validation.validate_preacquisition_next_action_report(
            repository,
            dataset,
            source,
        )


def test_decision_mutation_during_validation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    dataset = tmp_path / "dataset"
    repository.mkdir()
    dataset.mkdir()
    decision = _decision(repository, dataset)
    source = tmp_path / "next-action.json"
    _write(source, decision)

    def build(*args, **kwargs):
        del args, kwargs
        source.write_bytes(source.read_bytes() + b"\n")
        return deepcopy(decision)

    monkeypatch.setattr(
        validation,
        "build_preacquisition_operator_next_action",
        build,
    )

    with pytest.raises(ValueError, match="changed during validation"):
        validation.validate_preacquisition_next_action_report(
            repository,
            dataset,
            source,
        )


def test_validation_digest_is_mount_independent() -> None:
    report = {
        "repository_root": "/repo-a",
        "dataset_root": "/data-a",
        "decision_json": "/data-a/operator/next.json",
        "decision_file_sha256": "a" * 64,
        "decision_file_bytes": 123,
        "decision_evidence_sha256": "b" * 64,
        "decision_status_sha256": "c" * 64,
        "current_evidence_sha256": "b" * 64,
        "current_status_sha256": "d" * 64,
        "action_identity": {"action_id": "source", "execution_id": "one"},
        "current": True,
        "safe_to_execute": True,
    }
    relocated = deepcopy(report)
    relocated["repository_root"] = "/repo-b"
    relocated["dataset_root"] = "/data-b"
    relocated["decision_json"] = "/data-b/operator/next.json"
    relocated["decision_file_sha256"] = "e" * 64
    relocated["decision_file_bytes"] = 456
    relocated["decision_status_sha256"] = "f" * 64
    relocated["current_status_sha256"] = "0" * 64

    assert validation.next_action_validation_evidence_sha256(report) == (
        validation.next_action_validation_evidence_sha256(relocated)
    )


def test_cli_validates_and_writes_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = {
        "valid": True,
        "complete": True,
        "passed": True,
        "current": True,
        "safe_to_execute": True,
    }
    output = tmp_path / "validation.json"
    written: list[Path] = []
    monkeypatch.setattr(
        readiness_cli,
        "validate_preacquisition_next_action_report",
        lambda *args: report,
    )
    monkeypatch.setattr(
        readiness_cli,
        "write_preacquisition_next_action_validation",
        lambda path, value: written.append(Path(path)) or Path(path),
    )

    code = readiness_cli.main(
        [
            "next-action-validate",
            "/repo",
            "/dataset",
            "/dataset/operator/next-action.json",
            "--output-json",
            str(output),
        ]
    )

    assert code == 0
    assert written == [output]


def test_validation_report_writer_is_atomic(tmp_path: Path) -> None:
    output = tmp_path / "operator" / "validation.json"
    report = {
        "valid": True,
        "complete": True,
        "passed": True,
        "target_outcomes_used": False,
    }

    assert (
        validation.write_preacquisition_next_action_validation(
            output,
            report,
        )
        == output
    )
    assert json.loads(output.read_text(encoding="utf-8")) == report
