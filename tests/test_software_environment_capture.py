from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

import causal4d.software_environment_capture as capture
from causal4d.preacquisition_readiness_contracts import (
    GATE_EVIDENCE_ARTIFACT_KIND,
    GATE_EVIDENCE_SCHEMA_VERSION,
)
from causal4d.stack_lock import build_stack_lock, write_stack_lock


REVISIONS = {
    "prob4d": "1" * 40,
    "bayesian-phystwin": "2" * 40,
    "causal4d": "3" * 40,
}


def _wheel(tmp_path: Path, name: str, version: str = "1.0") -> Path:
    filename_name = name.replace("-", "_")
    path = tmp_path / f"{filename_name}-{version}-py3-none-any.whl"
    dist_info = f"{filename_name}-{version}.dist-info"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{dist_info}/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
            "Tag: py3-none-any\n",
        )
    return path


def _stack(tmp_path: Path) -> tuple[Path, tuple[Path, ...], dict]:
    wheels = tuple(_wheel(tmp_path, name) for name in REVISIONS)
    lock = build_stack_lock(wheels, source_revisions=REVISIONS)
    lock_path = tmp_path / "stack-lock.json"
    write_stack_lock(lock_path, lock)
    return lock_path, wheels, lock


def _gate_template(dataset: Path) -> Path:
    path = dataset / "preacquisition" / "software_environment.json"
    path.parent.mkdir(parents=True)
    payload = {
        "schema_version": GATE_EVIDENCE_SCHEMA_VERSION,
        "artifact_kind": GATE_EVIDENCE_ARTIFACT_KIND,
        "gate_id": "software_environment_locked",
        "protocol_id": "protocol-v1",
        "protocol_design_sha256": "a" * 64,
        "preacquisition_plan_id": "plan-v4",
        "preacquisition_amendment_sha256": "b" * 64,
        "status": "template",
        "completed_at_utc": None,
        "locked_before_confirmatory_collection": None,
        "target_outcomes_used": None,
        "checks": {},
        "evidence": [],
        "approval": {
            "approved": False,
            "approver_id": None,
            "approved_at_utc": None,
        },
        "artifact_sha256": None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _context() -> capture._CaptureContext:
    return capture._CaptureContext(
        protocol={
            "protocol_id": "protocol-v1",
            "design_sha256": "a" * 64,
        },
        v4={"plan_id": "plan-v4", "amendment_sha256": "b" * 64},
        method_freeze={
            "causal4d": {"commit_sha": REVISIONS["causal4d"]},
            "bayesian_phystwin": {
                "commit_sha": REVISIONS["bayesian-phystwin"]
            },
        },
        method_freeze_sha256="c" * 64,
        method_freeze_validation_sha256="d" * 64,
    )


def _installed_records(lock: dict) -> tuple[dict, ...]:
    return tuple(
        {
            "name": entry["name"],
            "version": entry["version"],
            "installer": "pip",
            "editable": False,
            "direct_url": {},
            "archive_sha256": entry["wheel"]["sha256"],
        }
        for entry in lock["distributions"]
    )


def _runtime_snapshot(**kwargs):
    del kwargs
    python_record = {
        "version": "3.12.13",
        "implementation": "CPython",
        "platform": "test-platform",
    }
    runtime_record = {
        "resolved_dependency_report": None,
        "execution_backend": "numpy_cpu",
        "containerized": False,
        "container_image_digest": None,
        "numpy_version": "2.2.6",
        "scipy_version": "1.17.1",
        "torch_version": None,
        "warp_version": None,
        "opencv_version": None,
        "cuda_runtime_version": None,
        "cuda_driver_version": None,
    }
    report = {
        "schema_name": capture.CAPTURE_SCHEMA_NAME,
        "schema_version": capture.CAPTURE_SCHEMA_VERSION,
        "generated_by": capture.CAPTURE_GENERATOR,
        "distributions": [],
    }
    return python_record, runtime_record, report


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, tuple[Path, ...], dict]:
    repository = tmp_path / "repository"
    dataset = tmp_path / "dataset"
    repository.mkdir()
    dataset.mkdir()
    _gate_template(dataset)
    lock_path, wheels, lock = _stack(tmp_path)
    monkeypatch.setattr(capture, "_capture_context", lambda *_: _context())
    monkeypatch.setattr(
        capture,
        "_installed_distribution_records",
        lambda: _installed_records(lock),
    )
    monkeypatch.setattr(capture, "_runtime_snapshot", _runtime_snapshot)
    return repository, dataset, wheels, lock


def test_capture_populates_but_does_not_approve_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, dataset, wheels, lock = _prepare(tmp_path, monkeypatch)
    lock_path = tmp_path / "stack-lock.json"

    result = capture.capture_software_environment_template(
        repository,
        dataset,
        lock_path,
        wheels,
        execution_backend="numpy_cpu",
        observation_producer_name="registered-rgbd-prefix",
        observation_producer_version="1",
        observation_artifact_contract="phys4d.observation_belief.v1",
        prob4d_used=False,
        prob4d_unused_reason="fresh real provider is not admitted",
        completed_at_utc="2026-08-08T12:00:00+00:00",
    )

    gate_path = dataset / "preacquisition" / "software_environment.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert result["ready_to_seal"] is True
    assert result["approved"] is False
    assert gate["status"] == "template"
    assert gate["approval"]["approved"] is False
    assert gate["artifact_sha256"] is None
    assert gate["target_outcomes_used"] is False
    assert gate["checks"]["stack_lock_id"] == lock["lock_id"]
    assert gate["checks"]["prob4d"] == {
        "used": False,
        "reason": "fresh real provider is not admitted",
    }
    assert gate["checks"]["runtime_environment"][
        "resolved_dependency_report"
    ].endswith("/resolved-environment.json")
    assert len(gate["evidence"]) == 6
    for descriptor in gate["evidence"]:
        evidence_path = dataset / descriptor["path"]
        assert evidence_path.is_file()
        assert not evidence_path.is_symlink()
    assert "seal-gate" in result["next_command"]


def test_capture_records_used_prob4d_from_locked_wheel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, dataset, wheels, lock = _prepare(tmp_path, monkeypatch)

    capture.capture_software_environment_template(
        repository,
        dataset,
        tmp_path / "stack-lock.json",
        wheels,
        execution_backend="numpy_cpu",
        observation_producer_name="Prob4D",
        observation_producer_version="1.0",
        observation_artifact_contract="phys4d.observation_belief.v1",
        prob4d_used=True,
        prob4d_observation_contract_version="phys4d.observation_belief.v1",
    )

    gate = json.loads(
        (dataset / "preacquisition" / "software_environment.json").read_text()
    )
    declaration = gate["checks"]["prob4d"]
    assert declaration["used"] is True
    assert declaration["commit_sha"] == REVISIONS["prob4d"]
    assert declaration["version"] == lock["distributions"][0]["version"]
    assert declaration["distribution"]["path"].endswith(".whl")


def test_capture_rejects_stack_revision_different_from_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, dataset, wheels, lock = _prepare(tmp_path, monkeypatch)
    wrong = _context()
    wrong = capture._CaptureContext(
        protocol=wrong.protocol,
        v4=wrong.v4,
        method_freeze={
            **wrong.method_freeze,
            "bayesian_phystwin": {"commit_sha": "9" * 40},
        },
        method_freeze_sha256=wrong.method_freeze_sha256,
        method_freeze_validation_sha256=wrong.method_freeze_validation_sha256,
    )
    monkeypatch.setattr(capture, "_capture_context", lambda *_: wrong)

    with pytest.raises(ValueError, match="BayesianPhysTwin revision differs"):
        capture.capture_software_environment_template(
            repository,
            dataset,
            tmp_path / "stack-lock.json",
            wheels,
            execution_backend="numpy_cpu",
            observation_producer_name="registered-rgbd-prefix",
            observation_producer_version="1",
            observation_artifact_contract="contract-v1",
            prob4d_used=False,
            prob4d_unused_reason="not admitted",
        )
    assert not (dataset / "preacquisition" / "software_environment" / lock["lock_id"]).exists()


def test_capture_is_non_overwriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, dataset, wheels, _ = _prepare(tmp_path, monkeypatch)
    kwargs = dict(
        execution_backend="numpy_cpu",
        observation_producer_name="registered-rgbd-prefix",
        observation_producer_version="1",
        observation_artifact_contract="contract-v1",
        prob4d_used=False,
        prob4d_unused_reason="not admitted",
    )
    capture.capture_software_environment_template(
        repository,
        dataset,
        tmp_path / "stack-lock.json",
        wheels,
        **kwargs,
    )
    with pytest.raises(ValueError, match="already sealed|already exists"):
        capture.capture_software_environment_template(
            repository,
            dataset,
            tmp_path / "stack-lock.json",
            wheels,
            **kwargs,
        )


def test_installed_core_stack_rejects_editable_or_wrong_archive(
    tmp_path: Path,
) -> None:
    _, _, lock = _stack(tmp_path)
    records = list(_installed_records(lock))
    records[0] = {**records[0], "editable": True}
    with pytest.raises(ValueError, match="installed editable"):
        capture._validate_installed_core_stack(lock, records)

    records = list(_installed_records(lock))
    records[1] = {**records[1], "archive_sha256": "f" * 64}
    with pytest.raises(ValueError, match="exact locked wheel archive"):
        capture._validate_installed_core_stack(lock, records)


def test_archive_sha256_supports_pep610_forms() -> None:
    digest = "e" * 64
    assert capture._archive_sha256(
        {"archive_info": {"hash": f"sha256={digest}"}}
    ) == digest
    assert capture._archive_sha256(
        {"archive_info": {"hashes": {"sha256": digest}}}
    ) == digest
    assert capture._archive_sha256({"archive_info": {}}) is None
