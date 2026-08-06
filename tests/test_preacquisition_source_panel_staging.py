from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

import causal4d.preacquisition_source_panel_control as source_control
import causal4d.preacquisition_source_panel_staging as staging
from causal4d.cli import preacquisition_readiness as readiness_cli
from causal4d.preacquisition_readiness_contracts import (
    SOURCE_PANEL_MANIFEST_PATH,
    SOURCE_PANEL_MANIFEST_TEMPLATE_PATH,
    source_panel_execution_manifest_template,
)


def _registered_values() -> tuple[dict, dict, dict]:
    protocol = {
        "protocol_id": "test-protocol",
        "design_sha256": "a" * 64,
    }
    profiles = [{"id": "lift_high", "amplitude_m": 0.08}]
    executions = [
        {
            "execution_id": f"source-lift_high-r{replicate}",
            "session_id": f"source-lift_high-r{replicate}",
            "command_profile_id": "lift_high",
            "contact_region_id": "upper_torso",
            "realization_condition_id": "nominal",
            "replicate": replicate,
            "fresh_reset_and_fresh_grasp": True,
            "confirmatory_fold_member": False,
        }
        for replicate in range(1, 13)
    ]
    v2 = {
        "preacquisition_signature_panel": {
            "executions": executions,
            "profiles": profiles,
        }
    }
    v4 = {
        "plan_id": "test-preacquisition-v4",
        "amendment_sha256": "b" * 64,
    }
    return protocol, v2, v4


def _patch_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, dict, dict]:
    protocol, v2, v4 = _registered_values()

    def load(repository_root):
        del repository_root
        return protocol, v2, {}, v4

    monkeypatch.setattr(staging, "load_registered_preacquisition_chain", load)
    monkeypatch.setattr(
        source_control,
        "load_registered_preacquisition_chain",
        load,
    )
    return protocol, v2, v4


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _descriptor(root: Path, relative: str) -> dict:
    data = (root / relative).read_bytes()
    return {
        "path": relative,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _scaffold(
    root: Path,
    protocol: dict,
    v2: dict,
    v4: dict,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for execution in v2["preacquisition_signature_panel"]["executions"]:
        relative = SOURCE_PANEL_MANIFEST_TEMPLATE_PATH.format(
            execution_id=execution["execution_id"]
        )
        _write_json(
            root / relative,
            source_panel_execution_manifest_template(
                execution,
                protocol,
                v4,
            ),
        )


def _completed_manifest(
    root: Path,
    execution: dict,
    protocol: dict,
    v4: dict,
    *,
    bad_digest: bool = False,
) -> dict:
    execution_id = execution["execution_id"]
    artifact_relative = (
        f"preacquisition/source_panel/executions/{execution_id}/raw.bin"
    )
    artifact = root / artifact_relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(f"physical-source:{execution_id}".encode())
    descriptor = _descriptor(root, artifact_relative)
    if bad_digest:
        descriptor["sha256"] = "0" * 64
    manifest = source_panel_execution_manifest_template(
        execution,
        protocol,
        v4,
    )
    manifest.update(
        {
            "status": "complete",
            "included": True,
            "quality_gate_failures": [],
            "started_at_utc": "2026-08-04T08:00:00Z",
            "ended_at_utc": "2026-08-04T08:01:00Z",
            "artifacts": [descriptor],
        }
    )
    return manifest


def _stage(
    root: Path,
    execution: dict,
    manifest: dict,
) -> Path:
    path = root / "staging" / f"{execution['execution_id']}.json"
    _write_json(path, manifest)
    return path


def test_staged_verification_is_read_only_and_content_addressed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    source = _stage(
        tmp_path,
        first,
        _completed_manifest(tmp_path, first, protocol, v4),
    )
    final = tmp_path / SOURCE_PANEL_MANIFEST_PATH.format(
        execution_id=first["execution_id"]
    )
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    report = staging.verify_staged_source_panel_manifest(
        tmp_path,
        tmp_path,
        source,
    )

    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert not final.exists()
    assert report["passed"] is True
    assert report["execution_id"] == first["execution_id"]
    assert report["mutated_dataset"] is False
    assert report["target_outcomes_used"] is False
    assert report["staged_manifest"]["path"] == (
        f"staging/{first['execution_id']}.json"
    )
    assert report["evidence_sha256"] == (
        staging.staged_source_verification_evidence_sha256(report)
    )
    assert report["status_sha256"] == (
        staging.staged_source_verification_status_sha256(report)
    )
    assert report["publication_command"]["argv_template"][-1] == (
        f"${{DATASET_ROOT}}/staging/{first['execution_id']}.json"
    )


def test_staged_verification_rejects_bad_artifact_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    source = _stage(
        tmp_path,
        first,
        _completed_manifest(
            tmp_path,
            first,
            protocol,
            v4,
            bad_digest=True,
        ),
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        staging.verify_staged_source_panel_manifest(
            tmp_path,
            tmp_path,
            source,
        )

    final = tmp_path / SOURCE_PANEL_MANIFEST_PATH.format(
        execution_id=first["execution_id"]
    )
    assert not final.exists()


def test_staged_verification_requires_registered_staging_location(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    source = tmp_path / "other" / f"{first['execution_id']}.json"
    _write_json(
        source,
        _completed_manifest(tmp_path, first, protocol, v4),
    )

    with pytest.raises(ValueError, match="directly below"):
        staging.verify_staged_source_panel_manifest(
            tmp_path,
            tmp_path,
            source,
        )


def test_staged_verification_requires_exact_execution_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    source = tmp_path / "staging" / "wrong.json"
    _write_json(
        source,
        _completed_manifest(tmp_path, first, protocol, v4),
    )

    with pytest.raises(ValueError, match="filename must match"):
        staging.verify_staged_source_panel_manifest(
            tmp_path,
            tmp_path,
            source,
        )


def test_staged_verification_rejects_wrong_registered_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first, second = v2["preacquisition_signature_panel"]["executions"][:2]
    source = _stage(
        tmp_path,
        first,
        _completed_manifest(tmp_path, second, protocol, v4),
    )

    with pytest.raises(ValueError, match="not the next registered"):
        staging.verify_staged_source_panel_manifest(
            tmp_path,
            tmp_path,
            source,
        )


def test_staged_verification_rejects_nested_target_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    manifest = _completed_manifest(tmp_path, first, protocol, v4)
    manifest["artifacts"][0]["target_metrics"] = {"rmse": 0.0}
    source = _stage(tmp_path, first, manifest)

    with pytest.raises(ValueError, match="target-outcome fields"):
        staging.verify_staged_source_panel_manifest(
            tmp_path,
            tmp_path,
            source,
        )


def test_staged_verification_rejects_symlinked_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    real = tmp_path / "real.json"
    _write_json(
        real,
        _completed_manifest(tmp_path, first, protocol, v4),
    )
    source = tmp_path / "staging" / f"{first['execution_id']}.json"
    source.parent.mkdir(parents=True)
    source.symlink_to(real)

    with pytest.raises(ValueError, match="symlink component"):
        staging.verify_staged_source_panel_manifest(
            tmp_path,
            tmp_path,
            source,
        )


def test_staged_verification_report_writer_is_atomic(
    tmp_path: Path,
) -> None:
    report = {
        "valid": True,
        "passed": True,
        "target_outcomes_used": False,
    }
    output = tmp_path / "reports" / "staged.json"

    written = staging.write_staged_source_panel_verification(
        output,
        report,
    )

    assert written == output
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_staged_verification_cli_writes_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {
        "valid": True,
        "passed": True,
        "target_outcomes_used": False,
    }
    output = tmp_path / "verification.json"
    written: list[tuple[Path, dict]] = []
    monkeypatch.setattr(
        readiness_cli,
        "verify_staged_source_panel_manifest",
        lambda *args: report,
    )
    monkeypatch.setattr(
        readiness_cli,
        "write_staged_source_panel_verification",
        lambda path, value: written.append((Path(path), value)) or Path(path),
    )

    code = readiness_cli.main(
        [
            "source-panel-verify-staged",
            "repository",
            "dataset",
            "dataset/staging/source.json",
            "--output-json",
            str(output),
        ]
    )

    assert code == 0
    assert written == [(output, report)]
    assert '"passed": true' in capsys.readouterr().out


def test_portable_digest_ignores_mount_and_local_status() -> None:
    report = {
        "repository_root": "/repo-a",
        "dataset_root": "/data-a",
        "source_panel_status_sha256": "a" * 64,
        "staged_manifest": {
            "path": "staging/source.json",
            "sha256": "b" * 64,
            "bytes": 12,
        },
        "passed": True,
    }
    relocated = deepcopy(report)
    relocated["repository_root"] = "/repo-b"
    relocated["dataset_root"] = "/data-b"
    relocated["source_panel_status_sha256"] = "c" * 64

    assert staging.staged_source_verification_evidence_sha256(report) == (
        staging.staged_source_verification_evidence_sha256(relocated)
    )
