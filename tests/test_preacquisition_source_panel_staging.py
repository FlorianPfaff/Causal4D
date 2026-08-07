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
    profiles = [
        {"id": "lift_high", "amplitude_m": 0.08},
        {"id": "lower_high", "amplitude_m": 0.08},
        {"id": "lift_high_slow", "amplitude_m": 0.08},
        {"id": "lift_high_long_hold", "amplitude_m": 0.08},
    ]
    executions: list[dict] = []
    for profile in profiles:
        for replicate in range(1, 4):
            identifier = f"source-{profile['id']}-r{replicate}"
            executions.append(
                {
                    "execution_id": identifier,
                    "session_id": identifier,
                    "command_profile_id": profile["id"],
                    "contact_region_id": "upper_torso",
                    "realization_condition_id": "nominal",
                    "replicate": replicate,
                    "fresh_reset_and_fresh_grasp": True,
                    "confirmatory_fold_member": False,
                }
            )
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


def _patch_chain(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, dict]:
    protocol, v2, v4 = _registered_values()

    def loader(repository_root):
        del repository_root
        return protocol, v2, {}, v4

    monkeypatch.setattr(
        source_control,
        "load_registered_preacquisition_chain",
        loader,
    )
    monkeypatch.setattr(
        staging,
        "load_registered_preacquisition_chain",
        loader,
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


def _scaffold(root: Path, protocol: dict, v2: dict, v4: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    executions = v2["preacquisition_signature_panel"]["executions"]
    for execution in executions:
        relative = SOURCE_PANEL_MANIFEST_TEMPLATE_PATH.format(
            execution_id=execution["execution_id"]
        )
        template = source_panel_execution_manifest_template(
            execution,
            protocol,
            v4,
        )
        _write_json(root / relative, template)


def _completed_manifest(
    root: Path,
    execution: dict,
    protocol: dict,
    v4: dict,
    *,
    bad_digest: bool = False,
) -> dict:
    execution_id = execution["execution_id"]
    artifact_relative = f"preacquisition/source_panel/executions/{execution_id}/raw.bin"
    artifact_path = root / artifact_relative
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(f"physical-source:{execution_id}".encode())
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


def _staged_path(root: Path, execution: dict) -> Path:
    return root / "staging" / f"{execution['execution_id']}.json"


def test_preflight_verifies_next_manifest_without_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    source = _staged_path(tmp_path, first)
    manifest = _completed_manifest(tmp_path, first, protocol, v4)
    _write_json(source, manifest)

    result = staging.verify_source_panel_manifest_staging(
        tmp_path,
        tmp_path,
        source,
    )

    final = tmp_path / SOURCE_PANEL_MANIFEST_PATH.format(
        execution_id=first["execution_id"]
    )
    assert result["safe_to_publish"] is True
    assert result["published"] is False
    assert result["claim_bearing_evidence_mutated"] is False
    assert result["changes_registered_method"] is False
    assert result["source_panel_status_stable"] is True
    assert result["execution_id"] == first["execution_id"]
    assert result["validated_executions_before"] == 0
    assert result["validated_executions_after_publication"] == 1
    assert result["artifact_count"] == 1
    assert result["artifacts"] == manifest["artifacts"]
    assert result["target_outcomes_used"] is False
    assert result["evidence_sha256"] == (
        staging.source_panel_staging_evidence_sha256(result)
    )
    assert result["status_sha256"] == (
        staging.source_panel_staging_status_sha256(result)
    )
    assert not final.exists()


def test_preflight_evidence_hash_is_mount_independent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    source = _staged_path(tmp_path, first)
    _write_json(source, _completed_manifest(tmp_path, first, protocol, v4))
    result = staging.verify_source_panel_manifest_staging(
        tmp_path,
        tmp_path,
        source,
    )
    relocated = deepcopy(result)
    relocated["repository_root"] = "/relocated/repository"
    relocated["dataset_root"] = "/relocated/dataset"
    relocated["source_json"] = (
        "/relocated/dataset/" + result["source_manifest_relative_path"]
    )
    relocated["source_panel_status_sha256_before"] = "0" * 64
    relocated["publication_command_argv"][4:] = [
        relocated["repository_root"],
        relocated["dataset_root"],
        relocated["source_json"],
    ]
    relocated["publication_command_text"] = "host-specific command"
    relocated.pop("evidence_sha256")
    relocated.pop("status_sha256")

    assert staging.source_panel_staging_evidence_sha256(result) == (
        staging.source_panel_staging_evidence_sha256(relocated)
    )


def test_preflight_requires_exact_staging_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    source = tmp_path / "staging" / "wrong.json"
    _write_json(source, _completed_manifest(tmp_path, first, protocol, v4))

    with pytest.raises(ValueError, match="filename must match"):
        staging.verify_source_panel_manifest_staging(
            tmp_path,
            tmp_path,
            source,
        )


def test_preflight_rejects_bad_artifact_hash_without_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    source = _staged_path(tmp_path, first)
    _write_json(
        source,
        _completed_manifest(
            tmp_path,
            first,
            protocol,
            v4,
            bad_digest=True,
        ),
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        staging.verify_source_panel_manifest_staging(tmp_path, tmp_path, source)

    final = tmp_path / SOURCE_PANEL_MANIFEST_PATH.format(
        execution_id=first["execution_id"]
    )
    assert not final.exists()


def test_preflight_rejects_nonnext_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first, second = v2["preacquisition_signature_panel"]["executions"][:2]
    source = _staged_path(tmp_path, first)
    _write_json(source, _completed_manifest(tmp_path, second, protocol, v4))

    with pytest.raises(ValueError, match="next registered execution"):
        staging.verify_source_panel_manifest_staging(tmp_path, tmp_path, source)


def test_preflight_rejects_symlinked_staging_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    actual = tmp_path / "actual.json"
    alias = _staged_path(tmp_path, first)
    _write_json(actual, _completed_manifest(tmp_path, first, protocol, v4))
    alias.parent.mkdir(parents=True)
    alias.symlink_to(actual)

    with pytest.raises(ValueError, match="symlink component"):
        staging.verify_source_panel_manifest_staging(tmp_path, tmp_path, alias)


def test_preflight_rejects_staging_file_outside_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    dataset = tmp_path / "dataset"
    _scaffold(dataset, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    source = tmp_path / f"{first['execution_id']}.json"
    _write_json(source, _completed_manifest(dataset, first, protocol, v4))

    with pytest.raises(ValueError, match="directly below dataset_root/staging"):
        staging.verify_source_panel_manifest_staging(tmp_path, dataset, source)


def test_preflight_rejects_file_inside_dataset_but_outside_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    source = tmp_path / "operator" / f"{first['execution_id']}.json"
    _write_json(source, _completed_manifest(tmp_path, first, protocol, v4))

    with pytest.raises(ValueError, match="directly below dataset_root/staging"):
        staging.verify_source_panel_manifest_staging(tmp_path, tmp_path, source)


def test_preflight_rejects_staging_file_mutation_during_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    source = _staged_path(tmp_path, first)
    _write_json(source, _completed_manifest(tmp_path, first, protocol, v4))
    validate = staging._validate_source_execution_manifest

    def validate_then_mutate(*args, **kwargs) -> None:
        validate(*args, **kwargs)
        source.write_bytes(source.read_bytes() + b"\n")

    monkeypatch.setattr(
        staging,
        "_validate_source_execution_manifest",
        validate_then_mutate,
    )

    with pytest.raises(ValueError, match="changed during validation"):
        staging.verify_source_panel_manifest_staging(tmp_path, tmp_path, source)

    final = tmp_path / SOURCE_PANEL_MANIFEST_PATH.format(
        execution_id=first["execution_id"]
    )
    assert not final.exists()


def test_preflight_rejects_artifact_mutation_during_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    source = _staged_path(tmp_path, first)
    manifest = _completed_manifest(tmp_path, first, protocol, v4)
    _write_json(source, manifest)
    artifact = tmp_path / manifest["artifacts"][0]["path"]
    validate = staging._validate_source_execution_manifest

    def validate_then_mutate(*args, **kwargs) -> None:
        validate(*args, **kwargs)
        artifact.write_bytes(artifact.read_bytes() + b"x")

    monkeypatch.setattr(
        staging,
        "_validate_source_execution_manifest",
        validate_then_mutate,
    )

    with pytest.raises(ValueError, match="artifacts changed during validation"):
        staging.verify_source_panel_manifest_staging(tmp_path, tmp_path, source)


def test_preflight_rejects_source_panel_status_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    source = _staged_path(tmp_path, first)
    _write_json(source, _completed_manifest(tmp_path, first, protocol, v4))
    build = staging.build_source_panel_status
    calls = 0

    def changing_status(*args, **kwargs):
        nonlocal calls
        calls += 1
        status = build(*args, **kwargs)
        if calls == 2:
            status = deepcopy(status)
            status["status_sha256"] = "0" * 64
        return status

    monkeypatch.setattr(staging, "build_source_panel_status", changing_status)

    with pytest.raises(ValueError, match="status changed"):
        staging.verify_source_panel_manifest_staging(tmp_path, tmp_path, source)


def test_preflight_rejects_nested_target_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    manifest = _completed_manifest(tmp_path, first, protocol, v4)
    manifest["artifacts"][0]["target_metrics"] = {"rmse": 0.0}
    source = _staged_path(tmp_path, first)
    _write_json(source, manifest)

    with pytest.raises(ValueError, match="target-outcome fields"):
        staging.verify_source_panel_manifest_staging(tmp_path, tmp_path, source)


def test_cli_exposes_preflight_and_writes_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = {
        "valid": True,
        "complete": True,
        "passed": True,
        "safe_to_publish": True,
    }
    written: list[Path] = []
    monkeypatch.setattr(
        readiness_cli,
        "verify_source_panel_manifest_staging",
        lambda *args: result,
    )
    monkeypatch.setattr(
        readiness_cli,
        "write_source_panel_staging_preflight",
        lambda path, value: written.append(Path(path)) or Path(path),
    )
    output = tmp_path / "preflight.json"

    code = readiness_cli.main(
        [
            "source-panel-verify-staged",
            "/repo",
            "/data/run",
            "/data/run/staging/source-01.json",
            "--output-json",
            str(output),
        ]
    )

    assert code == 0
    assert written == [output]
