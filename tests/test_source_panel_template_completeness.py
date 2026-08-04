from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import causal4d.preacquisition_source_panel_control as source_control
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
    monkeypatch.setattr(
        source_control,
        "load_registered_preacquisition_chain",
        lambda repository_root: (protocol, v2, {}, v4),
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
) -> dict:
    execution_id = execution["execution_id"]
    artifact_relative = (
        f"preacquisition/source_panel/executions/{execution_id}/raw.bin"
    )
    artifact_path = root / artifact_relative
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(f"physical-source:{execution_id}".encode())
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
            "artifacts": [_descriptor(root, artifact_relative)],
        }
    )
    return manifest


def _write_final_manifest(root: Path, execution: dict, manifest: dict) -> None:
    relative = SOURCE_PANEL_MANIFEST_PATH.format(
        execution_id=execution["execution_id"]
    )
    _write_json(root / relative, manifest)


def test_status_rejects_missing_template_before_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    template = tmp_path / SOURCE_PANEL_MANIFEST_TEMPLATE_PATH.format(
        execution_id=first["execution_id"]
    )
    template.unlink()

    status = source_control.build_source_panel_status(
        tmp_path,
        tmp_path,
        verify_file_hashes=True,
    )

    assert status["valid"] is False
    assert status["complete"] is False
    assert status["missing_template_ids"] == [first["execution_id"]]
    assert f"template_missing:{first['execution_id']}" in status["blockers"]


def test_complete_panel_rejects_deleted_template(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    executions = v2["preacquisition_signature_panel"]["executions"]
    for execution in executions:
        _write_final_manifest(
            tmp_path,
            execution,
            _completed_manifest(tmp_path, execution, protocol, v4),
        )
    first = executions[0]
    template = tmp_path / SOURCE_PANEL_MANIFEST_TEMPLATE_PATH.format(
        execution_id=first["execution_id"]
    )
    template.unlink()

    status = source_control.build_source_panel_status(
        tmp_path,
        tmp_path,
        verify_file_hashes=True,
    )

    assert status["validated_executions"] == 12
    assert status["valid"] is False
    assert status["complete"] is False
    assert status["passed"] is False
    assert status["missing_template_ids"] == [first["execution_id"]]
