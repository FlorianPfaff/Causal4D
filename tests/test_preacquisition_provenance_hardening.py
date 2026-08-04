from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

import causal4d.preacquisition_gate_validation as gate_validation
import causal4d.preacquisition_readiness_contracts as contracts
from causal4d.preacquisition_readiness import (
    GATE_PATHS,
    gate_evidence_sha256,
    gate_evidence_template,
)
from causal4d.preacquisition_readiness_contracts import (
    SOURCE_PANEL_MANIFEST_PATH,
    SOURCE_PANEL_MANIFEST_TEMPLATE_PATH,
    source_panel_execution_manifest_template,
)
from causal4d.preacquisition_source_validation import (
    _validate_source_execution_manifest,
)


def _registered_values() -> tuple[dict, dict, dict]:
    protocol = {
        "protocol_id": "test-protocol",
        "design_sha256": "a" * 64,
        "executions": [],
        "quality_gates": {"maximum_rgbd_actuator_sync_error_ms": 10.0},
    }
    profile = {"id": "test-profile", "amplitude_m": 0.08}
    executions = [
        {
            "execution_id": f"source-{index:02d}",
            "session_id": f"session-{index:02d}",
            "command_profile_id": profile["id"],
            "contact_region_id": "upper_torso",
            "realization_condition_id": "nominal",
            "replicate": index + 1,
            "fresh_reset_and_fresh_grasp": True,
            "confirmatory_fold_member": False,
        }
        for index in range(12)
    ]
    v2 = {
        "preacquisition_signature_panel": {
            "executions": executions,
            "profiles": [profile],
        }
    }
    v4 = {
        "plan_id": "test-preacquisition-v4",
        "amendment_sha256": "b" * 64,
    }
    return protocol, v2, v4


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _descriptor(root: Path, relative: str) -> dict:
    payload = (root / relative).read_bytes()
    return {
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _scaffold_complete_panel(
    root: Path,
    protocol: dict,
    v2: dict,
    v4: dict,
) -> list[dict]:
    evidence: list[dict] = []
    executions = v2["preacquisition_signature_panel"]["executions"]
    for execution in executions:
        execution_id = execution["execution_id"]
        template = source_panel_execution_manifest_template(
            execution,
            protocol,
            v4,
        )
        template_relative = SOURCE_PANEL_MANIFEST_TEMPLATE_PATH.format(
            execution_id=execution_id
        )
        _write_json(root / template_relative, template)
        artifact_relative = (
            f"preacquisition/source_panel/executions/{execution_id}/measurement.bin"
        )
        artifact_path = root / artifact_relative
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(execution_id.encode("utf-8"))
        manifest = deepcopy(template)
        manifest.update(
            {
                "status": "complete",
                "included": True,
                "started_at_utc": "2026-07-30T09:00:00Z",
                "ended_at_utc": "2026-07-30T09:01:00Z",
                "artifacts": [_descriptor(root, artifact_relative)],
            }
        )
        manifest_relative = SOURCE_PANEL_MANIFEST_PATH.format(execution_id=execution_id)
        _write_json(root / manifest_relative, manifest)
        evidence.append(_descriptor(root, manifest_relative))
    return evidence


def _signature_gate(
    root: Path,
    protocol: dict,
    v2: dict,
    v4: dict,
    *,
    sealed: bool,
) -> dict:
    gate = gate_evidence_template(
        "signature_panel_complete",
        protocol,
        v2,
        v4,
    )
    gate["completed_at_utc"] = "2026-07-30T10:00:00Z"
    gate["target_outcomes_used"] = False
    gate["evidence"] = _scaffold_complete_panel(
        root,
        protocol,
        v2,
        v4,
    )
    if sealed:
        gate["status"] = "passed"
        gate["locked_before_confirmatory_collection"] = True
        gate["approval"] = {
            "approved": True,
            "approver_id": "reviewer",
            "approved_at_utc": "2026-07-30T10:05:00Z",
        }
        gate["artifact_sha256"] = gate_evidence_sha256(gate)
    return gate


@pytest.mark.parametrize(
    "payload",
    [
        '{"value": 1, "value": 2}',
        '{"outer": {"value": 1, "value": 2}}',
    ],
)
def test_acquisition_json_rejects_duplicate_keys(
    tmp_path: Path,
    payload: str,
) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        contracts._read_json_mapping(path, name="duplicate fixture")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_id", 1),
        ("session_id", True),
    ],
)
def test_registered_source_ids_reject_coercible_values(
    field: str,
    value: object,
) -> None:
    _, v2, _ = _registered_values()
    v2["preacquisition_signature_panel"]["executions"][0][field] = value

    with pytest.raises(ValueError, match="id is invalid"):
        contracts._expected_source_panel(v2)


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_source_manifest_schema_requires_an_exact_integer(
    tmp_path: Path,
    schema_version: object,
) -> None:
    protocol, v2, v4 = _registered_values()
    execution = v2["preacquisition_signature_panel"]["executions"][0]
    _scaffold_complete_panel(tmp_path, protocol, v2, v4)
    relative = SOURCE_PANEL_MANIFEST_PATH.format(execution_id=execution["execution_id"])
    path = tmp_path / relative
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_version"] = schema_version
    _write_json(path, manifest)

    with pytest.raises(
        ValueError,
        match="unsupported source execution schema",
    ):
        _validate_source_execution_manifest(
            tmp_path,
            relative,
            protocol=protocol,
            v4=v4,
            execution_id=execution["execution_id"],
            session_id=execution["session_id"],
            verify_file_hashes=False,
        )


@pytest.mark.parametrize(
    "mutation",
    ["delete", "alter", "directory", "symlink"],
)
def test_signature_gate_revalidates_registered_templates(
    tmp_path: Path,
    mutation: str,
) -> None:
    protocol, v2, v4 = _registered_values()
    gate = _signature_gate(
        tmp_path,
        protocol,
        v2,
        v4,
        sealed=True,
    )
    execution_id = v2["preacquisition_signature_panel"]["executions"][0]["execution_id"]
    template = tmp_path / SOURCE_PANEL_MANIFEST_TEMPLATE_PATH.format(
        execution_id=execution_id
    )
    if mutation == "delete":
        template.unlink()
    elif mutation == "alter":
        payload = json.loads(template.read_text(encoding="utf-8"))
        payload["included"] = True
        _write_json(template, payload)
    elif mutation == "directory":
        template.unlink()
        template.mkdir()
    else:
        replacement_id = v2["preacquisition_signature_panel"]["executions"][1][
            "execution_id"
        ]
        replacement = tmp_path / SOURCE_PANEL_MANIFEST_TEMPLATE_PATH.format(
            execution_id=replacement_id
        )
        template.unlink()
        try:
            template.symlink_to(replacement)
        except OSError:
            pytest.skip("symlinks are unavailable on this platform")

    gate_path = tmp_path / GATE_PATHS["signature_panel_complete"]
    _write_json(gate_path, gate)
    result = gate_validation._validate_gate_file(
        "signature_panel_complete",
        gate_path,
        protocol=protocol,
        v2=v2,
        v4=v4,
        dataset_root=tmp_path,
        prerequisites={},
        verify_file_hashes=True,
    )

    assert result["valid"] is False
    assert "source-panel status is invalid" in str(result["error"])


def test_signature_gate_rejects_unexpected_execution_directory(
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _registered_values()
    gate = _signature_gate(
        tmp_path,
        protocol,
        v2,
        v4,
        sealed=True,
    )
    unexpected = (
        tmp_path
        / "preacquisition"
        / "source_panel"
        / "executions"
        / "unregistered-execution"
    )
    unexpected.mkdir(parents=True)
    gate_path = tmp_path / GATE_PATHS["signature_panel_complete"]
    _write_json(gate_path, gate)

    result = gate_validation._validate_gate_file(
        "signature_panel_complete",
        gate_path,
        protocol=protocol,
        v2=v2,
        v4=v4,
        dataset_root=tmp_path,
        prerequisites={},
        verify_file_hashes=True,
    )

    assert result["valid"] is False
    assert "unexpected_execution_directory" in str(result["error"])


def test_signature_gate_binds_portable_source_panel_digest(
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _registered_values()
    gate = _signature_gate(
        tmp_path,
        protocol,
        v2,
        v4,
        sealed=True,
    )
    gate_path = tmp_path / GATE_PATHS["signature_panel_complete"]
    _write_json(gate_path, gate)

    result = gate_validation._validate_gate_file(
        "signature_panel_complete",
        gate_path,
        protocol=protocol,
        v2=v2,
        v4=v4,
        dataset_root=tmp_path,
        prerequisites={},
        verify_file_hashes=True,
    )

    assert result["valid"] is True
    assert len(result["source_panel_evidence_sha256"]) == 64


def test_signature_gate_rejects_float_session_count(
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _registered_values()
    gate = _signature_gate(
        tmp_path,
        protocol,
        v2,
        v4,
        sealed=True,
    )
    gate["checks"]["independent_session_count"] = 12.0
    gate["artifact_sha256"] = gate_evidence_sha256(gate)
    gate_path = tmp_path / GATE_PATHS["signature_panel_complete"]
    _write_json(gate_path, gate)

    result = gate_validation._validate_gate_file(
        "signature_panel_complete",
        gate_path,
        protocol=protocol,
        v2=v2,
        v4=v4,
        dataset_root=tmp_path,
        prerequisites={},
        verify_file_hashes=True,
    )

    assert result["valid"] is False
    assert "12 independent sessions" in str(result["error"])


def test_sealing_refuses_a_deleted_registered_template(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _registered_values()
    gate = _signature_gate(
        tmp_path,
        protocol,
        v2,
        v4,
        sealed=False,
    )
    execution_id = v2["preacquisition_signature_panel"]["executions"][0]["execution_id"]
    template = tmp_path / SOURCE_PANEL_MANIFEST_TEMPLATE_PATH.format(
        execution_id=execution_id
    )
    template.unlink()
    gate_path = tmp_path / GATE_PATHS["signature_panel_complete"]
    _write_json(gate_path, gate)
    monkeypatch.setattr(
        gate_validation,
        "load_registered_preacquisition_chain",
        lambda repository_root: (protocol, v2, {}, v4),
    )
    monkeypatch.setattr(
        gate_validation,
        "build_real_evidence_status",
        lambda *args, **kwargs: {
            "prerequisites": {"method_freeze": {"valid": False}},
            "manifest_executions": 0,
            "acquired_executions": 0,
            "validated_executions": 0,
        },
    )

    with pytest.raises(
        ValueError,
        match="source-panel status is invalid",
    ):
        gate_validation.seal_preacquisition_gate(
            tmp_path,
            tmp_path,
            "signature_panel_complete",
            approved_by="reviewer",
            approved_at_utc="2026-07-30T10:05:00Z",
        )

    retained = json.loads(gate_path.read_text(encoding="utf-8"))
    assert retained["status"] == "template"
    assert retained["artifact_sha256"] is None
