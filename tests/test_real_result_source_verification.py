from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from causal4d.real_experiment_freeze import MILESTONE_ID, SCHEMA_VERSION
from causal4d.real_result_interpretation import (
    EXPECTED_PREACQUISITION_SHA256,
    EXPECTED_PROTOCOL_DESIGN_SHA256,
    EXPECTED_PROTOCOL_ID,
    RealResultGateSummary,
)
from causal4d.real_result_source_verification import (
    REGISTERED_ANALYSIS_ARTIFACT_KIND,
    validate_real_result_source_verification,
    verify_real_result_sources,
)


def _write_json(path: Path, payload: dict[str, object]) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_pair(
    tmp_path: Path,
    *,
    freeze_protocol_sha256: str = EXPECTED_PROTOCOL_DESIGN_SHA256,
) -> tuple[Path, Path, RealResultGateSummary]:
    freeze_path = tmp_path / "method-freeze.json"
    freeze_sha = _write_json(
        freeze_path,
        {
            "schema_version": SCHEMA_VERSION,
            "milestone_id": MILESTONE_ID,
            "status": "sealed",
            "locked_before_confirmatory_collection": True,
            "target_outcomes_observed_at_freeze": False,
            "protocol": {"design_sha256": freeze_protocol_sha256},
            "preacquisition": {"amendment_sha256": EXPECTED_PREACQUISITION_SHA256},
            "analysis_contract": {
                "target_outcomes_may_select_method_or_hyperparameters": False,
                "optional_branches_may_change_primary_analysis": False,
            },
        },
    )
    analysis_path = tmp_path / "analysis.json"
    analysis_sha = _write_json(
        analysis_path,
        {
            "schema_version": 1,
            "artifact_kind": REGISTERED_ANALYSIS_ARTIFACT_KIND,
            "analysis_id": "registered-real-analysis-v1",
            "protocol_id": EXPECTED_PROTOCOL_ID,
            "protocol_design_sha256": EXPECTED_PROTOCOL_DESIGN_SHA256,
            "preacquisition_amendment_sha256": EXPECTED_PREACQUISITION_SHA256,
            "method_freeze_sha256": freeze_sha,
            "primary_analysis_locked": True,
            "target_outcomes_may_select_method_or_hyperparameters": False,
            "optional_branches_may_change_primary_analysis": False,
        },
    )
    gates = RealResultGateSummary(
        protocol_id=EXPECTED_PROTOCOL_ID,
        protocol_design_sha256=EXPECTED_PROTOCOL_DESIGN_SHA256,
        preacquisition_amendment_sha256=EXPECTED_PREACQUISITION_SHA256,
        method_freeze_sha256=freeze_sha,
        analysis_manifest_sha256=analysis_sha,
        evidence_status="complete",
        factual_continuation="passed",
        same_grasp_transfer="passed",
        new_contact_transfer="passed",
        execution_block_calibration="passed",
    )
    return freeze_path, analysis_path, gates


def test_source_verification_is_portable_and_tamper_evident(tmp_path: Path) -> None:
    freeze_path, analysis_path, gates = _source_pair(tmp_path)

    verification = verify_real_result_sources(
        gates,
        method_freeze_path=freeze_path,
        analysis_manifest_path=analysis_path,
    )

    restored = validate_real_result_source_verification(verification)
    assert restored["method_freeze"]["sha256"] == gates.method_freeze_sha256
    assert "path" not in restored["method_freeze"]

    tampered = json.loads(json.dumps(verification))
    tampered["method_freeze"]["bytes"] += 1
    with pytest.raises(ValueError, match="source-verification SHA-256 mismatch"):
        validate_real_result_source_verification(tampered)


def test_source_verification_rejects_method_freeze_protocol_drift(
    tmp_path: Path,
) -> None:
    freeze_path, analysis_path, gates = _source_pair(
        tmp_path,
        freeze_protocol_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="protocol digest differs"):
        verify_real_result_sources(
            gates,
            method_freeze_path=freeze_path,
            analysis_manifest_path=analysis_path,
        )


def test_source_verification_rejects_symlinked_source(tmp_path: Path) -> None:
    freeze_path, analysis_path, gates = _source_pair(tmp_path)
    link = tmp_path / "analysis-link.json"
    try:
        link.symlink_to(analysis_path)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="must not be a symlink"):
        verify_real_result_sources(
            gates,
            method_freeze_path=freeze_path,
            analysis_manifest_path=link,
        )
