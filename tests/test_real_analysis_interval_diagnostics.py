from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from causal4d.cli.real_analysis_interval_diagnostics import main as diagnostics_main
from causal4d.real_analysis_interval_diagnostics import (
    bootstrap_t_sensitivity_interval,
    build_real_analysis_interval_diagnostics,
    student_t_sensitivity_interval,
)
from causal4d.real_analysis_reporting import (
    EXPECTED_OBJECT_ID,
    EXPECTED_PREACQUISITION_SHA256,
    EXPECTED_PROTOCOL_DESIGN_SHA256,
    EXPECTED_PROTOCOL_ID,
    build_real_analysis_effect_report,
    effect_table_id_for_payload,
)
from causal4d.real_experiment_freeze import MILESTONE_ID, SCHEMA_VERSION
from causal4d.real_result_source_verification import (
    REGISTERED_ANALYSIS_ARTIFACT_KIND,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/causal4d/sloth_multi_action_v1.json"


def _write_json(path: Path, payload: object) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_pair(tmp_path: Path) -> tuple[Path, Path, str, str]:
    freeze = tmp_path / "method-freeze.json"
    freeze_sha = _write_json(
        freeze,
        {
            "schema_version": SCHEMA_VERSION,
            "milestone_id": MILESTONE_ID,
            "status": "sealed",
            "locked_before_confirmatory_collection": True,
            "target_outcomes_observed_at_freeze": False,
            "protocol": {"design_sha256": EXPECTED_PROTOCOL_DESIGN_SHA256},
            "preacquisition": {"amendment_sha256": EXPECTED_PREACQUISITION_SHA256},
            "analysis_contract": {
                "target_outcomes_may_select_method_or_hyperparameters": False,
                "optional_branches_may_change_primary_analysis": False,
            },
        },
    )
    analysis = tmp_path / "registered-analysis.json"
    analysis_sha = _write_json(
        analysis,
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
    return freeze, analysis, freeze_sha, analysis_sha


def _factual_payload(
    *,
    freeze_sha: str,
    analysis_sha: str,
) -> dict[str, object]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    executions = {value["execution_id"]: value for value in protocol["executions"]}
    session_order: dict[str, int] = {}
    records = []
    for split in protocol["splits"]["factual_continuation"]:
        execution = executions[split["execution_id"]]
        session_id = execution["session_id"]
        if session_id not in session_order:
            session_order[session_id] = len(session_order)
        index = execution["acquisition_execution_index"]
        baseline = 2.0 + 0.01 * index
        improvement = 0.25 + 0.03 * session_order[session_id]
        records.append(
            {
                "unit_id": execution["execution_id"],
                "source_execution_id": None,
                "target_execution_id": execution["execution_id"],
                "session_id": session_id,
                "acquisition_execution_index": index,
                "action_id": execution["command_profile_id"],
                "contact_region_id": execution["contact_region_id"],
                "realization_condition_id": execution["realization_condition_id"],
                "included": True,
                "exclusion_reason": None,
                "baseline_value": baseline,
                "candidate_value": baseline - improvement,
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Causal4DRealAnalysisEffectTable",
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "protocol_design_sha256": EXPECTED_PROTOCOL_DESIGN_SHA256,
        "preacquisition_amendment_sha256": EXPECTED_PREACQUISITION_SHA256,
        "method_freeze_sha256": freeze_sha,
        "analysis_manifest_sha256": analysis_sha,
        "endpoint": "factual_continuation",
        "metric_id": "track_error_m",
        "metric_unit": "m",
        "lower_is_better": True,
        "target_outcomes_used": True,
        "target_informed_selection": False,
        "object_id": EXPECTED_OBJECT_ID,
        "records": records,
    }
    payload["effect_table_id"] = effect_table_id_for_payload(payload)
    return payload


def test_sensitivity_intervals_are_translation_equivariant() -> None:
    values = [-0.2, -0.05, 0.1, 0.25, 0.6]
    offset = 3.25
    shifted = [value + offset for value in values]

    for builder in (
        student_t_sensitivity_interval,
        bootstrap_t_sensitivity_interval,
    ):
        original = builder(values)
        translated = builder(shifted)
        assert translated["point_estimate"] == pytest.approx(
            original["point_estimate"] + offset
        )
        assert translated["lower"] == pytest.approx(original["lower"] + offset)
        assert translated["upper"] == pytest.approx(original["upper"] + offset)
        assert translated["may_change_primary_decision"] is False
        assert translated["finite_sample_coverage_guaranteed"] is False


def test_bootstrap_t_is_deterministic_and_finite() -> None:
    values = np.linspace(-0.3, 0.7, 18).tolist()
    first = bootstrap_t_sensitivity_interval(values)
    second = bootstrap_t_sensitivity_interval(values)

    assert first == second
    assert first["estimable"] is True
    assert first["replicates"] == 20_000
    assert first["seed"] == 20_260_726
    assert first["finite_studentized_replicate_fraction"] > 0.99
    assert np.isfinite(first["lower"])
    assert np.isfinite(first["upper"])
    assert first["lower"] <= first["point_estimate"] <= first["upper"]


def test_degenerate_samples_produce_explicit_point_intervals() -> None:
    values = [1.5] * 12

    for result in (
        student_t_sensitivity_interval(values),
        bootstrap_t_sensitivity_interval(values),
    ):
        assert result["estimable"] is True
        assert result["degenerate_sample"] is True
        assert result["lower"] == pytest.approx(1.5)
        assert result["upper"] == pytest.approx(1.5)


def test_companion_artifact_preserves_primary_interval_and_sources(
    tmp_path: Path,
) -> None:
    freeze, analysis, freeze_sha, analysis_sha = _source_pair(tmp_path)
    payload = _factual_payload(
        freeze_sha=freeze_sha,
        analysis_sha=analysis_sha,
    )
    effect_table = tmp_path / "effects.json"
    _write_json(effect_table, payload)

    primary = build_real_analysis_effect_report(
        effect_table,
        PROTOCOL,
        method_freeze_path=freeze,
        analysis_manifest_path=analysis,
    )
    first = build_real_analysis_interval_diagnostics(
        effect_table,
        PROTOCOL,
        method_freeze_path=freeze,
        analysis_manifest_path=analysis,
    )
    second = build_real_analysis_interval_diagnostics(
        effect_table,
        PROTOCOL,
        method_freeze_path=freeze,
        analysis_manifest_path=analysis,
    )

    assert first == second
    assert first["diagnostic_id"] == second["diagnostic_id"]
    assert first["source_primary_report_id"] == primary["report_id"]
    assert first["source_verification"] == primary["source_verification"]
    assert first["included_session_count"] == 18
    assert first["primary_percentile_interval"]["interval"] == primary[
        "primary_session_clustered_effect"
    ]["confidence_interval"]
    assert (
        first["primary_percentile_interval"][
            "finite_sample_coverage_guaranteed"
        ]
        is False
    )
    assert first["sensitivity_intervals"]["may_change_primary_decision"] is False
    assert (
        first["sensitivity_intervals"][
            "promotion_requires_explicit_preacquisition_amendment"
        ]
        is True
    )
    assert first["interpretation"]["target_informed_selection"] is False
    assert first["claim_boundary"]["physical_target_outcomes_used_to_choose_interval"] is False


def test_cli_publishes_atomically_and_requires_explicit_overwrite(
    tmp_path: Path,
) -> None:
    freeze, analysis, freeze_sha, analysis_sha = _source_pair(tmp_path)
    payload = _factual_payload(
        freeze_sha=freeze_sha,
        analysis_sha=analysis_sha,
    )
    effect_table = tmp_path / "effects.json"
    output = tmp_path / "interval-diagnostics.json"
    _write_json(effect_table, payload)
    arguments = [
        str(effect_table),
        str(PROTOCOL),
        str(output),
        "--method-freeze",
        str(freeze),
        "--analysis-manifest",
        str(analysis),
    ]

    assert diagnostics_main(arguments) == 0
    published = json.loads(output.read_text(encoding="utf-8"))
    assert published["artifact_kind"] == "Causal4DRealAnalysisIntervalDiagnostics"
    assert published["sensitivity_intervals"]["may_change_primary_decision"] is False

    with pytest.raises(FileExistsError):
        diagnostics_main(arguments)
    assert diagnostics_main([*arguments, "--overwrite"]) == 0
