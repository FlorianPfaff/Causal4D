from __future__ import annotations

from pathlib import Path

from causal4d.preacquisition_readiness import scaffold_preacquisition_readiness
from causal4d.preacquisition_source_panel_control import build_source_panel_status


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_registered_source_panel_scaffold_reports_exact_first_execution(
    tmp_path: Path,
) -> None:
    scaffold = scaffold_preacquisition_readiness(REPOSITORY_ROOT, tmp_path)

    status = build_source_panel_status(
        REPOSITORY_ROOT,
        tmp_path,
        verify_file_hashes=True,
    )

    assert scaffold["passed"] is True
    assert status["valid"] is True
    assert status["complete"] is False
    assert status["specified_executions"] == 12
    assert status["manifest_executions"] == 0
    assert status["validated_executions"] == 0
    assert status["completed_execution_ids"] == []
    assert status["invalid_execution_ids"] == []
    assert status["invalid_template_ids"] == []
    assert status["missing_template_ids"] == []
    assert status["registered_prefix_valid"] is True
    assert status["next_execution"]["execution_id"] == (
        "sloth-pre-v2-lift_high-r1"
    )
    assert status["next_execution"]["session_id"] == (
        "sloth-pre-v2-lift_high-r1"
    )
    assert status["next_execution"]["source_panel_execution_index"] == 0
    assert status["next_execution"]["profile"] == {
        "amplitude_m": 0.08,
        "direction_controller": [0.0, 0.0, 1.0],
        "hold_duration_s": 0.25,
        "id": "lift_high",
        "outbound_duration_s": 0.75,
        "post_return_settle_s": 1.5,
        "return_duration_s": 0.75,
        "waveform": "minimum_jerk_out_hold_minimum_jerk_return",
    }
    assert status["target_outcomes_used"] is False
