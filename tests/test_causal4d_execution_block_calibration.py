import json
from pathlib import Path

import numpy as np
import pytest

from causal4d.execution_block_calibration import (
    ExecutionBlockCalibrationCase,
    evaluate_execution_block_cases,
    fit_execution_block_conformal_calibration,
    load_execution_block_conformal_calibration,
    save_execution_block_conformal_calibration,
)


def _case(
    execution_id: str,
    session_id: str,
    role: str,
    standardized_residual: float,
    *,
    fold: str = "outer-fold-1",
) -> ExecutionBlockCalibrationCase:
    mean = np.zeros((4, 2, 3), dtype=float)
    variance = np.full_like(mean, 0.01**2)
    truth = np.full_like(mean, standardized_residual * 0.01)
    return ExecutionBlockCalibrationCase(
        execution_id=execution_id,
        session_id=session_id,
        outer_fold_id=fold,
        split_role=role,
        prediction_case_id="sloth-object",
        action_id="lift",
        contact_region_id="left_forepaw",
        mean_m=mean,
        variance_m2=variance,
        truth_m=truth,
        valid=np.ones(mean.shape[:2], dtype=bool),
        start_frame=1,
    )


def _fit_cases() -> tuple[ExecutionBlockCalibrationCase, ...]:
    return tuple(
        _case(f"fit-{index}", f"fit-session-{index}", "fit", 1.0) for index in range(3)
    )


def _calibration_cases(count: int = 9) -> tuple[ExecutionBlockCalibrationCase, ...]:
    return tuple(
        _case(
            f"cal-{index}",
            f"cal-session-{index}",
            "calibration",
            float(index + 1),
        )
        for index in range(count)
    )


def test_rank_nine_of_nine_uses_maximum_execution_score() -> None:
    calibration = fit_execution_block_conformal_calibration(
        _fit_cases(),
        _calibration_cases(),
        expected_fit_units=3,
    )
    scores = [value.score for value in calibration.calibration_scores]
    assert calibration.order_statistic_rank_one_based == 9
    assert calibration.threshold == max(scores)
    assert calibration.expected_calibration_units == 9
    assert calibration.claim_ready
    assert all(
        not value["finite_nominal_threshold"]
        for value in calibration.fragility_diagnostics[
            "leave_one_calibration_session_out"
        ]
    )


def test_calibration_artifact_is_invariant_to_input_order() -> None:
    fit = _fit_cases()
    heldout = _calibration_cases()
    first = fit_execution_block_conformal_calibration(
        fit,
        heldout,
        expected_fit_units=3,
    )
    second = fit_execution_block_conformal_calibration(
        tuple(reversed(fit)),
        tuple(reversed(heldout)),
        expected_fit_units=3,
    )
    assert first.calibration_id == second.calibration_id
    assert first.as_dict() == second.as_dict()


def test_eight_units_cannot_claim_ninety_percent_finite_coverage() -> None:
    with pytest.raises(ValueError, match="rank 9.*only 8"):
        fit_execution_block_conformal_calibration(
            _fit_cases(),
            _calibration_cases(8),
            expected_fit_units=3,
            expected_calibration_units=8,
        )


def test_session_cannot_cross_fit_and_calibration() -> None:
    fit = list(_fit_cases())
    heldout = list(_calibration_cases())
    heldout[0] = _case(
        "cal-overlap",
        fit[0].session_id,
        "calibration",
        1.0,
    )
    with pytest.raises(ValueError, match="session crosses fit and calibration"):
        fit_execution_block_conformal_calibration(
            fit,
            heldout,
            expected_fit_units=3,
        )


def test_target_evaluation_rejects_source_session_overlap() -> None:
    calibration = fit_execution_block_conformal_calibration(
        _fit_cases(),
        _calibration_cases(),
        expected_fit_units=3,
    )
    target = _case(
        "target-overlap",
        calibration.calibration_scores[0].session_id,
        "target",
        1.0,
    )
    with pytest.raises(ValueError, match="target sessions overlap"):
        evaluate_execution_block_cases((target,), calibration)


def test_target_evaluation_reports_execution_block_coverage() -> None:
    calibration = fit_execution_block_conformal_calibration(
        _fit_cases(),
        _calibration_cases(),
        expected_fit_units=3,
    )
    targets = (
        _case("target-a", "target-session-a", "target", 2.0),
        _case("target-b", "target-session-b", "target", 20.0),
    )
    result = evaluate_execution_block_cases(targets, calibration)
    assert result["target_execution_count"] == 2
    assert result["execution_block_coverage"] == 0.5
    assert result["cases"][0]["execution_block_covered"]
    assert not result["cases"][1]["execution_block_covered"]
    assert not result["pooled_coordinate_conformal_claimed"]
    assert not result["worst_group_coverage_guarantee_claimed"]


def test_execution_block_calibration_artifact_is_checksummed(tmp_path: Path) -> None:
    calibration = fit_execution_block_conformal_calibration(
        _fit_cases(),
        _calibration_cases(),
        expected_fit_units=3,
        metadata={"protocol_id": "locked-protocol"},
    )
    path = tmp_path / "block-calibration.json"
    save_execution_block_conformal_calibration(path, calibration)
    restored = load_execution_block_conformal_calibration(path)
    assert restored.calibration_id == calibration.calibration_id

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["threshold"] *= 2.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold|canonical|checksum"):
        load_execution_block_conformal_calibration(path)

    save_execution_block_conformal_calibration(path, calibration)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["coverage_claim"] = "unsupported stronger guarantee"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        load_execution_block_conformal_calibration(path)
