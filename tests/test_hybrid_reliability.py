import json

import numpy as np
import pytest

from causal4d.baselines import PredictiveDistribution, RidgeTrajectoryModel
from causal4d.hybrid_reliability import (
    HybridReliabilityCalibration,
    HybridReliabilityCase,
    apply_hybrid_reliability,
    fit_hybrid_reliability_calibration,
    load_hybrid_reliability_calibration,
    ridge_descriptor_leverage,
    save_hybrid_reliability_calibration,
)


def _prediction(
    method: str,
    values: float | np.ndarray,
    *,
    variance: float = 1.0,
) -> PredictiveDistribution:
    if np.isscalar(values):
        mean = np.full((6, 1, 2), float(values), dtype=float)
    else:
        mean = np.asarray(values, dtype=float)
    return PredictiveDistribution(
        method=method,
        mean=mean,
        variance=np.full_like(mean, variance, dtype=float),
    )


def _case(
    case_id: str,
    *,
    physics: float | np.ndarray = 0.20,
    hybrid: float | np.ndarray = 0.10,
    observations: np.ndarray | None = None,
    leverage: float = 0.20,
) -> HybridReliabilityCase:
    values = (
        np.zeros((6, 1, 2), dtype=float)
        if observations is None
        else np.asarray(observations, dtype=float)
    )
    return HybridReliabilityCase(
        case_id=case_id,
        physics=_prediction("physics_only", physics),
        hybrid=_prediction("hybrid", hybrid),
        observations=values,
        descriptor_leverage=leverage,
        prefix_frame_count=3,
    )


def _calibration() -> HybridReliabilityCalibration:
    return fit_hybrid_reliability_calibration(
        (
            _case("source-a", physics=0.20, hybrid=0.10, leverage=0.20),
            _case("source-b", physics=0.24, hybrid=0.12, leverage=0.40),
        )
    )


def test_ridge_descriptor_leverage_uses_the_stored_gram_representation() -> None:
    model = RidgeTrajectoryModel(
        feature_mean=np.zeros(2),
        feature_scale=np.ones(2),
        coefficients=np.zeros((3, 4)),
        gram_inverse=None,
        residual_variance=np.ones(4),
        output_shape=(2, 1, 2),
        gram_matrix=np.eye(3),
    )

    assert ridge_descriptor_leverage(model, np.asarray([1.0, 2.0])) == 6.0
    with pytest.raises(ValueError, match="descriptor shape"):
        ridge_descriptor_leverage(model, np.asarray([1.0]))


def test_source_calibration_is_content_addressed_and_round_trips(tmp_path) -> None:
    calibration = _calibration()
    output = tmp_path / "hybrid-reliability.json"

    save_hybrid_reliability_calibration(output, calibration)
    loaded = load_hybrid_reliability_calibration(output)

    assert loaded == calibration
    assert loaded.calibration_id == calibration.calibration_id
    assert loaded.hybrid_enabled
    assert loaded.source_futures_used
    assert not loaded.target_futures_used
    assert len(set(loaded.source_case_artifact_ids)) == 2
    assert len(set(loaded.source_prefix_input_ids)) == 2

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["source_case_ids"][0] = "relabeled-source"
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_hybrid_reliability_calibration(output)


def test_target_decision_accepts_supported_hybrid_without_reading_future() -> None:
    calibration = _calibration()
    future_a = np.zeros((6, 1, 2), dtype=float)
    future_b = future_a.copy()
    future_a[3:] = 1000.0
    future_b[3:] = -1000.0
    first = _case("target", observations=future_a, leverage=0.30)
    second = _case("target", observations=future_b, leverage=0.30)
    future_missing = np.zeros((6, 1, 2), dtype=float)
    future_missing[3:] = np.nan
    third = _case("target", observations=future_missing, leverage=0.30)

    first_prediction, first_decision = apply_hybrid_reliability(first, calibration)
    second_prediction, second_decision = apply_hybrid_reliability(second, calibration)
    third_prediction, third_decision = apply_hybrid_reliability(third, calibration)

    assert first_prediction is first.hybrid
    assert second_prediction is second.hybrid
    assert third_prediction is third.hybrid
    assert first_decision.accepted and second_decision.accepted
    assert third_decision.accepted
    assert first_decision.target_future_observations_read == 0
    assert not first_decision.target_future_outcomes_used
    assert first.target_prefix_input_id == second.target_prefix_input_id
    assert first.target_prefix_input_id == third.target_prefix_input_id
    assert first.source_case_artifact_id != second.source_case_artifact_id
    assert first_decision.diagnostics == second_decision.diagnostics
    assert first_decision.diagnostics == third_decision.diagnostics
    assert first_decision.decision_id == second_decision.decision_id
    assert first_decision.decision_id == third_decision.decision_id


def test_prefix_failure_returns_the_exact_physics_only_object() -> None:
    calibration = _calibration()
    target = _case("target", physics=0.20, hybrid=0.30, leverage=0.30)

    selected, decision = apply_hybrid_reliability(target, calibration)

    assert selected is target.physics
    assert not decision.accepted
    assert decision.selected_method == "physics_only"
    assert "prefix_point_score_not_supported" in decision.reasons
    assert "prefix_probabilistic_score_not_supported" in decision.reasons


def test_source_failure_disables_hybrid_and_preserves_exact_fallback() -> None:
    hybrid = np.full((6, 1, 2), 0.10, dtype=float)
    hybrid[3:] = 0.30
    calibration = fit_hybrid_reliability_calibration(
        (
            _case("source-a", hybrid=hybrid, leverage=0.20),
            _case("source-b", physics=0.24, hybrid=hybrid, leverage=0.40),
        )
    )
    target = _case("target", leverage=0.30)

    selected, decision = apply_hybrid_reliability(target, calibration)

    assert not calibration.hybrid_enabled
    assert selected is target.physics
    assert decision.reasons[0] == "no_source_future_gain"


def test_ood_correction_and_descriptor_support_fail_closed() -> None:
    calibration = _calibration()
    target = _case("target", physics=0.20, hybrid=0.01, leverage=1.0)

    selected, decision = apply_hybrid_reliability(target, calibration)

    assert selected is target.physics
    assert "correction_outside_source_scale" in decision.reasons
    assert "descriptor_outside_source_support" in decision.reasons


def test_target_prefix_cannot_reuse_source_measurements_under_a_new_label() -> None:
    sources = (
        _case("source-a", physics=0.20, hybrid=0.10, leverage=0.20),
        _case("source-b", physics=0.24, hybrid=0.12, leverage=0.40),
    )
    calibration = fit_hybrid_reliability_calibration(sources)
    relabeled = _case(
        "target-relabeled",
        physics=0.20,
        hybrid=0.10,
        leverage=0.20,
    )

    assert relabeled.target_prefix_input_id == sources[0].target_prefix_input_id
    with pytest.raises(ValueError, match="prefix reuses"):
        apply_hybrid_reliability(relabeled, calibration)


def test_case_owns_observations_and_rejects_nonfinite_prefix() -> None:
    observations = np.zeros((6, 1, 2), dtype=float)
    case = _case("owned", observations=observations)
    observations[...] = 9.0

    assert np.all(case.observations == 0.0)
    assert not case.observations.flags.writeable

    invalid = np.zeros((6, 1, 2), dtype=float)
    invalid[1] = np.nan
    with pytest.raises(ValueError, match="response prefix"):
        _case("invalid", observations=invalid)


def test_calibration_rejects_boolean_numeric_diagnostics() -> None:
    calibration = _calibration()
    payload = calibration.as_dict()
    payload.pop("calibration_id")
    payload.pop("schema_version")
    payload.pop("artifact_kind")
    payload["source_prefix_log_score_gains"] = (True, 0.1)

    with pytest.raises(ValueError, match="finite number"):
        HybridReliabilityCalibration(**payload)


def test_gate_remains_separate_from_the_frozen_counterfactual_runner() -> None:
    repository = __import__("pathlib").Path(__file__).resolve().parents[1]
    source = (repository / "src/causal4d/evaluation.py").read_text(encoding="utf-8")

    assert "hybrid_reliability" not in source
