from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from causal4d.action_conditioned_discrepancy import (
    ActionConditionedDiscrepancyFeatures,
)
from causal4d.action_support import (
    ActionSupportCalibration,
    ActionSupportSourceCase,
    evaluate_action_support,
    fit_action_support_calibration,
    load_action_support_calibration,
    load_claim_bearing_action_support_calibration,
    select_action_supported_candidate,
    write_action_support_calibration,
)


def features(values, *, ids=None, dt=0.1):
    return ActionConditionedDiscrepancyFeatures(
        names=("speed", "direction"),
        values=np.asarray(values, dtype=float),
        component_ids=ids,
        step_duration_s=dt,
        schema_id="signed_v2",
    )


def calibration(minimum_mass=0.9):
    first = ActionSupportSourceCase(
        "source-a",
        features([[1.0, -0.5], [1.2, -0.4], [1.1, -0.6]]),
    )
    second = ActionSupportSourceCase(
        "source-b",
        features([[2.0, 0.4], [2.2, 0.5], [2.1, 0.3]]),
    )
    third = ActionSupportSourceCase(
        "source-c",
        features([[1.5, 0.0], [1.6, 0.1], [1.4, -0.1]]),
    )
    return fit_action_support_calibration(
        (third, first, second),
        candidate_model_id="discrepancy-model-v1",
        support_margin=1.2,
        minimum_supported_component_mass=minimum_mass,
    )


def test_fit_is_source_order_invariant():
    cases = (
        ActionSupportSourceCase("b", features([[2.0, 0.2], [2.1, 0.3]])),
        ActionSupportSourceCase("a", features([[1.0, -0.2], [1.1, -0.1]])),
        ActionSupportSourceCase("c", features([[1.5, 0.0], [1.6, 0.1]])),
    )
    forward = fit_action_support_calibration(
        cases, candidate_model_id="discrepancy-model-v1"
    )
    reverse = fit_action_support_calibration(
        tuple(reversed(cases)),
        candidate_model_id="discrepancy-model-v1",
    )
    assert forward.calibration_id == reverse.calibration_id
    assert forward.source_case_ids == ("a", "b", "c")


def test_supported_query_is_accepted():
    fitted = calibration()
    target = features([[1.55, 0.02], [1.60, 0.05], [1.50, -0.02]])
    decision = evaluate_action_support(
        fitted, target, candidate_model_id="discrepancy-model-v1"
    )
    assert decision.accepted
    assert decision.supported_component_mass == pytest.approx(1.0)
    assert decision.future_observation_frames_read == 0


def test_out_of_support_query_returns_exact_baseline():
    fitted = calibration()
    target = features([[20.0, 3.0], [21.0, 3.5], [19.0, 2.5]])
    baseline = object()
    candidate = object()
    selection = select_action_supported_candidate(
        fitted,
        target,
        baseline=baseline,
        candidate=candidate,
        candidate_model_id="discrepancy-model-v1",
    )
    assert not selection.decision.accepted
    assert selection.deployed is baseline
    assert "summary_distance_out_of_support" in (selection.decision.rejection_reasons)
    assert "feature_value_out_of_support" in (selection.decision.rejection_reasons)


def test_supported_query_selects_candidate_by_identity():
    fitted = calibration()
    target = features([[1.5, 0.0], [1.6, 0.1], [1.4, -0.1]])
    baseline = object()
    candidate = object()
    selection = select_action_supported_candidate(
        fitted,
        target,
        baseline=baseline,
        candidate=candidate,
        candidate_model_id="discrepancy-model-v1",
    )
    assert selection.decision.accepted
    assert selection.deployed is candidate


def test_supported_component_mass_controls_acceptance():
    fitted = calibration(minimum_mass=0.8)
    target = features(
        [
            [[1.5, 0.0], [1.6, 0.1], [1.4, -0.1]],
            [[20.0, 3.0], [21.0, 3.5], [19.0, 2.5]],
        ],
        ids=("supported", "unsupported"),
    )
    accepted = evaluate_action_support(
        fitted,
        target,
        candidate_model_id="discrepancy-model-v1",
        component_weights=np.asarray([0.85, 0.15]),
        component_ids=("supported", "unsupported"),
    )
    rejected = evaluate_action_support(
        fitted,
        target,
        candidate_model_id="discrepancy-model-v1",
        component_weights=np.asarray([0.75, 0.25]),
        component_ids=("supported", "unsupported"),
    )
    assert accepted.accepted
    assert accepted.supported_component_mass == pytest.approx(0.85)
    assert not rejected.accepted
    assert rejected.supported_component_mass == pytest.approx(0.75)


def test_calibration_arrays_are_irreversibly_read_only():
    fitted = calibration()
    with pytest.raises(ValueError):
        fitted.summary_center[0] = 0.0
    with pytest.raises(ValueError):
        fitted.summary_center.setflags(write=True)


def test_round_trip_and_derived_tamper_rejection(tmp_path: Path):
    fitted = calibration()
    path = tmp_path / "support.json"
    write_action_support_calibration(path, fitted)
    loaded = load_action_support_calibration(path)
    assert loaded.calibration_id == fitted.calibration_id

    payload = json.loads(path.read_text())
    payload["maximum_supported_distance"] *= 2.0
    payload_without_id = dict(payload)
    payload_without_id.pop("calibration_id")
    # Even a caller that readdresses the outer artifact cannot alter derivations.
    encoded = json.dumps(
        payload_without_id,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["calibration_id"] = hashlib.sha256(encoded).hexdigest()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="maximum_supported_distance"):
        load_action_support_calibration(path)


def test_schema_mismatch_fails_closed():
    fitted = calibration()
    target = ActionConditionedDiscrepancyFeatures(
        names=("speed", "direction"),
        values=np.asarray([[1.5, 0.0], [1.6, 0.1]]),
        schema_id="magnitude_v1",
    )
    with pytest.raises(ValueError, match="schema"):
        evaluate_action_support(
            fitted, target, candidate_model_id="discrepancy-model-v1"
        )


def test_cadence_and_elapsed_horizon_are_part_of_support():
    fitted = calibration()
    target = features(
        [[1.5, 0.0], [1.6, 0.1], [1.4, -0.1]],
        dt=0.2,
    )
    decision = evaluate_action_support(
        fitted,
        target,
        candidate_model_id="discrepancy-model-v1",
    )
    assert not decision.accepted
    assert "feature_value_out_of_support" in decision.rejection_reasons


def test_candidate_model_identity_is_bound():
    fitted = calibration()
    target = features([[1.5, 0.0], [1.6, 0.1]])
    with pytest.raises(ValueError, match="candidate model"):
        evaluate_action_support(
            fitted,
            target,
            candidate_model_id="different-model",
        )


def test_component_specific_source_requires_weights():
    source = features(
        [
            [[1.0, 0.0], [1.1, 0.1]],
            [[2.0, 0.0], [2.1, 0.1]],
        ],
        ids=("a", "b"),
    )
    with pytest.raises(ValueError, match="required"):
        ActionSupportSourceCase("source", source)


def test_string_sequences_are_rejected():
    fitted = calibration()
    payload = fitted.as_dict()
    payload.pop("calibration_id")
    payload["feature_names"] = "speed"
    with pytest.raises(ValueError, match="sequence"):
        ActionSupportCalibration(
            **{
                key: value
                for key, value in payload.items()
                if key not in {"schema_version", "artifact_kind"}
            }
        )


def test_guarded_counterfactual_wrapper_preserves_exact_fallback(monkeypatch):
    import causal4d.action_support_counterfactual as guarded

    fitted = calibration()
    target = features([[20.0, 3.0], [21.0, 3.5], [19.0, 2.5]])
    baseline = object()

    class Candidate:
        physical = baseline
        weights = np.ones(1)
        component_ids = ("component-0",)
        discrepancy_model_id = "discrepancy-model-v1"

    candidate = Candidate()
    monkeypatch.setattr(
        guarded,
        "apply_action_conditioned_counterfactual_operator",
        lambda *args, **kwargs: candidate,
    )
    monkeypatch.setattr(
        guarded,
        "_component_features",
        lambda *args, **kwargs: target,
    )
    selection = guarded.apply_guarded_action_conditioned_counterfactual_operator(
        None,
        {},
        None,
        None,
        None,
        None,
        None,
        np.eye(1),
        np.zeros((1, 3)),
        fitted,
        frame_dt_s=0.1,
    )
    assert not selection.decision.accepted
    assert selection.deployed is baseline


def test_claim_bearing_loader_requires_independent_identity(tmp_path: Path):
    fitted = calibration()
    path = tmp_path / "support.json"
    write_action_support_calibration(path, fitted)
    loaded = load_claim_bearing_action_support_calibration(
        path,
        expected_calibration_id=fitted.calibration_id,
    )
    assert loaded.calibration_id == fitted.calibration_id
    with pytest.raises(ValueError, match="frozen expected identity"):
        load_claim_bearing_action_support_calibration(
            path,
            expected_calibration_id="0" * 64,
        )


def test_boolean_numeric_payloads_fail_closed(tmp_path: Path):
    fitted = calibration()
    path = tmp_path / "support.json"
    write_action_support_calibration(path, fitted)
    payload = json.loads(path.read_text())
    payload["source_case_summaries"][0][0] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Booleans"):
        load_action_support_calibration(path)
