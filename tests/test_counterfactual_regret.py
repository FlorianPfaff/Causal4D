from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from causal4d.counterfactual_regret import (
    CounterfactualRegretFeatures,
    CounterfactualRegretPrerequisite,
    CounterfactualRegretSourceCase,
    CounterfactualRegretTarget,
    evaluate_counterfactual_regret,
    fit_counterfactual_regret_certificate,
    load_claim_bearing_counterfactual_regret_certificate,
    load_counterfactual_regret_certificate,
    select_counterfactual_regret_candidate,
    write_counterfactual_regret_certificate,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _features(x: float, y: float | None = None) -> CounterfactualRegretFeatures:
    if y is None:
        y = 0.5 * x
    return CounterfactualRegretFeatures.from_mapping(
        {
            "action_support_distance": x,
            "query_null_response_fraction": y,
        }
    )


def _source(
    case_id: str,
    session_id: str,
    feature: float,
    gain: float,
    *,
    direction: str = "lower",
) -> CounterfactualRegretSourceCase:
    baseline_loss = 1.0
    candidate_loss = 1.0 - gain if direction == "lower" else 1.0 + gain
    return CounterfactualRegretSourceCase(
        case_id=case_id,
        session_id=session_id,
        protocol_id="counterfactual-regret-test-v1",
        endpoint="same_grasp_transfer",
        baseline_role="bayesian_phystwin_nominal_z",
        candidate_role="causal4d",
        metric_id="track_error_m",
        metric_unit="m",
        metric_direction=direction,
        baseline_artifact_id=_digest(f"baseline:{case_id}"),
        candidate_artifact_id=_digest(f"candidate:{case_id}"),
        features=_features(feature),
        baseline_loss=baseline_loss,
        candidate_loss=candidate_loss,
    )


def _sources() -> tuple[CounterfactualRegretSourceCase, ...]:
    return (
        _source("case-a", "session-a", 0.0, 0.20),
        _source("case-b", "session-b", 0.2, 0.15),
        _source("case-c", "session-c", 0.4, 0.10),
        _source("case-d", "session-d", 0.6, 0.12),
    )


def _prerequisites(*, accepted: bool = True):
    return (
        CounterfactualRegretPrerequisite(
            name="action_support",
            decision_id=_digest("action-support"),
            accepted=accepted,
        ),
        CounterfactualRegretPrerequisite(
            name="bayesian_phystwin_acceptance",
            decision_id=_digest("bpt"),
            accepted=True,
        ),
        CounterfactualRegretPrerequisite(
            name="intervention_identifiability",
            decision_id=_digest("identifiability"),
            accepted=True,
        ),
    )


def _certificate(source_cases=None, **kwargs):
    return fit_counterfactual_regret_certificate(
        _sources() if source_cases is None else source_cases,
        required_prerequisite_names=(
            "bayesian_phystwin_acceptance",
            "intervention_identifiability",
            "action_support",
        ),
        **kwargs,
    )


def _target(
    feature: float,
    *,
    prerequisites=None,
    case_id: str = "target",
) -> CounterfactualRegretTarget:
    return CounterfactualRegretTarget(
        case_id=case_id,
        session_id=f"session:{case_id}",
        protocol_id="counterfactual-regret-test-v1",
        endpoint="same_grasp_transfer",
        baseline_role="bayesian_phystwin_nominal_z",
        candidate_role="causal4d",
        metric_id="track_error_m",
        metric_unit="m",
        metric_direction="lower",
        baseline_artifact_id=_digest(f"baseline:{case_id}"),
        candidate_artifact_id=_digest(f"candidate:{case_id}"),
        features=_features(feature),
        prerequisites=(_prerequisites() if prerequisites is None else prerequisites),
    )


def test_feature_mapping_is_order_invariant_and_rejects_booleans() -> None:
    left = CounterfactualRegretFeatures.from_mapping({"b": 2.0, "a": 1.0})
    right = CounterfactualRegretFeatures.from_mapping({"a": 1.0, "b": 2.0})
    assert left.feature_id == right.feature_id
    assert left.names == ("a", "b")
    with pytest.raises(ValueError, match="finite number"):
        CounterfactualRegretFeatures.from_mapping({"a": True})


def test_source_case_identity_binds_future_loss() -> None:
    source = _source("case", "session", 0.0, 0.2)
    changed = _source("case", "session", 0.0, 0.1)
    assert source.source_case_artifact_id != changed.source_case_artifact_id
    assert source.relative_improvement == pytest.approx(0.2)


def test_higher_is_better_metric_direction() -> None:
    source = _source("case", "session", 0.0, 0.2, direction="higher")
    assert source.relative_improvement == pytest.approx(0.2)


def test_certificate_is_input_order_invariant_and_session_aware() -> None:
    sources = _sources()
    forward = _certificate(sources)
    reverse = _certificate(tuple(reversed(sources)))
    assert forward.certificate_id == reverse.certificate_id
    assert forward.candidate_enabled
    assert len({case.session_id for case in forward.source_cases}) == 4


def test_certificate_requires_three_independent_sessions() -> None:
    sources = (
        _source("case-a", "session-a", 0.0, 0.2),
        _source("case-b", "session-a", 0.1, 0.2),
        _source("case-c", "session-b", 0.2, 0.2),
    )
    with pytest.raises(ValueError, match="three independent sessions"):
        _certificate(sources)


def test_duplicate_source_case_id_is_rejected() -> None:
    sources = (
        _source("same", "session-a", 0.0, 0.2),
        _source("same", "session-b", 0.2, 0.2),
        _source("case-c", "session-c", 0.4, 0.2),
    )
    with pytest.raises(ValueError, match="source-case IDs must be unique"):
        _certificate(sources)


def test_in_support_target_is_accepted() -> None:
    certificate = _certificate()
    decision = evaluate_counterfactual_regret(_target(0.25), certificate)
    assert decision.accepted
    assert decision.reasons == ()
    assert decision.target_future_observations_read == 0
    assert len(set(decision.neighbor_session_ids)) == 3


def test_out_of_support_target_preserves_exact_baseline() -> None:
    certificate = _certificate()
    target = _target(100.0)
    baseline = object()
    candidate = object()
    selection = select_counterfactual_regret_candidate(
        certificate,
        target,
        baseline=baseline,
        candidate=candidate,
        baseline_artifact_id=target.baseline_artifact_id,
        candidate_artifact_id=target.candidate_artifact_id,
    )
    assert not selection.decision.accepted
    assert "feature_outside_source_support" in selection.decision.reasons
    assert selection.deployed is baseline


def test_rejected_prerequisite_preserves_exact_baseline() -> None:
    certificate = _certificate()
    target = _target(0.25, prerequisites=_prerequisites(accepted=False))
    baseline = object()
    candidate = object()
    selection = select_counterfactual_regret_candidate(
        certificate,
        target,
        baseline=baseline,
        candidate=candidate,
        baseline_artifact_id=target.baseline_artifact_id,
        candidate_artifact_id=target.candidate_artifact_id,
    )
    assert not selection.decision.accepted
    assert "prerequisite_rejected:action_support" in selection.decision.reasons
    assert selection.deployed is baseline


def test_prerequisite_inventory_is_frozen() -> None:
    certificate = _certificate()
    target = _target(0.25, prerequisites=_prerequisites()[:-1])
    with pytest.raises(ValueError, match="prerequisites differ"):
        evaluate_counterfactual_regret(target, certificate)


def test_globally_harmful_candidate_is_disabled() -> None:
    sources = tuple(
        _source(f"case-{index}", f"session-{index}", index / 10.0, -0.1)
        for index in range(4)
    )
    certificate = _certificate(sources)
    assert not certificate.candidate_enabled
    decision = evaluate_counterfactual_regret(_target(0.15), certificate)
    assert not decision.accepted
    assert "source_global_regret_policy_failed" in decision.reasons


def test_local_harmful_neighborhood_rejects_despite_global_gain() -> None:
    sources = (
        _source("near-a", "session-a", 0.0, -0.10),
        _source("near-b", "session-b", 0.1, -0.08),
        _source("near-c", "session-c", 0.2, 0.05),
        _source("far-d", "session-d", 10.0, 0.50),
        _source("far-e", "session-e", 11.0, 0.50),
        _source("far-f", "session-f", 12.0, 0.50),
    )
    certificate = _certificate(
        sources,
        support_margin=100.0,
        maximum_global_harmful_fraction=1.0 / 3.0,
        maximum_global_worst_relative_regret=0.10,
    )
    assert certificate.candidate_enabled
    decision = evaluate_counterfactual_regret(_target(0.1), certificate)
    assert not decision.accepted
    assert "local_harmful_fraction_exceeded" in decision.reasons
    assert "local_worst_regret_exceeded" in decision.reasons


def test_target_protocol_mismatch_is_rejected() -> None:
    certificate = _certificate()
    target = CounterfactualRegretTarget(
        case_id="target",
        session_id="session:target",
        protocol_id="counterfactual-regret-test-v1",
        endpoint="new_contact_transfer",
        baseline_role="bayesian_phystwin_nominal_z",
        candidate_role="causal4d",
        metric_id="track_error_m",
        metric_unit="m",
        metric_direction="lower",
        baseline_artifact_id=_digest("baseline"),
        candidate_artifact_id=_digest("candidate"),
        features=_features(0.2),
        prerequisites=_prerequisites(),
    )
    with pytest.raises(ValueError, match="does not match"):
        evaluate_counterfactual_regret(target, certificate)


def test_target_case_must_be_disjoint() -> None:
    certificate = _certificate()
    target = _target(0.2, case_id="case-a")
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_counterfactual_regret(target, certificate)


def test_target_session_must_be_disjoint() -> None:
    certificate = _certificate()
    target = CounterfactualRegretTarget(
        case_id="new-case",
        session_id="session-a",
        protocol_id="counterfactual-regret-test-v1",
        endpoint="same_grasp_transfer",
        baseline_role="bayesian_phystwin_nominal_z",
        candidate_role="causal4d",
        metric_id="track_error_m",
        metric_unit="m",
        metric_direction="lower",
        baseline_artifact_id=_digest("baseline:new-case"),
        candidate_artifact_id=_digest("candidate:new-case"),
        features=_features(0.2),
        prerequisites=_prerequisites(),
    )
    with pytest.raises(ValueError, match="target session"):
        evaluate_counterfactual_regret(target, certificate)


def test_object_identity_assertions_are_fail_closed() -> None:
    certificate = _certificate()
    target = _target(0.25)
    with pytest.raises(ValueError, match="baseline object identity"):
        select_counterfactual_regret_candidate(
            certificate,
            target,
            baseline=object(),
            candidate=object(),
            baseline_artifact_id=_digest("wrong"),
            candidate_artifact_id=target.candidate_artifact_id,
        )


def test_certificate_round_trip_and_claim_bearing_identity(tmp_path: Path) -> None:
    certificate = _certificate()
    path = tmp_path / "regret-certificate.json"
    write_counterfactual_regret_certificate(path, certificate)
    restored = load_counterfactual_regret_certificate(path)
    assert restored.certificate_id == certificate.certificate_id
    admitted = load_claim_bearing_counterfactual_regret_certificate(
        path,
        expected_certificate_id=certificate.certificate_id,
    )
    assert admitted.certificate_id == certificate.certificate_id
    with pytest.raises(ValueError, match="frozen identity"):
        load_claim_bearing_counterfactual_regret_certificate(
            path,
            expected_certificate_id=_digest("other"),
        )


def test_readdressed_derived_tampering_is_rejected(tmp_path: Path) -> None:
    certificate = _certificate()
    path = tmp_path / "regret-certificate.json"
    write_counterfactual_regret_certificate(path, certificate)
    payload = json.loads(path.read_text())
    payload["global_mean_relative_improvement"] += 0.1
    payload_without_id = dict(payload)
    payload_without_id.pop("certificate_id")
    payload["certificate_id"] = hashlib.sha256(
        json.dumps(
            payload_without_id,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_counterfactual_regret_certificate(path)


def test_source_case_nested_checksum_tampering_is_rejected(tmp_path: Path) -> None:
    certificate = _certificate()
    path = tmp_path / "regret-certificate.json"
    write_counterfactual_regret_certificate(path, certificate)
    payload = json.loads(path.read_text())
    payload["source_cases"][0]["candidate_loss"] += 0.1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source-case checksum mismatch"):
        load_counterfactual_regret_certificate(path)
