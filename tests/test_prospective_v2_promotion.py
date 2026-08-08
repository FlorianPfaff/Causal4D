from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from causal4d.decision_trace import DECISION_TRACE_ENDPOINTS
from causal4d.prospective_v2_promotion import (
    PROSPECTIVE_V2_CANDIDATE_KINDS,
    ProspectiveV2CandidateV1,
    ProspectiveV2EvaluationUnitV1,
    ProspectiveV2PromotionFreezeV1,
    ProspectiveV2PromotionPolicyV1,
    ProspectiveV2PromotionResultV1,
    ProspectiveV2UnitMetricsV1,
    evaluate_prospective_v2_promotion_v1,
)


def _id(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _candidates() -> tuple[ProspectiveV2CandidateV1, ...]:
    return tuple(
        ProspectiveV2CandidateV1(
            candidate_id=f"candidate-{index}",
            candidate_kind=kind,
            artifact_id=_id(f"artifact:{kind}"),
            source_configuration_id=_id(f"configuration:{kind}"),
            components=(kind,),
        )
        for index, kind in enumerate(PROSPECTIVE_V2_CANDIDATE_KINDS)
    )


def _units() -> tuple[ProspectiveV2EvaluationUnitV1, ...]:
    return tuple(
        ProspectiveV2EvaluationUnitV1(
            unit_id=f"{endpoint}-{index}",
            endpoint=endpoint,
            independent_group_id=f"{endpoint}-group-{index}",
        )
        for endpoint in DECISION_TRACE_ENDPOINTS
        for index in range(2)
    )


def _policy() -> ProspectiveV2PromotionPolicyV1:
    return ProspectiveV2PromotionPolicyV1(
        minimum_units_per_endpoint=2,
        minimum_mean_log_score_gain=0.01,
        maximum_mean_brier_change=0.0,
        maximum_mean_trajectory_regret_m=0.0,
        maximum_mean_coverage_error=0.15,
        maximum_mean_interval_width_ratio=1.25,
        minimum_accepted_update_rate=0.5,
        maximum_harmful_accepted_update_rate=0.25,
        maximum_fallback_rate=0.5,
    )


def _freeze() -> ProspectiveV2PromotionFreezeV1:
    return ProspectiveV2PromotionFreezeV1(
        experiment_id="prospective-v2-evaluation",
        stack_lock_id=_id("stack"),
        candidates=_candidates(),
        evaluation_units=_units(),
        policy=_policy(),
        source_artifact_ids=(_id("source-panel"),),
    )


def _metrics(
    freeze: ProspectiveV2PromotionFreezeV1,
    *,
    failing_candidate_id: str | None = None,
    all_candidates_fail: bool = False,
) -> tuple[ProspectiveV2UnitMetricsV1, ...]:
    opening_id = _id("single-evaluation-opening")
    metrics: list[ProspectiveV2UnitMetricsV1] = []
    baseline = freeze.candidates[0]
    for unit in freeze.evaluation_units:
        metrics.append(
            ProspectiveV2UnitMetricsV1(
                unit_id=unit.unit_id,
                endpoint=unit.endpoint,
                candidate_id=baseline.candidate_id,
                evaluation_opening_id=opening_id,
                log_score=-1.0,
                brier_score=0.30,
                trajectory_error_m=0.10,
                coverage_error=0.10,
                interval_width_m=0.20,
                candidate_accepted=True,
                harmful_update=False,
                fallback_used=False,
            )
        )
        for candidate in freeze.candidates[1:]:
            fails = (
                all_candidates_fail
                or candidate.candidate_id == failing_candidate_id
            )
            metrics.append(
                ProspectiveV2UnitMetricsV1(
                    unit_id=unit.unit_id,
                    endpoint=unit.endpoint,
                    candidate_id=candidate.candidate_id,
                    evaluation_opening_id=opening_id,
                    log_score=-1.10 if fails else -0.90,
                    brier_score=0.35 if fails else 0.25,
                    trajectory_error_m=0.12 if fails else 0.08,
                    coverage_error=0.20 if fails else 0.08,
                    interval_width_m=0.30 if fails else 0.18,
                    candidate_accepted=True,
                    harmful_update=fails,
                    fallback_used=False,
                )
            )
    return tuple(metrics)


def test_highest_passing_candidate_is_selected() -> None:
    freeze = _freeze()
    result = evaluate_prospective_v2_promotion_v1(
        freeze,
        _metrics(freeze),
    )

    expected = freeze.candidates[-1]
    assert result.selected_candidate_id == expected.candidate_id
    assert result.selected_candidate_kind == expected.candidate_kind
    assert result.selected_artifact_id == expected.artifact_id
    assert all(candidate.accepted for candidate in result.candidate_results)
    assert result.one_target_opening_verified
    assert result.exact_artifact_fallback_verified


def test_failed_top_candidate_selects_highest_remaining_pass() -> None:
    freeze = _freeze()
    result = evaluate_prospective_v2_promotion_v1(
        freeze,
        _metrics(freeze, failing_candidate_id=freeze.candidates[-1].candidate_id),
    )

    expected = freeze.candidates[-2]
    assert result.selected_candidate_id == expected.candidate_id
    assert result.selected_artifact_id == expected.artifact_id
    assert not result.candidate_results[-1].accepted
    assert any(
        reason.endswith("harmful_accepted_update_rate_exceeds_limit")
        for reason in result.candidate_results[-1].reasons
    )


def test_all_failed_candidates_preserve_exact_registered_baseline() -> None:
    freeze = _freeze()
    result = evaluate_prospective_v2_promotion_v1(
        freeze,
        _metrics(freeze, all_candidates_fail=True),
    )

    baseline = freeze.candidates[0]
    assert result.selected_candidate_id == baseline.candidate_id
    assert result.selected_candidate_kind == "registered_v1"
    assert result.selected_artifact_id == baseline.artifact_id
    assert not any(candidate.accepted for candidate in result.candidate_results)


def test_incomplete_or_multiple_opening_metrics_fail_closed() -> None:
    freeze = _freeze()
    metrics = _metrics(freeze)
    with pytest.raises(ValueError, match="frozen unit/candidate product"):
        evaluate_prospective_v2_promotion_v1(freeze, metrics[:-1])

    changed = replace(metrics[-1], evaluation_opening_id=_id("second-opening"))
    with pytest.raises(ValueError, match="one target opening"):
        evaluate_prospective_v2_promotion_v1(
            freeze,
            (*metrics[:-1], changed),
        )


def test_evaluation_units_must_be_independent_within_endpoint() -> None:
    candidates = _candidates()
    duplicate_group_units = list(_units())
    duplicate_group_units[1] = replace(
        duplicate_group_units[1],
        independent_group_id=duplicate_group_units[0].independent_group_id,
    )
    with pytest.raises(ValueError, match="unique within each endpoint"):
        ProspectiveV2PromotionFreezeV1(
            experiment_id="prospective-v2-evaluation",
            stack_lock_id=_id("stack"),
            candidates=candidates,
            evaluation_units=tuple(duplicate_group_units),
            policy=_policy(),
            source_artifact_ids=(_id("source-panel"),),
        )


def test_source_freeze_rejects_target_outcome_metadata() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        ProspectiveV2CandidateV1(
            candidate_id="candidate",
            candidate_kind="registered_v1",
            artifact_id=_id("artifact"),
            source_configuration_id=_id("configuration"),
            components=("registered",),
            metadata={"nested": {"target_loss": 1.0}},
        )


def test_unit_metrics_require_target_use_and_consistent_acceptance() -> None:
    values = {
        "unit_id": "unit",
        "endpoint": "factual_continuation",
        "candidate_id": "candidate",
        "evaluation_opening_id": _id("opening"),
        "log_score": -1.0,
        "brier_score": 0.2,
        "trajectory_error_m": 0.1,
        "coverage_error": 0.1,
        "interval_width_m": 0.2,
        "candidate_accepted": True,
        "harmful_update": False,
        "fallback_used": False,
    }
    with pytest.raises(ValueError, match="must be true"):
        ProspectiveV2UnitMetricsV1(**values, target_outcomes_used=False)
    inconsistent = {**values, "candidate_accepted": True, "fallback_used": True}
    with pytest.raises(ValueError, match="complement"):
        ProspectiveV2UnitMetricsV1(**inconsistent)


def test_result_construction_rejects_invented_selection() -> None:
    freeze = _freeze()
    result = evaluate_prospective_v2_promotion_v1(freeze, _metrics(freeze))
    with pytest.raises(ValueError, match="highest accepted"):
        replace(
            result,
            selected_candidate_id=result.baseline_candidate_id,
            selected_candidate_kind="registered_v1",
            selected_artifact_id=result.baseline_artifact_id,
        )
    assert isinstance(result, ProspectiveV2PromotionResultV1)


def test_result_identity_binds_metrics_and_selection() -> None:
    freeze = _freeze()
    first = evaluate_prospective_v2_promotion_v1(freeze, _metrics(freeze))
    changed_metrics = list(_metrics(freeze))
    changed_metrics[-1] = replace(changed_metrics[-1], log_score=-0.89)
    second = evaluate_prospective_v2_promotion_v1(freeze, tuple(changed_metrics))
    assert first.result_id != second.result_id
