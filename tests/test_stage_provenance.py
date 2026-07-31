import numpy as np
import pytest

from causal4d.contracts import CounterfactualQuery, build_causal_context
from causal4d.stage_provenance import (
    CounterfactualQueryContext,
    EvaluationTarget,
    FactualEvidenceContext,
    build_counterfactual_query_context,
    build_evaluation_target,
    build_factual_evidence_context,
    validate_stage_contexts,
)


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations = np.arange(8 * 2 * 3, dtype=np.float64).reshape(8, 2, 3)
    actions = np.arange(8 * 1 * 3, dtype=np.float64).reshape(8, 1, 3)
    counterfactual = actions.copy()
    counterfactual[4:, :, 0] *= -1.0
    return observations, actions, counterfactual


def _context(
    observations: np.ndarray,
    actions: np.ndarray,
    counterfactual: np.ndarray,
):
    return build_causal_context(
        protocol_id="stage-provenance-unit",
        case_id="unit-case",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=counterfactual,
        intervention_frame=4,
    )


def _query(context, counterfactual: np.ndarray) -> CounterfactualQuery:
    return CounterfactualQuery(
        context=context,
        controller_points_m=counterfactual[4:],
        horizon_frames=4,
        contact_policy="new_contact",
        source_factual_intervention_id="a" * 64,
        language="move the free endpoint",
        query_node_indices=np.asarray([0, 2]),
    )


def test_held_out_suffix_changes_only_the_evaluation_target_identity() -> None:
    observations, actions, counterfactual = _arrays()
    changed = observations.copy()
    changed[6:] += 1000.0

    context = _context(observations, actions, counterfactual)
    changed_context = _context(changed, actions, counterfactual)

    factual = build_factual_evidence_context(
        context,
        observations,
        actions,
        evidence_frame_stop=6,
    )
    changed_factual = build_factual_evidence_context(
        changed_context,
        changed,
        actions,
        evidence_frame_stop=6,
    )
    query = build_counterfactual_query_context(_query(context, counterfactual))
    changed_query = build_counterfactual_query_context(
        _query(changed_context, counterfactual)
    )
    target = build_evaluation_target(
        context,
        observations,
        target_frame_start=6,
    )
    changed_target = build_evaluation_target(
        changed_context,
        changed,
        target_frame_start=6,
    )

    assert factual.artifact_id == changed_factual.artifact_id
    assert query.artifact_id == changed_query.artifact_id
    assert target.artifact_id != changed_target.artifact_id
    validate_stage_contexts(factual, query, target)
    validate_stage_contexts(changed_factual, changed_query, changed_target)


def test_alternative_query_changes_only_the_query_identity() -> None:
    observations, actions, counterfactual = _arrays()
    alternative = counterfactual.copy()
    alternative[4:, :, 1] += 0.25

    context = _context(observations, actions, counterfactual)
    alternative_context = _context(observations, actions, alternative)

    factual = build_factual_evidence_context(
        context,
        observations,
        actions,
        evidence_frame_stop=6,
    )
    alternative_factual = build_factual_evidence_context(
        alternative_context,
        observations,
        actions,
        evidence_frame_stop=6,
    )
    query = build_counterfactual_query_context(_query(context, counterfactual))
    alternative_query = build_counterfactual_query_context(
        _query(alternative_context, alternative)
    )
    target = build_evaluation_target(
        context,
        observations,
        target_frame_start=6,
    )
    alternative_target = build_evaluation_target(
        alternative_context,
        observations,
        target_frame_start=6,
    )

    assert factual.artifact_id == alternative_factual.artifact_id
    assert query.artifact_id != alternative_query.artifact_id
    assert target.artifact_id == alternative_target.artifact_id


def test_admitted_prefix_changes_only_the_factual_identity() -> None:
    observations, actions, counterfactual = _arrays()
    changed = observations.copy()
    changed[5] += 10.0

    context = _context(observations, actions, counterfactual)
    changed_context = _context(changed, actions, counterfactual)
    factual = build_factual_evidence_context(
        context,
        observations,
        actions,
        evidence_frame_stop=6,
    )
    changed_factual = build_factual_evidence_context(
        changed_context,
        changed,
        actions,
        evidence_frame_stop=6,
    )
    target = build_evaluation_target(
        context,
        observations,
        target_frame_start=6,
    )
    changed_target = build_evaluation_target(
        changed_context,
        changed,
        target_frame_start=6,
    )

    assert factual.artifact_id != changed_factual.artifact_id
    assert target.artifact_id == changed_target.artifact_id


def test_builders_reject_arrays_that_disagree_with_the_v1_context() -> None:
    observations, actions, counterfactual = _arrays()
    context = _context(observations, actions, counterfactual)

    wrong_observations = observations.copy()
    wrong_observations[0] += 1.0
    with pytest.raises(ValueError, match="declared O- digest"):
        build_factual_evidence_context(
            context,
            wrong_observations,
            actions,
            evidence_frame_stop=6,
        )

    wrong_actions = actions.copy()
    wrong_actions[7] += 1.0
    with pytest.raises(ValueError, match="declared u_obs digest"):
        build_factual_evidence_context(
            context,
            observations,
            wrong_actions,
            evidence_frame_stop=6,
        )

    wrong_target = observations.copy()
    wrong_target[7] += 1.0
    with pytest.raises(ValueError, match=r"declared O\+ digest"):
        build_evaluation_target(
            context,
            wrong_target,
            target_frame_start=6,
        )


def test_stage_contexts_round_trip_and_validate_as_one_chain() -> None:
    observations, actions, counterfactual = _arrays()
    context = _context(observations, actions, counterfactual)
    factual = build_factual_evidence_context(
        context,
        observations,
        actions,
        evidence_frame_stop=6,
    )
    query = build_counterfactual_query_context(_query(context, counterfactual))
    target = build_evaluation_target(
        context,
        observations,
        target_frame_start=6,
    )

    restored_factual = FactualEvidenceContext.from_dict(factual.as_dict())
    restored_query = CounterfactualQueryContext.from_dict(query.as_dict())
    restored_target = EvaluationTarget.from_dict(target.as_dict())

    assert restored_factual.artifact_id == factual.artifact_id
    assert restored_query.artifact_id == query.artifact_id
    assert restored_target.artifact_id == target.artifact_id
    validate_stage_contexts(restored_factual, restored_query, restored_target)


def test_stage_chain_rejects_a_target_with_the_wrong_boundary() -> None:
    observations, actions, counterfactual = _arrays()
    context = _context(observations, actions, counterfactual)
    factual = build_factual_evidence_context(
        context,
        observations,
        actions,
        evidence_frame_stop=6,
    )
    query = build_counterfactual_query_context(_query(context, counterfactual))
    target = build_evaluation_target(
        context,
        observations,
        target_frame_start=5,
    )

    with pytest.raises(ValueError, match="factual evidence boundary"):
        validate_stage_contexts(factual, query, target)
