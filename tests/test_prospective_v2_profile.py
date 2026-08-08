from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from causal4d.decision_trace import (
    DecisionTraceArtifact,
    DecisionTraceDecision,
    DecisionTraceStage,
    UnifiedDecisionTrace,
    build_unified_decision_trace,
)
from causal4d.prospective_v2_profile import (
    PROSPECTIVE_V2_PROFILE_ID,
    PROSPECTIVE_V2_REQUIRED_DECISION_NAMES,
    build_prospective_v2_decision_trace_v1,
    require_prospective_v2_decision_trace_v1,
    validate_prospective_v2_decision_trace_v1,
)


def _id(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _artifact(role: str, producer: str) -> DecisionTraceArtifact:
    return DecisionTraceArtifact(
        artifact_id=_id(role),
        artifact_kind=f"test.{role}",
        role=role,
        producer=producer,  # type: ignore[arg-type]
    )


def _decision(
    name: str,
    producer: str,
    *,
    accepted: bool = True,
) -> DecisionTraceDecision:
    reasons = () if accepted else ("source_gate_rejected",)
    return DecisionTraceDecision(
        name=name,
        decision_id=_id(f"decision:{name}:{accepted}"),
        decision_kind=f"test.{name}",
        producer=producer,  # type: ignore[arg-type]
        accepted=accepted,
        reasons=reasons,
    )


def _parts(
    *,
    rejected: str | None = None,
    wrong_stage: str | None = None,
    legacy_inventory: bool = False,
):
    factual = _artifact("factual_evidence_context", "causal4d")
    query = _artifact("counterfactual_query_context", "causal4d")
    observation = _artifact("prob4d_observation", "prob4d")
    belief = _artifact("bayesian_phystwin_belief", "bayesian-phystwin")
    baseline_prediction = _artifact("baseline_prediction", "bayesian-phystwin")
    factual_posterior = _artifact("causal4d_factual_posterior", "causal4d")
    candidate_prediction = _artifact("candidate_prediction", "causal4d")

    names = PROSPECTIVE_V2_REQUIRED_DECISION_NAMES
    if legacy_inventory:
        names = (
            "prob4d_provider_acceptance",
            "bayesian_phystwin_acceptance",
            "intervention_identifiability",
            "action_support",
            "contact_v2_support",
            "query_calibration",
            "counterfactual_regret",
        )

    decision_by_name = {
        name: _decision(
            name,
            (
                "prob4d"
                if name == "prob4d_provider_acceptance"
                else "bayesian-phystwin"
                if name == "bayesian_phystwin_acceptance"
                else "causal4d"
            ),
            accepted=name != rejected,
        )
        for name in names
    }

    prob4d_decisions = [decision_by_name["prob4d_provider_acceptance"]]
    bpt_decisions = [decision_by_name["bayesian_phystwin_acceptance"]]
    abduction_names = [
        "joint_covariance_admission",
        "functional_support",
        "intervention_identifiability",
        "action_support",
        "contact_v2_support",
    ]
    counterfactual_names = [
        "conditional_uncertainty_calibration",
        "query_calibration",
    ]
    deployment_names = ["counterfactual_regret"]

    if wrong_stage is not None:
        for groups in (
            abduction_names,
            counterfactual_names,
            deployment_names,
        ):
            if wrong_stage in groups:
                groups.remove(wrong_stage)
        counterfactual_names.append(wrong_stage)

    stages = (
        DecisionTraceStage(
            stage_name="prob4d observation",
            stage_kind="prob4d_observation",
            producer="prob4d",
            input_artifact_ids=(factual.artifact_id,),
            output_artifacts=(observation,),
            decisions=tuple(prob4d_decisions),
        ),
        DecisionTraceStage(
            stage_name="bayesian physical belief",
            stage_kind="bayesian_phystwin_belief",
            producer="bayesian-phystwin",
            input_artifact_ids=(observation.artifact_id,),
            output_artifacts=(belief, baseline_prediction),
            decisions=tuple(bpt_decisions),
        ),
        DecisionTraceStage(
            stage_name="causal factual abduction",
            stage_kind="causal4d_abduction",
            producer="causal4d",
            input_artifact_ids=(observation.artifact_id, belief.artifact_id),
            output_artifacts=(factual_posterior,),
            decisions=tuple(
                decision_by_name[name]
                for name in abduction_names
                if name in decision_by_name
            ),
        ),
        DecisionTraceStage(
            stage_name="causal counterfactual",
            stage_kind="causal4d_counterfactual",
            producer="causal4d",
            input_artifact_ids=(factual_posterior.artifact_id, query.artifact_id),
            output_artifacts=(candidate_prediction,),
            decisions=tuple(
                decision_by_name[name]
                for name in counterfactual_names
                if name in decision_by_name
            ),
        ),
        DecisionTraceStage(
            stage_name="deployment",
            stage_kind="deployment",
            producer="causal4d",
            input_artifact_ids=(
                baseline_prediction.artifact_id,
                candidate_prediction.artifact_id,
            ),
            decisions=tuple(
                decision_by_name[name]
                for name in deployment_names
                if name in decision_by_name
            ),
        ),
    )
    return (
        (factual, query),
        stages,
        names,
        baseline_prediction.artifact_id,
        candidate_prediction.artifact_id,
    )


def test_complete_profile_selects_exact_candidate_object() -> None:
    roots, stages, _, baseline_id, candidate_id = _parts()
    baseline = object()
    candidate = object()
    result = build_prospective_v2_decision_trace_v1(
        trace_name="case trace",
        protocol_id="protocol-v2",
        case_id="case-001",
        session_id="session-001",
        endpoint="same_grasp_transfer",
        stack_lock_id=_id("stack"),
        root_artifacts=roots,
        stages=stages,
        baseline=baseline,
        candidate=candidate,
        deployed=candidate,
        baseline_artifact_id=baseline_id,
        candidate_artifact_id=candidate_id,
    )

    assert result.deployed is candidate
    assert result.trace.selection.candidate_selected
    assert result.trace.metadata["decision_profile_id"] == PROSPECTIVE_V2_PROFILE_ID
    validation = validate_prospective_v2_decision_trace_v1(result.trace)
    assert validation.accepted
    assert validation.rejected_required_decision_names == ()


def test_rejected_v2_gate_preserves_exact_baseline_object() -> None:
    roots, stages, _, baseline_id, candidate_id = _parts(
        rejected="conditional_uncertainty_calibration"
    )
    baseline = object()
    candidate = object()
    result = build_prospective_v2_decision_trace_v1(
        trace_name="case trace",
        protocol_id="protocol-v2",
        case_id="case-001",
        session_id="session-001",
        endpoint="new_contact_transfer",
        stack_lock_id=_id("stack"),
        root_artifacts=roots,
        stages=stages,
        baseline=baseline,
        candidate=candidate,
        deployed=baseline,
        baseline_artifact_id=baseline_id,
        candidate_artifact_id=candidate_id,
    )

    assert result.deployed is baseline
    assert not result.trace.selection.candidate_selected
    validation = validate_prospective_v2_decision_trace_v1(result.trace)
    assert validation.accepted
    assert validation.rejected_required_decision_names == (
        "conditional_uncertainty_calibration",
    )


def test_legacy_inventory_is_not_a_v2_deployment_profile() -> None:
    roots, stages, names, baseline_id, candidate_id = _parts(legacy_inventory=True)
    baseline = object()
    candidate = object()
    generic = build_unified_decision_trace(
        trace_name="legacy trace",
        protocol_id="protocol-v2",
        case_id="case-001",
        session_id="session-001",
        endpoint="same_grasp_transfer",
        stack_lock_id=_id("stack"),
        root_artifacts=roots,
        stages=stages,
        required_decision_names=names,
        baseline=baseline,
        candidate=candidate,
        deployed=candidate,
        baseline_artifact_id=baseline_id,
        candidate_artifact_id=candidate_id,
        metadata={"decision_profile_id": PROSPECTIVE_V2_PROFILE_ID},
    ).trace

    validation = validate_prospective_v2_decision_trace_v1(generic)
    assert not validation.accepted
    assert "missing_required_decision:joint_covariance_admission" in (
        validation.reasons
    )
    assert "missing_required_decision:functional_support" in validation.reasons
    assert (
        "missing_required_decision:conditional_uncertainty_calibration"
        in validation.reasons
    )
    assert "selection_required_decision_inventory_mismatch" in validation.reasons
    with pytest.raises(ValueError, match="does not satisfy"):
        require_prospective_v2_decision_trace_v1(generic)


def test_required_decision_must_appear_at_the_registered_stage() -> None:
    roots, stages, names, baseline_id, candidate_id = _parts(
        wrong_stage="functional_support"
    )
    baseline = object()
    candidate = object()
    generic = build_unified_decision_trace(
        trace_name="wrong-stage trace",
        protocol_id="protocol-v2",
        case_id="case-001",
        session_id="session-001",
        endpoint="factual_continuation",
        stack_lock_id=_id("stack"),
        root_artifacts=roots,
        stages=stages,
        required_decision_names=names,
        baseline=baseline,
        candidate=candidate,
        deployed=candidate,
        baseline_artifact_id=baseline_id,
        candidate_artifact_id=candidate_id,
        metadata={"decision_profile_id": PROSPECTIVE_V2_PROFILE_ID},
    ).trace

    validation = validate_prospective_v2_decision_trace_v1(generic)
    assert not validation.accepted
    assert (
        "required_decision_wrong_stage:functional_support:causal4d_counterfactual"
    ) in validation.reasons


def test_profile_metadata_is_content_bound_and_cannot_be_redeclared() -> None:
    roots, stages, _, baseline_id, candidate_id = _parts()
    baseline = object()
    candidate = object()
    with pytest.raises(ValueError, match="different decision profile"):
        build_prospective_v2_decision_trace_v1(
            trace_name="case trace",
            protocol_id="protocol-v2",
            case_id="case-001",
            session_id="session-001",
            endpoint="same_grasp_transfer",
            stack_lock_id=_id("stack"),
            root_artifacts=roots,
            stages=stages,
            baseline=baseline,
            candidate=candidate,
            deployed=candidate,
            baseline_artifact_id=baseline_id,
            candidate_artifact_id=candidate_id,
            metadata={"decision_profile_id": "other-profile"},
        )

    valid = build_prospective_v2_decision_trace_v1(
        trace_name="case trace",
        protocol_id="protocol-v2",
        case_id="case-001",
        session_id="session-001",
        endpoint="same_grasp_transfer",
        stack_lock_id=_id("stack"),
        root_artifacts=roots,
        stages=stages,
        baseline=baseline,
        candidate=candidate,
        deployed=candidate,
        baseline_artifact_id=baseline_id,
        candidate_artifact_id=candidate_id,
    ).trace
    tampered = replace(valid, metadata={"decision_profile_id": "other-profile"})
    validation = validate_prospective_v2_decision_trace_v1(tampered)
    assert not validation.accepted
    assert "decision_profile_id_mismatch" in validation.reasons
    assert isinstance(tampered, UnifiedDecisionTrace)
