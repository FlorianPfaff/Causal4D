from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from causal4d.decision_trace import (
    DECISION_TRACE_STAGE_KINDS,
    DecisionTraceArtifact,
    DecisionTraceDecision,
    DecisionTraceSelection,
    DecisionTraceStage,
    UnifiedDecisionTrace,
    build_unified_decision_trace,
    load_claim_bearing_decision_trace,
    load_decision_trace,
    require_decision_trace_stack_lock,
    write_decision_trace,
)


def _id(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _artifact(
    role: str,
    producer: str,
    *,
    label: str | None = None,
    metadata: dict[str, object] | None = None,
) -> DecisionTraceArtifact:
    return DecisionTraceArtifact(
        artifact_id=_id(role if label is None else label),
        artifact_kind=f"test.{role}",
        role=role,
        producer=producer,  # type: ignore[arg-type]
        metadata={} if metadata is None else metadata,
    )


def _decision(
    name: str,
    producer: str,
    *,
    accepted: bool = True,
    reasons: tuple[str, ...] = (),
) -> DecisionTraceDecision:
    return DecisionTraceDecision(
        name=name,
        decision_id=_id(f"decision:{name}:{accepted}:{reasons}"),
        decision_kind=f"test.{name}",
        producer=producer,  # type: ignore[arg-type]
        accepted=accepted,
        reasons=reasons,
    )


def _trace_parts(
    *,
    rejected_decision: str | None = None,
) -> tuple[
    tuple[DecisionTraceArtifact, ...],
    tuple[DecisionTraceStage, ...],
    tuple[str, ...],
    object,
    object,
]:
    factual = _artifact("factual_evidence_context", "causal4d")
    query = _artifact("counterfactual_query_context", "causal4d")
    observation = _artifact("prob4d_observation", "prob4d")
    belief = _artifact("bayesian_phystwin_belief", "bayesian-phystwin")
    baseline_prediction = _artifact(
        "baseline_prediction",
        "bayesian-phystwin",
    )
    factual_posterior = _artifact("causal4d_factual_posterior", "causal4d")
    candidate_prediction = _artifact("candidate_prediction", "causal4d")

    names = (
        "prob4d_provider_acceptance",
        "bayesian_phystwin_acceptance",
        "intervention_identifiability",
        "action_support",
        "contact_v2_support",
        "query_calibration",
        "counterfactual_regret",
    )

    def decision(name: str, producer: str) -> DecisionTraceDecision:
        rejected = name == rejected_decision
        return _decision(
            name,
            producer,
            accepted=not rejected,
            reasons=("source_regret_exceeds_limit",) if rejected else (),
        )

    stages = (
        DecisionTraceStage(
            stage_name="prob4d observation",
            stage_kind="prob4d_observation",
            producer="prob4d",
            input_artifact_ids=(factual.artifact_id,),
            output_artifacts=(observation,),
            decisions=(decision(names[0], "prob4d"),),
        ),
        DecisionTraceStage(
            stage_name="bayesian physical belief",
            stage_kind="bayesian_phystwin_belief",
            producer="bayesian-phystwin",
            input_artifact_ids=(observation.artifact_id,),
            output_artifacts=(belief, baseline_prediction),
            decisions=(decision(names[1], "bayesian-phystwin"),),
        ),
        DecisionTraceStage(
            stage_name="causal factual abduction",
            stage_kind="causal4d_abduction",
            producer="causal4d",
            input_artifact_ids=(observation.artifact_id, belief.artifact_id),
            output_artifacts=(factual_posterior,),
            decisions=(
                decision(names[2], "causal4d"),
                decision(names[3], "causal4d"),
                decision(names[4], "causal4d"),
            ),
        ),
        DecisionTraceStage(
            stage_name="causal counterfactual",
            stage_kind="causal4d_counterfactual",
            producer="causal4d",
            input_artifact_ids=(factual_posterior.artifact_id, query.artifact_id),
            output_artifacts=(candidate_prediction,),
            decisions=(decision(names[5], "causal4d"),),
        ),
        DecisionTraceStage(
            stage_name="deployment",
            stage_kind="deployment",
            producer="causal4d",
            input_artifact_ids=(
                baseline_prediction.artifact_id,
                candidate_prediction.artifact_id,
            ),
            decisions=(decision(names[6], "causal4d"),),
        ),
    )
    return (factual, query), stages, names, object(), object()


def _build_trace(
    *,
    rejected_decision: str | None = None,
) -> tuple[UnifiedDecisionTrace, object, object]:
    roots, stages, names, baseline, candidate = _trace_parts(
        rejected_decision=rejected_decision
    )
    deployed = baseline if rejected_decision is not None else candidate
    result = build_unified_decision_trace(
        trace_name="case trace",
        protocol_id="protocol-v2",
        case_id="case-001",
        session_id="session-001",
        endpoint="same_grasp_transfer",
        stack_lock_id=_id("stack lock"),
        root_artifacts=roots,
        stages=stages,
        required_decision_names=names,
        baseline=baseline,
        candidate=candidate,
        deployed=deployed,
        baseline_artifact_id=stages[1].output_artifacts[1].artifact_id,
        candidate_artifact_id=stages[3].output_artifacts[0].artifact_id,
        metadata={"operator": {"version": 1}},
    )
    assert result.deployed is deployed
    return result.trace, baseline, candidate


def test_candidate_trace_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    trace, _, _ = _build_trace()
    assert trace.selection.candidate_selected
    assert trace.selection.deployed_artifact_id == (
        trace.selection.candidate_artifact_id
    )
    assert trace.target_future_observations_read == 0
    assert not trace.target_future_outcomes_used
    assert tuple(stage.stage_kind for stage in trace.stages) == (
        DECISION_TRACE_STAGE_KINDS
    )
    assert trace.trace_id == UnifiedDecisionTrace.from_dict(trace.as_dict()).trace_id

    path = tmp_path / "trace.json"
    write_decision_trace(path, trace)
    loaded = load_decision_trace(path)
    assert loaded.trace_id == trace.trace_id
    assert loaded.as_dict() == trace.as_dict()


def test_rejected_gate_selects_exact_baseline_object() -> None:
    trace, baseline, candidate = _build_trace(rejected_decision="counterfactual_regret")
    assert not trace.selection.candidate_selected
    assert trace.selection.deployed_artifact_id == trace.selection.baseline_artifact_id
    roots, stages, names, _, _ = _trace_parts(rejected_decision="counterfactual_regret")
    result = build_unified_decision_trace(
        trace_name="case trace",
        protocol_id="protocol-v2",
        case_id="case-001",
        session_id="session-001",
        endpoint="same_grasp_transfer",
        stack_lock_id=_id("stack lock"),
        root_artifacts=roots,
        stages=stages,
        required_decision_names=names,
        baseline=baseline,
        candidate=candidate,
        deployed=baseline,
        baseline_artifact_id=stages[1].output_artifacts[1].artifact_id,
        candidate_artifact_id=stages[3].output_artifacts[0].artifact_id,
    )
    assert result.deployed is baseline


def test_builder_rejects_wrong_runtime_object() -> None:
    roots, stages, names, baseline, candidate = _trace_parts(
        rejected_decision="action_support"
    )
    with pytest.raises(ValueError, match="deployed object"):
        build_unified_decision_trace(
            trace_name="case trace",
            protocol_id="protocol-v2",
            case_id="case-001",
            session_id="session-001",
            endpoint="same_grasp_transfer",
            stack_lock_id=_id("stack lock"),
            root_artifacts=roots,
            stages=stages,
            required_decision_names=names,
            baseline=baseline,
            candidate=candidate,
            deployed=candidate,
            baseline_artifact_id=stages[1].output_artifacts[1].artifact_id,
            candidate_artifact_id=stages[3].output_artifacts[0].artifact_id,
        )


def test_builder_rejects_same_baseline_and_candidate_object() -> None:
    roots, stages, names, _, _ = _trace_parts()
    shared = object()
    with pytest.raises(ValueError, match="must differ"):
        build_unified_decision_trace(
            trace_name="case trace",
            protocol_id="protocol-v2",
            case_id="case-001",
            session_id="session-001",
            endpoint="same_grasp_transfer",
            stack_lock_id=_id("stack lock"),
            root_artifacts=roots,
            stages=stages,
            required_decision_names=names,
            baseline=shared,
            candidate=shared,
            deployed=shared,
            baseline_artifact_id=stages[1].output_artifacts[1].artifact_id,
            candidate_artifact_id=stages[3].output_artifacts[0].artifact_id,
        )


def test_missing_required_decision_is_rejected() -> None:
    roots, stages, names, baseline, candidate = _trace_parts()
    with pytest.raises(ValueError, match="absent from stages"):
        build_unified_decision_trace(
            trace_name="case trace",
            protocol_id="protocol-v2",
            case_id="case-001",
            session_id="session-001",
            endpoint="same_grasp_transfer",
            stack_lock_id=_id("stack lock"),
            root_artifacts=roots,
            stages=stages,
            required_decision_names=(*names, "missing_gate"),
            baseline=baseline,
            candidate=candidate,
            deployed=candidate,
            baseline_artifact_id=stages[1].output_artifacts[1].artifact_id,
            candidate_artifact_id=stages[3].output_artifacts[0].artifact_id,
        )


def test_direct_trace_cannot_disagree_with_gate_conjunction() -> None:
    trace, _, _ = _build_trace(rejected_decision="action_support")
    invalid_selection = replace(
        trace.selection,
        deployed_artifact_id=trace.selection.candidate_artifact_id,
        candidate_selected=True,
    )
    with pytest.raises(ValueError, match="conjunction"):
        replace(trace, selection=invalid_selection)


def test_forward_artifact_reference_is_rejected() -> None:
    trace, _, _ = _build_trace()
    first = trace.stages[0]
    invalid_first = replace(
        first,
        input_artifact_ids=(trace.selection.candidate_artifact_id,),
    )
    with pytest.raises(ValueError, match="unavailable or forward"):
        replace(trace, stages=(invalid_first, *trace.stages[1:]))


def test_required_input_role_is_enforced() -> None:
    trace, _, _ = _build_trace()
    bpt = trace.stages[1]
    factual_id = trace.root_artifacts[0].artifact_id
    invalid_bpt = replace(bpt, input_artifact_ids=(factual_id,))
    with pytest.raises(ValueError, match="missing input roles"):
        replace(trace, stages=(trace.stages[0], invalid_bpt, *trace.stages[2:]))


def test_required_output_role_is_enforced() -> None:
    trace, _, _ = _build_trace()
    observation_stage = trace.stages[0]
    wrong = replace(
        observation_stage.output_artifacts[0],
        role="unrelated_observation",
    )
    invalid_stage = replace(observation_stage, output_artifacts=(wrong,))
    with pytest.raises(ValueError, match="missing output roles"):
        replace(trace, stages=(invalid_stage, *trace.stages[1:]))


def test_duplicate_output_artifact_id_is_rejected() -> None:
    trace, _, _ = _build_trace()
    bpt = trace.stages[1]
    duplicate = replace(
        bpt.output_artifacts[1],
        artifact_id=bpt.output_artifacts[0].artifact_id,
    )
    with pytest.raises(ValueError, match="artifact IDs"):
        replace(bpt, output_artifacts=(bpt.output_artifacts[0], duplicate))


def test_required_roles_must_be_unique_globally() -> None:
    trace, _, _ = _build_trace()
    duplicate_root = _artifact(
        "factual_evidence_context",
        "causal4d",
        label="duplicate factual",
    )
    with pytest.raises(ValueError, match="exactly one artifact"):
        replace(trace, root_artifacts=(*trace.root_artifacts, duplicate_root))


def test_stage_order_and_producer_are_fixed() -> None:
    trace, _, _ = _build_trace()
    with pytest.raises(ValueError, match="fixed Prob4D"):
        replace(trace, stages=(trace.stages[1], trace.stages[0], *trace.stages[2:]))
    with pytest.raises(ValueError, match="must be produced"):
        replace(trace.stages[0], producer="causal4d")


def test_deployment_has_no_new_output_artifact() -> None:
    trace, _, _ = _build_trace()
    with pytest.raises(ValueError, match="has no outputs"):
        replace(
            trace.stages[-1],
            output_artifacts=(_artifact("deployment_receipt", "causal4d"),),
        )


def test_held_out_target_artifacts_and_metadata_are_forbidden() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        _artifact("evaluation_target", "causal4d")
    with pytest.raises(ValueError, match="target-safe"):
        _artifact(
            "factual_evidence_context",
            "causal4d",
            metadata={"target_loss": 1.0},
        )
    trace, _, _ = _build_trace()
    with pytest.raises(ValueError, match="target-safe"):
        replace(trace, metadata={"nested": {"target_future": [1, 2]}})


def test_target_future_flags_fail_closed() -> None:
    trace, _, _ = _build_trace()
    with pytest.raises(ValueError, match="must not read"):
        replace(trace, target_future_observations_read=1)
    with pytest.raises(ValueError, match="must not use"):
        replace(trace, target_future_outcomes_used=True)


def test_decision_reason_contract_is_strict() -> None:
    with pytest.raises(ValueError, match="accepted decisions"):
        _decision("gate", "causal4d", accepted=True, reasons=("not_allowed",))
    with pytest.raises(ValueError, match="require at least one"):
        _decision("gate", "causal4d", accepted=False)


def test_selection_identity_contract_is_strict() -> None:
    trace, _, _ = _build_trace()
    with pytest.raises(ValueError, match="must be verified"):
        replace(trace.selection, exact_object_identity_verified=False)
    with pytest.raises(ValueError, match="disagrees"):
        DecisionTraceSelection(
            baseline_artifact_id=trace.selection.baseline_artifact_id,
            candidate_artifact_id=trace.selection.candidate_artifact_id,
            deployed_artifact_id=trace.selection.baseline_artifact_id,
            candidate_selected=True,
            exact_object_identity_verified=True,
            required_decision_names=trace.selection.required_decision_names,
        )


def test_nested_ids_detect_tampering() -> None:
    trace, _, _ = _build_trace()
    stage_payload = trace.stages[0].as_dict()
    stage_payload["stage_name"] = "tampered"
    with pytest.raises(ValueError, match="stage_id"):
        DecisionTraceStage.from_dict(stage_payload)

    selection_payload = trace.selection.as_dict()
    selection_payload["candidate_selected"] = False
    selection_payload["deployed_artifact_id"] = selection_payload[
        "baseline_artifact_id"
    ]
    with pytest.raises(ValueError, match="selection_id"):
        DecisionTraceSelection.from_dict(selection_payload)

    trace_payload = trace.as_dict()
    trace_payload["trace_name"] = "tampered"
    with pytest.raises(ValueError, match="trace_id"):
        UnifiedDecisionTrace.from_dict(trace_payload)


def test_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    trace, _, _ = _build_trace()
    serialized = json.dumps(trace.as_dict(), sort_keys=True)
    duplicate = serialized[:-1] + ',"trace_name":"duplicate"}'
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_decision_trace(path)


def test_claim_bearing_loader_requires_independent_pins(tmp_path: Path) -> None:
    trace, _, _ = _build_trace()
    path = tmp_path / "trace.json"
    write_decision_trace(path, trace)
    assert (
        load_claim_bearing_decision_trace(
            path,
            expected_trace_id=trace.trace_id,
            expected_stack_lock_id=trace.stack_lock_id,
            expected_protocol_id=trace.protocol_id,
        ).trace_id
        == trace.trace_id
    )
    with pytest.raises(ValueError, match="expected_trace_id"):
        load_claim_bearing_decision_trace(
            path,
            expected_trace_id=_id("wrong"),
            expected_stack_lock_id=trace.stack_lock_id,
        )
    with pytest.raises(ValueError, match="expected_stack_lock_id"):
        load_claim_bearing_decision_trace(
            path,
            expected_trace_id=trace.trace_id,
            expected_stack_lock_id=_id("wrong lock"),
        )
    with pytest.raises(ValueError, match="expected_protocol_id"):
        load_claim_bearing_decision_trace(
            path,
            expected_trace_id=trace.trace_id,
            expected_stack_lock_id=trace.stack_lock_id,
            expected_protocol_id="wrong-protocol",
        )


def test_stack_lock_binding_uses_validated_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace, _, _ = _build_trace()

    def validate(value: object) -> dict[str, object]:
        assert value == {"lock_id": trace.stack_lock_id}
        return {"lock_id": trace.stack_lock_id}

    monkeypatch.setattr("causal4d.stack_lock.validate_stack_lock", validate)
    assert (
        require_decision_trace_stack_lock(
            trace,
            {"lock_id": trace.stack_lock_id},
        )
        is trace
    )

    def mismatch(value: object) -> dict[str, object]:
        del value
        return {"lock_id": _id("other stack")}

    monkeypatch.setattr("causal4d.stack_lock.validate_stack_lock", mismatch)
    with pytest.raises(ValueError, match="does not match"):
        require_decision_trace_stack_lock(trace, {})


def test_required_root_producers_are_fixed() -> None:
    roots, stages, names, baseline, candidate = _trace_parts()
    bad_factual = replace(roots[0], producer="prob4d")
    with pytest.raises(ValueError, match="root role.*must be produced"):
        build_unified_decision_trace(
            trace_name="case trace",
            protocol_id="protocol-v2",
            case_id="case-001",
            session_id="session-001",
            endpoint="same_grasp_transfer",
            stack_lock_id=_id("stack lock"),
            root_artifacts=(bad_factual, roots[1]),
            stages=stages,
            required_decision_names=names,
            baseline=baseline,
            candidate=candidate,
            deployed=candidate,
            baseline_artifact_id=stages[1].output_artifacts[1].artifact_id,
            candidate_artifact_id=stages[3].output_artifacts[0].artifact_id,
        )


def test_selection_must_be_validated_type() -> None:
    trace, _, _ = _build_trace()
    with pytest.raises(ValueError, match="selection must be"):
        replace(trace, selection=object())


def test_metadata_is_deeply_immutable() -> None:
    trace, _, _ = _build_trace()
    with pytest.raises(TypeError):
        trace.metadata["operator"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        trace.metadata["operator"]["version"] = 2  # type: ignore[index]


def test_metadata_order_does_not_change_content_identity() -> None:
    trace, _, _ = _build_trace()
    reordered = replace(
        trace,
        metadata={"z": 2, "a": {"y": 1, "x": 0}},
    )
    same = replace(
        trace,
        metadata={"a": {"x": 0, "y": 1}, "z": 2},
    )
    assert reordered.trace_id == same.trace_id


def test_write_is_non_overwriting_by_default(tmp_path: Path) -> None:
    trace, _, _ = _build_trace()
    path = tmp_path / "trace.json"
    write_decision_trace(path, trace)
    with pytest.raises(FileExistsError):
        write_decision_trace(path, trace)
    write_decision_trace(path, trace, overwrite=True)
