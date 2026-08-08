# Unified Prob4D → BayesianPhysTwin → Causal4D Decision Trace

## Purpose

`causal4d.decision_trace` provides an additive, content-addressed audit contract for
one complete prediction/deployment decision across the three-repository stack.

The trace answers:

> Which exact observation, physical-belief, causal-inference, support, calibration,
> and regret artifacts led to the deployed result, and was the candidate or the
> declared baseline selected?

It does not embed model payloads, trajectories, target continuations, or held-out
losses. It records immutable artifact and decision identities and validates their
topological relationships.

The frozen estimator and the registered 18-session/36-execution physical protocol
are unchanged.

## Fixed stage order

A trace contains exactly five stages:

1. `prob4d_observation`
2. `bayesian_phystwin_belief`
3. `causal4d_abduction`
4. `causal4d_counterfactual`
5. `deployment`

The producer is fixed by stage:

| Stage | Producer |
|---|---|
| `prob4d_observation` | `prob4d` |
| `bayesian_phystwin_belief` | `bayesian-phystwin` |
| `causal4d_abduction` | `causal4d` |
| `causal4d_counterfactual` | `causal4d` |
| `deployment` | `causal4d` |

The trace validates the graph incrementally. A stage may consume only root
artifacts or outputs of earlier stages. Forward references, duplicate output
identities, missing hand-offs, and producer mismatches fail closed.

## Required artifact roles

The root contains exactly one principal artifact for each of:

```text
factual_evidence_context
counterfactual_query_context
```

The stage outputs must establish exactly one principal artifact for each of:

```text
prob4d_observation
bayesian_phystwin_belief
baseline_prediction
causal4d_factual_posterior
candidate_prediction
```

Additional stage-specific artifacts are allowed, but every principal role must
remain unique.

The required hand-offs are:

```text
factual_evidence_context
  -> prob4d_observation

prob4d_observation
  -> bayesian_phystwin_belief
  -> baseline_prediction

prob4d_observation + bayesian_phystwin_belief
  -> causal4d_factual_posterior

causal4d_factual_posterior + counterfactual_query_context
  -> candidate_prediction

baseline_prediction + candidate_prediction
  -> deployment
```

This contract exposes accidental bypasses. For example, a causal abduction stage
cannot claim to use BayesianPhysTwin while omitting the exact physical-belief
artifact from its inputs.

## Decisions and exact fallback

Each stage may reference content-addressed decisions. The generic decision-trace
builder accepts an experiment-specific required inventory. The versioned
prospective V2 deployment profile is:

```text
causal4d.prospective-v2-deployment-profile/v1
```

It requires the following complete inventory:

```text
prob4d_provider_acceptance
joint_covariance_admission
bayesian_phystwin_acceptance
functional_support
intervention_identifiability
action_support
contact_v2_support
conditional_uncertainty_calibration
query_calibration
counterfactual_regret
```

The profile also fixes the stage and producer for every decision. Use
`build_prospective_v2_decision_trace_v1` for that path; it inserts the profile ID,
uses the generic exact-object builder, and then validates the stricter inventory.
See `docs/prospective_v2_promotion.md` for the profile and promotion protocol.

For other experiments, the required inventory is frozen directly in
`DecisionTraceSelection`. Candidate deployment is valid if and only if every
named required decision is present and accepted.

At runtime, the generic `build_unified_decision_trace` checks Python object
identity:

```python
result = build_unified_decision_trace(
    trace_name="rope-001/same-grasp",
    protocol_id=protocol_id,
    case_id=case_id,
    session_id=session_id,
    endpoint="same_grasp_transfer",
    stack_lock_id=stack_lock["lock_id"],
    root_artifacts=root_artifacts,
    stages=stages,
    required_decision_names=required_decisions,
    baseline=baseline_prediction,
    candidate=causal4d_candidate,
    deployed=deployed_prediction,
    baseline_artifact_id=baseline_prediction_id,
    candidate_artifact_id=causal4d_candidate_id,
)

trace = result.trace
```

When a required decision rejects, `deployed` must be the exact baseline object.
When every required decision accepts, `deployed` must be the exact candidate
object. Reconstructed approximations do not satisfy the runtime identity check.

The serialized selection records:

```text
baseline_artifact_id
candidate_artifact_id
deployed_artifact_id
candidate_selected
exact_object_identity_verified
required_decision_names
selection_id
```

## Target-information boundary

A deployment trace has no target-continuation or target-loss field. It always
records:

```text
target_future_observations_read = 0
target_future_outcomes_used = false
```

Held-out target artifact roles are rejected. Metadata recursively rejects keys
such as:

```text
evaluation_target
held_out_target
target_continuation
target_future
target_loss
target_outcome
target_outcomes
```

The trace may include a target case or session identity because those are
necessary for audit and source/target separation. It may not contain the held-out
future used to score that target.

Evaluation targets and losses belong in separate registered result artifacts,
not in the deployment decision trace.

## Content addressing

The following objects are independently content-addressed:

```text
DecisionTraceStage.stage_id
DecisionTraceSelection.selection_id
UnifiedDecisionTrace.trace_id
```

The top-level trace ID covers:

- protocol, case, session, and endpoint;
- the exact stack-lock identity;
- root artifact references;
- ordered stage payloads and stage IDs;
- all decision identities, states, and reason codes;
- final selection and required-decision inventory;
- target-information declarations; and
- finite immutable metadata.

Changing any of these values invalidates the recorded ID.

## Stack-lock binding

The trace references one exact three-repository stack lock:

```python
require_decision_trace_stack_lock(trace, stack_lock)
```

This function first validates the complete external stack lock and then requires
its `lock_id` to equal `trace.stack_lock_id`.

The trace does not duplicate wheel hashes or repository revisions. Those remain
authoritative in the existing stack-lock artifact. This avoids two competing
copies of the same compatibility declaration.

## Artifact references

An artifact reference contains:

```text
artifact_id
artifact_kind
role
producer
metadata
```

`artifact_id` is a lowercase SHA-256 identity supplied by the producing contract.
Typical references include:

- a Prob4D observation-factor lineage or provider artifact;
- a BayesianPhysTwin provider/belief artifact;
- a Causal4D factual posterior or rollout-bank artifact;
- a physical/Bayesian baseline forecast;
- a Causal4D counterfactual forecast; and
- stage-specific support or calibration artifacts.

Payload bytes remain in their native repositories or result bundles.

## Decision references

A decision reference contains:

```text
name
decision_id
decision_kind
producer
accepted
reasons
metadata
```

Accepted decisions cannot contain rejection reasons. Rejected decisions must
contain at least one stable reason code.

The trace references existing decision artifacts rather than redefining their
scientific semantics. For example:

- Prob4D provider/lineage acceptance remains governed by the Prob4D contract;
- joint-covariance admission remains governed by the admitted covariance
  contract;
- BayesianPhysTwin provider acceptance remains governed by the BPT contract;
- functional support remains governed by the rollout-space and projected
  functional-support certificates;
- action support remains governed by `ActionSupportDecision`;
- contact-patch finite support remains governed by `ContactV2SupportDecision`;
- conditional uncertainty remains governed by its source-calibration artifact;
- baseline-relative benefit remains governed by
  `CounterfactualRegretDecision`.

## Construction example

```python
from causal4d import (
    DecisionTraceArtifact,
    DecisionTraceDecision,
    DecisionTraceStage,
    build_unified_decision_trace,
)

factual_context = DecisionTraceArtifact(
    artifact_id=factual_evidence_context.artifact_id,
    artifact_kind="causal4d.FactualEvidenceContext",
    role="factual_evidence_context",
    producer="causal4d",
)
query_context = DecisionTraceArtifact(
    artifact_id=counterfactual_query_context.artifact_id,
    artifact_kind="causal4d.CounterfactualQueryContext",
    role="counterfactual_query_context",
    producer="causal4d",
)

prob4d_observation = DecisionTraceArtifact(
    artifact_id=observation_lineage.artifact_id,
    artifact_kind="prob4d.ObservationFactorLineage",
    role="prob4d_observation",
    producer="prob4d",
)

prob4d_stage = DecisionTraceStage(
    stage_name="prob4d observation admission",
    stage_kind="prob4d_observation",
    producer="prob4d",
    input_artifact_ids=(factual_context.artifact_id,),
    output_artifacts=(prob4d_observation,),
    decisions=(
        DecisionTraceDecision(
            name="prob4d_provider_acceptance",
            decision_id=prob4d_acceptance_id,
            decision_kind="prob4d.ProviderAcceptance",
            producer="prob4d",
            accepted=True,
        ),
    ),
)
```

The remaining stages follow the required role graph described above.

## Publication and claim-bearing loading

```python
write_decision_trace("decision-trace.json", trace)

loaded = load_decision_trace("decision-trace.json")

claim_trace = load_claim_bearing_decision_trace(
    "decision-trace.json",
    expected_trace_id=registered_trace_id,
    expected_stack_lock_id=registered_stack_lock_id,
    expected_protocol_id=registered_protocol_id,
)
```

Publication is atomic and non-overwriting by default. Loading rejects:

- duplicate JSON object keys;
- non-finite JSON;
- missing or unexpected fields;
- malformed SHA-256 identities;
- stage, selection, or trace ID mismatches;
- invalid stage order or producer;
- missing required roles or hand-offs;
- forward artifact references;
- missing or duplicated decisions;
- selection/gate disagreement;
- target-future access; and
- stack-lock mismatch in claim-bearing use.

## Recommended retained evidence

For every prospective target execution, retain:

1. the exact three-repository stack lock and verification report;
2. native Prob4D observation/lineage artifacts;
3. the BayesianPhysTwin belief/provider artifacts;
4. Causal4D factual and counterfactual artifacts;
5. all support, identifiability, query-calibration, and regret decisions;
6. the unified decision trace; and
7. a separate result artifact if a held-out target is later opened.

This allows an auditor to reconstruct the decision graph without conflating
deployment inputs with evaluation outcomes.

## Scientific boundary

A valid trace proves internal identity and decision-chain consistency. It does
not prove:

- that any upstream model is scientifically correct;
- that the selected source-calibration panel is representative;
- that the required-decision inventory is sufficient;
- that counterfactual intervals are calibrated;
- that accepted candidates have zero regret;
- that the physical experiment is complete; or
- that a prospective V2 method is eligible to revise registered results.

The required-decision inventory, source panels, thresholds, and claim-bearing
trace/stack identities must be frozen independently before target outcomes are
opened.
