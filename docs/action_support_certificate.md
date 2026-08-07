# Source-Frozen Action-Support Admission

## Purpose

Action-conditioned discrepancy dynamics are meaningful only inside the action and
realization neighborhood on which their source model was selected. A positive
semidefinite covariance increment is numerically valid outside that neighborhood,
but it is not evidence that the transferred discrepancy mean or uncertainty is
calibrated there.

`causal4d.action_support` adds an opt-in, source-only support certificate for the
action features already used by Causal4D discrepancy forecasts. It does not alter
the frozen counterfactual operator, the registered physical estimator, or any
existing result.

## Calibration boundary

Each `ActionSupportSourceCase` contains only:

- a source case identity;
- an `ActionConditionedDiscrepancyFeatures` trajectory;
- posterior component weights when the features are component-specific;
- the exact candidate discrepancy-model identity; and
- the feature schema, component identities, and physical step duration already
  bound by that feature artifact.

No source or target object-response continuation is accepted by the contract.
Every source case and the resulting calibration are content addressed.

The fitter augments every feature trajectory with physical step duration and
elapsed query time, then summarizes each source case by the posterior-weighted
temporal mean and root-mean-square value of every support dimension. It records:

- canonical source-case summaries and raw feature ranges;
- robust source-summary scales;
- the largest leave-one-source nearest-neighbour distance;
- an explicit multiplicative support margin;
- expanded per-feature source bounds; and
- the minimum posterior component mass that must remain supported.

The loader independently recomputes every derived threshold and bound from the
retained source summaries. A self-rehashed artifact with modified derived values
is rejected. Claim-bearing use must additionally bind the expected calibration ID
in a separately frozen protocol or method manifest and load it through
`load_claim_bearing_action_support_calibration`.

## Target decision

A target decision reads only the proposed action/intervention feature trajectory
and component weights. For each posterior component it computes:

1. the nearest standardized source-summary distance; and
2. whether every feature value remains inside the expanded source range.

A component is supported only when both conditions pass. The candidate is admitted
only when supported posterior mass reaches the frozen minimum. The decision records
all distances, supported flags, component weights, supported mass, rejection
reasons, and `future_observation_frames_read = 0`.

The frozen mass threshold is a deployment gate for the complete candidate. Passing
the gate deploys the candidate exactly as constructed; failing it preserves the
complete baseline. The selector does not drop unsupported components, renormalize
the supported subset, or construct a hybrid posterior. Such component-level
modification would be a different method requiring its own source calibration,
registration, and validation.

## Exact fallback

The generic selector preserves the caller-provided baseline object by identity:

```python
selection = select_action_supported_candidate(
    calibration,
    target_features,
    baseline=physical_posterior,
    candidate=action_conditioned_posterior,
    candidate_model_id=action_conditioned_posterior.discrepancy_model_id,
    component_weights=action_conditioned_posterior.weights,
    component_ids=action_conditioned_posterior.component_ids,
)

assert selection.deployed is (
    selection.candidate if selection.decision.accepted else selection.baseline
)
```

For the complete counterfactual path, use:

```python
from causal4d import apply_guarded_action_conditioned_counterfactual_operator

selection = apply_guarded_action_conditioned_counterfactual_operator(
    bank,
    manifest,
    twin_belief,
    factual_intervention,
    query,
    graph_discrepancy_belief,
    discrepancy_model,
    graph_basis,
    control_anchor_m,
    action_support_calibration,
    frame_dt_s=1.0 / 30.0,
    feature_schema="signed_v2",
)
```

A calibration cannot be reused across a different candidate-model identity. A
rejection deploys the exact ordinary `PhysicalPosterior` contained in the
candidate. The action-conditioned readout remains available for diagnostics but
is not the deployed result.

## Validation contract

The focused regression suite covers deterministic calibration identity, strict
source/model/schema binding, component-wise range and distance admission,
posterior-mass thresholds, target-future independence, immutable arrays, strict
JSON loading, derived-value tamper rejection, and exact baseline identity on
fallback. Pull-request validation additionally runs Ruff lint and formatting,
MyPy with the repository's Python 3.12 type-check target, and warning-as-error
byte compilation. These are software and contract checks, not predictive evidence.

## Scientific boundary

This is prospective method-development infrastructure. A green support decision
shows only that the query features lie inside a source-defined envelope. It does
not establish provider competence, intervention identifiability, beneficial
counterfactual correction, calibrated coverage, deployment safety, or state of
the art. A claim-bearing pipeline still needs independent observation admission,
BayesianPhysTwin accept/fallback, Causal4D identifiability, a baseline-relative
regret certificate, and held-out query calibration.
