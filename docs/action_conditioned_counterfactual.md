# Action-Conditioned Counterfactual Readout Uncertainty

## Purpose

The standard Causal4D counterfactual operator transports the factual posterior
into a finite PhysTwin rollout bank and returns a provenance-complete
`PhysicalPosterior`. Its discrepancy mean and variance are intentionally static
for frozen milestone compatibility.

`apply_action_conditioned_counterfactual_operator` adds an opt-in extension that
keeps the standard operator authoritative for state trajectories, intervention
transport, contact handling, component weights, and provenance, but replaces the
readout moments with:

- a graph-persistent discrepancy mean;
- component-specific, action-conditioned positive-semidefinite covariance growth;
- a full temporal marginal variance with shape `(component, frame, node, 3)`.

No held-out object observation enters the extension.

## Component alignment

A `GraphDiscrepancyBelief` may identify either:

1. every counterfactual rollout component directly; or
2. the `TwinBelief` particles.

In the second case, the implementation expands the discrepancy belief through
the `twin_particle_indices` of the physical posterior. This preserves the joint
rollout support while allowing all contact/action hypotheses associated with one
physical particle to share its endpoint discrepancy belief.

## Readout moments

For graph basis `U`, discrepancy coefficient mean `c`, and component-specific
action features `f_t`, the mean remains persistent while covariance evolves as

```text
E[c_t] = E[c_0]
P_{t+1} = P_t + Q(f_t).
```

`Q(f_t)` is supplied by `ActionConditionedDiscrepancyModel` and is positive
semidefinite by construction. The returned readout is

```text
Y_t = X_t + U E[c_t]
Var[Y_t] = diag(U P_t U^T) + projection_variance + rollout_variance_floor.
```

Zero innovation covariance and zero feature weights preserve the endpoint
readout mean and variance exactly across the horizon.

## API

```python
from causal4d import apply_action_conditioned_counterfactual_operator

posterior = apply_action_conditioned_counterfactual_operator(
    bank,
    manifest,
    twin_belief,
    factual_intervention,
    query,
    graph_discrepancy_belief,
    discrepancy_model,
    graph_basis,
    control_anchor_m,
    frame_dt_s=1.0 / 30.0,
)
```

The result wraps the original `PhysicalPosterior` and exposes temporal
`readout_trajectories_m`, `readout_variance_m2`, and discrepancy coefficient
covariances. The frozen standard operator and milestone artifacts are unchanged.

## Claim boundary

This is integration machinery, not a promoted discrepancy model. Feature
weights, base innovation covariance, graph rank, and variance caps must be
selected from source-only or preregistered data. Confirmatory use still requires
held-out action/contact evaluation and execution-level calibration. Outside the
validated action neighborhood, the caller must widen uncertainty further or
abstain rather than interpreting covariance growth as calibrated transfer.
