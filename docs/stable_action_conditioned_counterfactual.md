# Stable Action-Conditioned Counterfactual Discrepancy

## Purpose

The repository previously contained two complementary opt-in components:

1. an action-conditioned counterfactual posterior that propagated a persistent
   discrepancy mean with feature-conditioned covariance growth; and
2. a stable discrepancy transition that represented graph-mode rotation,
   contraction, bounded drift, and exact identity fallback.

`apply_stable_action_conditioned_counterfactual_operator` connects those
components. The ordinary `apply_counterfactual_operator` remains authoritative
for simulator state trajectories, intervention transport, contact semantics,
posterior weights, and provenance. The new operator replaces only the
counterfactual discrepancy-aware readout moments.

## Dynamics

For graph coefficients `d_t`, action/intervention features `f_t`, and innovation
covariance `Q(f_t)`, the operator propagates

```text
d_(t+1) = A(f_t) d_t + b(f_t) + epsilon_t
P_(t+1) = A(f_t) P_t A(f_t)^T + Q(f_t)
```

where the stable transition model constructs

```text
A(f_t) = expm(S(f_t) - C(f_t)).
```

`S` is skew-symmetric and permits graph-mode rotation. `C` is positive
semidefinite and permits contraction without expansive dynamics. Optional drift
is norm capped. The identity constructor produces exact graph persistence.

The graph coefficients are projected to node space and added only to readout
moments. They are never injected into simulator position or velocity.

## Example

```python
posterior = apply_stable_action_conditioned_counterfactual_operator(
    bank,
    manifest,
    twin_belief,
    factual_intervention,
    counterfactual_query,
    graph_discrepancy_belief,
    innovation_model,
    transition_model,
    graph_basis,
    controller_anchor,
    frame_dt_s=0.1,
)
```

Use `StableDiscrepancyTransitionModel.identity(...)` to obtain the exact
persistent-mean fallback. Tests require that this fallback reproduces the
existing action-conditioned counterfactual readout and covariance exactly.

## Evidence boundary

This is inference machinery, not promoted real-data evidence. Transition
features, graph rank, covariance parameters, drift caps, and all model-selection
choices must be fitted on source executions or preregistered. No held-out future
observation is read. A failed prospective mechanism or calibration gate must
retain the exact persistence operator.
