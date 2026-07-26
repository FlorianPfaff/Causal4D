# Regime-conditioned graph discrepancy

## Purpose

Graph persistence is the current real-data discrepancy baseline, but the frozen
state diagnostics show interaction-dependent contraction, rotation, and transfer
between graph modes. This opt-in module adds a stable mean transition conditioned
on the declared contact regime and action features. It does not alter the frozen
`v0.3.0-causal4d-aip` path or promote a new real-data mechanism.

## Stable transition

For contact regime `r` and feature vector `f_t`, the coefficient transition is

```text
rate_r(f_t) = base_rate_r + (w_r^T f_t)^2
alpha_r(f_t) = 1 - exp(-rate_r(f_t))
A_r(f_t) = (1 - alpha_r) I + alpha_r T_r
```

Each declared target operator `T_r` must have spectral norm at most one. The
convex combination is therefore non-expansive, while allowing contraction,
rotation, and mode mixing. Zero base rates and zero feature weights return the
identity exactly, giving byte-exact coefficient persistence when no innovation
covariance is supplied.

The forecast propagates

```text
a_(t+1) = A_r(f_t) a_t
P_(t+1) = A_r(f_t) P_t A_r(f_t)^T + Q(f_t)
```

where `Q(f_t)` may be supplied by the existing
`ActionConditionedDiscrepancyModel`. This reuses its positive-semidefinite,
action-conditioned covariance growth while extending the discrepancy mean.

## Contact-path interface

`forecast_regime_conditioned_discrepancy` accepts either one shared regime path
with shape `(H,)` or component-specific paths with shape `(K, H)`. The indices
follow `causal4d.dynamic_contact.ContactRegime`:

- inactive;
- sticking;
- slipping;
- detached.

The routine also accepts shared or component-specific action features and checks
component IDs, graph-basis hashes, model rank, feature schemas, regime support,
and non-expansiveness.

## Example

```python
from causal4d import (
    RegimeConditionedDiscrepancyTransitionModel,
    forecast_regime_conditioned_discrepancy,
)

transition = RegimeConditionedDiscrepancyTransitionModel(
    feature_names=features.names,
    target_matrices=source_frozen_targets,
    base_activation_rates=source_frozen_rates,
    feature_weights=source_frozen_weights,
)

forecast = forecast_regime_conditioned_discrepancy(
    discrepancy_belief,
    transition,
    features,
    contact_path.regime_paths,
    graph_basis,
    innovation_model=action_conditioned_covariance,
)
```

## Evidence boundary

Target futures must not select target matrices, activation rates, feature
weights, basis rank, or innovation parameters. Candidate transitions should be
fit on source executions, shrunk toward identity, and compared prospectively
against exact graph persistence in leave-one-action and leave-one-contact folds.
The registered same-object multi-action protocol remains the required promotion
gate.
