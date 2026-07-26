# Action-conditioned graph discrepancy

This opt-in module keeps the empirically successful graph-persistent discrepancy
as its exact fallback while allowing both uncertainty and the low-rank readout
mean to evolve with the commanded and realized intervention. It does not alter
the frozen `v0.3.0-causal4d-aip` inference path.

## Typed belief

`GraphDiscrepancyBelief` stores component-wise graph coefficients and their full
low-rank covariance:

```text
coefficient_mean_m        (K, rank, 3)
coefficient_covariance_m2 (K, 3, rank, rank)
projection_variance_m2    (3,)
```

The artifact binds the basis by SHA-256, names the transition and innovation
models, optionally identifies its source physical posterior, and uses a
checksummed non-pickled NPZ payload.

## Feature-conditioned covariance

For transition feature vector `f`, the covariance model uses

```text
Q(f) = Q0 + sum_j (w_j^T f)^2 v_j v_j^T.
```

This is positive semidefinite by construction. With zero feature weights it is
exactly the declared base persistence model. The provided feature builder uses
controller speed, acceleration and direction together with gain, delay, frame
rotation, slip, attachment shift, and the same-grasp/new-contact policy. It
consumes no future object outcome.

## Stable discrepancy-mean movement

`ActionConditionedMeanTransitionModel` adds an opt-in transition for the graph
coefficient mean. It separates mode rotation, contraction, and bounded additive
movement:

```text
Omega(f) = sum_j (r_j^T f) G_j,       G_j^T = -G_j
D(f)     = sum_j (c_j^T f)^2 d_j d_j^T
b(f)     = B f
A(f)     = exp(Omega(f)) exp(-D(f))
c_next   = A(f) c + b(f)
```

The rotation factor is orthogonal and the contraction factor is non-expansive,
so the homogeneous transition cannot amplify the coefficient norm. Optional
Frobenius-norm caps bound both transition generators and additive movement. The
same transition propagates coefficient covariance as

```text
P_next = A(f) P A(f)^T + Q(f).
```

`forecast_action_conditioned_movement` applies this model component by component
and returns graph-readout mean and variance without injecting a correction into
simulator position or velocity. A zero-weight
`ActionConditionedMeanTransitionModel.persistence(...)` delegates directly to
the existing persistence forecast and returns it unchanged, including its model
identifier and array values.

This parameterization is intended to test the interaction-dependent contraction,
rotation, and graph-mode transfer observed in the released discrepancy-location
diagnostics. It is not evidence that such a transition transfers. All movement
weights, caps, feature normalization, graph rank, and covariance settings must be
fitted on source executions or preregistered before confirmatory targets are
opened. Rejection must retain exact graph persistence.

## Forecast API

```python
from causal4d import (
    ActionConditionedMeanTransitionModel,
    forecast_action_conditioned_movement,
)

mean_model = ActionConditionedMeanTransitionModel.persistence(
    features.names,
    belief.rank,
)
forecast = forecast_action_conditioned_movement(
    belief,
    mean_model,
    covariance_model,
    features,
    basis,
)
```

Replace the persistence model only with a source-fitted and prospectively gated
model. Zero movement weights remain the fail-closed fallback.

## Correlation-aware factual abduction

`abduct_factual_intervention_graph_mode` is a separate opt-in replacement for the
coordinatewise pseudo-likelihood. It projects the allowed response prefix onto a
fixed graph basis and scores multivariate Student-t innovations. A supplied
source covariance can represent correlated graph modes. The dynamic score
includes the endpoint-to-first-response increment.

The legacy `abduct_factual_intervention` function and its default likelihood are
unchanged, so frozen artifacts remain reproducible. Promotion of the graph-mode
likelihood or discrepancy movement requires source-only selection and a sealed
held-out comparison.
