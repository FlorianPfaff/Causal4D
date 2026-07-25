# Action-conditioned graph discrepancy

This opt-in module keeps the empirically successful graph-persistent discrepancy
mean while allowing uncertainty to grow with the commanded and realized
intervention. It does not alter the frozen `v0.3.0-causal4d-aip` inference path.

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

For transition feature vector `f`, the model uses

```text
Q(f) = Q0 + sum_j (w_j^T f)^2 v_j v_j^T.
```

This is positive semidefinite by construction. With zero feature weights it is
exactly the declared base persistence model. The discrepancy mean remains fixed;
only covariance changes. The provided feature builder uses controller speed,
acceleration and direction together with gain, delay, frame rotation, slip,
attachment shift, and the same-grasp/new-contact policy. It consumes no future
object outcome.

`forecast_action_conditioned_persistence` returns coefficient and graph-readout
moments for every posterior component. This is intended for source-fitted or
preregistered covariance models. Target outcomes must not select feature weights,
directions, variance caps, or feature normalization.

## Correlation-aware factual abduction

`abduct_factual_intervention_graph_mode` is a separate opt-in replacement for the
coordinatewise pseudo-likelihood. It projects the allowed response prefix onto a
fixed graph basis and scores multivariate Student-t innovations. A supplied
source covariance can represent correlated graph modes. The dynamic score
includes the endpoint-to-first-response increment.

The legacy `abduct_factual_intervention` function and its default likelihood are
unchanged, so frozen artifacts remain reproducible. Promotion of the graph-mode
likelihood requires source-only temperature/covariance selection and a sealed
held-out comparison.
