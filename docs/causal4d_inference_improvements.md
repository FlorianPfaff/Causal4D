# Session-Aware Abduction and Stable Discrepancy Dynamics

This development extension addresses three limitations without changing the
frozen `v0.3.0-causal4d-aip` path.

## Session-aware hierarchical abduction

`abduct_hierarchical_interventions` accepts `session_ids`. Executions in the
same grasp or reset session share nuisance errors, so they should not each count
as an independent unit of evidence for shared `(theta, phi)` variables.

For a session with `n_s` executions, each marginalized execution log evidence is
weighted by `1/n_s`. The local execution likelihood remains unpowered when
recovering its `kappa_e` posterior. Omitting `session_ids` reproduces the original
independent-execution product exactly. Explicit `execution_evidence_powers` can
be supplied for a source-frozen alternative composite likelihood.

The result metadata records the session IDs, evidence powers, session count, and
evidence mode.

## Scale-invariant and partial identifiability

`assess_intervention_identifiability` accepts characteristic
`parameter_scale`. Intervention sensitivity columns are multiplied by that
scale before nuisance projection and information analysis. This makes the
result invariant to unit conversions such as degrees versus radians or frames
versus seconds, provided the corresponding scale is converted consistently.

The result now retains identifiable and nullspace bases. The helper
`preserve_prior_within_unidentified_subspace` removes unsupported posterior
distinctions while retaining evidence between distinguishable projection
groups:

- full rank returns the supplied update;
- rank zero returns the normalized prior exactly;
- partial rank preserves prior-relative weights within each indistinguishable
  group.

The frozen guarded-abduction behavior remains conservative: this helper is
opt-in and does not silently replace exact-prior abstention.

## Stable discrepancy mean dynamics

`StableDiscrepancyTransitionModel` augments the existing action-conditioned
innovation covariance with a mean transition

```text
d_(t+1) = A(f_t) d_t + b(f_t) + epsilon_t.
```

The transition is built from

```text
G(f) = S(f) - C(f)
A(f) = expm(G(f)),
```

where `S` is skew-symmetric and `C` is positive semidefinite. This permits
source-fitted graph-mode rotation and contraction while keeping the transition
non-expansive. A feature-conditioned drift is optional and norm capped.

`StableDiscrepancyTransitionModel.identity(...)` gives exact graph persistence.
`forecast_action_conditioned_dynamics` propagates

```text
m_(t+1) = A m_t + b
P_(t+1) = A P_t A^T + Q(f_t),
```

using the existing `ActionConditionedDiscrepancyModel` as the innovation model.
Transition generators, feature weights, and drift caps must be selected on
source executions or preregistered before confirmatory evaluation.

## Claim boundary

These additions are inference machinery, not new real-data evidence. Promotion
still requires the locked multi-action protocol, independent-session
calibration, source-only mechanism selection, and exact persistence/prior
fallback controls.
