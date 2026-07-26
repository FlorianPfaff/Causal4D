# Session-Aware Abduction and Stable Discrepancy Dynamics

This development extension addresses correlated execution evidence, partial
intervention identifiability, and action-dependent readout discrepancy without
changing the frozen `v0.3.0-causal4d-aip` path.

## Session-aware hierarchical abduction

`abduct_hierarchical_interventions` accepts `session_ids`. Executions in the
same grasp or reset session share nuisance errors, so they should not each count
as an independent unit of evidence for shared `(theta, phi)` variables.

For a session with `n_s` executions, each marginalized execution log evidence is
weighted by `1/n_s`. The local execution likelihood remains unpowered when
recovering its `kappa_e` posterior. Omitting `session_ids` reproduces the original
independent-execution product exactly. Explicit `execution_evidence_powers` can
be supplied for a source-frozen alternative composite likelihood.

The result metadata records session IDs, evidence powers, session count, and the
evidence mode. Equal-session weighting is intentionally described as a
composite likelihood rather than an exact random-effects posterior. A future
hierarchical session-nuisance model may replace it after source-only validation.

## Scale-invariant and partial identifiability

`assess_intervention_identifiability` accepts characteristic
`parameter_scales`. Intervention sensitivity columns are multiplied by those
scales before nuisance projection and information analysis. The diagnostic is
therefore invariant to equivalent unit conversions, such as degrees versus
radians or frames versus seconds, when scales are converted consistently.

The result retains identifiable and nullspace bases. It can additionally score
a held-out query sensitivity and report the fraction of query response lying in
unresolved intervention directions. Full parameter recovery is therefore not
required when the requested prediction is insensitive to the unresolved
subspace.

`preserve_prior_within_unidentified_subspace` removes unsupported posterior
distinctions while retaining evidence between distinguishable projection
groups:

- full rank returns the supplied update;
- rank zero returns the normalized prior exactly;
- partial rank preserves prior-relative weights within each indistinguishable
  group.

The frozen guarded-abduction behavior remains conservative: partial updates are
opt-in and do not silently replace exact-prior abstention.

## Correlation-aware graph evidence

`GraphDiscrepancyBelief` retains full low-rank coefficient covariance.
`graph_discrepancy_group_covariances` maps it into grouped observation
covariances while preserving persistent cross-frame correlation. The grouped
Student-t likelihood accepts these full component-specific covariances instead
of forcing graph uncertainty into independent coordinate variances.

The graph basis is hash-bound to both the belief and grouped covariance mapping.
Target futures must not select covariance rank, temperature, feature scales, or
group construction.

## Stable discrepancy mean dynamics

`StableDiscrepancyTransitionModel` augments action-conditioned innovation
covariance with a mean transition

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

using `ActionConditionedDiscrepancyModel` as the innovation model.

## Integrated counterfactual operator

`apply_action_conditioned_counterfactual_operator` first calls the ordinary
counterfactual operator. That operator remains authoritative for simulator state
trajectories, posterior weights, `phi` transport, contact handling, and artifact
provenance. The extension replaces only discrepancy-aware readout moments.

Omitting `transition_model` preserves graph-persistent discrepancy means with
action-conditioned positive-semidefinite covariance growth. Supplying a stable
transition propagates both mean and covariance. Neither path injects discrepancy
into simulator position or velocity, and neither reads held-out object outcomes.

## Same-patch counterfactual semantics

The default `same_grasp` behavior remains `fixed_kappa`: the complete factual
contact and slip variable is reused. A query may explicitly set

```text
same_grasp_semantics = evolve_slip
```

in its metadata. This preserves the factual posterior over `(theta, phi,
contact patch)` while resampling counterfactual slip from the query-bank prior
conditional on `(phi, patch)`. `new_contact` continues to sample a fresh complete
contact event. Posterior metadata records whether the patch, slip, or complete
`kappa` was reused.

## Claim boundary

These additions are inference machinery, not new real-data evidence. Promotion
still requires the locked multi-action protocol, independent-session
calibration, source-only mechanism selection, and exact persistence/prior
fallback controls. Transition generators, graph rank, feature normalization,
drift caps, innovation covariance, slip priors, and admission thresholds must be
source-frozen or preregistered before confirmatory evaluation.
