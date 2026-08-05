# Horizon-conditioned Bayesian-PhysTwin discrepancy

## Status and claim boundary

This is an additive development path for consuming Bayesian-PhysTwin's
source-calibrated horizon discrepancy provider. It does not change the frozen
Causal4D estimator, the registered 36-execution protocol, or any completed
result. In particular, it does not describe raw covariance as calibrated and it
does not use target outcomes to choose horizon dynamics.

The motivation is the completed real undercoverage audit: discrepancy uncertainty
changes materially with prediction horizon, while one global affine covariance
multiplier transferred harmfully across actions. The new adapter preserves the
source-frozen mean-retention and process-growth semantics instead of applying one
post-hoc target inflation.

## Provider boundary

`causal4d.belief_provider_v2_contract` validates the additive
`bayesian_phystwin.causal4d_belief_provider_v2` manifest before any residual or
prediction payload is consumed. The required capabilities include:

- evidence-weighted endpoint model averaging;
- horizon-dependent predictive covariance;
- source-calibrated horizon discrepancy;
- mean-reverting discrepancy prediction;
- per-track component evidence; and
- immutable endpoint artifacts.

The original fixed-anchor provider-v1 contract remains unchanged and continues to
own frozen reproductions.

## Bank construction

`build_horizon_discrepancy_bank()` takes:

- one immutable Causal4D `TwinBelief`;
- one Bayesian-PhysTwin `ModelAveragedEndpointPosteriorV1` per physical particle;
- one source-only `HorizonDiscrepancyCalibrationV1`;
- a registered horizon set containing zero;
- the fixed tracked-to-state readout map; and
- an optional discrepancy-norm cap fixed before target access.

For every physical particle and horizon, Bayesian-PhysTwin predicts tracked-node
mean and covariance. Causal4D then lifts those moments to the complete physical
state. If an untracked state node is represented by fixed readout weights
`w_j`, the current covariance lift is

```text
mu_extra = sum_j w_j mu_j
Sigma_extra = sum_j w_j^2 Sigma_j
```

The covariance expression assumes conditionally independent tracked-node
errors. This assumption is retained explicitly in the artifact metadata; the
adapter does not manufacture unobserved cross-node covariance.

The returned `HorizonDiscrepancyBankV1` stores:

```text
mean_m                         (P, H, N, 3)
covariance_m2                  (P, H, N, 3, 3)
horizon_steps                  (H,)
mean_retention                 (H,)
additional_axis_variance_m2    (H, 3)
particle_weights               (P,)
```

It binds the exact `TwinBelief` artifact, particle identities and weights,
calibration artifact, provider revision, source-group declaration, readout
semantics, optional cap, and every numerical array into one SHA-256 identity.

## Horizon-zero parity

Every bank must contain horizon zero. At horizon zero, Bayesian-PhysTwin must
return the endpoint posterior moments without process growth:

```text
mean_retention = 1
additional_axis_variance = 0
```

This supplies an executable parity check between the endpoint posterior and the
horizon-conditioned path before any positive-horizon result is interpreted.

## Information boundary

The adapter receives already inferred causal-prefix posteriors. It reads no
future observations and records `future_observations_read = 0`. The calibration
contract itself requires at least two independent source groups and rejects
interval-calibration, confirmation, or target outcomes as selectors of horizon
dynamics.

A future claim-bearing protocol must still keep source fitting, interval
calibration, and target evaluation as separate stages. Passing the provider and
artifact contracts establishes lineage and numerical semantics, not empirical
coverage.

## Current limitations

This first consumer conditions discrepancy on horizon only. It does not yet fit
or select graph-region, action-family, or contact-regime calibration. It also
propagates tracked-node covariance through a fixed independent-readout
approximation. These limitations must be reported rather than hidden by a global
inflation factor.

The next evidence step is a source/calibration-separated comparison of:

1. raw model-averaged endpoint covariance;
2. horizon-conditioned discrepancy covariance;
3. horizon plus predeclared graph-region conditioning; and
4. exact uncalibrated fallback.

The evaluation unit must be the independent execution or session, not frames,
vertices, or coordinates. Report coverage, interval width, NLL or energy score,
and worst-group coverage together.
