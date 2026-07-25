# Hierarchical Multi-Execution Abduction

## Purpose

The original real-data path abduced one realized intervention from one response
prefix. The multi-action protocol instead supplies repeated executions in which
physical parameters and persistent actuation variables can be shared while
contact and slip remain execution-specific.

The new backend-neutral implementation represents

\[
p(\theta,\phi,\kappa_{1:E}\mid O^+_{1:E},u_{1:E}),
\]

with:

- shared physical-particle index `theta`;
- shared persistent realization variables `phi` (gain, delay, frame rotation);
- local rollout hypotheses, including execution-specific contact and slip
  variables `kappa_e`.

It does not form the Cartesian product over every execution-specific contact
hypothesis. For each shared `(theta, phi)` value it marginalizes the local
hypotheses of each execution, multiplies the resulting execution evidences, and
then reconstructs a normalized local posterior for every execution.

## Prefix likelihood

`causal4d.prefix_likelihood` replaces the factual-abduction prefix update with a
correlation-aware composite Student-t score. The position block uses response
frames after the branch endpoint. The dynamic block differences the complete
allowed prefix and therefore includes the endpoint-to-first-response increment,
which the previous implementation omitted.

For adjacent observation-error correlation `rho`, the difference scale is

\[
\sigma_{\Delta}=\sigma\sqrt{2(1-\rho)}.
\]

A graph-persistent discrepancy mean cancels in the dynamic block. Its declared
static variance inflates only the position block; it is not counted twice as
independent velocity uncertainty.

The existing `FactualAbductionConfig` remains the public real-backend
configuration and now maps exactly to `PrefixLikelihoodConfig`. Existing
settings remain valid. The additional parameters are:

- `position_likelihood_weight`, default `1.0`;
- `difference_correlation`, default `0.0`.

The real abduction CLI exposes both values. Confirmatory use should still load
values from a source-locked protocol rather than select them on target outcomes.

## Hierarchical API

```python
from causal4d.hierarchical_abduction import (
    abduct_hierarchical_interventions,
)
from causal4d.prefix_likelihood import PrefixLikelihoodConfig

result = abduct_hierarchical_interventions(
    banks,
    observations,
    prefix_frame_counts=prefix_frame_counts,
    masks=masks,
    config=PrefixLikelihoodConfig(
        observation_scale_m=0.01,
        likelihood_power=12.0,
        dynamic_likelihood_weight=0.25,
    ),
)

phi_posterior = result.phi_marginal
parameter_posterior = result.parameter_marginal
local_execution_posteriors = result.execution_joint_weights
```

All banks must expose the same physical particles and the same finite `phi`
support. Hypothesis order and local `kappa` support may differ. By default,
`phi` is read from the contact metadata fields `gain_multiplier`, `delay_steps`,
and `rotation_degrees`; an explicit `phi_values_by_bank` matrix can be supplied
for another backend.

Only frames before each declared `prefix_frame_count` enter the likelihood.
Future changes therefore leave both the shared posterior and every reconstructed
execution posterior unchanged.

## Identifiability gate

`causal4d.identifiability.evaluate_response_identifiability` compares a local
intervention-response matrix with a nuisance-response matrix containing reset
state, discrepancy, or observation-bias directions. It reports:

- the minimum principal angle between the response subspaces;
- maximum canonical correlation;
- the fraction of intervention-response energy projected into the nuisance
  subspace;
- the smallest singular value remaining after nuisance projection.

The result fails closed when any configured threshold is violated. Optional row
weights can whiten heteroscedastic responses or remove predeclared unreliable
coordinates. The gate diagnoses local separation only; passing it does not by
itself establish calibrated intervention recovery.

## Claim boundary

These additions provide the inference machinery required by the same-object,
multi-action protocol. They do not convert the existing single-interaction
sloth diagnostics into multi-execution evidence, and they do not promote a new
physical discrepancy mechanism. Real claims still require the preregistered
independent executions, held-out action/contact evaluation, and the declared
calibration boundary.
