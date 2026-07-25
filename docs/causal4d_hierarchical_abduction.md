# Hierarchical Multi-Execution Abduction

## Purpose

The existing real-data path abduces one realized intervention from one response
prefix. A same-object multi-action protocol instead supplies repeated executions
in which physical parameters and persistent actuation variables can be shared,
while contact and slip remain execution-specific.

The backend-neutral implementation represents

\[
p(\theta,\phi,\kappa_{1:E}\mid O^+_{1:E},u_{1:E}),
\]

with:

- a shared physical-particle index `theta`;
- shared persistent realization variables `phi` such as gain, delay, and frame
  rotation;
- local rollout hypotheses containing execution-specific contact and slip
  variables `kappa_e`.

It does not form the Cartesian product over all execution-specific contact
hypotheses. For each shared `(theta, phi)` value, it marginalizes each
execution's local hypotheses, multiplies the resulting execution evidences, and
then reconstructs a normalized local posterior for every execution.

## Correlation-aware prefix likelihood

`causal4d.prefix_likelihood` provides an opt-in composite Student-t score. The
position block uses response frames after the branch endpoint. The dynamic block
differences the complete allowed prefix and therefore includes the
endpoint-to-first-response increment.

For adjacent observation-error correlation `rho`, the difference scale is

\[
\sigma_{\Delta}=\sigma\sqrt{2(1-\rho)}.
\]

A graph-persistent discrepancy mean cancels in the dynamic block. Its declared
static variance inflates only the position block; it is not counted as
independent velocity uncertainty. Particle-specific scales retain the
Student-t scale-normalization term, so broader discrepancy uncertainty is not
rewarded without a likelihood penalty.

This scorer is separate from the merged grouped-observation likelihood and from
the frozen legacy factual-abduction path. It is intended for the hierarchical
finite-bank API and controlled comparisons. Hyperparameters must be selected on
source data or preregistered before confirmatory use.

## API

```python
from causal4d import (
    PrefixLikelihoodConfig,
    abduct_hierarchical_interventions,
)

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

All banks must expose the same physical particles and finite `phi` support.
Hypothesis order and local `kappa` support may differ. By default, `phi` is read
from the contact metadata fields `gain_multiplier`, `delay_steps`, and
`rotation_degrees`; an explicit `phi_values_by_bank` matrix can be supplied for
another backend.

Only frames before each declared `prefix_frame_count` enter the likelihood.
Changing held-out observations therefore leaves both the shared posterior and
every reconstructed execution posterior unchanged.

## Existing identifiability guard

Hierarchical pooling does not replace the grouped-abduction identifiability guard
already present on `main`. Before pooling executions, the permitted response
coordinates and admitted nuisance directions should pass that source-frozen,
fail-closed guard. Passing it establishes only local separability, not calibrated
intervention recovery.

## Claim boundary

These additions provide inference machinery for a same-object multi-action
protocol. They do not convert existing single-interaction diagnostics into
multi-execution evidence and do not promote a new physical discrepancy
mechanism. Real claims still require preregistered independent executions,
held-out action/contact evaluation, and the declared calibration boundary.
