# Grouped Robust Abduction and Action-Conditioned Discrepancy

## Scope

This development path adds two opt-in models without changing frozen Causal4D
results or the default command behavior:

1. effective grouped observation factors with a nominal/outlier multivariate
   Student-t mixture and conservative composite weights;
2. action-conditioned low-rank graph-discrepancy increments around graph
   persistence.

The legacy factual-abduction likelihood remains the default. Existing milestone
commands therefore reproduce their recorded results unless
`--observation-model grouped_robust` is supplied explicitly.

## Grouped robust observation factors

`causal4d.grouped_observations.ObservationGroup` records one effective factor,
including its frame, node, covariance, source/view provenance, prior nominal
probability, outlier scale multiplier, and composite weight. A group may also
represent an explicit frame increment by naming a reference frame.

The likelihood is

\[
\ell_g = \rho_g t_\nu(r_g;0,\Psi_g)
       +(1-\rho_g)t_\nu(r_g;0,\lambda_{\rm out}\Psi_g),
\]

where the supplied covariance is converted to the corresponding Student-t scale
matrix. Consequently, grouped mode requires `nu > 2`. The dense-prefix adapter
uses one node/frame vector per group and normalizes composite weights by the
number of valid coordinates, so increasing graph density does not silently
increase the total likelihood power. Position increments are disabled by
default because they reuse position evidence; when enabled, they enter as
separate, explicitly downweighted factors.

Run the new path with:

```bash
causal4d-abduct-phystwin-intervention \
  rollout_bank.npz belief.npz final_data.pkl factual.npz evaluation.json \
  --observation-model grouped_robust \
  --robust-nominal-probability 0.95 \
  --robust-outlier-scale-multiplier 25
```

The grouped updater accepts either a static discrepancy field with shape
`(particle, node, coordinate)` or a time-varying field with shape
`(particle, frame, node, coordinate)`. A static discrepancy and its uncertainty
cancel exactly in displacement factors; temporal marginal variances are added
conservatively because cross-time covariance is not yet represented.

## Action-conditioned graph discrepancy

`ActionConditionedGraphDiscrepancyModel` implements

\[
c_{t+1}=c_t+d+B\psi_t+w_t,
\]

where `c_t` contains graph-Laplacian coefficients and `psi_t` contains measured
or abduced causal features. Drift and input maps are coordinate-specific. The
null model is graph persistence (`d=0`, `B=0`), and forecast covariance grows
from the fitted innovation covariance plus projection uncertainty.

The implementation deliberately does not inject this field into simulator
position or velocity. It supplies a source-fitted readout-discrepancy candidate
that can enter the existing cross-fitted mechanism gate. Appropriate features
include measured applied translation, signed speed, hold/return phase, registered
contact-patch coordinates, and slip probability. Features must be frozen before
confirmatory outcomes are opened.

## Claim boundary

These additions are software and source-development capabilities. They do not
promote a mechanism, establish calibration, or alter the frozen Causal4D paper
claim. Promotion still requires the preregistered source-panel shrinkage,
held-out prediction, plausibility, transfer, and calibration gates, with exact
fallback to graph persistence on rejection.
