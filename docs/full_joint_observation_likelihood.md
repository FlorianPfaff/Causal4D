# Full-joint observation likelihoods

`causal4d.joint_observation` provides an exact Gaussian likelihood for an
observation vector whose covariance spans every selected row. It complements the
robust grouped likelihood: grouped evidence remains appropriate when groups have
independent outlier mixtures or contributor-aware power caps, while the joint
path is intended for an admitted producer such as Prob4D that explicitly exports
cross-window or cross-view covariance.

## Evidence model

`LinearJointObservationEvidence` stores a sparse linear operator from a rollout
ending in `(frame, node, coordinate)` to one observation vector:

```text
y = H x + epsilon

epsilon ~ Normal(0, B + U U^T)
```

`B` is a positive-definite dense base covariance in square metres. The optional
factor `U` has units of metres and represents shared positive-semidefinite
uncertainty without materializing `U U^T`. The operator can represent direct
coordinates, increments, differences, projections, or other registered linear
observations. Endpoint frame zero is admitted only in a zero-sum contrast, so it
cannot be reused as an additional observation.

For finite trajectory support, use:

```python
posterior, diagnostics = posterior_weights_from_joint_observation(
    prior_weights,
    predicted_components_m,
    evidence,
    prefix_frame_count=prefix_frame_count,
    component_independent_variance_m2=component_variance,
    component_joint_covariance_m2=component_dense_covariance,
    component_joint_covariance_factor_m=component_factor,
)
```

The dense and low-rank component terms are additive and must represent distinct
uncertainty sources. Diagonal uncertainty on rollout scalars is propagated
through the full sparse operator. If one uncertain scalar contributes to several
observation rows, the resulting off-diagonal covariance is retained.

## Numerical method

The implementation uses Cholesky whitening for the dense base covariance. For a
factor `U`, it then applies the matrix determinant lemma and Woodbury identity to
the rank-sized system:

```text
I + U^T B^-1 U
```

No covariance inverse is formed, and the structured path does not call a dense
`slogdet` routine. Evidence and component factors are concatenated before the
single joint marginalization, preserving their cross-row effects.

## Required ablation

`block_diagonalize_covariance` creates an explicit labelled block-diagonal
control. A scientific comparison should report at least:

1. independent or block-diagonal observations;
2. the admitted full joint covariance; and
3. any low-rank approximation used for scale.

The comparison should use proper log score, normalized innovations, coverage,
posterior calibration, factual prediction error, and counterfactual transfer
error. Calibration and covariance selection must remain source-only.

## Invariances and fail-closed behavior

Tests establish that:

- dense `B + U U^T` and structured `B, U` calculations agree;
- a simultaneous permutation of rows, covariance, factor, and operator leaves
  scores and posterior weights unchanged;
- changing metres to millimetres leaves normalized posterior weights unchanged;
- exact zero prior support is never recreated;
- malformed dimensions, future-prefix access, non-positive-definite covariance,
  invalid endpoint reuse, and incomplete operator metadata fail closed.

This module is prospective inference infrastructure. It does not modify the
registered 18-session/36-execution estimator, count physical evidence, authorize
confirmatory execution 1, or change `claim_ready`.
