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

The positive-definite base `B` can be represented either as a dense `(D, D)`
matrix or as fixed covariance blocks `(B, C, C)` with `B * C == D`. The latter
keeps Prob4D's local 3-D covariance at `O(row_count)` storage instead of
materializing an `O(row_count^2)` matrix. The optional factor `U` has units of
metres and carries shared positive-semidefinite uncertainty across all blocks
without forming `U U^T`.

The sparse operator can represent direct coordinates, increments, differences,
projections, or other registered linear observations. Endpoint frame zero is
admitted only in a zero-sum contrast, so it cannot be reused as an additional
observation.

For finite trajectory support, use:

```python
posterior, diagnostics = posterior_weights_from_joint_observation(
    prior_weights,
    predicted_components_m,
    evidence,
    prefix_frame_count=prefix_frame_count,
    component_independent_variance_m2=component_variance,
    component_joint_covariance_m2=component_covariance,
    component_joint_covariance_factor_m=component_factor,
)
```

For a block base, `component_joint_covariance_m2` must use the same block shape.
The dense/block and low-rank component terms are additive and must represent
distinct uncertainty sources. Diagonal uncertainty on rollout scalars is
propagated through the full sparse operator. If one uncertain scalar contributes
to several observation rows, the resulting off-diagonal covariance is retained.
A scalar reused across separate declared blocks fails closed rather than silently
dropping the induced cross-block covariance.

## Prob4D adapter

`causal4d.prob4d_joint_observation.joint_observation_from_prob4d` converts a
strict, independently validated Prob4D causal observation into the scalable
representation:

```python
evidence, adapter_diagnostics = joint_observation_from_prob4d(
    descriptor,
    arrays,
    rollout_frame_ids=rollout_frame_ids,
    entity_to_node=registered_entity_mapping,
    reliability_policy="require_neutral",
)
```

The adapter:

- invokes `validate_prob4d_causal_observation_metadata` before conversion;
- requires an explicit absolute-frame-to-rollout mapping;
- requires an explicit Prob4D-entity-to-physical-node mapping;
- preserves every local `(3, 3)` covariance block;
- reshapes the single shared Prob4D gauge factor into `(3 * rows, rank)`;
- records source revision, source artifact, mappings, provider validation, and
  hashes of every reliability/composite-weight array; and
- rejects more than one factor group for the strict joint contract.

Prob4D reliability and composite weights are not Gaussian covariance. The
default `require_neutral` policy therefore rejects any nonunit association,
prior-reliability, group-nominal, or composite-weight values. An exploratory
analysis may select `record_only`, but that choice is explicit in both the
evidence metadata and diagnostics; it is not claim-bearing calibration and must
be preregistered separately.

## Numerical method

Dense evidence uses Cholesky whitening of `B`. Block evidence factorizes each
small covariance block independently. Both paths apply the matrix determinant
lemma and Woodbury identity to the rank-sized system:

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
error. Calibration, reliability treatment, and covariance selection must remain
source-only.

## Invariances and fail-closed behavior

Tests establish that:

- dense `B + U U^T` and structured `B, U` calculations agree;
- dense and materialized block-diagonal bases agree;
- a simultaneous permutation of rows, covariance, factor, and operator leaves
  scores and posterior weights unchanged;
- changing metres to millimetres leaves normalized posterior weights unchanged;
- exact zero prior support is never recreated;
- shared selector variance produces the correct cross-row covariance;
- the repository's Prob4D joint-observation fixture passes through the real
  independent validator and the full-joint likelihood; and
- malformed dimensions, future-prefix access, non-positive-definite covariance,
  invalid endpoint reuse, ambiguous mappings, multiple factor groups, and silent
  reliability reinterpretation fail closed.

This module is prospective inference infrastructure. It does not modify the
registered 18-session/36-execution estimator, count physical evidence, authorize
confirmatory execution 1, or change `claim_ready`.
