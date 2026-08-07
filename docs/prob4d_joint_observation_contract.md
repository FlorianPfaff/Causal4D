# Prob4D Joint-Observation Contract

Causal4D validates Prob4D observation artifacts without importing Prob4D or
Bayesian-PhysTwin. Validation therefore remains available in the lightweight
base installation and fails before a `TwinBelief` is bound to inconsistent
observation lineage.

## Supported covariance encodings

Two encodings are accepted:

1. `legacy_per_window_sim3_marginals_v1` uses the seven factor names
   `gauge_latent_0` through `gauge_latent_6` and assigns one independent factor
   group per source window.
2. `sequential_joint_spanning_tree_v1` uses dynamically ranked
   `joint_gauge_latent_####` factors and exactly one shared factor group. Rows
   from different windows therefore use different blocks of the same latent
   covariance root, preserving represented cross-window covariance.

The joint encoding must state and satisfy its full `7K` gauge dimension,
exported rank, retained covariance-trace threshold, parent-window ordering, and
cross-window covariance flags. The legacy fixed-lag block-diagonal encoding is
accepted only when it explicitly declares approximate boundary covariance and
makes no cross-window covariance claim.

## Causal and metric invariants

Every accepted artifact must bind:

- an exact Prob4D source revision and content digest;
- an external metric gauge anchor tied to the first selected window;
- an exclusive causal frame cutoff;
- every selected source window and its payload/frame digests;
- zero opened future prediction payloads;
- observation rows contained in their declared source windows.

Unknown factor names, per-window substitution of a joint factor group, rank or
parent-lineage drift, inconsistent covariance flags, insufficient retained
trace, and future-dependent lineage are rejected.

## Full-joint inference consumption

Validation and inference remain separate boundaries. After the strict causal
validator accepts an artifact,
`causal4d.prob4d_joint_observation.joint_observation_from_prob4d` can adapt the
joint encoding to `LinearJointObservationEvidence`.

The adapter keeps `local_covariance_m2` as one positive-definite `(3, 3)` block
per Prob4D row and reshapes `low_rank_factor_m` into one factor over all scalar
coordinates. Consequently, the likelihood uses

```text
blockdiag(local_covariance_m2) + U U^T
```

without allocating the dense block-diagonal matrix or discarding represented
cross-window covariance. Cholesky whitening is blockwise, and the determinant
lemma/Woodbury update operates only on the exported factor rank.

The adapter additionally requires explicit mappings from absolute source frames
to rollout indices and from Prob4D entity IDs to registered physical nodes.
Index equality is never assumed. It invokes the validator before conversion and
records the exact mappings, source revision, source artifact, validation result,
and reliability-array hashes in the resulting content-addressed evidence.

`prior_reliability`, `association_probability`, group nominal probabilities, and
composite group weights are not covariance entries. The default inference policy
therefore requires them to be neutral. A separately registered exploratory path
may select `record_only`, which records but does not reinterpret those values.
That option is not a calibration claim and cannot silently enter a claim-bearing
physical analysis.

See `docs/full_joint_observation_likelihood.md` for the inference API, numerical
method, invariance tests, and required full-versus-block covariance ablation.

## Cross-repository fixture

`tests/fixtures/prob4d_joint_observation_v1.json` is copied byte-for-byte into
Prob4D, Bayesian-PhysTwin, and Causal4D. Each repository independently computes
and checks the same artifact ID while testing its own producer, estimator, or
lineage-consumer responsibilities.

The Causal4D fixture test now also runs the admitted artifact through the real
full-joint adapter and likelihood. It checks block-local covariance, shared
factor rank, represented cross-window covariance, finite scores, and preference
for an exact matching trajectory over a perturbed control.

The fixture is an interoperability contract, not an empirical accuracy or
calibration result.
