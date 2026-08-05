# Prob4D factor-bundle lineage

Causal4D independently validates portable Prob4D observation-factor bundles
without importing Prob4D or BayesianPhysTwin. This is a lineage and provenance
boundary, not an estimator implementation.

## Supported schemas

The validator accepts two explicit wire representations:

- schema v3, the frozen marginal-block compatibility format; and
- schema v4, which adds an ordered joint `7K x 7K` gauge covariance and explicit
  `marginal-blocks-only` or `joint-cross-window` semantics.

Schema v3 retains its historical content-address calculation and binding
metadata. Schema v4 includes the schema version in the pair identity and binds
the covariance semantics and cross-window preservation flag into the resulting
`TwinBelief` metadata.

## Schema-v4 checks

For schema v4, Causal4D verifies all of the following before returning lineage:

- the manifest and payload use closed field sets and exact JSON scalar types;
- duplicate JSON keys and non-finite JSON numbers are rejected;
- the NPZ payload disables pickle and contains exactly the referenced arrays;
- gauge, point, covariance, probability, and mask arrays use the canonical
  `float64`, `int64`, and Boolean dtypes;
- the ordered gauge IDs match the gauge records exactly;
- the joint covariance is finite, symmetric, positive semidefinite, and has
  shape `7K x 7K`;
- every diagonal `7 x 7` block matches the corresponding gauge marginal; and
- `marginal-blocks-only` rejects any nonzero cross-window block, while
  `joint-cross-window` requires the preservation flag.

Payload paths must be safe POSIX-relative paths and may not escape through
parent components or symlinks.

## Architectural boundary

Direct factor-bundle lineage validation is useful for audits and compatibility.
The claim-bearing execution path remains:

```text
Prob4D factor bundle
    -> BayesianPhysTwin validation, physical update, guard, and exact fallback
    -> accepted or fallback TwinBelief
    -> Causal4D counterfactual inference
```

A valid factor-bundle lineage does not establish provider competence, calibrated
physical uncertainty, a beneficial Bayesian update, or intervention-prediction
accuracy.
