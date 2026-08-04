# Result-bundle reproducibility

Causal4D uses two separate reproducibility contracts. They answer different
questions and must not be conflated.

## Frozen artifact identity

A frozen artifact is identified by its exact archived bytes and recorded SHA-256
digest under the historical environment. This contract is used for release
archives, milestone inventories, evidence citations, and chain-of-custody checks.

A later run on another supported numerical stack is not allowed to rewrite,
round, reserialize, or rehash the frozen payload merely to make its bytes match.

## Independent numerical reproduction

An independent reproduction may differ in final-decimal floating-point text while
remaining scientifically identical. The field-aware comparator therefore requires:

- the same files, JSON keys, CSV headers, row order, list order, and categorical
  values;
- exact schema versions, seeds, counts, frame indices, and other discrete numeric
  fields;
- exact registered gate names, comparisons, thresholds, and reported decisions;
- internally consistent gate records whose value is on the reported side of the
  threshold;
- floating-point agreement under the declared relative and absolute tolerances;
- a separate near-zero direction-angle tolerance for the conditioning of
  `arccos`.

A tolerance can never change, revise, or rescue a scientific gate. A regenerated
value that crosses a registered threshold is a semantic mismatch even when a
caller supplies deliberately loose generic tolerances.

Run the comparison with:

```bash
python scripts/ci/compare_result_bundles.py \
  milestones/v0.3.0-causal4d-aip/results/controlled \
  regenerated-controlled \
  --output regenerated-controlled-comparison.json
```

The report exposes `all_payload_bytes_match` and `semantic_match` separately. It
also records the tolerance-policy identifier, runtime Python/platform metadata,
NumPy and SciPy versions when installed, numeric-policy counts, gate-check count,
maximum drift, and every retained mismatch up to the reporting limit.

## Field classes

### Exact structure and values

The following are exact:

- file presence and supported file names;
- JSON object keys and array order;
- CSV headers, row count, and row order;
- strings, booleans, nulls, method labels, object identities, and conditions;
- schema versions, seeds, counts, discrete delays, and frame indices;
- gate names, comparison operators, thresholds, and decisions.

### Field-aware floating-point values

Ordinary calculated floating-point values use the comparator's declared relative
and absolute tolerances. Near-zero `direction_error_deg` values use the separately
declared angle tolerance because machine-level cosine differences are amplified by
`arccos`.

Raw values remain unrounded for all calculations. Display rounding, when used in
Markdown or figures, is not a scientific comparison contract.

## Adversarial guarantees

The regression suite demonstrates that semantic comparison rejects:

- substantive trajectory changes;
- reordered CSV rows;
- schema drift;
- changed registered thresholds;
- changed or internally inconsistent gate decisions; and
- threshold crossings hidden behind permissive numeric tolerances.

It separately demonstrates that a known near-zero direction-angle perturbation can
pass the semantic contract while remaining explicitly non-byte-identical.

## Environment evidence

The comparison report records the environment executing the comparison. Producers
of new deterministic bundles should additionally retain their Python, NumPy,
SciPy, platform, BLAS/LAPACK or numerical-backend identity, and applicable
Torch/Warp/CUDA identities in the surrounding run manifest or workflow evidence.

This policy concerns reproducibility and serialization only. It does not modify
the frozen estimator, result values, registered protocol, target-access boundary,
or scientific claim.
