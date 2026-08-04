# Result-bundle reproducibility contract

Causal4D distinguishes two different claims that must not be conflated:

1. **Frozen artifact identity** means the archived files, byte counts, and SHA-256 digests are exactly the ones recorded under the historical environment.
2. **Independent numerical reproduction** means a newly generated bundle has the same schema, identities, ordering, categorical decisions, and scientific gate outcomes, while explicitly declared floating-point diagnostics may differ within narrow field-aware tolerances.

A semantic reproduction never rewrites, rehashes, or replaces a frozen artifact. Byte differences remain visible in the comparison report even when the numerical reproduction passes.

## Exact frozen identity

The embedded result manifest remains the authority for the exact payload inventory:

```bash
python scripts/ci/verify_result_bundle.py \
  results/controlled/manifest.json
```

Verification is fail-closed. The manifest schema, benchmark, artifact names, byte counts, and lowercase SHA-256 digests must be valid. Every declared payload must be an ordinary file, the bundle may contain no undeclared file, directory, special entry, or symlink, and each payload must match its recorded size and digest.

Historical milestone manifests and payloads are immutable. In particular, the `v0.3.0-causal4d-aip` files are not regenerated merely to obtain cross-platform byte equality.

## Runtime-bound reproduction sidecar

A new reproduction records the numerical environment in an external sidecar. The sidecar is intentionally written outside the verified result directory so it cannot alter the bundle inventory:

```bash
python scripts/ci/write_reproduction_manifest.py \
  outputs/workstation2/frozen-controlled \
  --output outputs/workstation2/frozen-controlled.reproduction.json \
  --repository IPS-Stuttgart/Causal4D \
  --commit-sha "$GITHUB_SHA" \
  --workflow-run-id "$GITHUB_RUN_ID" \
  --runner-name "$RUNNER_NAME"
```

The sidecar binds the exact result manifest and every payload by byte count and SHA-256. It records:

- Python implementation, version, compiler, build, and executable;
- operating-system, architecture, processor, and C-library identity;
- installed Causal4D, NumPy, and SciPy distribution versions;
- NumPy and SciPy numerical-backend configuration;
- relevant BLAS, thread, CUDA, and hash-seed environment variables; and
- repository, commit, workflow-run, and runner identities when supplied.

The sidecar itself is validated against the result bundle during comparison. Its schema is exact: invented, omitted, or incorrectly typed runtime and source fields are rejected. A stale sidecar cannot be reused after any payload or embedded result manifest changes.

On a self-hosted runner, the workflow uses the runner's qualified Python tool cache but creates a fresh virtual environment for every evaluation. It intentionally avoids sharing the `setup-python` pip cache across concurrent research workflows; the exact resolved environment is preserved in `pip-freeze.txt` and the reproduction sidecar instead.

## Independent semantic comparison

```bash
python scripts/ci/compare_result_bundles.py \
  milestones/v0.3.0-causal4d-aip/results/controlled \
  outputs/workstation2/frozen-controlled \
  --actual-reproduction-manifest \
  outputs/workstation2/frozen-controlled.reproduction.json \
  --require-actual-reproduction-manifest \
  --output outputs/workstation2/frozen-bundle-comparison.json
```

The comparison first verifies each bundle independently against its own embedded manifest. It then applies comparison contract version 2.

### Exact semantic fields

No tolerance is applied to:

- result-manifest schema, benchmark, or artifact inventory;
- JSON value types, object keys, list lengths, or list order;
- JSON integers or CSV integer lexemes;
- strings, booleans, nulls, categories, CSV headers, or CSV row order;
- success-gate names, comparison operators, thresholds, individual decisions, or the overall decision; and
- internal gate truth consistency between `value`, `comparison`, `threshold`, and `passed`.

Duplicate JSON object keys, non-finite JSON numbers, non-finite CSV numeric tokens, malformed rows, and symlinked payloads are rejected.

### Floating-point fields

Ordinary finite floating-point diagnostics use:

```text
relative tolerance: 2e-12
absolute tolerance: 2e-15
```

Only the field `direction_error_deg` receives the additional absolute tolerance:

```text
near-zero direction-angle tolerance: 2e-6 degrees
```

This special case addresses the conditioning of `arccos` near a cosine of one. It does not apply to trajectory errors, probabilities, thresholds, identities, or any other field.

A tolerance cannot rescue a scientific gate. Thresholds and decisions are exact, and every gate is independently recomputed from its recorded value and operator. A value that crosses a threshold while retaining the old decision is an invalid bundle; changing the decision is an exact semantic mismatch.

## Reading the comparison report

The report keeps separate fields for separate claims:

- `result_manifests_byte_identical`: exact equality of the embedded manifests;
- `all_payload_bytes_match`: exact equality of every payload;
- `semantic_match`: success under the strict field-aware contract;
- `maximum_absolute_difference` and `maximum_relative_difference`: largest accepted or rejected floating drift;
- `maximum_direction_angle_difference_deg`: largest observed direction-angle drift;
- `mismatches`: exact structural, identity, gate, or numerical failures; and
- `reproduction_manifests`: runtime-sidecar validation and recorded runtime identity.

A valid cross-platform reproduction may therefore report:

```text
all_payload_bytes_match=false
semantic_match=true
```

That outcome is evidence of numerical reproduction, not byte identity. It cannot alter the frozen result, a registered threshold, the real-data protocol, or claim readiness.

## Additive diagnostic compatibility

The frozen controlled bundle predates the tie-aware contact diagnostics added by
Causal4D PR #118. An independently regenerated current bundle may therefore carry
a strictly enumerated set of additional contact-recovery columns and aggregate
fields. These fields are non-claim diagnostics and are projected out of the
frozen claim comparison while remaining visible in
`additive_diagnostic_fields` and in the byte mismatch.

This is not a general schema relaxation. The comparator admits only the named
contact diagnostics in `contact_recovery.csv` and
`summary.json.aggregate.contact_recovery`. Missing frozen fields, changed field
order, an unknown extra field, or any extra field elsewhere remains a semantic
mismatch. Registered gates, thresholds, decisions, identities, and all retained
claim-bearing values remain exact or tolerance-governed exactly as specified
above.
