# Topology-aware contact-posterior diagnostics

The controlled latent-contact estimator is frozen. Independent seeds `100:120`
retain strong trajectory gains but expose topology-dependent exact-node recovery
and mild overconfidence. This diagnostic distinguishes a genuinely wrong
realized-contact posterior from a neighboring or structurally symmetric contact
hypothesis without changing the registered exact-node gate.

## Run

First produce the ordinary independent-seed bundle with the existing benchmark
surface:

```bash
python -m causal4d.cli.latent_contact_benchmark \
  --output-dir outputs/contact-posterior/independent-seeds \
  --seeds 100:120 \
  --frames 56 \
  --training-repeats 2 \
  --parameter-grid-count 5 \
  --contact-parameter-particles 12 \
  --observation-fraction 0.20 \
  --observation-noise-mm 1.5
```

Then recompute and analyze the posterior using the exact configuration and seeds
stored in that bundle:

```bash
python scripts/ci/analyze_contact_posterior.py \
  outputs/contact-posterior/independent-seeds \
  --output-dir outputs/contact-posterior/diagnostics
```

The manual `Contact-posterior diagnostics` GitHub Actions workflow runs the same
sequence on `workstation2`.

## Fail-closed source integrity

The analysis command verifies the source bundle itself before it reconstructs any
posterior. This check is part of the command and therefore also applies outside
GitHub Actions; the workflow's standalone bundle verifier is defense in depth.
The command rejects:

- a manifest with the wrong benchmark or schema, an unsafe artifact name, or an
  artifact inventory other than the six declared latent-contact payloads;
- any missing, additional, symlinked, byte-count-mismatched, or checksum-mismatched
  payload;
- duplicate JSON keys, duplicate or empty CSV headers, blank rows, and rows whose
  width differs from the declared header;
- undeclared or duplicate seed/case identities, unsupported settings or world
  conditions, inconsistent observation fractions, malformed node labels,
  noncanonical booleans or integers, and non-finite required diagnostics;
- intervention rows that do not match the online recovery cases, change source
  identities within a case, or omit either the paired `nominal_physics` or
  `latent_contact` trajectory; and
- a `success_gates.json` payload that differs from the gate record embedded in
  `summary.json`.

The resulting integrity report, including the exact source-manifest SHA-256, is
embedded in the diagnostic artifact under `source_bundle.integrity_verification`.
Rehashing a contradictory CSV or JSON payload does not bypass the semantic checks.

## Fail-closed recomputation

The diagnostic does not trust an independently reconstructed posterior merely
because it used the same seed range. After source-integrity verification it compares
every recomputed online posterior with the retained `contact_recovery.csv` rows:

- seed, object, condition, and setting keys must be identical;
- node truth, MAP node, correctness, credible coverage, delay MAP, and delay
  correctness must match exactly;
- confidence, truth probability, Brier score, joint effective sample size, and
  normalized joint entropy must agree under strict numerical tolerances.

Analysis stops on any mismatch.

## Recovery notions

The diagnostic reports several notions in parallel:

- **Exact-node recovery:** the original registered metric. Its threshold is not
  changed.
- **One-hop-patch recovery:** the optimally matched MAP nodes are at most one graph
  edge from the truth.
- **Graph-diffusion force-field proxy:** each contact node is propagated through
  `(I + alpha L)^-1`, where `L` is the unnormalised graph Laplacian and `alpha`
  is recorded in the diagnostic configuration. Cosine recovery is reported at
  fixed thresholds 0.80, 0.90, and 0.95.
- **Structural symmetry proxy:** nodes share degree, neighboring-degree, and
  all-node graph-distance signatures.
- **Sensor-conditioned symmetry proxy:** the structural signature also includes
  distances to registered sensor nodes.

The symmetry tests are conservative graph signatures, not a proof of graph
automorphism. The diffusion score is an analysis proxy, not a replacement contact
label or a claim of physical force equivalence.

## Reported evidence

The JSON, CSV, and Markdown artifacts include:

- per-topology and per-seed confusion matrices;
- exact-node, one-hop-patch, and force-field-proxy recovery;
- shortest matched graph distance;
- posterior entropy, normalized entropy, effective sample size, support size,
  credible-set size, confidence, Brier score, and truth probability;
- truth and MAP node degree;
- graph distance to the nearest registered sensor;
- rest-geometry and commanded-action-direction error;
- trajectory RMSE gain for correct and incorrect node-MAP subsets; and
- a diagnostic categorization into exact recovery, symmetry-metric limitation,
  trajectory-equivalent neighbor under the declared proxy, or genuinely wrong /
  unresolved posterior.

All categories are diagnostic. They do not revise the frozen five-seed result,
registered exact-node gate, thresholds, target IDs, or the 36-execution real
experiment. Any estimator or calibration change must be proposed separately and
tested on newly generated seeds.