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
The two source verifiers run both before and after numerical recomputation, and
their complete reports must remain identical. This rejects a bundle that changes
during analysis. Unit-interval diagnostics admit only declared `1e-12` numerical
boundary drift and still reject material range violations.

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

## Independent-seed result

The complete `100:120` panel ran on `workstation2` in workflow run
`30892181513`. The archived artifact is `8885901047`; the uploaded archive digest
is
`sha256:9c860ff37d4a068777b0ca9b397b49f7689bd0759d7918ae1c6cc2382c13daba`.
The admitted source bundle retained manifest SHA-256
`7b832886ef45e5eb469fa6933b83613337343cdd55f37d13ef3d17c14c888769`.
Admission, exact byte identity, domain row contracts, pre/post-analysis source
stability, and recomputation parity all passed. Recomputation checked `720` exact
fields and `600` floating fields across `120` online rows with zero numerical
difference.

Across the `60` shifted-contact cases:

- exact-node recovery was `75.0%` (`45/60`), while one-hop-patch recovery was
  `100%`;
- graph-diffusion proxy recovery was `81.67%` at cosine `0.80` and `75.0%` at
  `0.90` and `0.95`;
- credible coverage was `81.67%`, mean confidence was `91.29%`, and confidence
  calibration error was `16.29` percentage points;
- all `45` exact-MAP cases improved trajectory RMSE, with mean relative gain
  `81.85%`;
- all `15` incorrect-MAP cases also improved trajectory RMSE, with mean relative
  gain `30.12%`; and
- the diagnostic categories contained `45` exact recoveries, `3` conservative
  symmetry-metric limitations, and `12` genuinely wrong or unresolved posteriors
  under the declared `0.90` force-proxy threshold.

The topology dependence is substantial:

| Topology | Exact node | One hop | Credible coverage | Incorrect-MAP trajectory gain | Diagnostic misses |
| --- | ---: | ---: | ---: | ---: | --- |
| cloth | `60%` | `100%` | `80%` | `36.55%` relative | `3` symmetry limitations, `5` unresolved |
| rope | `100%` | `100%` | `100%` | no incorrect cases | none |
| soft block | `65%` | `100%` | `65%` | `22.77%` relative | `7` unresolved |

The bounded interpretation is therefore more precise than either extreme. Exact
node identity understates useful neighboring-contact recovery, because every miss
is one hop away and every miss still yields a beneficial trajectory posterior.
However, neighboring and trajectory-beneficial does not by itself prove physical
equivalence: only three misses satisfy the conservative sensor-conditioned
symmetry category, while twelve remain unresolved under the registered diagnostic
proxy. Rope is fully identifiable in this panel; cloth and soft block retain a real
topology-dependent discrete-posterior limitation.

All categories are diagnostic. They do not revise the frozen five-seed result,
registered exact-node gate, thresholds, target IDs, or the 36-execution real
experiment. Any estimator or calibration change must be proposed separately and
tested on newly generated seeds.