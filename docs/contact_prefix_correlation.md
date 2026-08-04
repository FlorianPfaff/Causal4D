# Contact-prefix correlation diagnostic

## Purpose

This diagnostic addresses issue #134 without changing the registered latent-contact
estimator. The current prefix likelihood adds position, velocity, and acceleration
residual energy across frames and material nodes. Those terms are derived from the
same short trajectory and can be strongly correlated. The experiment asks whether
that repeated evidence makes the contact posterior overconfident and whether a
source-only correlation treatment improves a proper score without sacrificing
future prediction.

The evaluation panel is fixed to seeds `300:320`. Seeds `0:5`, `100:120`, and
`200:220` are excluded from both selection and evaluation.

## Compared policies

Every policy uses the exact likelihood scale, likelihood power, dynamic residual
weight, and posterior temperature selected by the existing source-topology fold
calibration.

1. `registered_exact` reproduces the current prefix update exactly and fails the run
   if the independently reconstructed residual energy produces different weights.
2. `temporal_frame_blocks` partitions each residual order into contiguous blocks and
   averages evidence within each block instead of summing every frame. Block sizes
   `2`, `3`, and `4` are selected by source-only mean node Brier score.
3. `graph_distance_node_blocks` groups material nodes by graph distance from the
   nominal contact nodes, averages within each distance stratum, and gives every
   stratum one contribution.
4. `source_residual_whitening` estimates a position/velocity/acceleration residual
   correlation matrix from source-topology cases only. It uses the nearest available
   realization state with the correct source contact nodes, preserves the registered
   likelihood scale, and selects correlation shrinkage from
   `0.10, 0.25, 0.50, 0.75` on source cases.
5. `generalized_bayes_learning_rate` scales only the likelihood contribution, not
   the prior. Rates `0.25, 0.50, 0.75, 1.00` are selected on source cases.

All policies preserve exact zero prior support.

## Reported evidence

The diagnostic retains row-level and topology-stratified results for matched and
shifted contacts. It reports:

- exact-node accuracy and confidence calibration error;
- node Brier score, truth probability, and log score;
- credible-set coverage;
- posterior entropy and entropy-effective support;
- future trajectory RMSE from the end of the observed prefix;
- effective temporal, node, and total residual block counts;
- selected source-only candidates and their full candidate score tables;
- source residual correlation matrices, eigenvalues, condition numbers, shrinkage
  records, and truth-support proxy distances;
- Python, platform, NumPy, SciPy, installed-distribution, commit, workflow, and runner
  identities;
- SHA-256 and byte counts for every claim-bearing output.

## Predeclared joint decision rule

A policy is only marked as a candidate for a separately versioned method and another
untouched panel when all of the following hold:

- shifted-contact mean Brier improves by at least `0.005`;
- matched-contact mean Brier worsens by at most `0.010`;
- matched and shifted exact-node accuracy each decline by at most `0.020`;
- matched and shifted credible coverage each decline by at most `0.050`;
- matched and shifted future trajectory RMSE each worsen by no more than the larger
  of `5%` and `0.05 mm`.

A passing diagnostic does not modify or promote the estimator in this repository.
It identifies a candidate that still requires a new method version and another
untouched evaluation panel. If no policy meets the joint rule, the result is retained
as a bounded or negative result.

## Workflow

The implementation is CPU/NumPy based, so the permanent `300:320` workflow uses a
GitHub-hosted `ubuntu-latest` runner rather than occupying the GPU workstation.
`actions/setup-python` enables the pip download cache using `pyproject.toml` as the
cache dependency path. Scientific outputs are never restored from cache.

Manual execution:

```bash
python scripts/ci/run_contact_correlation_diagnostic.py \
  --output-dir outputs/contact-prefix-correlation/results \
  --seeds 300:320 \
  --frame-block-sizes 2,3,4 \
  --whitening-shrinkages 0.10,0.25,0.50,0.75 \
  --generalized-bayes-rates 0.25,0.50,0.75,1.00
```

The workflow uploads the complete output directory, including the checksummed
scientific artifacts, console summary, environment record, and exact `pip freeze`.

## Completed fresh-panel result

GitHub-hosted capacity was queued repository-wide, so the exact predeclared panel
was also executed once on the read-only `workstation2` research runner. The
permanent cached GitHub-hosted workflow remains the portability check; the temporary
self-hosted workflow was removed after its artifact was published.

Evidence identity:

- workflow run: `30904897390`;
- evaluated PR merge commit: `3d424418f908a551a80a0e38a46994bdfa11bb1d`;
- implementation head in that merge: `e4d92ade607e0c171ee6bd40bc470795ecbfb72c`;
- base commit in that merge: `f7c30685f8b438ee86e0883efd2586bc6beadbfb`;
- artifact ID: `8890872509`;
- uploaded archive SHA-256:
  `d18a895192f443c333a3dd4e5b8c7b0ff35ae0364cd957004641b3b6b5ee31a2`;
- archive size: `131501` bytes;
- runtime: Python `3.12.13`, NumPy `2.2.6`, SciPy `1.17.1`, Causal4D `0.5.0`;
- evidence inventory: 600 evaluation rows, 300 source-selection rows, and 60
  source-whitening records.

The registered method remained best under the predeclared joint rule. No exploratory
policy improved shifted-contact Brier, and `any_promotion_candidate` was `false`.
The aggregate shifted-contact result was:

| Policy | Brier | Log score | Accuracy | Calibration error | Coverage | Future RMSE |
|---|---:|---:|---:|---:|---:|---:|
| `registered_exact` | 0.30641 | 0.81224 | 83.33% | 7.02 pp | 90.00% | 0.9656 mm |
| `source_residual_whitening` | 0.32233 | 0.78189 | 81.67% | 8.81 pp | 91.67% | 1.0021 mm |
| `graph_distance_node_blocks` | 0.34596 | 0.88397 | 80.00% | 1.37 pp | 91.67% | 0.9547 mm |
| `generalized_bayes_learning_rate` | 0.35014 | 0.88170 | 76.67% | 10.03 pp | 90.00% | 1.0133 mm |
| `temporal_frame_blocks` | 0.41652 | 1.06362 | 71.67% | 12.39 pp | 86.67% | 1.0091 mm |

Relative to `registered_exact`:

- graph-distance blocks reduced calibration error by `5.65` percentage points and
  improved future RMSE by `0.0108 mm`, but worsened Brier by `0.03955`, log score by
  `0.07173`, and exact-node accuracy by `3.33` percentage points. This is direct
  evidence that confidence-calibration error alone is not a sufficient promotion
  metric;
- residual whitening improved aggregate log score by `0.03034`, while worsening
  Brier by `0.01592`, exact-node accuracy by `1.67` percentage points, and future
  RMSE by `0.0365 mm`;
- generalized Bayes worsened shifted Brier by `0.04373` and exact-node accuracy by
  `6.67` percentage points. Source-only selection chose the unchanged rate `1.0` in
  `48/60` folds, giving little support for universal likelihood downweighting;
- temporal blocking was the clearest negative result, worsening shifted Brier by
  `0.11011` and exact-node accuracy by `11.67` percentage points.

The one scientifically useful exception was topology-specific. On shifted
`soft_block` cases, source-residual whitening improved Brier from `0.46579` to
`0.42882` and log score from `0.99259` to `0.79538`, kept exact-node accuracy at
`75%`, increased coverage from `85%` to `90%`, and increased future RMSE by only
`0.0170 mm`. The same policy worsened Brier on cloth and rope. This supports a
narrower next hypothesis: residual correlation is topology dependent, and a single
cross-topology covariance treatment is misspecified.

Claim-bearing output identities:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `contact-correlation-diagnostic.json` | 44095 | `97e1b2e92622939ca205e9e6aa7a9f257dd9c6aa01e24e87c2c973e2815de446` |
| `contact-correlation-rows.csv` | 187599 | `e2a469fe9e8e667bbabf9050b3a699a963e07746acddcac1d1440ac6af044050` |
| `contact-correlation-selection.csv` | 514051 | `88171eadd645381ea3fc1dc3bf52c88dd1ebfffc45b89262e26f4f10530f09ad` |
| `contact-correlation-whitening.csv` | 68371 | `aa4b565668b0e01323ff7da87350456ac353506cbcea92c13fdc1837ebf47bb2` |

## Scientific boundary

This is an exploratory controlled diagnostic. It cannot revise the frozen five-seed
result, the independent topology panel, the concentration-softening result, the
registered exact-node gate, or the locked 36-execution physical experiment. Target
outcomes are not used for candidate selection. The completed result does not justify
changing the frozen estimator. A topology-conditioned residual model would require a
separately versioned method and another untouched panel.
