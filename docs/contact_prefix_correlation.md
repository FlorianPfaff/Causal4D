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

The implementation is CPU/NumPy based, so the full `300:320` panel runs on a
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

## Scientific boundary

This is an exploratory controlled diagnostic. It cannot revise the frozen five-seed
result, the independent topology panel, the concentration-softening result, the
registered exact-node gate, or the locked 36-execution physical experiment. Target
outcomes are not used for candidate selection.
