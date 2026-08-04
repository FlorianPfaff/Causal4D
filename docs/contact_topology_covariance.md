# Topology-conditioned contact-prefix covariance diagnostic

## Purpose

The completed `300:320` contact-prefix correlation study rejected every global
likelihood correction under its predeclared joint rule. Its only positive local
signal was topology-specific: source-residual whitening improved shifted
`soft_block` proper scores while worsening cloth and rope. This diagnostic tests
that narrower hypothesis without modifying the registered latent-contact
estimator.

The method is selected only on the already-open `300:320` development panel and
is evaluated once on the disjoint, previously unopened `400:420` panel. The
registered likelihood remains an explicit baseline and every whitening family
contains an exact identity/no-op candidate.

## Predeclared panels

| Role | Seeds | Outcome access |
| --- | --- | --- |
| Open development | `300:320` | Used for covariance estimation and nested candidate selection |
| Untouched evaluation | `400:420` | Forbidden until the complete method, grid, workflow, metrics, and decision rule are committed |
| Excluded prior panels | `0:5`, `100:120`, `200:220` | Not used for fitting or evaluation |

The development and evaluation units are complete benchmark seeds. Each seed
contains rope, cloth, and soft-block folds with matched and shifted contacts.
No frame, node, coordinate, or individual residual is treated as an independent
evaluation unit.

## Compared policies

1. `registered_exact` reproduces the current prefix likelihood and registered
   posterior concentration exactly.
2. `development_global_residual_whitening` estimates one residual correlation
   matrix from all development topologies. Identity shrinkage is selected by
   leave-one-development-seed-out Brier score.
3. `development_topology_residual_whitening` estimates one residual correlation
   matrix per topology and shrinks it hierarchically toward the shared matrix:

   ```text
   R_g(lambda, rho) =
       (1 - rho) * ((1 - lambda) * R_g + lambda * R_shared)
       + rho * I
   ```

   `lambda` is selected from `0, 0.25, 0.50, 0.75, 1.00`; `rho` is selected from
   `0.10, 0.25, 0.50, 0.75, 1.00`. The exact `rho=1` candidate yields the
   identity precision and therefore reproduces the registered residual energy.

Every topology candidate is selected independently by leave-one-development-
seed-out mean node Brier score, then future trajectory RMSE, then fixed candidate
order. Correlation matrices are estimated from truth-proxy residuals only on the
opened development panel. The evaluation panel contributes no covariance,
shrinkage, likelihood-scale, concentration, prior, or threshold selection.

## Preserved estimator boundary

All policies use the registered source-fold values for:

- likelihood scale;
- likelihood power;
- dynamic residual weight;
- posterior concentration;
- contact prior and exact-zero support;
- physical parameter particles;
- prefix length and forecast boundary.

The diagnostic changes only the residual precision used by its separately
labelled whitening policies. It does not alter `ContactRolloutBank.update_weights`
or any frozen result.

## Metrics

Matched and shifted contacts are reported separately, both overall and by
topology. The retained metrics are:

- node Brier score and log score;
- exact-node accuracy and truth probability;
- confidence calibration error;
- credible-set coverage;
- posterior entropy and effective support;
- future trajectory RMSE;
- selected hierarchical weights and identity shrinkage;
- empirical, effective, and inverse correlation matrices;
- source sample counts, eigenvalues, conditioning, and truth-proxy distances.

Row-level evaluation, leave-one-seed-out candidate tables, final covariance
records, runtime identity, and SHA-256 manifests are retained.

## Joint decision rule

A candidate is not supported merely because calibration error improves. It must:

- improve aggregate shifted-contact Brier by at least `0.005` relative to the
  registered policy;
- worsen matched-contact Brier by at most `0.010`;
- reduce matched or shifted exact-node accuracy by at most `0.020`;
- reduce matched or shifted credible coverage by at most `0.050`;
- worsen matched and shifted future RMSE by no more than the larger of `5%` and
  `0.05 mm`;
- worsen shifted-contact Brier by at most `0.010` on every topology.

The topology-conditioned policy must additionally improve aggregate shifted
Brier by at least `0.002` over the global development covariance. A pass supports
only a separately versioned method candidate. It does not revise the frozen
five-seed result or enter the locked 36-execution physical protocol.

## Execution

The permanent workflow is read-only and uses GitHub-hosted CPU capacity. Manual
execution is:

```bash
python scripts/ci/run_contact_topology_covariance_diagnostic.py \
  --output-dir outputs/contact-topology-covariance/results \
  --development-seeds 300:320 \
  --evaluation-seeds 400:420 \
  --shared-correlation-weights 0,0.25,0.50,0.75,1.00 \
  --identity-shrinkages 0.10,0.25,0.50,0.75,1.00
```

The workflow uploads the summary, row-level evaluation, complete nested
selection table, covariance matrices, environment record, exact `pip freeze`,
and checksummed manifest.

## Scientific boundary

This is an exploratory controlled diagnostic. It cannot alter the registered
likelihood, prior panels, exact-node gate, real protocol, target identities,
thresholds, frozen artifacts, or physical evidence count. A negative result is a
completed boundary: it rejects topology-conditioned residual covariance under
this controlled model and redirects effort to physical discrepancy or the
registered real experiment.
