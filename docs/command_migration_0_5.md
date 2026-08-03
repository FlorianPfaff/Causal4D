# Command migration for Causal4D 0.5

Causal4D 0.5 installs exactly one executable:

```text
causal4d
```

The 67 historical `causal4d-*` executables are no longer installed. Their
functionality remains available through typed, lazily imported grouped routes.
Frozen tags and the environments recorded under `milestones/` retain their
historical command surfaces; those immutable artifacts were not rewritten.

Inspect the complete machine-readable catalog with:

```bash
causal4d commands list --json
causal4d commands list --removed-only --json
causal4d commands describe causal4d-real-protocol --json
causal4d commands migrate causal4d-real-protocol
causal4d commands validate --require-installed
```

## Migration table

| Removed executable | Current invocation | Lifecycle |
| --- | --- | --- |
| `causal4d-abduct-phystwin-intervention` | `causal4d experiment phystwin abduct-intervention` | experimental |
| `causal4d-adaptive-molmo-task-posterior` | `causal4d experiment semantic adaptive-task-posterior` | experimental |
| `causal4d-aggregate-phystwin-discrepancy-location` | `causal4d diagnostic discrepancy aggregate-localization` | diagnostic |
| `causal4d-aggregate-phystwin-propagated-state` | `causal4d diagnostic state aggregate-propagated` | diagnostic |
| `causal4d-aggregate-real-failure-attribution` | `causal4d diagnostic real failure-attribution` | diagnostic |
| `causal4d-audit-mechanism-gate-controls` | `causal4d diagnostic mechanism-gate-controls` | diagnostic |
| `causal4d-audit-parameter-support` | `causal4d diagnostic parameter-support` | diagnostic |
| `causal4d-audit-real-oracle-gap` | `causal4d diagnostic real oracle-gap` | diagnostic |
| `causal4d-build-molmo-task-posterior` | `causal4d experiment semantic build-task-posterior` | experimental |
| `causal4d-build-phystwin-canonical-graph` | `causal4d experiment rest-geometry build-canonical-graph` | experimental |
| `causal4d-calibrate-actuator-realization` | `causal4d protocol actuator-realization` | stable |
| `causal4d-contact-registration` | `causal4d protocol contact-registration` | stable |
| `causal4d-counterfactual-benchmark` | `causal4d benchmark counterfactual` | stable |
| `causal4d-counterfactual-phystwin` | `causal4d experiment phystwin counterfactual` | experimental |
| `causal4d-deform360-contact` | `causal4d public deform360 contact` | public-study |
| `causal4d-deform360-phystwin-feasibility` | `causal4d public deform360 phystwin-feasibility` | public-study |
| `causal4d-deform360-pooling-control` | `causal4d public deform360 pooling-control` | public-study |
| `causal4d-deform360-preflight` | `causal4d public deform360 preflight` | public-study |
| `causal4d-deform360-replication` | `causal4d public deform360 replication` | public-study |
| `causal4d-deform360-rope-evaluate` | `causal4d public deform360 rope-evaluate` | public-study |
| `causal4d-deform360-rope-fit` | `causal4d public deform360 rope-fit` | public-study |
| `causal4d-deform360-rope-future` | `causal4d public deform360 rope-future` | public-study |
| `causal4d-deform360-rope-observation` | `causal4d public deform360 rope-observation` | public-study |
| `causal4d-deform360-rope-oracle` | `causal4d public deform360 rope-oracle` | public-study |
| `causal4d-deform360-rope-predict` | `causal4d public deform360 rope-predict` | public-study |
| `causal4d-deform360-rope-prefix` | `causal4d public deform360 rope-prefix` | public-study |
| `causal4d-deform360-rope-sequence` | `causal4d public deform360 rope-sequence` | public-study |
| `causal4d-deform360-sam2-masks` | `causal4d public deform360 sam2-masks` | public-study |
| `causal4d-deform360-sam2-prefix` | `causal4d public deform360 sam2-prefix` | public-study |
| `causal4d-deform360-sam2-suffix` | `causal4d public deform360 sam2-suffix` | public-study |
| `causal4d-deform360-sam2-views` | `causal4d public deform360 sam2-views` | public-study |
| `causal4d-deform360-source-qa` | `causal4d public deform360 source-qa` | public-study |
| `causal4d-deform360-splat-probe` | `causal4d public deform360 splat-probe` | public-study |
| `causal4d-diagnose-phystwin-discrepancy-location` | `causal4d diagnostic discrepancy localize` | diagnostic |
| `causal4d-diagnose-phystwin-propagated-state` | `causal4d diagnostic state propagated` | diagnostic |
| `causal4d-dynamic-contact-benchmark` | `causal4d benchmark dynamic-contact` | experimental |
| `causal4d-evaluate-graph-temporal-discrepancy` | `causal4d diagnostic discrepancy graph-temporal` | diagnostic |
| `causal4d-evaluate-molmo-acceptance` | `causal4d diagnostic semantic acceptance` | diagnostic |
| `causal4d-evaluate-physical-counterfactual` | `causal4d evidence physical-counterfactual evaluate` | stable |
| `causal4d-evaluate-phystwin-molmo` | `causal4d diagnostic semantic phystwin-evaluation` | diagnostic |
| `causal4d-evaluate-rest-geometry` | `causal4d diagnostic rest-geometry evaluate` | diagnostic |
| `causal4d-execution-block-calibration` | `causal4d calibration execution-block` | stable |
| `causal4d-export-bpt-belief` | `causal4d evidence bpt-belief export` | stable |
| `causal4d-fit-semantic-trust` | `causal4d experiment semantic fit-trust` | experimental |
| `causal4d-latent-contact-benchmark` | `causal4d benchmark latent-contact` | stable |
| `causal4d-molmo-phystwin-forecast` | `causal4d archive semantic forecast-v1` | archive |
| `causal4d-molmo-phystwin-forecast-v2` | `causal4d experiment semantic forecast` | experimental |
| `causal4d-observation-lineage` | `causal4d evidence observation-lineage` | stable |
| `causal4d-phystwin-rest-geometry-transfer` | `causal4d experiment rest-geometry transfer` | experimental |
| `causal4d-phystwin-rollout-bank` | `causal4d experiment phystwin rollout-bank` | experimental |
| `causal4d-pokeflex-fixture` | `causal4d public pokeflex fixture` | public-study |
| `causal4d-pokeflex-preflight` | `causal4d public pokeflex preflight` | public-study |
| `causal4d-pokeflex-source-qa` | `causal4d public pokeflex source-qa` | public-study |
| `causal4d-pokeflex-warp-source` | `causal4d public pokeflex warp-source` | public-study |
| `causal4d-preacquisition-protocol` | `causal4d archive preacquisition v2` | archive |
| `causal4d-preacquisition-protocol-v3` | `causal4d archive preacquisition v3` | archive |
| `causal4d-preacquisition-protocol-v4` | `causal4d protocol preacquisition-v4` | stable |
| `causal4d-real-calibration` | `causal4d calibration real` | diagnostic |
| `causal4d-real-experiment-freeze` | `causal4d protocol freeze` | stable |
| `causal4d-real-protocol` | `causal4d protocol real` | stable |
| `causal4d-rest-geometry-candidate-evidence` | `causal4d diagnostic rest-geometry candidate-evidence` | diagnostic |
| `causal4d-rest-geometry-cross-action` | `causal4d diagnostic rest-geometry cross-action` | diagnostic |
| `causal4d-rest-geometry-protocol` | `causal4d experiment rest-geometry protocol` | experimental |
| `causal4d-rest-geometry-protocol-result` | `causal4d diagnostic rest-geometry protocol-result` | diagnostic |
| `causal4d-rest-geometry-register-graph` | `causal4d protocol rest-geometry register-graph` | stable |
| `causal4d-rest-geometry-source-correction` | `causal4d experiment rest-geometry source-correction` | experimental |
| `causal4d-structural-protocol` | `causal4d protocol structural` | stable |

## Compatibility policy

- Current releases install no `causal4d-*` wrappers or aliases.
- Historical names remain metadata only and can be resolved by `commands migrate`.
- Frozen releases and recorded milestone environments remain reproducible with
  their original executables.
- New commands must add a grouped route to the authoritative registry and must
  not add another `[project.scripts]` entry.
- The installed-artifact CI enumerates every grouped route and requires its
  `--help` surface to work from both wheel and source distribution installations.
