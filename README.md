# Causal4D

Causal4D is a research framework for **Bayesian abduction of realized
interventions and held-out interventional prediction for deformable-object
dynamics**.

The central distinction is:

```text
commanded action u
        |
        v
realized intervention z = (actuation realization phi, contact state kappa)
        |
        v
uncertain physical rollout and observable future
```

The software implements explicit abduction, intervention, and prediction:

1. begin from an uncertain physical-twin belief;
2. infer how an observed command was physically realized from an allowed
   response prefix;
3. branch at the prefix endpoint under a held-out action;
4. propagate physical, intervention, and unresolved discrepancy uncertainty;
5. optionally reweight safe rollouts with a separately gated semantic prior.

This repository is the canonical home of Causal4D. It was extracted with
history from
[Bayesian-PhysTwin](https://github.com/FlorianPfaff/Bayesian-PhysTwin);
the migration boundary is recorded in
[docs/migration_from_bayesian_phystwin.md](docs/migration_from_bayesian_phystwin.md).

## Project Map

### Causal4D core

`src/causal4d/` owns the typed posterior contracts, controlled benchmark,
latent-contact inference, intervention abduction, counterfactual operators,
discrepancy transfer, semantic trust gates, physical validation, and
prospective mechanism gates.

### Bayesian-PhysTwin integration

[Bayesian-PhysTwin](https://github.com/FlorianPfaff/Bayesian-PhysTwin) supplies
the uncertain deformable-object twin: state and parameter particles, graph
geometry, PhysTwin/Warp replay, and perception/discrepancy artifacts. Causal4D
consumes those artifacts and owns the intervention and counterfactual
inference. The dependency points from Causal4D to Bayesian-PhysTwin.

Install the `phystwin` extra for these adapters. Core controlled benchmarks do
not require Warp or the PhysTwin checkout.

### Public-data studies

`src/causal4d_public/` contains source-locked Deform360 and PokeFlex adapters,
preflight checks, technical-failure accounting, shared-physics controls, and
the frozen public-data protocols. These studies are evidence about specific
model classes and information boundaries; they are not all positive
confirmations.

### Prob4D

[Prob4D](https://github.com/FlorianPfaff/Prob4D) is a separate, newly developed
probabilistic 4D observation and calibration feeder. It is not assumed prior
literature and is not part of Causal4D's core causal claim. Causal4D may consume
versioned Prob4D observation artifacts through a narrow interface, but camera
evidence is admitted only through source-calibrated gates with exact fallback.

## Installation

Core and public protocol code:

```bash
python -m pip install -e .
```

Development tools:

```bash
python -m pip install -e ".[dev]"
```

Bayesian-PhysTwin adapters:

```bash
python -m pip install -e ".[phystwin]"
```

The `phystwin` extra pins the compatible Bayesian-PhysTwin provider revision.
Visual, Warp-runtime, and actuator-calibration dependencies remain separate so
the controlled benchmark stays lightweight.

## Quick Start

Run the controlled counterfactual benchmark:

```bash
causal4d-counterfactual-benchmark \
  --output-dir runs/causal4d-counterfactual-v1
```

Run the latent-contact benchmark:

```bash
causal4d-latent-contact-benchmark \
  --output-dir runs/causal4d-latent-contact-v1
```

Validate the locked same-object real protocol:

```bash
causal4d-real-protocol validate-protocol \
  configs/causal4d/sloth_multi_action_v1.json
```

The full PhysTwin abduction chain is documented in
[docs/causal4d_abduction_intervention_prediction.md](docs/causal4d_abduction_intervention_prediction.md).

## Next Scientific Milestone

The controlled result has passed. The next first-paper milestone is the locked
same-object physical experiment: 18 grasp sessions, 36 command executions, and
independent-execution calibration. Primary-method development is frozen for
this result; another discrepancy mechanism, semantic component, planner, or
public-data branch cannot replace it.

After scaffolding the acquisition dataset, seal the exact clean Causal4D commit,
Bayesian-PhysTwin pin, protocol files, analysis boundary, and reporting contract:

```bash
causal4d-real-experiment-freeze seal \
  . \
  /data/causal4d-sloth-multi-action-v1/method_freeze.json \
  --frozen-by "<operator-or-principal-investigator>"
```

The experiment must report either successful transfer/calibration or a
well-powered negative result without target-informed method selection. See
[docs/causal4d_real_experiment_milestone.md](docs/causal4d_real_experiment_milestone.md)
for the freeze, acquisition, and reporting workflow.

## Evidence Boundary

- `milestones/v0.3.0-causal4d-aip/` is the frozen controlled and first real
  abduction-intervention-prediction milestone.
- The released PhysTwin interactions are diagnostic-only after their recorded
  audits; they must not be reused for further model selection.
- Deform360 and PokeFlex artifacts preserve source/target access boundaries,
  retained technical failures, and unsealable cases separately.
- Graph persistence remains the unresolved-discrepancy fallback unless a
  physical mechanism passes the prospective held-out shrinkage, prediction,
  plausibility, transfer, and calibration gates.
- MolmoMotion or another semantic prior has zero influence unless its locked
  trust gate passes; rejection gives exact physical-posterior fallback.
- The 36-execution same-object real protocol is now the decisive pending
  milestone; optional branches cannot alter or rescue its primary result.

See [docs/causal4d_paper_scope.md](docs/causal4d_paper_scope.md) for the narrow
paper claim and the other documents in `docs/` for protocol-specific details.

## Repository Layout

```text
src/causal4d/          core inference, contracts, and PhysTwin adapters
src/causal4d_public/   Deform360 and PokeFlex public-data studies
configs/               locked protocol and registration artifacts
docs/                  formulation, protocols, diagnostics, and claim limits
milestones/            immutable research milestones and evidence manifests
runs/                  small checked-in diagnostic result bundles
scripts/remote/        reproducible remote execution wrappers
tests/                 unit, protocol, parity, and artifact-boundary tests
```
