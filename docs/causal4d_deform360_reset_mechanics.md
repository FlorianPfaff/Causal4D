# Deform360 observed-reset mechanics competence

## Purpose

The frozen Deform360 source-backend study failed before target access. A later
causal prefix-velocity diagnostic reproduced the zero-velocity baseline but did
not improve it. The next narrow question is therefore not another initial-state
search:

> When the sparse official-Warp backend is reset to an observed source geometry,
> can its fixed physical candidate beat constant persistence over short horizons?

This diagnostic separates an initialization/state-estimation failure from an
instantaneous or accumulating mechanics/contact-realization failure. It does not
change the registered Causal4D estimator, the Deform360 replication decision, or
the 36-execution physical protocol.

## Locked source panel

The diagnostic uses the same five source-complete objects and all 30 already-opened
source episodes as the completed prefix-kinematics study:

- `002-rope-silk`;
- `081-stripe-rope`;
- `083-blanket-cloth`;
- `085-scarf-cloth`; and
- `170-spider`.

For each episode, it selects the existing quality-constrained source oracle from
the frozen 200-candidate source grid, falling back to the finite source oracle only
when no quality-valid candidate exists. No new parameter selection is performed.
The exact object set, protocol identity, source milestone manifest, terminal source
decision, thresholds, and information boundary are content-addressed in
`configs/causal4d_public/deform360_reset_mechanics_v1.json`.

## Availability-only reset ladder

Let `f_0, ..., f_n` be the available source hull frames and let the largest
registered horizon contain six future hull observations. Three reset ordinals are
chosen deterministically from the eligible prefix:

```text
0,
integer midpoint of the eligible range,
latest ordinal with six future observations.
```

Only hull availability determines these ordinals. Chamfer errors, strains,
physical predictions, and target identities cannot influence reset selection.
The first reset is the original source prefix endpoint and must reproduce the
archived candidate before the diagnostic can be interpreted.

At every reset, the backend is rebuilt from the observed hull at that frame. The
contact patch is registered from the contemporaneous robot pose and object graph;
the future controller trajectory and source tactile schedule are then replayed.
Initial object velocity remains exactly zero. This is intentional: the preceding
source study already rejected global and graph-harmonic causal velocity repairs.

## Metrics and statistical unit

The candidate and a constant-persistence baseline are scored after the next:

- one available hull observation;
- three available hull observations; and
- six available hull observations.

Each reset also retains full-remainder Chamfer and strain diagnostics. For every
registered horizon, reset scores are first averaged inside one episode. The episode
is the statistical unit; three resets are never treated as three independent
samples.

The retained metrics include:

- symmetric Chamfer distance;
- persistence-relative improvement;
- episode win fraction;
- horizon duration in raw frames and seconds;
- p99 and maximum relative edge strain;
- quality-valid fraction;
- prefix-baseline reproduction error; and
- per-object and per-episode records through the complete result artifact.

## Predeclared decision

Every horizon must satisfy all of the following:

- at least 24 complete source episodes;
- at least 5% equal-episode mean improvement over persistence;
- at least 60% episode wins;
- at least 90% quality-valid reset rollouts; and
- successful reproduction of every original prefix baseline within `0.5 mm`
  Chamfer and `0.01` p99-strain tolerance.

The result is classified by the first failed boundary:

1. `baseline_reproduction_failure` — no interpretation is permitted;
2. `instantaneous_mechanics_or_contact_realization_failure` — the first horizon
   fails despite an observed reset;
3. `multi_step_dynamics_accumulation_failure` — shorter horizons pass but a later
   horizon fails; or
4. `observed_reset_mechanics_competence_supported` — all registered reset horizons
   pass, redirecting the next study toward initialization, state estimation, or
   contact-state inference.

A passing result supports only a separately versioned mechanism candidate. It does
not promote the current backend to target evaluation and does not alter any frozen
method.

## Execution

The hosted contract lane checks formatting, typing, the immutable lock, aggregation
semantics, target-closed validation, and the relevant replication contracts. The
GPU lane uses the same pinned conditional reproduction runtime as the completed
prefix-kinematics study and verifies the exact BayesianPhysTwin, Deform360, official
PhysTwin, dataset, protocol, and source-milestone identities.

Manual execution on the research runner is:

```bash
bash scripts/remote/run_deform360_reset_mechanics_workflow.sh \
  "$PWD" \
  "$PWD/_bpt" \
  "$PWD/_deform360" \
  "$PWD/_official_phystwin" \
  "$PWD/deform360-reset-mechanics"
```

The output directory contains `result.json`, `result.runtime.json`, the runtime
selection record, logs, and `SHA256SUMS`.

## Information and claim boundary

The diagnostic may read already-opened source candidate outcomes, source future
geometry for scoring, source robot trajectories, and source tactile schedules. It
must not read calibration outcomes, target prefixes, target future geometry, or
target tactile data. It does not modify the registered replication result, the
frozen 36-execution estimator, or the physical evidence count.

A negative result is complete evidence for the current sparse representation. It
should redirect the next mechanism study toward finite-area/moving contact,
support and friction, topology-specific mass and bending, or volumetric constraints
rather than another parameter or initial-velocity sweep on the same source cohort.
