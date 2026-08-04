# Deform360 Source Prefix-Kinematics Diagnostic

Status: implementation and source-only lock complete; the registered Deform360
replication remains terminally negative and target-closed.

## Motivation

The frozen six-object source-backend study found that zero of six objects passed
the competence gate. Even an episode-specific, strain-constrained candidate beat
constant persistence on only 6 of 28 source episodes with an admissible
candidate. The sparse official-Warp adapter initializes every object node with
zero velocity at the prefix endpoint, although the endpoint usually occurs
during active robot motion.

This diagnostic tests one narrow failure hypothesis before changing material
representation, contact logic, discrepancy dynamics, thresholds, or target
access:

> Does a causal estimate of object velocity from the robot prefix materially
> improve the already-selected source-episode simulation?

A negative result rules out missing initial velocity as a large, general repair
for the current sparse backend. A positive source-only result identifies a
backend component worth evaluating in a separately registered replication; it
does not reopen the frozen target split.

## Locked comparison

The exact controls are stored in
[`configs/causal4d_public/deform360_prefix_kinematics_v1.json`](
../configs/causal4d_public/deform360_prefix_kinematics_v1.json).
The lock binds:

- the canonical replication config identity;
- the complete frozen source-backend milestone manifest;
- the terminal source-backend decision;
- five objects with complete source grids (`002`, `081`, `083`, `085`, `170`);
- all 30 available source episodes, including the two blanket episodes whose
  original candidates violated the strain limit;
- one fixed candidate per episode, selected before the new rollouts as the best
  strain-valid source candidate, or the best finite source candidate when no
  strain-valid candidate existed; and
- the policy parameters, baseline reproduction tolerances, and decision gate.

The volumetric squirrel object is excluded because its source geometry failed
before candidate-grid construction. The diagnostic cannot answer a geometry
failure with an initial-condition change.

## Causal velocity evidence

For each controller, the method reuses the taxel subset and contact registration
sealed in the source grid. It evaluates the registered contact patch at frames
`t-3` and `t`, where `t` is the prefix endpoint, and computes

```text
u_c = (p_c(t) - p_c(t-3)) / (3 * dt).
```

Only robot states at or before the prefix endpoint are loaded or validated for
`u_c`; the future robot-state suffix is ignored. A controller is considered
recently active when its source tactile schedule is active in the three-frame
causal tail. Controller and node speeds are radially capped at the
locked 2 m/s bound.

The three policies are:

1. `zero_v1`: the exact registered zero-velocity baseline;
2. `global_contact_translation_v1`: every node receives the mean velocity of
   recently active contact patches; and
3. `graph_harmonic_contact_v1`: active patch velocities are propagated through
   the prefix stretch/shear graph by solving

```text
argmin_V
    100 * sum_c ||V[node(c)] - u_c||^2
  +   1 * sum_(i,j) w_ij ||V_i - V_j||^2
  + 0.1 * sum_i ||V_i||^2,

w_ij = median_edge_length / edge_length_ij.
```

The candidate parameters, controller trajectory, source contact schedule,
visual-hull scoring frames, simulator configuration, and p99 strain threshold
remain unchanged. This isolates initial velocity rather than rerunning a new
joint model search.

## Decision gate

The graph-harmonic policy is a source-supported repair candidate only when all
of the following hold:

- every zero-velocity rerun reproduces its archived source score within 0.5 mm
  Chamfer and 0.01 absolute p99 strain;
- at least 24 episodes have finite zero and graph-harmonic results;
- mean Chamfer improves by at least 5% over the rerun zero baseline;
- at least 60% of common finite episodes are strain-valid wins over zero; and
- the number of strain-valid episodes does not decrease.

The rigid-translation policy is a control and cannot satisfy the primary gate.
All episodes and failures remain in the artifact; no result-dependent object or
execution exclusion is allowed.

## Information and claim boundary

This is a diagnostic over source episodes whose future geometry and tactile data
were already opened by the completed source-backend study. It reads no
calibration outcome, target prefix, target future geometry, or target future
tactile stream. It cannot alter the frozen replication decision, admit target
access, change the physical 36-execution evidence count, or support a target
prediction claim.

## Reproduction

Run the unit and contract checks:

```bash
python -m pytest -q \
  tests/test_deform360_prefix_kinematics.py \
  tests/test_deform360_prefix_kinematics_diagnostic.py
```

Run the source diagnostic on the research runner with the pinned Deform360,
BayesianPhysTwin, and official PhysTwin environments:

```bash
python scripts/remote/run_deform360_prefix_kinematics.py \
  --data-root /path/to/deform360-replication-derived \
  --bayesian-phystwin-repo /path/to/BayesianPhysTwin \
  --deform360-repo /path/to/deform360 \
  --official-phystwin-repo /path/to/PhysTwin \
  --output runs/deform360-prefix-kinematics/result.json
```

The `Deform360 source prefix kinematics` workflow provides the same locked
execution and uploads the result, runtime sidecar, log, and checksums.
