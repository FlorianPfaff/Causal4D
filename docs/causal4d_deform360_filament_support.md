# Deform360 filament graph support boundary

## Purpose

The locked observed-reset mechanics study reproduced all 30 original prefix
baselines, but the registered filament graph extractor failed on eight resets
from seven rope episodes because its maximum-neighbour graph remained
disconnected. Only 23 complete episodes therefore remained, below the locked
24-episode support requirement. This follow-up asks one narrower question:

> Can a minimal connectivity fallback produce auditable filament graphs for the
> exact failed resets without changing any reset that the registered extractor
> already supports?

This is a structure-only source diagnostic. It does not rerun mechanics, alter
contact realization, change a physical candidate, relax the prior support gate,
or access calibration or target data.

## Frozen parent result

The study is bound to the checked-in compact record under
`milestones/deform360-reset-mechanics-v1/`. The prior result is immutable:

```text
classification = insufficient_common_episode_support
registered successful resets = 28
registered failed resets = 8
registered failure episodes = 7
```

The exact failure identities, raw reset frames, exception types, and error
messages must reproduce before the new construction can be interpreted.

## Policies

`registered_v1` is the unchanged centerline extractor. It increases k-nearest
neighbour support from 6 to at most 24 and fails if the graph is still
disconnected.

`component_bridge_v1` first calls `registered_v1`. A connected input therefore
returns the exact registered centerline and graph arrays. Only the exact frozen
maximum-neighbour disconnection error activates the additive fallback:

1. apply the registered density filter;
2. construct the maximum-24-neighbour graph;
3. identify its connected components;
4. compute the closest point pair for every component pair;
5. connect components by a deterministic component-level minimum spanning tree;
6. extract the diameter of the resulting point-level minimum spanning tree; and
7. apply the registered resampling and refinement operations unchanged.

No bridge is selected from mechanics scores. No future hull contributes to the
graph at a reset; only the contemporaneous reset hull is used.

## Predeclared admission gates

All 36 filament resets from the two source-complete rope objects are evaluated.
The candidate is admitted only when every gate passes:

- the registered extractor reproduces the exact frozen set of 28 successes and
  eight failures;
- the candidate completes all 36 resets;
- all 28 common resets have exact graph-content parity;
- every repaired reset has two to four pre-bridge components and exactly
  `component_count - 1` bridges;
- the longest bridge is at most 12 times the local maximum-kNN edge scale;
- the longest bridge is at most 25% and all bridges together at most 35% of the
  resulting centerline length;
- repaired p95 point-to-centerline distance is at most 1.5 times the same
  object's registered-success p95-q95 boundary;
- repaired centerline length is between 75% and 125% of the same object's
  registered-success median; and
- edge-length coefficient of variation is at most 0.05.

The thresholds were frozen before opening the eight repaired structures. A
failure is retained as a complete negative result and cannot be rescued by
lowering a threshold on the same source resets.

## Decisions

The first failed boundary determines the result:

1. `registered_failure_boundary_changed`;
2. `component_bridge_incomplete_support`;
3. `component_bridge_common_case_parity_failure`;
4. `component_bridge_nonlocal_structure_failure`;
5. `component_bridge_geometry_admission_failure`; or
6. `component_bridge_filament_support_admitted`.

Admission establishes only a separately versioned source-side graph candidate.
It does not authorize mechanics rescoring, target evaluation, or a change to the
registered physical method. A later mechanics study would need a new lock and
must retain both the registered graph and persistence controls.

## Execution

The permanent workflow validates the lock, exact-parity contract, decision
logic, and target-closed boundary on GitHub-hosted Linux. The source lane runs on
`workstation2`, selects the already qualified conditional reproduction Python,
locates the derived Deform360 source data, evaluates the frozen 36-reset panel,
and uploads the checksummed result and runtime sidecar.

Manual execution is:

```bash
bash scripts/remote/run_deform360_filament_support_workflow.sh \
  "$PWD" \
  "$PWD/deform360-filament-support"
```

## Information and claim boundary

The study may read the already-opened source reset hulls and source-grid
identities. It does not read future mechanics scores, calibration outcomes,
target prefixes, target geometry, or target tactile data. It does not modify the
frozen reset result, the replication result, the 36-execution estimator, or the
physical evidence count, which remains `0/36`.
