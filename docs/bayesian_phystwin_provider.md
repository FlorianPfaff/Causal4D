# Bayesian-PhysTwin provider boundary

Causal4D currently consumes Bayesian-PhysTwin through two versioned public
modules:

- `bayesian_phystwin.causal4d_provider_v1` for the existing belief-artifact,
  replay, and diagnostic compatibility path;
- `bayesian_phystwin.causal4d_graph_provider_v1` for the NumPy-only spring-graph
  value type, graph construction, and released controller grouping semantics.

The graph module is explicitly parented to Bayesian-PhysTwin's immutable
`causal4d_provider_v2` contract. This PR does not silently switch the existing
Causal4D rollout backend from replay provider v1 to v2; that replay migration is
a separate compatibility change. Provider v1 remains necessary for frozen and
current diagnostic paths.

Production code no longer imports graph/controller implementations or
underscore-prefixed Bayesian-PhysTwin functions and modules directly.

## Compatibility contract

Normal development accepts Bayesian-PhysTwin versions in the range
`>=0.4,<0.5`. Compatibility is not inferred from the package version alone.
Causal4D validates the current replay provider for:

- provider API/schema version 1;
- the capabilities required for artifact checksums, parameter particles,
  particle-specific endpoint state, replay, residual lifting, target validity,
  and the migrated diagnostics;
- `TwinBelief` and `GraphBelief` artifact schema version 1.

The graph provider is checked separately for:

- graph-provider API/schema version 1;
- `phystwin_spring_graph` and `controller_grouping` capabilities;
- `PhysTwinSpringGraph` artifact schema version 1;
- the exact public graph-provider identity; and
- the exact parent `bayesian_phystwin.causal4d_provider_v2` identity and API
  version 2.

The replay manifest is loaded with
`load_bayesian_phystwin_provider_manifest()` and checked with
`validate_bayesian_phystwin_provider()`. The graph manifest is loaded with
`load_bayesian_phystwin_graph_provider_manifest()` and checked with
`validate_bayesian_phystwin_graph_provider()`. A version, capability, artifact,
graph-provider, or parent-provider mismatch fails closed and is reported
explicitly.

## Execution API

The current `PhysTwinReplayProvider` v1 protocol remains the execution boundary
for Causal4D's main BPT belief exporter and rollout-bank backend. They use only
these operations:

- set grouped spring log-scales;
- set a controller trajectory;
- replay from the released initial state;
- replay from an explicit position/velocity endpoint;
- release runtime resources.

`create_official_replay_provider()` constructs BPT's Warp-backed v1
implementation. Existing specialized diagnostics use public compatibility
functions from the same versioned module while they are incrementally moved
onto higher-level execution protocols.

Bayesian-PhysTwin provider v2 already supplies immutable request-complete replay
DTOs, velocity-bearing replay trajectories, and owned replay/geometry/hash
modules. The graph child contract reuses v2's canonical package metadata and
declares v2 as its parent without re-exporting or redefining the replay
protocol. A later Causal4D replay migration can therefore adopt v2 without
changing this graph/controller contract.

Graph and controller geometry stay outside the accelerator-facing protocol.
Causal4D imports `PhysTwinSpringGraph`, `PhysTwinSpringGraphConfig`,
`build_phystwin_spring_graph()`, `controller_hand_count()`, and
`infer_controller_groups()` only from `causal4d_graph_provider_v1`. The graph
surface depends only on NumPy and can therefore be validated in core CI without
Torch, Warp, OpenCV, or SciPy.

The official rollout manifest records both the current replay-provider manifest
and the graph-provider manifest. Frozen evidence can thus identify the exact
replay implementation and the exact graph/controller contract independently.

## Development installation

For sibling checkouts, install Bayesian-PhysTwin first and then Causal4D:

```bash
python -m pip install -e "../Bayesian-PhysTwin[graph]"
python -m pip install -e ".[dev]"
CAUSAL4D_REQUIRE_BPT_PROVIDER=1 python -m pytest -q \
  tests/test_bpt_provider_integration.py \
  tests/test_bpt_graph_provider_integration.py
```

Package-based installations may use `python -m pip install ".[phystwin]"`;
the extra encodes the supported `>=0.4,<0.5` range rather than one Git commit.
The cross-repository workflows test the current development branches against
each other's public contracts. An AST boundary test also rejects future direct
imports from `phystwin_graph` or `phystwin_controller_sensitivity` in production
source and scripts.

## Frozen experiments

A frozen experiment must lock the complete two-repository stack, not combine
current Causal4D with an old provider snapshot. The historical pre-provider-API
stack remains available as:

```bash
python -m pip install -r requirements/frozen/causal4d-0.3.0.txt
```

That file locks Causal4D and Bayesian-PhysTwin to exact Git commits. Existing
milestone tags and recorded environments remain unchanged. New experiments
should record the exact BPT revision in both provider manifests in addition to
using the normal compatibility range during development.
