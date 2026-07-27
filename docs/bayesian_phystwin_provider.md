# Bayesian-PhysTwin provider boundary

Causal4D consumes Bayesian-PhysTwin through two versioned public modules:

- `bayesian_phystwin.causal4d_provider_v1` for belief artifacts, replay, and
  compatibility diagnostics;
- `bayesian_phystwin.causal4d_graph_provider_v1` for the NumPy-only spring-graph
  value type, graph construction, and released controller grouping semantics.

Production code no longer imports graph/controller implementations or
underscore-prefixed Bayesian-PhysTwin functions and modules directly.

## Compatibility contract

Normal development accepts Bayesian-PhysTwin versions in the range
`>=0.4,<0.5`. Compatibility is not inferred from the package version alone.
Causal4D validates the replay provider for:

- provider API/schema version 1;
- the capabilities required for artifact checksums, parameter particles,
  particle-specific endpoint state, replay, residual lifting, target validity,
  and the migrated diagnostics;
- `TwinBelief` and `GraphBelief` artifact schema version 1.

The graph provider is checked separately for:

- graph-provider API/schema version 1;
- `phystwin_spring_graph` and `controller_grouping` capabilities;
- `PhysTwinSpringGraph` artifact schema version 1; and
- the exact public provider-module identity.

The replay manifest is loaded with
`load_bayesian_phystwin_provider_manifest()` and checked with
`validate_bayesian_phystwin_provider()`. The graph manifest is loaded with
`load_bayesian_phystwin_graph_provider_manifest()` and checked with
`validate_bayesian_phystwin_graph_provider()`. A version, capability, artifact,
or provider-module mismatch fails closed and is reported explicitly.

## Execution API

The `PhysTwinReplayProvider` protocol is the execution boundary. Causal4D's
main BPT belief exporter and rollout-bank backend use only these operations:

- set grouped spring log-scales;
- set a controller trajectory;
- replay from the released initial state;
- replay from an explicit position/velocity endpoint;
- release runtime resources.

`create_official_replay_provider()` constructs BPT's Warp-backed
implementation. Existing specialized diagnostics use public compatibility
functions from the same versioned module while they are incrementally moved
onto higher-level execution protocols.

Graph and controller geometry stay outside that accelerator-facing protocol.
Causal4D imports `PhysTwinSpringGraph`, `PhysTwinSpringGraphConfig`,
`build_phystwin_spring_graph()`, `controller_hand_count()`, and
`infer_controller_groups()` only from `causal4d_graph_provider_v1`. The graph
surface depends only on NumPy and can therefore be validated in core CI without
Torch, Warp, OpenCV, or SciPy.

The official rollout manifest records both provider manifests. Frozen evidence
can thus identify the exact replay implementation and the exact graph/controller
contract independently.

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
