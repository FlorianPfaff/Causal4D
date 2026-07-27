# Bayesian-PhysTwin provider boundary

Causal4D consumes Bayesian-PhysTwin only through explicit versioned public
modules:

- `bayesian_phystwin.causal4d_provider_v1` for frozen scientific and diagnostic
  compatibility names;
- `bayesian_phystwin.causal4d_provider_v2` for all production initial and restart
  replay execution through immutable request-complete contracts;
- `bayesian_phystwin.causal4d_graph_provider_v1` for the NumPy-only spring-graph
  value type, graph construction, and released controller grouping semantics;
- `bayesian_phystwin.causal4d_artifacts_v1` for hash-locked released pickle
  inputs and immutable raw-track correspondence;
- `bayesian_phystwin.causal4d_public_provider_v1` for source-locked public-data
  diagnostics that still reuse BPT experiment semantics.

The graph module is explicitly parented to Bayesian-PhysTwin's immutable
`causal4d_provider_v2` contract. Causal4D's belief exporter, rollout-bank backend,
and resumable cache now execute replay exclusively through provider v2. Provider
v1 remains only for frozen scientific and diagnostic compatibility operations.

Production source and scripts no longer import any unversioned
Bayesian-PhysTwin implementation module. An AST allowlist makes new direct
experiment imports a blocking test failure.

## Compatibility contract

Normal development accepts Bayesian-PhysTwin versions in the range
`>=0.4,<0.5`. Compatibility is not inferred from the package version alone.
Causal4D validates two deliberately separate provider manifests:

- scientific provider API/schema version 1 for frozen compatibility names,
  fixed-anchor inference, and migrated diagnostics; and
- replay provider API/schema version 2 for typed initial/restart requests,
  immutable position/velocity trajectories, frame provenance, and stateless
  replay execution.

The scientific manifest requires its existing `TwinBelief` and `GraphBelief`
artifact schemas. The replay manifest additionally requires `ReplayRequest` and
`ReplayTrajectory` schema version 1 and every provider-v2 replay capability.

The graph provider is checked separately for:

- graph-provider API/schema version 1;
- `phystwin_spring_graph` and `controller_grouping` capabilities;
- `PhysTwinSpringGraph` artifact schema version 1;
- the exact public graph-provider identity; and
- the exact parent `bayesian_phystwin.causal4d_provider_v2` identity and API
  version 2.

The scientific manifest is loaded with
`load_bayesian_phystwin_provider_manifest()` and checked with
`validate_bayesian_phystwin_provider()`. The replay manifest is loaded with
`load_bayesian_phystwin_replay_provider_manifest()` and checked with
`validate_bayesian_phystwin_replay_provider()`. The graph manifest is loaded with
`load_bayesian_phystwin_graph_provider_manifest()` and checked with
`validate_bayesian_phystwin_graph_provider()`. A version, capability, artifact,
graph-provider, or parent-provider mismatch fails closed and is reported
explicitly.

## Execution API

Production simulation uses `PhysTwinReplayProvider` from the explicitly versioned
`causal4d_provider_v2` module. Each invocation is one immutable
`InitialReplayRequestV1` or `RestartReplayRequestV1` containing:

- a content-addressed request identifier;
- the exact simulator-configuration and initial-state identifiers;
- grouped spring log-scales and the complete controller trajectory;
- for restarts, the particle-specific endpoint position and velocity; and
- the complete requested frame interval.

Causal4D independently validates every `ReplayTrajectoryV1` response against the
request ID, configuration ID, state ID, frame IDs, timestep, shapes, and finite
position/velocity values. The resumable cache stores and hashes positions,
velocities, frame provenance, timestep, and all three identities. A cache hit can
therefore reconstruct a complete provider-v2 response without instantiating Warp.

Provider v1 is not the production replay boundary. It remains a versioned
compatibility facade for frozen diagnostics and scientific operations that have no
request-complete replay role. Graph and controller geometry remain in
`causal4d_graph_provider_v1`, which is NumPy-only and declares replay provider v2
as its parent contract.

The official rollout manifest records the scientific provider, replay-provider-v2,
and graph-provider manifests separately. It also records source-artifact hashes,
simulator/state identifiers, every request ID, exact frame provenance, and position
and velocity digests. Public-data studies additionally record the public-study
provider manifest, and Molmo query preparation requires trusted SHA-256 identities
for both `final_data.pkl` and `calibrate.pkl`. This provenance separation improves
upgrade auditability; it is not an empirical accuracy, calibration, or
causal-prediction claim.

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
each other's public contracts. An AST boundary test rejects every BPT import in
production source and scripts unless its exact module is present in the
versioned allowlist above.

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
