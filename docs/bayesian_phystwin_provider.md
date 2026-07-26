# Bayesian-PhysTwin provider boundary

Causal4D consumes Bayesian-PhysTwin through the versioned public module
`bayesian_phystwin.causal4d_provider_v1`. It no longer imports underscore-
prefixed Bayesian-PhysTwin functions or modules.

## Compatibility contract

Normal development accepts Bayesian-PhysTwin versions in the range
`>=0.4,<0.5`. Compatibility is not inferred from the package version alone.
Causal4D also validates:

- provider API/schema version 1;
- the capabilities required for artifact checksums, parameter particles,
  particle-specific endpoint state, replay, residual lifting, target validity,
  and the migrated diagnostics;
- `TwinBelief` and `GraphBelief` artifact schema version 1.

The provider manifest is loaded with
`load_bayesian_phystwin_provider_manifest()` and checked with
`validate_bayesian_phystwin_provider()`. A version, capability, or artifact
mismatch fails closed and is reported explicitly.

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

## Development installation

For sibling checkouts, install Bayesian-PhysTwin first and then Causal4D:

```bash
python -m pip install -e "../Bayesian-PhysTwin[graph]"
python -m pip install -e ".[dev]"
CAUSAL4D_REQUIRE_BPT_PROVIDER=1 python -m pytest -q \
  tests/test_bpt_provider_integration.py
```

Package-based installations may use `python -m pip install ".[phystwin]"`;
the extra encodes the supported `>=0.4,<0.5` range rather than one Git commit.
The cross-repository workflows test the current development branches against
each other's public contract.

## Frozen experiments

A frozen experiment must lock the complete two-repository stack, not combine
current Causal4D with an old provider snapshot. The historical pre-provider-API
stack remains available as:

```bash
python -m pip install -r requirements/frozen/causal4d-0.3.0.txt
```

That file locks Causal4D and Bayesian-PhysTwin to exact Git commits. Existing
milestone tags and recorded environments remain unchanged. New experiments
should record the exact BPT revision in the provider manifest in addition to
using the normal compatibility range during development.
