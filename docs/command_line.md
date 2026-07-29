# Command-line interface

Causal4D provides one grouped executable while retaining every historical
`causal4d-*` entry point for frozen manifests and compatibility.

```bash
causal4d --help
causal4d --version
causal4d commands list
causal4d commands list --json --include-legacy
```

The grouped routes cover the stable benchmark, protocol, evidence, and
calibration workflows:

```bash
causal4d benchmark counterfactual --output-dir runs/counterfactual
causal4d benchmark latent-contact --output-dir runs/latent-contact
causal4d protocol real validate-protocol configs/causal4d/sloth_multi_action_v1.json
causal4d protocol freeze validate method_freeze.json protocol.json checkout/
causal4d evidence observation-lineage validate observation.npz twin_belief.npz
```

Command modules are imported only after a route is selected. Therefore root
help, version reporting, and registry inspection stay independent of optional
Bayesian-PhysTwin, Warp, vision, and GPU dependencies.

## Registry and migration

The registry records the route, Python target, lifecycle, owner, optional
extras, and historical executable name. Inspect one entry with:

```bash
causal4d commands describe protocol/real
causal4d commands migrate causal4d-real-protocol
```

`commands migrate` reports the preferred grouped spelling without changing the
historical executable. To invoke an installed compatibility entry point through
the root command, use its suffix after `legacy`; all remaining arguments are
passed through unchanged:

```bash
causal4d legacy real-protocol --help
```

Historical executable names remain runnable and continue to be stored verbatim
in frozen result manifests. New stable command families should add a grouped
route instead of adding another top-level executable. A legacy script may be
removed from ordinary installations only in a versioned compatibility change;
frozen tags and recorded environments must remain reproducible.
