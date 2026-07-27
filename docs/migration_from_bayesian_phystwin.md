# Migration from Bayesian-PhysTwin

Causal4D began as an independent subproject inside the Bayesian-PhysTwin
repository. This repository was produced by a path-filtered history extraction
from Bayesian-PhysTwin commit
`2a0431025e7c7ede02efdc5c9d4492985bae9442`.

The extraction preserves the original commits and frozen Causal4D, Deform360,
and PokeFlex milestone tags. No experiment artifact was regenerated during the
migration.

The corresponding Bayesian-PhysTwin cleanup is commit
`c7ad36aad7e592ce8a391c9ca2d4db7389dee3ac`. The historical two-repository
stack remains reproducible through
`requirements/frozen/causal4d-0.3.0.txt`, which locks both repositories to
exact commits.

## Ownership after migration

Causal4D owns:

- causal posterior contracts and artifact serialization;
- factual intervention abduction;
- same-grasp and new-contact intervention semantics;
- counterfactual rollout construction and evaluation;
- semantic trust and exact-fallback gates;
- prospective real-data protocols and mechanism gates;
- Causal4D-specific PhysTwin diagnostics;
- Causal4D public-data adapters and frozen evidence.

Bayesian-PhysTwin owns:

- perception and observation construction;
- state and physical-parameter estimation;
- graph construction and discrepancy primitives;
- PhysTwin/Warp simulation and replay;
- the versioned physical-belief artifacts consumed by Causal4D.

Prob4D remains an independent observation/calibration project. Its artifacts
may feed Bayesian-PhysTwin and then Causal4D, but its method and claims are not
silently incorporated into either project.

## Compatibility

The Python packages remain `causal4d` and `causal4d_public`. Existing
`causal4d-*` command names are preserved. Five diagnostics formerly exposed as
`bpt-*` commands moved with their Causal4D ownership:

| Previous command | Canonical command |
| --- | --- |
| `bpt-structural-protocol` | `causal4d-structural-protocol` |
| `bpt-diagnose-phystwin-discrepancy-location` | `causal4d-diagnose-phystwin-discrepancy-location` |
| `bpt-aggregate-phystwin-discrepancy-location` | `causal4d-aggregate-phystwin-discrepancy-location` |
| `bpt-diagnose-phystwin-propagated-state` | `causal4d-diagnose-phystwin-propagated-state` |
| `bpt-aggregate-phystwin-propagated-state` | `causal4d-aggregate-phystwin-propagated-state` |

Frozen tags continue to reproduce their historical tree. New development uses
`bayesian_phystwin.causal4d_provider_v2` for immutable replay requests,
`bayesian_phystwin.causal4d_provider_v1` only for frozen scientific compatibility,
and the BPT compatibility range `>=0.4,<0.5`. See
[bayesian_phystwin_provider.md](bayesian_phystwin_provider.md) for the
development and frozen-install policies.
