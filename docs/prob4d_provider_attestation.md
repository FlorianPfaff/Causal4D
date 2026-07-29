# Prob4D provider-v2 attestation validation

Causal4D independently validates portable observation content, causal source
lineage, and the Prob4D stream contract before binding an observation to a
`TwinBelief`. New prospective Prob4D evidence can additionally require the
self-contained calibrated provider-v2 producer contract.

The validator in `causal4d.prob4d_provider_attestation` deliberately does not
import Prob4D. It recomputes the embedded provider-manifest content address and
checks the exact neutral JSON semantics covered by the observation artifact ID.

## Compatibility mode

`load_observation_lineage` and
`validate_prob4d_causal_observation_metadata` remain backward compatible:

- historical provider-v1 observations without an attestation remain valid for
  frozen reproduction;
- any provider-v2 attestation that is present is validated and fails closed when
  malformed; and
- the resulting `ObservationLineage.provider_validation` contains a compact
  validated provider summary rather than duplicating the full manifest.

The independent checks cover:

- attestation schema and exact declared fields;
- provider-manifest SHA-256, identity, exact source revision, API version,
  repository, import boundary, and observation schema versions;
- capabilities and limitations required by the current provider-v2 contract;
- calibrated versus exploratory mode consistency;
- gauge and point calibration artifact IDs;
- covariance-root and composition-Jacobian modes; and
- matched, independently verified runtime revision evidence.

An artifact that labels itself claim-bearing is checked against the complete
strict boundary even when it enters through the compatibility loader. It cannot
use the looser entry point to bypass calibration or fallback checks.

## Prospective claim-bearing boundary

New Prob4D-to-Bayesian-PhysTwin-to-Causal4D evidence should use the strict loader:

```python
from causal4d.claim_bearing_observation_lineage import (
    load_claim_bearing_prob4d_observation_lineage,
)

lineage = load_claim_bearing_prob4d_observation_lineage(
    "observation_belief.npz"
)
```

For an already loaded lineage:

```python
from causal4d.claim_bearing_observation_lineage import (
    require_claim_bearing_prob4d_lineage,
)

lineage = require_claim_bearing_prob4d_lineage(lineage)
```

The strict boundary requires all of the following:

- an explicitly declared causal stream contract version 2 using the exclusive
  frame-stop convention; inferred legacy or joint-stream versions are
  insufficient;
- the sequential joint spanning-tree covariance model, canonical shared factor
  names, one shared factor group, and preserved cross-window covariance;
- metric-anchor uncertainty represented inside the joint factor;
- calibrated covariance metadata with valid gauge and point calibration IDs
  identical to the IDs embedded in the provider attestation;
- `gauge_calibrated_alignment_count == alignment_count`;
- both uncalibrated covariance and pointwise fallback permissions set to false;
- an empty covariance-fallback count, proving that no fallback was used; and
- matched, independently verified runtime revision evidence.

It therefore rejects:

- provider-v1 artifacts without a provider-v2 attestation;
- exploratory provider-v2 exports;
- fixed-lag products without the strict causal-stream contract;
- missing or incompatible covariance calibrations;
- partially calibrated gauge alignments;
- calibration-identity drift between metadata and attestation;
- permission for uncalibrated or pointwise covariance fallback;
- any recorded covariance fallback use;
- legacy covariance roots or composition derivatives; and
- environment-only, mismatched, unavailable, or dirty runtime provenance.

The installed-wheel three-repository workflow exercises the same artifact
through Prob4D's strict loader, Bayesian-PhysTwin's claim-bearing adapter and
update, and Causal4D's strict lineage loader. It also requires all three layers
to reject the same tampered and fallback-bearing variants.

This validation establishes provenance and contract consistency, not empirical
observation quality, covariance calibration, or downstream physical-prediction
benefit. Those remain separate held-out gates.
