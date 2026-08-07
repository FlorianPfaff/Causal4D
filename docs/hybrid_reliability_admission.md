# Claim-bearing hybrid-reliability calibration admission

`load_hybrid_reliability_calibration` establishes internal validity: exact-byte
JSON loading, closed schema, canonical source order, at least two source cases,
recomputed thresholds and source-future summaries, and agreement with the
embedded content ID.

Internal validity is necessary but not sufficient for a promoted run. A changed
source panel or policy can be made internally consistent and assigned a new
content ID. Claim-bearing use must therefore compare the reconstructed identity
to an independently frozen expected identity:

```python
from causal4d.hybrid_reliability_admission import (
    load_claim_bearing_hybrid_reliability_calibration,
)

calibration = load_claim_bearing_hybrid_reliability_calibration(
    "hybrid-reliability-calibration.json",
    expected_calibration_id=protocol.hybrid_reliability_calibration_id,
)
```

The expected ID must come from a protocol or source-manifest artifact frozen
independently of the calibration file being admitted. Copying the ID from the
same calibration file does not establish independent admission.

The separation is deliberate:

- exploratory tooling may reconstruct and inspect any internally valid
  calibration with the ordinary loader;
- promoted or claim-bearing tooling must use the admission wrapper and supply
  the independently frozen expected identity; and
- neither loader authorizes target access or changes the registered physical
  estimator.

A calibration whose policy threshold, source diagnostics, derived values, and
outer checksum were all changed consistently is still rejected by the
claim-bearing loader when it differs from the protocol-frozen identity.
