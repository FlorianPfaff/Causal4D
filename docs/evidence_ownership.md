# Cross-stage evidence ownership

Causal4D can consume a Bayesian physical-twin belief and then multiply that
posterior by independent actuator or contact-sensor factors. The second stage is
valid only when the supposedly independent factor has not already influenced the
BayesianPhysTwin state update, directly or through a correlated proxy.

`causal4d.evidence_ownership` provides an opt-in, content-addressed boundary for
new experiments. It does not change any frozen estimator or historical artifact.

## Contract

Each `EvidenceConsumptionV1` records:

- the exact consumed artifact ID and the underlying raw-factor ID;
- source repository, revision, and optional source-file SHA-256;
- sensor family, stream, clock, causal frame interval, and correlation group;
- one declared role: state update, actuator abduction, contact abduction, an
  explicitly joint state/intervention update, calibration only, or evaluation
  only.

`ConsumedEvidenceLedgerV1` binds the records to one protocol, case, and causal
frame stop. Its content identity is independent of input ordering. Construction
fails before inference when:

- one evidence artifact or raw factor appears twice;
- identical source bytes are relabelled as different evidence;
- evidence crosses the admitted causal prefix;
- one correlation group is consumed in incompatible posterior stages; or
- an explicitly joint factor is also consumed through an independent path.

Multiple factors may share a correlation group within one stage, because that
stage may evaluate them jointly. The ledger forbids only cross-stage independent
multiplication. A genuinely joint state/intervention likelihood must be represented
by the single `joint_state_intervention_update` role.

## Strict independent-sensor path

The existing sensor-factorized abduction API remains unchanged. New
claim-bearing experiments can use:

```python
from causal4d.evidence_ownership import (
    ConsumedEvidenceLedgerV1,
    evidence_consumption_for_independent_sensor,
    strict_reweight_factual_intervention_with_independent_sensors,
)

ledger = ConsumedEvidenceLedgerV1(
    protocol_id=factual.context.protocol_id,
    case_id=factual.context.case_id,
    causal_frame_stop=factual.evidence_frame_stop,
    entries=state_update_consumptions,
)
actuator_use = evidence_consumption_for_independent_sensor(
    actuator_evidence,
    source_repository="acquisition/repository",
    source_revision="exact-revision",
    correlation_group_id="independent-robot-encoder",
)
result = strict_reweight_factual_intervention_with_independent_sensors(
    factual,
    prior_evidence_ledger=ledger,
    actuator_evidence=actuator_evidence,
    actuator_consumption=actuator_use,
    predicted_actuator_positions_m=predictions,
)
```

The strict wrapper validates the proposed complete ledger before evaluating the
sensor likelihood. When the factor is component-invariant, invalid, absent, or
otherwise uninformative, it preserves both the original factual artifact and the
prior ledger exactly. When an update is informative, the returned factual
artifact embeds the complete resulting ledger and its content identity.

## Cross-repository use

For the intended `Prob4D -> BayesianPhysTwin -> Causal4D` path:

1. assign every raw observation, tactile, force, and robot-state source a stable
   raw-factor ID and correlation group at production time;
2. record all factors admitted by BayesianPhysTwin with role `state_update`;
3. pass that ledger with the `TwinBelief` orchestration record;
4. describe any actuator or wrench factor proposed for Causal4D abduction; and
5. use the strict wrapper so duplicate or correlated reuse fails before posterior
   weights are changed.

A factor used jointly by BayesianPhysTwin and Causal4D must be modelled as one
joint likelihood and recorded once. Relabelling it as two independent factors is
not admissible.

## Scientific boundary

This contract establishes auditable evidence accounting. Passing it is not
accuracy, calibration, transfer, or physical-experiment evidence. The frozen
36-execution protocol and all existing result identities remain unchanged.
