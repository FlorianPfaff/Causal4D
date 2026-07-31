# Stage-specific provenance identities

Causal4D version-1 artifacts retain one complete `CausalContext` containing
`O-`, `O+`, `u_obs`, and `u_cf`. That contract is preserved for frozen
milestones and existing result bundles.

For new provenance and future version-2 contracts, the three logically distinct
stages can now be content-addressed independently with
`causal4d.stage_provenance`:

```text
FactualEvidenceContext
├── O-
├── admitted O+ response prefix
└── known observed command u_obs

CounterfactualQueryContext
├── do(u_cf)
├── contact-resampling policy
└── optional semantic/query-node specification

EvaluationTarget
└── held-out observation suffix
```

The separation prevents an unchanged factual input from receiving a different
stage identity merely because a held-out target or alternative query changed.
It is additive provenance infrastructure: it does not change a posterior,
likelihood, rollout, registered analysis, or frozen artifact identifier.

## Construction from version-1 contracts

```python
from causal4d.stage_provenance import (
    build_counterfactual_query_context,
    build_evaluation_target,
    build_factual_evidence_context,
    validate_stage_contexts,
)

factual_context = build_factual_evidence_context(
    legacy_context,
    observations,
    observed_actions,
    evidence_frame_stop=prefix_stop,
)
query_context = build_counterfactual_query_context(counterfactual_query)
target = build_evaluation_target(
    legacy_context,
    observations,
    target_frame_start=prefix_stop,
)
validate_stage_contexts(factual_context, query_context, target)
```

`build_factual_evidence_context` verifies `O-` and the complete known
`u_obs` trajectory against their version-1 digests, but reads and hashes only the
admitted `O+` prefix. It can therefore be sealed before the held-out target is
opened.

`build_counterfactual_query_context` copies only the query action, contact
policy, language, and selected query nodes. The target-bearing version-1 context
is deliberately not retained in the new identity.

`build_evaluation_target` is called only after evaluation data are available. It
verifies the version-1 `O+` digest and addresses exactly the held-out suffix.

## Required invariants

The regression tests enforce the following behavior:

| Change | Factual identity | Query identity | Target identity |
| --- | --- | --- | --- |
| perturb held-out observations only | unchanged | unchanged | changed |
| change only `u_cf` | unchanged | changed | unchanged |
| perturb the admitted response prefix | changed | unchanged | unchanged |

`validate_stage_contexts` additionally requires one protocol and case, an exact
factual/target boundary, and a counterfactual action interval that covers the
held-out target.

## Adoption boundary

Existing `TwinBelief`, `FactualIntervention`, `CounterfactualQuery`, and
`PhysicalPosterior` artifact IDs remain version-1 identities. Frozen milestone
files must not be rewritten.

A future version-2 artifact family can bind:

1. a `TwinBeliefV2` to pre-intervention evidence;
2. a `FactualInterventionV2` to `FactualEvidenceContext`;
3. a `PhysicalPosteriorV2` to the factual posterior and
   `CounterfactualQueryContext`; and
4. an evaluation result to the physical posterior and `EvaluationTarget`.

That migration should be introduced under new schema and contract-type names,
with explicit parity tests against the frozen version-1 numerical path.
