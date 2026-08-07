# Two-person source-panel publication receipt

A successful staged preflight proves that the next source manifest and all
referenced artifacts are valid at one instant. It does not prove that a second
person inspected the result, and it must not allow the same person to approve
and publish claim-bearing evidence under two operator aliases.

The review-receipt flow adds a registered two-person boundary before the
existing exactly-once publisher.

## Operator sequence

```text
acquire registered source execution
        ↓
hash-verify staged manifest and artifacts
        ↓
registered reviewer seals a content-addressed receipt
        ↓
distinct registered publisher reruns validation and publishes exactly once
        ↓
recompute readiness and the next action
```

The reviewer must be active in the sealed operator registry with either the
`gate_approver` or `independent_verifier` role. The publisher must be another
active registered person. Distinct operator IDs are insufficient: the two
records must have different `person_identity_sha256` values.

## Review command

After the staged preflight succeeds, a reviewer runs:

```bash
causal4d protocol readiness source-panel-review-staged \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/staging/<execution-id>.json \
  --reviewed-by <registered-reviewer-id>
```

The command reruns the complete preflight. It writes the receipt exactly once to:

```text
staging/reviews/<execution-id>.json
```

The review timestamp must follow execution completion, and the sealed operator
registry must predate the review. The receipt binds:

- protocol and pre-acquisition amendment identities;
- execution and independent-session identities;
- staged manifest path, SHA-256, and byte count;
- source execution completion timestamp;
- portable and host-local staged-preflight identities;
- the source-panel evidence identity before publication;
- the exact operator-registry identity;
- reviewer operator ID, person identity, and active roles;
- review timestamp; and
- the explicit no-target-outcomes and no-method-change boundary.

The receipt carries its own canonical `artifact_sha256`. Publication refuses a
modified or relocated receipt.

## Publication command

A distinct registered publisher runs:

```bash
causal4d protocol readiness source-panel-publish \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/staging/<execution-id>.json \
  --review-receipt \
  /data/causal4d-sloth-multi-action-v1/staging/reviews/<execution-id>.json \
  --published-by <registered-publisher-id>
```

The claim-bearing CLI has no receipt-free route. Before publication it:

1. reruns the complete staged preflight;
2. verifies the receipt path and canonical digest;
3. verifies every receipt field against the current preflight;
4. revalidates the sealed operator registry;
5. resolves reviewer and publisher as active operators;
6. proves person-level independence;
7. checks receipt stability while validation runs; and
8. invokes the existing exactly-once publisher, which validates all source
   artifacts again immediately before the final write.

The publication result includes both operator identities, the review-receipt
file descriptor, the bound preflight evidence identity, and
`independent_people=true`.

## Failure handling

A receipt becomes stale if the staged manifest, any referenced artifact, the
source-panel prefix, or the operator registry changes. Generate a fresh
preflight and a new receipt. A receipt path is non-overwriting; do not edit or
replace an existing receipt to rescue a failed publication.

Self-review, duplicate person identities under different operator IDs, inactive
operators, unsupported reviewer roles, review before execution completion,
receipt-field drift, target-outcome fields, symlinks, and noncanonical receipt
paths fail closed before the final manifest can be created.

## Scientific boundary

The receipt does not make an execution valid, increment evidence count, reserve
the next source slot, alter the estimator, or authorize confirmatory collection.
It records only that a registered second person reviewed the current source
preflight before a distinct registered person invoked exactly-once publication.
