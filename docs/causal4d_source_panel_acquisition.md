# Physical source-panel acquisition

The registered pre-acquisition amendment requires 12 source-only physical
executions before the method freeze and before any confirmatory target execution.
They use four command profiles with three independent reset/grasp sessions per
profile at the registered upper-torso contact.

This panel estimates repeatability and source-only mechanism signatures. It is
not part of a confirmatory fold and does not increment the `0/36` confirmatory
evidence count.

## Safety boundary

The source-panel control surface enforces these invariants:

- the protocol, v2, v3, and v4 registration chain must validate;
- execution and session identities are taken from that chain, not supplied by an
  operator;
- completed manifests must form the exact registered prefix;
- the next execution includes its complete registered command profile;
- every referenced artifact is an ordinary file below the dataset root and is
  checked by SHA-256 and byte count before publication;
- publication creates `manifest.json` exactly once and never overwrites it;
- target-outcome fields are forbidden recursively;
- templates do not count as completed evidence; and
- source-panel completion requires all 12 manifests plus file-hash verification.

An invalid template, an out-of-order manifest, an unexpected execution entry, a
stale digest, or a malformed completed manifest fails closed.

## Scaffold once

First create the registered confirmatory dataset structure and the separate
pre-acquisition evidence templates:

```bash
causal4d protocol real scaffold \
  configs/causal4d/sloth_multi_action_v1.json \
  /data/causal4d-sloth-multi-action-v1

causal4d protocol readiness scaffold \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1
```

The readiness scaffold writes one immutable worksheet template per registered
source execution:

```text
preacquisition/source_panel/executions/<execution-id>/manifest.template.json
```

Do not rename, edit in place, or promote that worksheet by a filesystem move.
Create a separate completed JSON file for validation and exactly-once
publication.

## Inspect the next execution

Before every source session, derive the current status from disk:

```bash
causal4d protocol readiness source-panel-status \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  --verify-file-hashes \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/preacquisition/source-panel-status.json
```

The report identifies `next_execution` and includes the exact registered
profile, contact, realization condition, replicate, execution ID, and independent
session ID. Use that record as the operator instruction. Do not choose another
profile or repair the ordering manually.

A valid incomplete panel is an expected state. With `--require-complete`, the
command returns:

- `0` when all 12 source executions validate with file hashes;
- `3` when the panel is valid but incomplete; and
- `2` when present evidence is malformed or contradictory.

## Acquire and describe one source execution

Use a fresh reset and fresh grasp for the displayed execution. Store raw sensor,
controller, timing, registration, gripper, contact, and technical-quality
artifacts below its registered execution directory. Preserve technical failures;
do not silently replace a registered source execution.

Copy the corresponding worksheet to an operator staging path and complete only
its values. A completed `SourcePanelExecutionManifest` must retain the exact
schema and registered identities and state:

```json
{
  "status": "complete",
  "fresh_reset_and_fresh_grasp": true,
  "confirmatory_fold_member": false,
  "target_outcomes_used": false,
  "included": true,
  "quality_gate_failures": [],
  "started_at_utc": "<UTC ISO-8601 timestamp>",
  "ended_at_utc": "<UTC ISO-8601 timestamp>",
  "artifacts": [
    {
      "path": "preacquisition/source_panel/executions/<id>/<artifact>",
      "sha256": "<64 lowercase hexadecimal characters>",
      "bytes": 123
    }
  ]
}
```

The other protocol, plan, execution, and session fields must remain byte-for-byte
consistent with the worksheet. `artifacts` must be nonempty. Descriptor paths are
relative to the dataset root; absolute paths, `..`, symlinks, directories, stale
hashes, and incorrect byte counts are rejected.

If a preregistered quality gate fails, retain the files and failure record for
operator review. Do not set `included=true` and do not publish a manifest that
pretends the gate passed.

## Publish exactly once

Validate every referenced artifact and atomically create the final registered
manifest:

```bash
causal4d protocol readiness source-panel-publish \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/staging/<execution-id>.json
```

The publisher admits only the exact `next_execution`. It validates the completed
JSON through the same source-panel contract used by final readiness, verifies
all artifact hashes, fsyncs the temporary manifest, and creates:

```text
preacquisition/source_panel/executions/<execution-id>/manifest.json
```

The destination must not already exist. A second publication, an out-of-order
execution, or a failed temporary validation leaves the final path untouched.
The command then recomputes the complete source-panel status and reports the next
registered execution.

## Complete and seal the source-panel gate

After the twelfth publication, require terminal validation:

```bash
causal4d protocol readiness source-panel-status \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  --verify-file-hashes \
  --require-complete \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/preacquisition/source-panel-status.json
```

Only then complete the `signature_panel.json` gate record, bind all 12 final
manifest descriptors as evidence, and obtain the registered independent approval:

```bash
causal4d protocol readiness seal-gate \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  signature_panel_complete \
  --approved-by "<registered-gate-approver-id>"
```

The actuator synchronization, support/gravity registration, and nonconfirmatory
end-to-end dry-run gates remain separate prerequisites. All four operational
gates and the operator registry must predate the method freeze. The software
environment gate must follow the freeze and independent attestation.

## Evidence interpretation

A green source-panel status means only that the registered 12 source executions
exist in order and their bound files validate. It is pre-acquisition evidence,
not confirmatory performance evidence. It cannot change a method, threshold,
exclusion, target identity, or paper claim, and it cannot authorize execution 1
by itself. Confirmatory collection remains forbidden until the final readiness
status reports both:

```text
ready=true
first_confirmatory_execution_allowed=true
```
