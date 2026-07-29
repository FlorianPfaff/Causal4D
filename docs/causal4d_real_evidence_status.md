# Real-Evidence Status and Claim-Readiness Gate

The same-object multi-action protocol contains 18 same-grasp sessions and 36
preregistered executions. Scaffolded files are acquisition templates, not
evidence. The version-2 status contract therefore separates acquisition
progress, evidence completeness, statistical analysability, and claim readiness.

## Required evidence tree

A claim-ready dataset contains the locked protocol and acquisition schedule plus:

- `object_registration.json` and every hashed canonical contact-node set;
- approved schema-3 `contact_registration.json`, including multiview overlays,
  rejected attachment candidates, independent reviews, and source checksums that
  bind the simple registration and each canonical node set;
- `slip_pilot.json` with the preregistered bounded-slip decision;
- approved `timebase_calibration.json` for the exact timestamped-stream set and
  one common `clock_domain_id`;
- sealed `method_freeze.json`, verified against a clean checkout at the frozen
  Causal4D commit and its Bayesian-PhysTwin pin;
- independent `method_freeze_validation.json`, signed by someone other than the
  freezer and bound to the exact freeze-file SHA-256;
- one completed `sessions/<session-id>/session.json` for each of the 18 sessions;
- one completed and hash-verified execution manifest for each of the 36 locked
  executions.

Create the independent freeze attestation from a second operator account or
review step after sealing:

```bash
causal4d-real-experiment-freeze attest \
  /data/causal4d-sloth-multi-action-v1/method_freeze.json \
  configs/causal4d/sloth_multi_action_v1.json \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1/method_freeze_validation.json \
  --verified-by "<independent-verifier>"
```

The command revalidates the clean checkout, every locked file hash, and the
Bayesian-PhysTwin pin before writing the attestation. It rejects a verifier ID
that matches the original freezer.

The physical registration uses `PhysicalContactRegistration` schema 3 as the
authoritative contact record. Its `source_checksums` mapping must include
`object_registration.json` and `contact_node_set:<region-id>` entries. The
simpler registration remains as a compact acquisition index and is checked
against the authoritative artifact.

## Scaffold

Create the acquisition tree with the CLI so the version-2 prerequisite and
same-grasp templates are installed as well as the execution templates:

```bash
causal4d-real-protocol scaffold \
  configs/causal4d/sloth_multi_action_v1.json \
  /data/causal4d-sloth-multi-action-v1
```

`session.template.json`, `timebase_calibration.template.json`,
`method_freeze_validation.template.json`, and every `manifest.template.json`
remain explicitly incomplete. Copy a template to its non-template name only
after recording the required evidence; changing a filename alone never makes it
valid.

## Execution and session continuity

Every completed execution manifest retains the version-1 fields and adds these
members under `acquisition`:

```json
{
  "ended_at_utc": "2026-07-28T10:31:12Z",
  "acquisition_execution_index": 0,
  "grasp_instance_id": "grasp-session-001",
  "clock_domain_id": "ptp-clock-0"
}
```

All timestamps must be UTC. Every present timestamped artifact must use the same
clock domain. Numeric quality values must be finite and nonnegative, and
`dropped_rgbd_frames` must be a nonnegative integer; JSON `NaN` and infinities
are rejected before validation.

Each session manifest binds:

- the exact two execution IDs in locked order;
- a unique `grasp_instance_id` shared by both executions;
- the approved timebase and contact-registration SHA-256 values;
- the exact SHA-256 of both execution manifests;
- chronological, non-overlapping execution timestamps;
- successful neutral-state checks before, between, and after the pair;
- `same_grasp_confirmed: true` and `release_between_executions: false`;
- an approval timestamp after session completion.

A grasp identifier reused across sessions is rejected.

## Progress report

Write an observational progress snapshot at any stage:

```bash
causal4d-real-protocol status \
  configs/causal4d/sloth_multi_action_v1.json \
  /data/causal4d-sloth-multi-action-v1 \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/evidence-status.json
```

The report includes prerequisite, execution, and session validation details and
uses separate decisions:

- `acquisition_complete`: the original protocol, schedule, simple registration,
  slip pilot, and all 36 executions are present, valid, and accounted for;
- `evidence_complete`: every version-2 prerequisite and all 18 session records
  also validate, there are no unexpected directories, and all requested hashes
  were checked;
- `analysis_ready`: every registered fold still contains at least one included
  fit, calibration, and target execution and at least one same-grasp pair remains;
- `full_registered_power`: no execution was excluded;
- `claim_ready`: the evidence tree is complete and hash verified.

`claim_ready` does not silently discard a negative or exclusion-limited result.
It can be true while `analysis_ready` or `full_registered_power` is false, so the
mandatory report can state the registered limitation explicitly.

## Fail-closed final gate

Run the final gate from, or point it at, a clean checkout of the exact frozen
Causal4D commit:

```bash
causal4d-real-protocol status \
  configs/causal4d/sloth_multi_action_v1.json \
  /data/causal4d-sloth-multi-action-v1 \
  --repository-root /opt/causal4d-frozen \
  --verify-file-hashes \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/evidence-status.json \
  --require-complete
```

The command returns:

- `0` when the report is produced and, with `--require-complete`, is claim-ready;
- `2` when an input or locked contract cannot be interpreted;
- `3` when `--require-complete` is requested but any evidence blocker remains.

`validate-dataset` applies the same version-2 contract. Without
`--skip-file-hashes`, it fails unless the complete evidence tree is claim-ready.
The status command never creates evidence, repairs failed gates, replaces an
excluded execution, or converts a template into a completed record.
