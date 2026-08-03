# Acquisition doctor and flight recorder

The acquisition tooling protects the locked 36-execution experiment from
operational failures without changing the estimator, protocol, split, exclusion
rules, analysis, or scientific thresholds.

It has three surfaces:

```text
causal4d protocol acquisition doctor ...
causal4d protocol acquisition snapshot ...
causal4d protocol acquisition journal ...
```

All outputs explicitly state `target_outcomes_used=false`. Journal payloads and
health snapshots reject target-error, held-out-metric, coverage, NLL, Chamfer,
and oracle-result fields. The tooling is therefore suitable for collection
health and recovery, not target-informed method selection.

## Pre-session doctor

Run the doctor immediately before every session from the exact deployed
checkout:

```bash
causal4d protocol acquisition doctor \
  configs/causal4d/sloth_multi_action_v1.json \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  --minimum-free-gib 100 \
  --write-probe-mib 64 \
  --minimum-write-mib-s 100 \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/operator/doctor-latest.json \
  --overwrite \
  --require-ready
```

The doctor fails closed when:

- the checkout is dirty or differs from `method_freeze.json`;
- any locked method file differs from the freeze;
- the sealed pre-acquisition readiness report is absent, invalid, or did not
  permit execution 1;
- the dataset or journal traverses a symlink;
- free space or measured synchronous write throughput is below the operator
  threshold;
- the registered evidence status or any referenced artifact hash is invalid;
- validated execution manifests do not form the locked acquisition prefix;
- the next execution or session directory was not scaffolded; or
- the next session journal is already sealed while an execution remains
  incomplete.

An existing valid, unsealed journal is reported as a warning rather than erased.
It blocks `--require-ready` until the operator reviews the chain and explicitly
reruns the doctor with `--allow-resume`. The doctor verifies that the journal
identifies the registered protocol and current session, that its terminal
execution events agree with hash-validated manifests, and that any active
execution is the next locked execution. The doctor never overwrites evidence and
its temporary write probe is created with exclusive, no-follow semantics,
fsynced, and deleted.

With `--require-ready`, exit codes follow the other operational gates:

| Code | Meaning |
| ---: | --- |
| `0` | Ready to record, or all registered executions are complete. |
| `2` | Invalid or contradictory operational state. |
| `3` | Valid invocation, but collection is blocked or requires review. |

## Live health snapshots

Hardware-specific capture processes can periodically publish a small JSON
snapshot and ask Causal4D to evaluate it:

```bash
causal4d protocol acquisition snapshot health.json \
  --maximum-heartbeat-age-s 2 \
  --maximum-clock-offset-ms 5 \
  --maximum-dropped-frames 0 \
  --minimum-free-gib 100 \
  --minimum-write-mib-s 100 \
  --require-healthy
```

A snapshot has this shape:

```json
{
  "schema_version": 1,
  "artifact_kind": "Causal4DAcquisitionHealthSnapshot",
  "protocol_id": "causal4d-sloth-multi-action-v1",
  "session_id": "sloth-v1-c1-s6",
  "execution_id": "sloth-v1-c1-s6-e1",
  "captured_at_utc": "2026-08-03T08:00:00+00:00",
  "target_outcomes_used": false,
  "streams": {
    "rgbd": {
      "required": true,
      "alive": true,
      "heartbeat_age_s": 0.12,
      "dropped_frames": 0,
      "clock_offset_ms": 1.4
    },
    "actuator": {
      "required": true,
      "alive": true,
      "heartbeat_age_s": 0.03,
      "dropped_frames": 0,
      "clock_offset_ms": -0.8
    }
  },
  "storage": {
    "free_bytes": 500000000000,
    "write_mib_s": 420.0
  }
}
```

Required dead or stale streams, dropped-frame excess, clock-offset excess, low
free space, or low write rate produce a failed decision. Optional unhealthy
streams produce warnings. The capture system decides whether to stop safely; the
snapshot checker does not issue actuator commands.

## Append-only session journal

The journal is JSON Lines with one canonical JSON object per event. Every event
contains:

- protocol, session, and optional execution identities;
- a contiguous sequence number;
- UTC and monotonic timestamps;
- an event type and source;
- finite JSON operational payload;
- the previous event SHA-256; and
- its own canonical SHA-256.

The first event must be `session_started`. Append operations use an exclusive
file lock on POSIX, `O_APPEND`, `O_NOFOLLOW` where available, `flush`, and
`fsync`. Existing bytes are never rewritten.

Start a journal:

```bash
causal4d protocol acquisition journal append \
  /data/causal4d-sloth-multi-action-v1/sessions/sloth-v1-c1-s6/acquisition.jsonl \
  session_started \
  --protocol-id causal4d-sloth-multi-action-v1 \
  --session-id sloth-v1-c1-s6 \
  --source acquisition-supervisor
```

Append an execution transition or a health sample:

```bash
causal4d protocol acquisition journal append \
  /data/causal4d-sloth-multi-action-v1/sessions/sloth-v1-c1-s6/acquisition.jsonl \
  execution_started \
  --protocol-id causal4d-sloth-multi-action-v1 \
  --session-id sloth-v1-c1-s6 \
  --execution-id sloth-v1-c1-s6-e1 \
  --source controller
```

Payloads are supplied as a separate JSON object:

```bash
causal4d protocol acquisition journal append \
  SESSION/acquisition.jsonl stream_heartbeat \
  --protocol-id causal4d-sloth-multi-action-v1 \
  --session-id sloth-v1-c1-s6 \
  --execution-id sloth-v1-c1-s6-e1 \
  --source rgbd-recorder \
  --payload-json heartbeat.json
```

Use coarse operational events or periodic heartbeats. Raw sensor samples remain
in their registered artifacts and should not be duplicated into JSONL.

Validate at any time:

```bash
causal4d protocol acquisition journal validate SESSION/acquisition.jsonl
```

Finish with exactly one terminal disposition:

```bash
causal4d protocol acquisition journal append \
  SESSION/acquisition.jsonl session_completed \
  --protocol-id causal4d-sloth-multi-action-v1 \
  --session-id sloth-v1-c1-s6 \
  --source acquisition-supervisor

causal4d protocol acquisition journal seal \
  SESSION/acquisition.jsonl \
  --sealed-by operator.primary
```

For a technical failure, use `session_aborted` instead of pretending the session
completed. The deterministic seal is written as
`acquisition.jsonl.seal.json`. It binds the journal byte count, journal SHA-256,
final event SHA-256, event count, execution IDs, session outcome, signer, and
seal time. After a terminal event or seal, further appends fail.

Revalidate an archived journal and seal together:

```bash
causal4d protocol acquisition journal validate \
  SESSION/acquisition.jsonl \
  --require-sealed
```

## Recovery policy

A crash may leave a valid unsealed journal. Do not delete or replace it.

1. Run `journal validate` and the pre-session doctor.
2. Record `recovery_started` with the detected process and artifact state.
3. Resume only hardware streams whose registered artifacts support safe append or
   create a new preregistered technical-failure record.
4. Record `recovery_completed`.
5. Rerun the doctor with `--allow-resume` after reviewing the validated chain.
6. Complete or abort the session explicitly, then seal.

A corrupt journal is operational evidence of a failed acquisition path. It is
not silently repaired. Preserve the bytes, record the technical failure, and
follow the registered replacement/exclusion policy.

## Scientific boundary

The doctor, snapshots, journals, and seals are operational provenance. They do
not increment the acquired or validated execution count, change inclusion, tune
thresholds, inspect held-out outcomes, or support a physical-prediction claim.
Only the registered execution/session manifests and evidence-status validator
establish the confirmatory evidence count.
