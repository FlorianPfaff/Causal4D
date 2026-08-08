# Live acquisition snapshot integrity

The live acquisition-health route evaluates operational capture health only. It
must not read prediction errors, held-out metrics, coverage, oracle outputs, or
other target outcomes.

## Exact-byte input boundary

Use the grouped CLI route:

```bash
causal4d protocol acquisition snapshot health.json \
  --maximum-snapshot-age-s 5 \
  --maximum-heartbeat-age-s 2 \
  --maximum-clock-offset-ms 5 \
  --maximum-dropped-frames 0 \
  --minimum-free-gib 100 \
  --minimum-write-mib-s 100 \
  --output-json health-decision.json \
  --require-healthy
```

The command opens `health.json` as one ordinary file without following a final
symbolic link, reads and hashes the exact bytes through that descriptor, and
parses strict finite UTF-8 JSON. Duplicate object keys are rejected. The decision
records `snapshot_sha256`, `snapshot_byte_count`, and its own
`decision_sha256`.

`--output-json` uses the repository's atomic JSON publication helper. Existing
outputs are not replaced unless `--overwrite` is supplied. The printed decision
and the optional stored decision are identical.

## Freshness and heartbeat semantics

The producer reports each stream heartbeat age at `captured_at_utc`. A live
checker must also account for transport and scheduling delay between capture and
evaluation. The effective age is therefore

```text
effective_heartbeat_age_s = heartbeat_age_s + snapshot_age_s.
```

Required streams are tested against the effective age. A snapshot also fails
when its own age exceeds `--maximum-snapshot-age-s`, or when its capture time is
farther in the future than `--maximum-future-skew-s` permits. This prevents an
old file with apparently recent producer-local heartbeats from passing after the
capture process has stalled.

The Python mapping API remains useful for deterministic archived diagnostics:
when `evaluated_at_utc` is omitted from `evaluate_health_snapshot`, evaluation is
anchored to the snapshot capture time. Live file evaluation uses
`evaluate_health_snapshot_file`, which defaults to the current UTC time and binds
the exact source bytes.

## Decision boundary

A healthy decision is operational evidence only. It does not increment the
registered execution count, validate an execution manifest, determine inclusion,
or support a physical-prediction claim. The capture supervisor remains
responsible for a safe stop or abort when the decision fails.
