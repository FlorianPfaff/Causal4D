# Pre-acquisition next-action freshness validation

A `next-action` report is a content-addressed snapshot of the registered
pre-acquisition state. It does not reserve a physical execution, freeze another
operator out, or remain valid after evidence changes. Before performing a
physical or claim-bearing action, validate that the saved report is still the
current hash-verified decision.

## Usage

Generate the operator report:

```bash
causal4d protocol readiness next-action \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/operator/next-action.json
```

Immediately before acting, run:

```bash
causal4d protocol readiness next-action-validate \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/operator/next-action.json \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/operator/next-action-validation.json
```

Only `current=true` and `safe_to_execute=true` authorize use of the saved action.
The validator always rebuilds readiness and source-panel status with file hashing
enabled; there is no structure-only mode.

## Validation contract

The command requires that:

- the decision is an ordinary, nonsymlinked JSON file;
- duplicate keys and non-finite JSON values are absent;
- the artifact kind and operator-flow schema are supported;
- target outcomes and registered-method changes remain forbidden;
- the decision's portable evidence and host-local status SHA-256 values
  recompute exactly;
- its recorded repository and dataset roots equal the current roots;
- a newly built, hash-verified next-action decision has the same portable
  evidence SHA-256;
- the action category, action ID, source execution ID, and source session ID are
  unchanged; and
- the decision file remains byte-identical while current evidence is rebuilt.

Any changed prerequisite, newly published source execution, altered operational
gate, method-freeze transition, changed artifact hash, or competing operator's
progress makes the decision stale.

## Result artifact

A successful validation records:

- the decision file path, byte count, and SHA-256;
- the decision and current portable evidence identities;
- the decision and current host-local status identities;
- the exact action, execution, and session identity;
- the current readiness and source-panel evidence identities;
- `file_hashes_verified=true`;
- `current=true`;
- `safe_to_execute=true`;
- `changes_registered_method=false`; and
- `target_outcomes_used=false`.

The report has a portable `evidence_sha256` that excludes mount-local paths,
file bytes, and host-local status values, plus an exact host-local
`status_sha256`.

## Stale-action handling

A stale decision returns exit code `2` through the grouped readiness CLI. Do not
perform its physical acquisition, approval, freeze, or publication command.
Regenerate `next-action.json`, inspect the newly selected action, and validate it
again immediately before use.

This prevents duplicated source sessions when two operators read the same old
status, prevents acting after a gate or method-freeze transition, and ensures
that a relocated report cannot silently retain commands for the wrong checkout
or evidence tree.

## Relationship to staged publication

Freshness validation occurs before the physical operation. For a source-panel
execution, the later staged-manifest preflight remains mandatory:

```text
validate current next action
        ↓
perform the registered physical source execution
        ↓
validate staged manifest and every referenced artifact
        ↓
review the preflight report
        ↓
publish exactly once
```

The two checks cover different races. Next-action validation prevents acting on
stale registered progress; staged preflight prevents publishing malformed or
changing evidence after acquisition.

## Scientific boundary

Freshness validation is operational provenance only. It does not reserve an
execution, modify evidence, update the estimator, change a threshold or split,
or increment any physical evidence count. A successful report proves only that
the saved operator action matched the current hash-verified state at validation
time.
