# Real-Evidence Status and Claim-Readiness Gate

The same-object multi-action protocol contains 36 preregistered executions. A
scaffolded directory is only an acquisition template; it is not evidence. The
status command therefore reports templates, completed manifests, validated
executions, inclusion/exclusion accounting, and verified artifact hashes as
separate quantities.

## Progress report

After scaffolding the dataset, write a machine-readable progress snapshot:

```bash
causal4d-real-protocol status \
  configs/causal4d/sloth_multi_action_v1.json \
  /data/causal4d-sloth-multi-action-v1 \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/evidence-status.json
```

`manifest.template.json` files are never counted as acquired executions. An
execution is acquired only when `manifest.json` exists and explicitly declares
`acquisition_status: complete`. It is validated only when the completed manifest
passes the frozen protocol, quality-gate, exclusion, artifact-descriptor, and
information-boundary checks.

The report includes:

- prerequisite status for the dataset protocol, locked acquisition schedule,
  object registration, and slip pilot;
- specified, manifest-present, acquired, validated, included, and excluded
  execution counts;
- ordered missing, incomplete, and invalid execution IDs;
- the next unresolved execution in the preregistered acquisition order;
- per-execution validation errors and quality-gate failures;
- unexpected directories under `executions/`;
- `complete`, `file_hashes_verified`, and `claim_ready` decisions.

## Fail-closed completion gate

Before analysis or release of a multi-action real-data claim, rehash every
registered artifact and require claim readiness:

```bash
causal4d-real-protocol status \
  configs/causal4d/sloth_multi_action_v1.json \
  /data/causal4d-sloth-multi-action-v1 \
  --verify-file-hashes \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/evidence-status.json \
  --require-complete
```

Exit codes are:

- `0`: the report was produced and, when `--require-complete` is present, the
  evidence is claim-ready;
- `2`: the command or locked protocol could not be interpreted;
- `3`: `--require-complete` was requested but the evidence is not claim-ready.

`complete=true` means that all prerequisites and all 36 execution manifests are
present, explicitly complete, valid, and fully accounted for. `claim_ready=true`
additionally requires `--verify-file-hashes` and successful checksum validation
of every execution artifact and registered contact-node set.

The status command is observational. It does not create completed manifests,
repair failed gates, replace excluded executions, or transform scaffolded
placeholders into evidence.
