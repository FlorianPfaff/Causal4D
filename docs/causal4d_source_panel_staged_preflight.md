# Staged source-panel manifest preflight

The 12-execution physical source panel uses exactly-once publication. A completed
manifest cannot be overwritten after it becomes part of the registered evidence
prefix. `source-panel-verify-staged` therefore performs the complete admission
check against a staging file without creating the final manifest.

## Operator workflow

After one registered source execution is complete, populate:

```text
<dataset-root>/staging/<execution-id>.json
```

The filename must be the exact execution ID currently reported by
`source-panel-status`. Run the read-only preflight:

```bash
causal4d protocol readiness source-panel-verify-staged \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/staging/<execution-id>.json \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/operator/<execution-id>-preflight.json
```

A successful report contains a placeholder-based publication command. Substitute
the same repository and dataset roots, then publish explicitly:

```bash
causal4d protocol readiness source-panel-publish \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/staging/<execution-id>.json
```

Preflight does not make publication automatic. The operator retains a deliberate
boundary between validation and the irreversible evidence write.

## Admission checks

The command requires all of the following:

- the existing source-panel status is valid and hash verified;
- the source panel is incomplete and has one registered next execution;
- the registered template for that execution is present and unchanged;
- the staging path is an ordinary file directly below `dataset_root/staging`;
- the filename is exactly `<next-execution-id>.json`;
- the JSON object has the exact schema-1 manifest fields;
- the execution and session identities match the next registered entry;
- no target-outcome field occurs anywhere in the JSON object;
- every referenced artifact is an ordinary dataset file with the declared byte
  count and SHA-256 digest;
- the prospective final manifest path is absent and is not a symlink or other
  non-file entry; and
- source-panel status is byte-identical before and after preflight.

Duplicate JSON keys, non-finite JSON values, path traversal, symlink components,
wrong execution order, malformed timestamps, failed inclusion semantics, and bad
artifact hashes fail before a report is produced.

## Verification artifact

The JSON report binds:

- protocol and pre-acquisition amendment identities;
- execution and independent-session identities;
- the staged file path, byte count, and SHA-256 digest;
- the current portable and host-local source-panel status identities;
- the prospective final manifest path;
- a publication-command template;
- a mount-independent `evidence_sha256`; and
- an exact host-local `status_sha256`.

The report states:

```json
{
  "mutated_dataset": false,
  "final_manifest_present": false,
  "changes_registered_method": false,
  "target_outcomes_used": false
}
```

The report is derived operator evidence. It does not increment the 12-execution
source-panel count and cannot satisfy the signature-panel gate.

## Failure handling

A failed preflight leaves the final path absent. Correct the staging manifest or
its referenced artifacts and rerun the command. Do not bypass the preflight by
copying a file directly into the registered execution directory.

If the current status is invalid, the source panel is already complete, the
registered next execution changed, or a final manifest already exists, stop and
resolve that evidence boundary before attempting publication.

## Scientific boundary

This command changes no estimator, intervention posterior, physical parameter,
likelihood, threshold, exclusion, source/target split, acquisition order, or
paper claim. It only reduces the operational risk of discovering a malformed
manifest after an exactly-once publication attempt.
