# Acquisition campaign status and reports

The acquisition campaign surface turns one hash-verified pre-session doctor
artifact into concise operator decisions and deterministic human-readable
reports. It does not rerun validation, inspect target outcomes, or change the
locked protocol. The source doctor report remains authoritative.

## Generate the source doctor report

Run the doctor from the exact deployed checkout immediately before a session:

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

The campaign commands first verify the exact schema-1 doctor contract: artifact
kind, canonical SHA-256, timestamp and protocol identity, threshold types, the
complete check inventory, derived validity/readiness/completion flags, execution
counts, next-execution identity and index, resume semantics, and the recursive
`target_outcomes_used=false` boundary. A modified, self-rehashed but
contradictory, or outcome-contaminated doctor artifact is rejected.

## Compact machine-readable status

```bash
causal4d protocol acquisition campaign status \
  /data/causal4d-sloth-multi-action-v1/operator/doctor-latest.json \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/operator/campaign-latest.json \
  --overwrite \
  --require-ready
```

The summary reports:

- `state`: `invalid`, `blocked`, `ready`, or `complete`;
- completed, remaining, and total registered executions;
- progress as a fraction of the frozen protocol;
- the complete next registered execution descriptor;
- blocking checks and warnings;
- the exact source doctor report SHA-256; and
- `target_outcomes_used=false`.

`--require-ready` returns exit code 3 for a structurally valid but blocked
campaign.

## Show only the next decision

```bash
causal4d protocol acquisition campaign next \
  /data/causal4d-sloth-multi-action-v1/operator/doctor-latest.json
```

This emits the campaign state, the next registered execution, blocking checks,
and the source doctor identity. It returns:

| Code | Meaning |
| ---: | --- |
| `0` | The next execution is ready, or collection is complete. |
| `2` | The doctor artifact is invalid, tampered, or contradictory. |
| `3` | The campaign is valid but blocked or requires operator review. |

A blocked `next` result must not be used as permission to record.

## Render an operator report

Markdown is selected by default unless the output name ends in `.html`:

```bash
causal4d protocol acquisition campaign report \
  /data/causal4d-sloth-multi-action-v1/operator/doctor-latest.json \
  /data/causal4d-sloth-multi-action-v1/operator/campaign-latest.md \
  --overwrite

causal4d protocol acquisition campaign report \
  /data/causal4d-sloth-multi-action-v1/operator/doctor-latest.json \
  /data/causal4d-sloth-multi-action-v1/operator/campaign-latest.html \
  --format html \
  --overwrite
```

Both formats contain the progress, next execution, blockers, warnings, complete
doctor-check table, source doctor digest, and explicit scientific boundary.
Operator-controlled strings are escaped before rendering. Publication is
atomic; without `--overwrite`, an existing destination is preserved and the
command fails.

A campaign output must be distinct from its source doctor report. The command
rejects the same pathname as well as an existing symbolic-link or hard-link
alias, even when `--overwrite` is supplied. This prevents a derived summary or
rendered report from replacing the authoritative doctor artifact.

## Workflow usage

A collection supervisor can use this sequence:

```bash
set -euo pipefail

doctor=operator/doctor-latest.json
summary=operator/campaign-latest.json
report=operator/campaign-latest.html

causal4d protocol acquisition doctor \
  configs/causal4d/sloth_multi_action_v1.json \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  --output-json "$doctor" \
  --overwrite

causal4d protocol acquisition campaign status \
  "$doctor" \
  --output-json "$summary" \
  --overwrite \
  --require-ready

causal4d protocol acquisition campaign next "$doctor"

causal4d protocol acquisition campaign report \
  "$doctor" \
  "$report" \
  --overwrite
```

Retain the doctor JSON beside any rendered report. The rendered files are
operator views derived from the doctor identity; they do not replace readiness,
method-freeze, execution-manifest, journal, or evidence-status artifacts.

## Scientific boundary

Campaign status and reports are operational provenance only. They do not:

- increment the acquired or validated execution count;
- alter the registered order, split, thresholds, exclusions, or estimator;
- inspect or summarize held-out target outcomes;
- authorize target-informed method selection; or
- establish physical-prediction accuracy or calibration.

Only the registered execution and session manifests, together with the
hash-verified evidence-status validator, establish confirmatory evidence.
