# Execution-Block Conformal Calibration

The confirmatory Causal4D real protocol treats an independent execution/session,
not an individual coordinate or point-frame, as the calibration unit. The
`causal4d-execution-block-calibration` command implements that registered
boundary without changing the older affine-calibration diagnostic.

## Method

For each outer fold, the source data have two disjoint roles:

1. **fit executions** determine an affine variance transform
   `variance_fit = a * variance_raw + b` by equal-execution Gaussian NLL;
2. **calibration executions** each contribute exactly one nonconformity score,
   the maximum absolute standardized coordinate residual over the complete
   valid held-out execution block.

For `n` independent calibration sessions and nominal coverage `c`, the threshold
uses the split-conformal order statistic

```text
k = ceil((n + 1) * c).
```

The locked 90% analysis uses nine calibration sessions, so `k = 9` and the
threshold is the largest of the nine execution scores. Eight sessions cannot
produce a finite nominal-90% threshold; the command fails rather than silently
using the eighth score.

## Source manifest

The source manifest is schema version 1 and identifies one outer fold. Every
case must have an `execution_id`, a `session_id`, and the existing prediction
inputs accepted by `causal4d-real-calibration`.

```json
{
  "schema_version": 1,
  "outer_fold_id": "hold-left_forepaw-lift_high",
  "protocol_id": "causal4d-sloth-multi-action-v1",
  "protocol_design_sha256": "...",
  "preacquisition_plan_id": "causal4d-sloth-preacquisition-v4",
  "preacquisition_amendment_sha256": "...",
  "method_freeze_sha256": "...",
  "fit": [
    {
      "execution_id": "source-fit-001",
      "session_id": "source-fit-session-001",
      "case_id": "source-fit-001",
      "moments_npz": "...",
      "final_data": "..."
    }
  ],
  "calibration": [
    {
      "execution_id": "source-cal-001",
      "session_id": "source-cal-session-001",
      "case_id": "source-cal-001",
      "moments_npz": "...",
      "final_data": "..."
    }
  ]
}
```

`protocol_id`, `protocol_design_sha256`, `preacquisition_plan_id`, and
`preacquisition_amendment_sha256` are mandatory. Digests must be lowercase
SHA-256 values. `method_freeze_sha256` is optional before the final method seal,
but once supplied it becomes part of the immutable source calibration identity.
The exact source-manifest digest is also embedded in the checksummed calibration
artifact.

Fit and seal the calibration artifact with:

```bash
causal4d-execution-block-calibration fit \
  source-fold.json \
  execution-block-calibration.json \
  --confidence-level 0.90 \
  --expected-calibration-units 9
```

`--expected-fit-units` may additionally lock the fit count. The command rejects
repeated session IDs, execution overlap, session overlap between fit and
calibration, mixed outer folds, incomplete frozen counts, unattainable finite
conformal ranks, and incomplete or malformed protocol provenance.

## Target evaluation

The target manifest uses the same schema and outer-fold ID with a `target` list.
No target execution or session may occur in the fit or calibration source. It
must repeat the exact source `protocol_id`, protocol-design digest,
preacquisition plan ID, and amendment digest. If either source or target names a
`method_freeze_sha256`, both must name the same digest. A same-named fold from a
different protocol, amendment, or method freeze is rejected before any target
case is scored.

```bash
causal4d-execution-block-calibration evaluate \
  execution-block-calibration.json \
  target-fold.json \
  target-evaluation.json
```

The output contains a canonical `frozen_source_target_binding` record with the
source and target manifest digests and the verified protocol identities. This
record is written alongside the execution-block coverage result so paper-facing
evidence cannot silently combine calibration and target artifacts from different
registered designs.

The output reports execution-block coverage as the primary quantity. Pointwise
coordinate coverage, RMSE, track error, and interval width are diagnostics only.
No pooled-coordinate conformal guarantee or worst-group coverage guarantee is
claimed.

## Fragility diagnostics

The checksummed calibration artifact records:

- largest and second-largest calibration scores;
- maximum-to-median score ratio;
- leave-one-calibration-session-out diagnostics;
- whether the requested nominal threshold remains finite after each removal.

These diagnostics cannot select or revise the registered threshold.
