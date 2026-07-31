# Predeclared interpretation of the real Causal4D result

## Purpose

The 36-execution same-object protocol can produce several scientifically useful
outcomes. Those outcomes must not be combined after target access into a broader
claim than the registered gates support.

`causal4d evidence interpret-real-result` applies one versioned interpretation
tree to the completed gate summary. It reads **gate decisions and provenance,
not raw target metrics**. The output is recomputable, content-addressed, and
fails validation if a headline, supported claim, limitation, or next action is
edited by hand.

This reporting contract does not change the estimator, protocol, thresholds,
folds, exclusion policy, or target outcomes. It fixes how their registered gate
results may be described.

## Bound inputs

A gate summary identifies:

- protocol `causal4d-sloth-multi-action-v1` and its exact design SHA-256;
- the final v4 pre-acquisition amendment SHA-256;
- the sealed method-freeze SHA-256;
- the registered analysis-manifest SHA-256;
- complete versus incomplete evidence status;
- factual-continuation, same-grasp-transfer, new-contact-transfer, and
  execution-block-calibration gate results;
- the preregistered oracle diagnosis;
- technical-failure and preregistered-exclusion counts; and
- whether target-informed method or threshold selection occurred.

Every primary gate is one of `passed`, `failed`, or `not_estimable`. The oracle
diagnosis is one of `intervention_headroom`, `model_discrepancy_dominant`,
`mixed`, or `not_estimable`.

## Decision tree

The tree is evaluated in this order:

| Result pattern | Classification | Permitted headline |
| --- | --- | --- |
| Target-informed selection occurred | `confirmatory_boundary_violated` | No confirmatory claim; report the violation and issue a new protocol. |
| Evidence registry incomplete | `incomplete_evidence` | No confirmatory claim; retain missing, failed, and excluded executions. |
| Factual-continuation gate fails | `primary_chain_not_supported` | Negative primary result; transfer or calibration cannot rescue it. |
| Factual, same-grasp, new-contact, and calibration gates pass | `full_chain_supported` | Complete registered evidence chain passes, bounded to this protocol. |
| Factual and both transfer gates pass; calibration fails | `transfer_supported_calibration_limited` | Transfer is supported, but no calibrated-risk or hardware-safety claim. |
| Factual and both transfer gates pass; calibration not estimable | `transfer_supported_calibration_unresolved` | Bounded transfer evidence with unresolved independent-execution calibration. |
| Factual and same-grasp pass; new-contact fails | `persistent_transfer_only` | Persistent realized actuation transfers; fresh-contact/event transfer does not. |
| Factual and new-contact pass; same-grasp fails | `incoherent_transfer_pattern` | Report the inconsistent arms separately; no general transfer claim. |
| Factual passes; both transfer gates fail | `factual_only` | Prefix-conditioned factual prediction improves; transfer is unsupported. |
| Factual passes; at least one transfer endpoint is not estimable | `partial_transfer_evidence` | Report the estimable subset without inferring the complete chain. |

A passed calibration gate does not rescue failed factual or transfer gates.
Oracle diagnostics add mandatory limitations and a post-reporting research
focus; they cannot change the primary classification.

## Input format

```json
{
  "schema_version": 1,
  "artifact_kind": "Causal4DRealResultGateSummary",
  "protocol_id": "causal4d-sloth-multi-action-v1",
  "protocol_design_sha256": "6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968",
  "preacquisition_amendment_sha256": "0e167538a7824e5ec053031d8359d4e9b4ff89ad61a85666400a86c2a88ac42f",
  "method_freeze_sha256": "<64 lowercase hex>",
  "analysis_manifest_sha256": "<64 lowercase hex>",
  "evidence_status": "complete",
  "factual_continuation": "passed",
  "same_grasp_transfer": "passed",
  "new_contact_transfer": "failed",
  "execution_block_calibration": "failed",
  "oracle_diagnosis": "model_discrepancy_dominant",
  "technical_failure_count": 0,
  "preregistered_exclusion_count": 0,
  "target_informed_selection": false
}
```

The gate summary should be generated from the registered analysis and evidence
registry. It is not an opportunity to redefine thresholds or manually choose a
more favorable qualitative label.

An explicitly invalid operator template is available at
`configs/causal4d/real_result_gate_summary_v1.template.json`. It must be copied,
completed, and changed to artifact kind `Causal4DRealResultGateSummary`; the
template itself cannot pass validation.

## Command

```bash
causal4d evidence interpret-real-result \
  /data/causal4d-sloth-multi-action-v1/real-result-gates.json \
  /data/causal4d-sloth-multi-action-v1/real-result-interpretation.json \
  --require-complete
```

The output contains:

- the exact gate and provenance identities;
- the matched rule and stable classification ID;
- paper status: `positive`, `bounded_positive`, `negative`, or `incomplete`;
- supported claims;
- mandatory limitations;
- globally prohibited claims;
- the next action allowed by the registered result; and
- SHA-256 identities for the interpretation contract and result.

The command returns exit code `3` with `--require-complete` when the resulting
paper status is `incomplete`. Existing output is not overwritten unless
`--overwrite` is supplied.

## Non-claims retained for every outcome

The interpretation artifact always prohibits:

- individual-level real counterfactual ground truth;
- an overall state-of-the-art claim beyond the registered same-object protocol;
- calibration of the raw physical-posterior covariance;
- general robot-execution safety or hardware-control success;
- Prob4D physical-prediction benefit without its separate prospective test; and
- real contact recovery without independent contact instrumentation.

A negative or bounded result is complete scientific evidence. It must not be
retuned on the same target executions.
