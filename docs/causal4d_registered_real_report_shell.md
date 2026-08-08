# Target-free registered real-analysis report shell

The physical Causal4D study has a sealed analysis manifest, but a sealed manifest
alone does not prove that the final paper tables, figures, failure accounting,
and interpretation paths can be rendered without ad hoc target-side work.

`causal4d.registered_real_report_shell` creates a deterministic report shell from
one validated registered-analysis manifest **before confirmatory outcomes are
opened**. The shell is a derived operator artifact. It is not evidence, it does
not increment the 36-execution registry, and it cannot approve or interpret a
scientific result.

The supported entry point is the stable, non-claim-bearing
`causal4d evidence real-report-shell` route in the authoritative single-executable
command catalog.

## Render the shell

```bash
causal4d evidence real-report-shell render \
  /data/causal4d-sloth-multi-action-v1/registered-analysis.json \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/operator/real-report-shell.json \
  --output-markdown \
  /data/causal4d-sloth-multi-action-v1/operator/real-report-shell.md
```

The command validates the registered analysis with the existing closed schema,
binds the exact analysis bytes and software revisions, and writes:

- a content-addressed JSON contract for the report layout; and
- a Markdown dry rendering with visibly empty result cells.

Both destinations are checked before publication. Existing outputs are rejected
unless `--overwrite` is supplied. The Markdown derivative is published first
and the validated JSON shell is published last as the completion marker. On the
default no-overwrite path, an in-process JSON publication failure removes the
new Markdown draft instead of leaving a misleading half-published pair.

## Validate the shell against its source

Standalone validation checks the schema, content identity, safety boundary, and
that every table, figure, and result narrative remains empty or unselected:

```bash
causal4d evidence real-report-shell validate \
  /data/causal4d-sloth-multi-action-v1/operator/real-report-shell.json
```

For acquisition use, bind the shell back to both the exact registered analysis
and the separately stored Markdown rendering:

```bash
causal4d evidence real-report-shell validate \
  /data/causal4d-sloth-multi-action-v1/operator/real-report-shell.json \
  --analysis-manifest \
  /data/causal4d-sloth-multi-action-v1/registered-analysis.json \
  --markdown \
  /data/causal4d-sloth-multi-action-v1/operator/real-report-shell.md
```

This form rebuilds the expected shell from the validated source bytes, requires
exact shell equality, regenerates the deterministic Markdown, and compares its
exact UTF-8 bytes. Readdressing an embedded contract and its shell ID therefore
cannot substitute either a different analysis manifest or a stale/manually
edited report rendering. The validation summary reports the Markdown SHA-256 and
byte count for archiving.

## What is rendered

The shell proves that the registered analysis can produce a complete result
without manual panel selection. It includes empty layouts for:

- all 36 execution records and every preregistered exclusion or technical failure;
- factual-continuation effects;
- same-grasp transfer effects;
- new-contact transfer effects;
- independent execution-block calibration, coverage, interval width, normalized
  NEES, and finite-calibration sensitivity;
- replay/reset variance;
- diagnostic intervention-oracle gap attribution;
- paired endpoint-effect figures;
- coverage and interval-width figures;
- execution accounting; and
- oracle-gap decomposition.

It also renders both predeclared interpretation paths:

1. registered success across factual prediction, both transfer endpoints, and
   independent-execution calibration; and
2. a bounded negative result that reports each failed gate separately and does
   not use calibration or an optional branch to rescue failed prediction.

Neither path is selected by the shell. Selection remains the responsibility of
the source-verified real-result interpretation after blind analysis.

## Fail-closed safety boundary

A valid shell contains exactly:

```json
{
  "target_outcomes_loaded": false,
  "target_metric_values_loaded": false,
  "confirmatory_execution_evidence_count": 0,
  "may_select_method_or_hyperparameters": false,
  "manual_table_or_figure_selection_allowed": false,
  "derived_artifact_is_claim_bearing": false,
  "report_shell_is_a_scientific_result": false
}
```

Every table has `rows=[]`, every figure has `values=[]`, and both result
narratives have `selected=false`. Adding a target value, selecting a narrative,
changing the layout, or altering the registered safety boundary invalidates the
artifact even when its `shell_id` is recomputed.

## Scientific boundary

This helper changes no estimator, intervention posterior, six-frame information
boundary, split, threshold, exclusion rule, calibration method, acquisition
order, or evidence count. The physical status remains `0/36` until genuine
registered executions are acquired and validated. A successful report-shell dry
run demonstrates analysis-operational completeness only; it is not empirical
support for Causal4D.
