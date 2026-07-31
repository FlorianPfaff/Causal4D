# Real-experiment failure attribution

The confirmatory Causal4D analysis compares nominal PhysTwin,
Bayesian-PhysTwin with nominal realized intervention, frozen Causal4D, and a
holdout-selected intervention oracle used only as a diagnostic ceiling. One
execution-level oracle audit can distinguish several sources of remaining
error, but the first physical-experiment report also needs a protocol-accounted
summary across all registered executions.

`causal4d diagnostic real-failure-attribution` aggregates the immutable JSON
outputs produced by `causal4d diagnostic real-oracle-gap`. It does not run a new
estimator, select an intervention, or alter the frozen analysis. The aggregate
is explicitly diagnostic and cannot be used to tune or rescue the confirmatory
method.

## Failure decomposition

For each metric, the per-execution audit defines:

```text
inference gap = frozen Causal4D posterior - current-bank oracle
proposal gap  = current-bank oracle - expanded-bank oracle
model gap     = expanded-bank oracle - labeled discrepancy ceiling
```

The aggregate reports equal-execution summaries of these gaps, their shares of
total diagnostic headroom, and the most frequent dominant gap. It also reports
paired prediction differences for:

- Bayesian-PhysTwin versus nominal PhysTwin;
- Causal4D versus Bayesian-PhysTwin with nominal `z`;
- Causal4D versus nominal PhysTwin.

Predictive-variance contributions and numerical-closure diagnostics are
aggregated separately. Oracle-selected quantities remain nondeployable even
when they identify a clear failure boundary.

## Exact execution accounting

Supply the registered protocol to require every scheduled execution to be
represented by exactly one valid audit or one explicit preregistered exclusion.
The command fails closed on missing, duplicate, extra, cross-protocol, or
contradictory cases. Exclusions require nonempty reasons and cannot overlap
included audits.

An exclusions file has this form:

```json
{
  "protocol_id": "causal4d-sloth-multi-action-v1",
  "exclusions": [
    {
      "case_id": "sloth-v1-c1-s1-e1",
      "reason": "failed registered synchronization gate"
    }
  ]
}
```

Run the aggregation after producing one oracle audit per included execution:

```bash
causal4d diagnostic real-failure-attribution \
  analysis/real-failure-attribution.json \
  analysis/real-failure-attribution.csv \
  analysis/oracle-audits/*.json \
  --protocol-json configs/causal4d/sloth_multi_action_v1.json \
  --exclusions-json analysis/registered-exclusions.json
```

The JSON output contains a content-derived evidence fingerprint, exact input
SHA-256 identities, execution accounting, equal-execution summaries, paired
comparisons, diagnostic gap attribution, variance diagnostics, and the claim
boundary. The CSV contains one compact row per included execution.

## Validation boundary

The aggregator rejects an input unless all audits agree on the protocol,
selection metric, admitted observation-prefix interval, abduction likelihood,
discrepancy cap, and gap definitions. It also verifies that:

- no predictor used holdout labels;
- evaluation and oracle selection are explicitly labeled;
- oracle outputs are diagnostic-only and nondeployable;
- current and expanded oracle contracts agree;
- the expanded rollout bank was verified as nested;
- the inference, proposal, model, and total-headroom arithmetic closes;
- variance contributions and closure values are finite.

A positive or negative confirmatory conclusion still comes only from the
registered physical-experiment analysis. This report localizes the empirical
boundary; it does not change it.
