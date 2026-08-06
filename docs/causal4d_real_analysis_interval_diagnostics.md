# Real-analysis interval diagnostics

The registered real-analysis report remains session-clustered and keeps its
frozen percentile-bootstrap interval unchanged.  A source-only operating-
characteristic audit completed before physical target acquisition showed that
an unstudentized percentile interval for the session mean can materially
undercover at the registered sample sizes of 12 and 18 sessions.

This companion artifact makes that limitation explicit and adds two
non-decision-making sensitivity intervals:

- a Student-t interval for a transparent symmetric small-sample check; and
- a deterministic bootstrap-t interval for better calibration under skew and
  other non-Gaussian session-effect distributions.

Neither sensitivity interval may rescue a failed primary endpoint, alter a
registered gate, or select a method after target access.  Promotion to primary
status requires a separate, content-addressed preacquisition amendment.

## Workflow evidence

All studies evaluated the immutable implementation at
`fa6a64b2442474321e453e9e8fdccd591e0a282d` and used no physical target
outcomes.

### Exact percentile-bootstrap operating characteristics

- workflow run: `31091137654`;
- audit ID:
  `7dbea2a9b99cbc98acd03fa28af9583f0e95d4d0772e58853af4f05d0584267a`;
- ten distribution/sample-size scenarios;
- 2,000 synthetic session panels per scenario;
- the frozen 20,000 bootstrap resamples and seed `20260726` for every panel;
- matrix implementation verified against the production `_bootstrap` function
  in every scenario.

For Gaussian session effects, nominal 95% percentile coverage was about 90.9%
at 12 sessions and 93.1% at 18 sessions.  Coverage was lower under strong
right skew.

### Interval-method comparison

- workflow run: `31091652355`;
- audit ID:
  `5a13c416d7efd522f5123f98afacaacd218838583d78256d463eeb5e1d478576`;
- 15,000 common synthetic panels;
- percentile, basic, Student-t, BCa, and bootstrap-t intervals compared on the
  same panels and bootstrap resamples.

Across the ten scenarios, bootstrap-t had the smallest mean absolute coverage
error, 0.019, and the best worst-case absolute coverage error, 0.042.  The
Student-t interval had the smallest maximum favorable one-sided type-I error,
about 0.0267.  Basic and BCa intervals did not improve the aggregate result.

These findings justify publishing bootstrap-t and Student-t as sensitivity
intervals; they do not change the frozen primary analysis.

## Build the companion artifact

```bash
python -m causal4d.cli.real_analysis_interval_diagnostics \
  effects.json \
  configs/causal4d/sloth_multi_action_v1.json \
  interval-diagnostics.json \
  --method-freeze method-freeze.json \
  --analysis-manifest registered-analysis.json
```

Use `--overwrite` only for an intentional regeneration of the same output
path.  The command first builds the complete primary report, including protocol
and source verification, and then derives the companion intervals from the
same equal-session-weighted effects.

## Output contract

The companion JSON records:

- the exact primary report and effect-table identities;
- the unchanged primary percentile interval;
- `finite_sample_coverage_guaranteed=false` for that interval;
- Student-t and bootstrap-t sensitivity bounds;
- the fixed bootstrap seed and replicate count;
- the completed workflow evidence and audit identifiers;
- `may_change_primary_decision=false` for every sensitivity interval; and
- the same one-object, non-safety claim boundary as the primary report.

The companion artifact is additive.  Existing report consumers and the frozen
primary percentile output are unaffected.
