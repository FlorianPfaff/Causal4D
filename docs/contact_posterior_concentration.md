# Contact-posterior concentration diagnostic

This controlled diagnostic compares the registered multiplicative log-weight scale
candidates with an expanded candidate set that also permits posterior softening.
It does not modify the frozen estimator or any registered result.

For each fresh seed and held-out target topology, ordinary likelihood settings are
calibrated on the two source topologies. Each candidate policy then selects its
log-weight scale on those source cases by mean node Brier score. Target outcomes
are evaluated only after selection.

The default evaluation panel is seeds `200:220`. It is separate from the frozen
seeds `0:5` and the earlier diagnostic seeds `100:120`.

Run:

```bash
python scripts/ci/run_contact_concentration_diagnostic.py \
  --output-dir outputs/contact-concentration \
  --seeds 200:220
```

Outputs include source-selection records, evaluation rows, aggregate calibration
and trajectory metrics, explicit expanded-minus-registered comparisons, and a
checksummed manifest.

## Fresh-seed result

The complete `200:220` panel ran on `workstation2` in workflow run
`30890114134`. The archived artifact is `8885503602`; the uploaded archive digest
is
`sha256:8e8ff4c55ac25dcf1dece24e552449e191e25e16b2d433423cab11e970f0ab01`.
All diagnostic-contract tests passed and the checksummed result bundle was
uploaded successfully.

For shifted-contact cases, the registered candidates and expanded candidates had
the same exact-node accuracy, `81.67%`. Allowing softening reduced confidence
calibration error from `11.03` to `4.03` percentage points and increased credible
coverage from `88.33%` to `90.00%`. However, it worsened the proper Brier score
from `0.2873` to `0.3299` and increased future trajectory RMSE from `0.859` to
`0.974` mm.

For matched-contact cases, both policies retained `100%` exact-node accuracy and
coverage. The registered policy remained substantially better calibrated and more
predictive: its Brier score was effectively zero and its trajectory RMSE was
`1.933` mm, versus `0.00166` and `2.186` mm for the expanded policy.

The expanded-minus-registered deltas therefore show a calibration tradeoff rather
than evidence for an estimator change:

- shifted confidence-calibration error: `-6.995` percentage points;
- shifted Brier score: `+0.04268`;
- shifted trajectory RMSE: `+0.1146` mm;
- matched confidence-calibration error: `+1.210` percentage points;
- matched trajectory RMSE: `+0.2538` mm.

The scientific conclusion is negative but useful: adding uniform posterior
softening is not supported as a replacement for the registered concentration
policy. It improves one marginal confidence diagnostic on shifted contacts while
worsening proper scoring and trajectory prediction. Any remaining overconfidence
should be investigated through topology dependence or correlated prefix evidence,
not by changing the frozen candidate set on this panel.

## Scientific boundary

This is exploratory fresh-seed evidence. It cannot revise the frozen five-seed
result, exact-node threshold, earlier independent panel, or locked 36-execution
physical experiment. Any later estimator change requires a new method version and
new evaluation data.
