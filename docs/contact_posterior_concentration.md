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

## Scientific boundary

This is exploratory fresh-seed evidence. It cannot revise the frozen five-seed
result, exact-node threshold, earlier independent panel, or locked 36-execution
physical experiment. Any later estimator change requires a new method version and
new evaluation data.
