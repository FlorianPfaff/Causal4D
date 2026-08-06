# Source-only observation clock-offset prior

Causal4D already estimates one actuator/measurement timestamp correction for
each source or dry-run execution through the PyRecEst-backed actuator
realization diagnostic. `ObservationClockOffsetPriorV1` aggregates several such
independent artifacts into a predictive prior that downstream observation
models can consume without changing the original calibration implementation or
any frozen physical evidence.

## Sign convention

The portable convention is fixed to

```text
aligned_observation_time_s = observation_time_s + offset_s
```

A negative mean therefore says that the observation timestamps must be shifted
earlier to align them with the reference clock. The sign must not be inferred
from a provider name or silently reversed by a consumer.

## Source-only construction

At least three independent source or dry-run executions are required. Each
input must be a valid content-addressed
`ActuatorRealizationCalibration` artifact with:

- hardware timestamps declared authoritative;
- `source_or_dry_run_only = true`;
- `target_outcomes_used = false`; and
- the original PyRecEst convention
  `aligned_measurement_time_s = measurement_time_s + offset_s`.

Executions receive equal weight. For offsets \(d_1,\ldots,d_n\), the predictive
variance is

\[
\left(1 + \frac{1}{n}\right)s_d^2 + \sigma_{\mathrm{grid}}^2,
\]

where \(s_d\) is the between-execution sample standard deviation and
\(\sigma_{\mathrm{grid}}\) is the largest source offset-grid step divided by
\(\sqrt{12}\). A declared positive floor prevents a zero-width prior when a
small source panel happens to return identical grid points.

```python
from causal4d.observation_clock_offset_prior import (
    fit_observation_clock_offset_prior,
    write_observation_clock_offset_prior,
)

prior = fit_observation_clock_offset_prior(
    source_calibration_records,
    clock_domain="camera-hardware-clock",
    reference_clock_domain="actuator-command-clock",
    time_scale="device-monotonic",
    source_revision=causal4d_commit,
)
write_observation_clock_offset_prior(prior, "camera-clock-prior.json")
```

`prior.bayesian_phystwin_prior_payload()` exports the clock domain, signed mean,
predictive standard deviation, content ID, and exact offset convention expected
by an explicit downstream timing latent.

## Integrity and information order

The prior binds the exact source revision, source artifact IDs, execution IDs,
source offsets, grid-quantization term, variance floor, derived statistics,
information boundary, and claim boundary. Input order is canonicalized by
execution ID. Loading recomputes all derived quantities and the content ID;
duplicate JSON keys, symlinks, non-finite values, forged summaries, target-used
inputs, and convention changes fail closed. Publication is idempotent for the
same content and refuses to overwrite a different artifact.

The source panel, clock domains, time scale, variance floor, and aggregation
rule must be frozen before a confirmation cohort is opened. A downstream model
that retains this shared offset explicitly must not add the same offset
uncertainty again to every local observation covariance.

## Claim boundary

This artifact is a predictive prior for one deployed clock relationship. It
does not identify contact slip, material relaxation, controller-frame physics,
provider competence, physical-state identifiability, calibrated physical-query
coverage, downstream intervention benefit, deployment safety, or state of the
art. If a downstream timing direction remains confounded with state, gauge, or
spatial bias, the physical update must retain its frozen fallback.
