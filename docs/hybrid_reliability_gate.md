# Prefix-only hybrid reliability gate

The controlled counterfactual benchmark shows a specific limitation: the
hybrid residual correction strongly outperforms the generative-only baseline,
but it can be worse than physics-only prediction under combined world and
contact mismatch. `causal4d.hybrid_reliability` provides an experimental,
source-calibrated abstention boundary for that failure mode.

This component is **not wired into the frozen controlled benchmark or the
registered 36-execution physical estimator**. It cannot modify the current
primary method, rescue a failed physical endpoint, or authorize target access.
A claim-bearing deployment would require a separately locked source calibration
and an explicit preacquisition amendment. The current surface is a Python API;
there is deliberately no stable command-line route or default runtime wiring.

## Information boundary

Calibration and target use are deliberately different:

- source calibration may use complete source futures to decide whether the
  hybrid correction has sufficient average gain and source-case win fraction;
- a target decision uses only response-prefix observations, the already
  generated physics/hybrid predictive distributions, correction scale, and
  residual-model descriptor leverage;
- no target future observation is required to construct or apply the decision;
- the target input identity hashes only the admitted prefix observations, never
  future observation bytes; and
- rejection returns the exact original `PredictiveDistribution` object from the
  physics-only path.

The source artifact identity separately binds the complete source observations,
physics and hybrid predictions, descriptor leverage, prefix length, and case
identity. The calibration stores both full source identities and prefix-input
identities, so relabelling an already used source prefix as a target fails
closed.

## Diagnostics

For each target case the gate records:

- physics-only and hybrid response-prefix RMSE;
- relative prefix RMSE improvement;
- Gaussian prefix log-score gain;
- hybrid correction RMS over the requested prediction horizon;
- correction RMS relative to the physics predictive standard deviation; and
- descriptor leverage under the fitted hybrid residual model.

`future_observation_frames_read` is always zero. Target decisions also record
`target_future_observations_read=0` and
`target_future_outcomes_used=false`.

## Source calibration

A calibration requires at least two uniquely identified source cases with one
common prefix length. Source cases are sorted by canonical `case_id` before any
aligned diagnostic vector or content identity is constructed. Reversing the
input sequence therefore produces byte-identical calibration records.

By default, the hybrid family is enabled only when:

```text
mean source-future relative RMSE improvement >= 0.5%
source-future win fraction >= 2/3
```

The source panel also fixes the admitted correction-scale and descriptor-
leverage envelopes. The target must meet nonnegative prefix point-score and
probabilistic-score gains and remain inside both source envelopes. All
thresholds and complete source diagnostics are content-addressed.

### Derived-field validation

Calibration schema version 2 stores the two prefix-score margins and the support
margin needed to reconstruct every operational boundary. Construction and
loading independently recompute from the stored source diagnostic vectors:

- minimum prefix RMSE relative improvement;
- minimum prefix Gaussian log-score gain;
- maximum correction-to-physics-standard-deviation ratio;
- maximum descriptor leverage;
- mean source-future relative improvement;
- source-future win fraction; and
- the resulting `hybrid_enabled` decision.

A matching outer SHA-256 is not sufficient. An artifact whose derived value was
changed and then re-addressed still fails closed because the value no longer
matches its source diagnostics. Directly constructed or loaded records also
reject fewer than two source cases and noncanonical source ordering. Version-1
calibration files are intentionally rejected rather than silently upgraded,
because they do not retain the margins required for independent reconstruction.

These defaults are development semantics, not a new registered gate. A
scientific experiment must freeze its source split, thresholds, support margin,
case identities, and calibration artifact before opening target outcomes.

## Python use

```python
from causal4d.hybrid_reliability import (
    HybridReliabilityCase,
    apply_hybrid_reliability,
    fit_hybrid_reliability_calibration,
    ridge_descriptor_leverage,
    save_hybrid_reliability_calibration,
    write_hybrid_reliability_decision,
)

source_cases = []
for case_id, physics, hybrid, source_observations, descriptor in source_panel:
    source_cases.append(
        HybridReliabilityCase(
            case_id=case_id,
            physics=physics,
            hybrid=hybrid,
            observations=source_observations,
            descriptor_leverage=ridge_descriptor_leverage(
                fitted.hybrid.residual_model,
                descriptor,
            ),
            prefix_frame_count=6,
        )
    )

calibration = fit_hybrid_reliability_calibration(source_cases)
save_hybrid_reliability_calibration(
    "hybrid-reliability-calibration.json",
    calibration,
)

target = HybridReliabilityCase(
    case_id="held-out-case",
    physics=physics_prediction,
    hybrid=hybrid_prediction,
    # Future entries may be NaN; only observations[1:prefix_frame_count] are used.
    observations=prefix_padded_observations,
    descriptor_leverage=ridge_descriptor_leverage(
        fitted.hybrid.residual_model,
        held_out_descriptor,
    ),
    prefix_frame_count=6,
)
selected_prediction, decision = apply_hybrid_reliability(target, calibration)
write_hybrid_reliability_decision("hybrid-reliability-decision.json", decision)
```

The calibration writer is atomic and non-overwriting by default. Reloading uses
an exact-byte, duplicate-key-rejecting JSON reader, reconstructs all derived
values, and verifies the content ID.

## Rejection reasons

A target falls back to physics-only when any of the following applies:

| Reason | Meaning |
| --- | --- |
| `no_source_future_gain` | The frozen source panel did not justify enabling hybrid correction. |
| `prefix_point_score_not_supported` | Hybrid did not meet the source-calibrated prefix RMSE boundary. |
| `prefix_probabilistic_score_not_supported` | Hybrid did not meet the prefix Gaussian log-score boundary. |
| `correction_outside_source_scale` | The proposed correction is too large relative to physics uncertainty. |
| `descriptor_outside_source_support` | The residual-model feature leverage exceeds source support. |

Multiple reasons are retained rather than collapsed into one label.

## Claim boundary

This gate is a controlled-method development component. Unit tests establish
prefix invariance, exact fallback, immutable inputs, source/target disjointness,
order-independent source calibration, independently reconstructed artifact
fields, and fail-closed support checks. They do not establish that the gate
improves real deformable-object prediction or calibrated uncertainty. That
requires fresh source calibration and prospective held-out evidence.
