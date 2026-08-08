# Prospective V2 promotion and deployment profile

## Scope

The modules described here are **prospective infrastructure**. They do not
change the frozen estimator, the registered physical-acquisition candidate, the
18-session/36-execution protocol, an existing prediction seal, or any recorded
evidence count.

They address three separate failure modes in the experimental V2 path:

1. a support reduction can preserve marginal behavior while changing a
   task-relevant joint readout;
2. a generic decision trace can omit a newly required V2 certificate; and
3. a candidate ladder can be evaluated or selected without one target opening,
   independent-unit aggregation, or exact baseline fallback.

The implementation is split across:

```text
causal4d.projected_functional_support_v1
causal4d.prospective_v2_profile
causal4d.prospective_v2_promotion
```

## Task-projected functional support

`functional_support_v1` remains the base rollout-space certificate. The
projected certificate adds a frozen set of linear task readouts over
`(frame, node, coordinate)` and checks, for every source action and projection:

- projected predictive variance relative error; and
- projected Gaussian-mixture interval endpoint error.

The projected variance includes all three contributions available at this
boundary:

1. coordinatewise conditional variance;
2. posterior mixture covariance between component trajectories; and
3. optional component-specific low-rank conditional trajectory modes.

For projection coefficients `a`, component mean `mu`, independent variance
`d`, and low-rank factors `F`, the conditional projected variance is

```text
sum(a**2 * d) + sum_r (a^T F_r)**2
```

The mixture variance then applies the law of total variance across support
components. This catches, for example, two supports with identical coordinate
marginals but opposite correlation along an endpoint or graph-mode readout.

A projected certificate is valid only when:

- the base certificate covers the same actions in the same order;
- the base certificate binds every base action artifact;
- the projected metrics contain the complete action-by-projection product;
- the base decision and every projected metric exactly imply the serialized
  decision; and
- no target outcome or target-loss field appears in source-only metadata.

Example:

```python
from causal4d.projected_functional_support_v1 import (
    FunctionalSupportProjectionV1,
    ProjectedFunctionalSupportActionV1,
    ProjectedFunctionalSupportPolicyV1,
    certify_projected_functional_support_v1,
)

projected_action = ProjectedFunctionalSupportActionV1(
    action=source_action,
    full_component_low_rank_factors_m=full_modes,
    reduced_component_low_rank_factors_m=reduced_modes,
)
projection = FunctionalSupportProjectionV1(
    projection_id="late-endpoint-displacement",
    coefficients=late_endpoint_coefficients,
)
certificate = certify_projected_functional_support_v1(
    (projected_action,),
    (projection,),
    policy=ProjectedFunctionalSupportPolicyV1(
        maximum_projected_variance_relative_error=0.10,
        maximum_projected_interval_endpoint_error_m=0.002,
    ),
    base_certificate=base_functional_support_certificate,
    source_artifact_ids=(source_projection_freeze_id,),
)
```

Passing a plain `FunctionalSupportActionV1` remains supported and is equivalent
to supplying no low-rank factors.

## Versioned prospective V2 decision profile

The generic `UnifiedDecisionTrace` intentionally supports experiment-specific
decision inventories. The prospective profile freezes the complete inventory
under:

```text
causal4d.prospective-v2-deployment-profile/v1
```

The required decisions are:

```text
prob4d_provider_acceptance
joint_covariance_admission
bayesian_phystwin_acceptance
functional_support
intervention_identifiability
action_support
contact_v2_support
conditional_uncertainty_calibration
query_calibration
counterfactual_regret
```

The profile also fixes the stage and producer for every decision. In
particular, joint-covariance admission and functional support must be recorded
at Causal4D abduction, while conditional-uncertainty calibration must be
recorded at counterfactual prediction.

Use `build_prospective_v2_decision_trace_v1` at runtime. It delegates to the
generic builder, preserves its exact Python-object identity check, inserts the
profile ID into content-addressed metadata, and then validates the stricter
profile.

A profile validation can be structurally accepted while a required decision is
rejected. That means the trace is a valid V2 trace and correctly deploys the
exact baseline object. It does **not** mean that the V2 candidate was accepted.
The rejected required-decision names are recorded explicitly.

## Two-phase promotion experiment

`ProspectiveV2PromotionFreezeV1` is written before target outcomes are opened.
It binds:

- the exact three-repository stack lock;
- the exact candidate and source-configuration artifact IDs;
- all independent evaluation units and endpoint assignments;
- the source-artifact inventory;
- every promotion threshold; and
- `target_outcomes_used = false`.

The candidate ladder is fixed and ordered:

```text
registered_v1
normalized_diagonal_or_block_covariance
normalized_full_joint_covariance
support_certified_contact_patch
complete_v2_structured_uncertainty
```

The result requires the complete Cartesian product of registered unit and
candidate. Every metric row must carry the same content-addressed evaluation
opening ID and must declare `target_outcomes_used = true`. Missing rows,
unregistered rows, duplicate rows, mixed openings, or endpoint mismatches fail
closed.

All gates are evaluated per endpoint at the registered independent-unit level:

- mean log-score gain relative to the registered baseline;
- mean Brier-score change;
- mean trajectory regret;
- mean coverage error;
- mean interval-width ratio;
- accepted-update rate;
- harmful accepted-update rate, defined conditionally on acceptance; and
- fallback rate.

The minimum accepted-update rate must be positive and the maximum fallback rate
must be below one, so a candidate cannot pass by abstaining on every unit.

The highest candidate in the frozen ladder that passes every endpoint is
selected. When no candidate passes, the result preserves the exact registered
baseline artifact ID. Direct result construction also verifies that the
serialized selection equals the highest accepted result or that exact baseline.

Example:

```python
from causal4d.prospective_v2_promotion import (
    ProspectiveV2PromotionFreezeV1,
    ProspectiveV2PromotionPolicyV1,
    evaluate_prospective_v2_promotion_v1,
    write_prospective_v2_promotion_freeze,
    write_prospective_v2_promotion_result,
)

freeze = ProspectiveV2PromotionFreezeV1(
    experiment_id="rope-v2-untouched-panel-1",
    stack_lock_id=stack_lock["lock_id"],
    candidates=candidate_ladder,
    evaluation_units=independent_units,
    policy=ProspectiveV2PromotionPolicyV1(
        minimum_units_per_endpoint=8,
        minimum_mean_log_score_gain=0.0,
        maximum_mean_brier_change=0.0,
        maximum_mean_trajectory_regret_m=0.0,
        maximum_mean_coverage_error=0.05,
        maximum_mean_interval_width_ratio=1.20,
        minimum_accepted_update_rate=0.50,
        maximum_harmful_accepted_update_rate=0.10,
        maximum_fallback_rate=0.50,
    ),
    source_artifact_ids=source_artifact_ids,
)
write_prospective_v2_promotion_freeze("promotion-freeze.json", freeze)

# Only after the freeze is independently sealed and the evaluation panel opens:
result = evaluate_prospective_v2_promotion_v1(freeze, evaluation_metrics)
write_prospective_v2_promotion_result("promotion-result.json", result)
```

## Metric orientation

The promotion API assumes that larger log score is better. Therefore,
`mean_log_score_gain` is candidate minus baseline. Brier change and trajectory
regret are also candidate minus baseline, so lower values are better. Coverage
error is an absolute error supplied per independent unit. Interval width is
compared as candidate divided by baseline after applying the frozen positive
width floor.

`harmful_update` is meaningful only when `candidate_accepted` is true.
`fallback_used` must be the exact Boolean complement of `candidate_accepted`.
The registered baseline rows must be accepted, nonfallback, and nonharmful.

## Scientific boundary

These contracts make the prospective experiment harder to accidentally change,
but they do not turn source-only calibration into target evidence and they do
not promote a method by themselves. The candidate ladder, projections,
thresholds, independent units, source artifacts, and stack lock must be reviewed
and sealed before opening the untouched panel. A negative result is complete and
must retain the registered baseline rather than trigger target-driven method
revision.
