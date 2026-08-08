# Prospective V2 decision profile

## Purpose

The generic `UnifiedDecisionTrace` supports experiment-specific decision
inventories. The prospective V2 path needs a stricter, versioned inventory so a
candidate cannot be selected while silently omitting joint-covariance,
functional-support, conditional-uncertainty, or baseline-relative-regret
evidence.

The profile identity is:

```text
causal4d.prospective-v2-deployment-profile/v1
```

## Required decisions

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

The profile fixes the stage and producer for every decision. In particular,
joint-covariance admission and functional support belong to Causal4D abduction,
while conditional-uncertainty calibration belongs to counterfactual prediction.

Use `build_prospective_v2_decision_trace_v1` to construct the runtime trace. It:

1. delegates the artifact graph and exact-object selection to the generic
   decision-trace builder;
2. inserts the profile identity into content-addressed metadata; and
3. validates the complete V2 inventory, stages, producers, and selection state.

A profile validation may be structurally accepted while one or more required
decisions are rejected. In that case the trace is a valid V2 trace that records
and deploys the exact baseline. It does not mean that the candidate passed. The
rejected required-decision names remain explicit in the validation artifact.

This profile is prospective integration infrastructure. It does not alter the
frozen acquisition candidate, registered 18-session/36-execution protocol,
target boundary, or evidence count.
