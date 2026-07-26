# Signed Continuous-Time Dynamics and Query Gates

## Scope

This development extension closes three gaps without changing the frozen
`v0.3.0-causal4d-aip` path:

1. discrepancy-mean dynamics need signed realization and action-phase features;
2. learned dynamics should not change merely because the camera frame rate
   changes;
3. local Jacobian identifiability should be complemented by a global check over
   the actual finite rollout support.

It also introduces a versioned capability contract for the Bayesian-PhysTwin
provider. None of these additions constitute new held-out real-data evidence.

## Feature schemas

`build_action_conditioned_features` retains `magnitude_v1` as its default and
byte-compatible semantic path. It uses magnitudes for gain deviation, delay,
rotation, slip, and attachment shift and remains appropriate for conservative
covariance inflation.

The opt-in `signed_v2` schema additionally preserves:

- signed gain deviation;
- signed rotation through sine and cosine-minus-one coordinates;
- signed mean attachment shift;
- radial velocity and distance from the controller anchor;
- the existing Cartesian direction, speed, acceleration, slip, and contact-policy
  fields.

The radial fields distinguish outbound, hold, and return behavior without
requiring post hoc phase labels. Every generated feature artifact records its
schema and physical step duration.

## Continuous-time parameterization

Both `ActionConditionedDiscrepancyModel` and
`StableDiscrepancyTransitionModel` accept:

```text
time_parameterization = per_step | per_second
```

`per_step` preserves all previous behavior. Under `per_second`, the stable mean
model is interpreted as

```text
dc/dt = G(f) c + b(f)
```

and discretized exactly with a matrix exponential for every declared frame
interval. The covariance model is interpreted as a covariance rate. When both
mean and covariance models are continuous-time, process covariance is
discretized with the Van Loan construction. Constant-feature predictions are
therefore invariant, up to numerical precision, to subdividing the same physical
interval into more frames.

The counterfactual readout operator accepts an optional stable transition model.
Omitting it preserves graph persistence exactly. A supplied transition model is
recorded in posterior metadata together with the feature schema, frame interval,
and time parameterizations.

## Finite-support query ambiguity

`assess_finite_query_ambiguity` compares every positive-mass pair in a finite
rollout support. A pair is ambiguous when its allowed-prefix responses are close
in whitened RMS Mahalanobis distance but its requested future-query responses are
far apart after declared query scaling.

The gate reports:

- ambiguous pair probability mass;
- pair-mass-weighted query divergence;
- maximum query divergence;
- the exact support pairs responsible for rejection.

Pair mass is defined as the probability of drawing the two components in either
order. Consequently, the aggregate is invariant to support permutation and to
splitting a component into exact clones whose weights sum to the original mass.
This is a global finite-support complement to local sensitivity-based
identifiability. Rejection should widen uncertainty or trigger fallback; it does
not authorize target-future tuning.

## Bayesian-PhysTwin provider contract

`PhysicalBeliefProviderManifest` records provider identity, revision, contract
schema, artifact-schema versions, and explicit capabilities. The compatibility
gate fails closed on:

- missing endpoint, parameter, or checksum capabilities;
- unsupported provider-contract versions;
- mismatched required artifact versions.

The existing pinned Bayesian-PhysTwin revision remains a reproducible installation
choice. The manifest supplies a stable semantic boundary so future compatible
providers do not need to be accepted solely by implementation commit identity.

## Promotion boundary

Signed dynamics, covariance rates, ambiguity thresholds, graph rank, and provider
requirements must be selected from source data or preregistered. The registered
same-object multi-action protocol remains the promotion gate for held-out
accuracy, transfer, and independent-execution calibration.
