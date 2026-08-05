# Factorized Multi-Contact Path Beliefs

## Status and scope

This is an experimental, backend-neutral extension of the dynamic contact-path
model. It does not modify any frozen Causal4D estimator, registered physical
protocol, or claim-bearing result. Its purpose is to remove a concrete modeling
restriction in the existing development prototype: one global contact regime
cannot represent independently changing left/right gripper contacts.

The implementation supports an arbitrary number of named contact channels. A
channel can represent a gripper, a support contact, or another independently
realized force-transmission path. The current prior factorizes across channels;
coupled contact mechanics are represented by the continuously simulated joint
trajectory, not by splicing marginal rollouts.

## Joint path enumeration

For contact channels `g = 1, ..., G`, the prior is

```text
p(kappa_1:T) = product_g p(kappa_g,1:T | a_g,1:T).
```

Each marginal path support is generated with the existing
`enumerate_contact_paths` implementation and can use its own
`ContactTransitionConfig`. `enumerate_multi_contact_paths` then extracts the
highest-probability joint products with a deterministic best-first heap. It does
not construct the full Cartesian product.

For a retained marginal mass `m_g` and selected normalized product mass `s`, the
reported joint mass is

```text
retained_prior_mass = s * product_g m_g.
```

This accounts for both marginal beam pruning and joint top-k pruning. The optional
`minimum_joint_probability` threshold is evaluated against the original prior
mass, before retained joint weights are renormalized. Increasing
`maximum_joint_paths` cannot reduce the retained mass; this invariant is tested.
The returned object also records the marginal path indices and the size of the
unpruned Cartesian support for auditing.

A one-contact call is exactly equivalent to the existing single-contact
enumerator, apart from the added contact axis and named joint path identifier.
Setting all transition hazards to zero returns one all-inactive joint path with
unit mass for any number of contacts.

## Continuous simulation and identity boundary

`MultiContactPathBank` associates each complete joint schedule with one
trajectory of shape `(T, N, C)`. The trajectory must be produced by one
continuous simulator execution for that schedule. It must not be assembled by
splicing independently simulated contact segments, because that would generally
break state, velocity, and internal-stress continuity.

The low-level constructor remains available for controlled or synthetic
providers:

```python
from causal4d.multi_contact import MultiContactPathBank

bank = MultiContactPathBank.from_prior(
    prior,
    trajectories_m,                 # shape (K, T, N, 3)
    base_variance_m2=base_variance,
    replay_result_identity=provider_result_identity,
    frame_times_s=frame_times_s,
)
```

For Bayesian-PhysTwin integration, use the verified adapter instead of supplying
an arbitrary replay identity:

```python
from causal4d.multi_contact import (
    MultiContactEnumerationConfig,
    enumerate_multi_contact_paths,
    replay_multi_contact_prior,
)

prior = enumerate_multi_contact_paths(
    command_activation,             # shape (G, T)
    contact_ids=("left", "right"),
    transition_configs=(left_config, right_config),
    config=MultiContactEnumerationConfig(maximum_joint_paths=128),
)

evidence = replay_multi_contact_prior(
    prior,
    scheduled_provider,
    request_id="case-17-contact-bank",
    simulator_configuration_id=simulator_configuration_id,
    initial_state_id=twin_endpoint_id,
    group_log_scales=particle_log_scales,
    controller_points_m=controller_points,
    position_m=endpoint_position,
    velocity_mps=endpoint_velocity,
    frame_times_s=frame_times,
    contact_node_indices=contact_patch_indices,
    contact_node_weights=contact_patch_weights,
    normal_stiffness_npm=normal_stiffness,
    tangential_stiffness_npm=tangential_stiffness,
    friction_coefficient=friction,
)
bank = evidence.bank
```

The adapter loads the additive contracts from
`bayesian_phystwin.causal4d_provider_v2`, verifies that the runtime provider
implements `ScheduledContactReplayProviderV1`, checks the provider configuration
before execution, constructs a content-addressed physical request, revalidates
the complete result, and then creates the rollout bank. Provider, request, replay,
configuration, state, schedule, path, regime, and timebase drift fail closed.

The schedule and rollout identities are deliberately separate:

- `schedule_identity` binds contact names, path identifiers, complete regime
  schedules, normalized prior weights, and retained prior mass;
- the Bayesian-PhysTwin request identity additionally binds endpoint state,
  controller motion, finite-area contact geometry, contact mechanics, physical
  parameters, and explicit timebase;
- the provider replay-result identity binds the request plus every returned
  trajectory, conditional-variance, and provider-revision byte; and
- Causal4D's `rollout_identity` additionally binds that verified replay identity
  to the consumed trajectory and variance arrays.

Changing a trajectory, variance tensor, replay identity, or timebase changes the
rollout identity even when the schedule remains unchanged. Equal arrays with
different NumPy memory layouts retain the same identity. Claim-bearing use can
require both a replay-result identity and a strictly increasing explicit
timebase.

The public contract is now implemented, but the ordinary official PhysTwin/Warp
provider does not yet advertise or implement dynamic scheduled contact. Contract
validity must not be confused with physical replay competence.

## Prefix-only normalized likelihood

`infer_multi_contact_posterior` performs robust Student-t reweighting using only
frames before `prefix_frame_count`. The known future activation sequence is part
of the intervention query and may be used for prior generation and uncertainty
propagation. Future object observations are never read.

For a residual `r`, scale `s`, and degrees of freedom `nu`, the compared score
retains the heteroscedastic normalization term:

```text
-log(s) - (nu + 1) / 2 * log(1 + r^2 / (nu * s^2)).
```

The `-log(s)` term is required because conditional uncertainty differs between
contact paths. Omitting it would allow a path to improve its likelihood merely
by inflating its variance. Constants that are shared by all compared paths are
omitted.

The posterior returns:

- joint path weights and predictive trajectory moments;
- per-contact, per-frame probabilities for inactive, sticking, slipping, and
  detached regimes;
- per-contact switch probabilities;
- the probability that any contact switches at each frame;
- an active-contact probability for each contact and frame;
- schedule and rollout identities; and
- retained-support and omitted-posterior diagnostics.

Changing held-out observations leaves weights and predictive moments byte-exact.
The test suite also checks contact-label permutation symmetry when the complete
joint support is retained.

## Retained-support admission and exact static fallback

Renormalizing a narrow retained beam must not silently turn it into complete
support. `MultiContactInferencePolicy` can require all of the following:

```python
from causal4d.multi_contact import MultiContactInferencePolicy

policy = MultiContactInferencePolicy(
    minimum_retained_prior_mass=0.95,
    maximum_omitted_posterior_mass=0.05,
    require_replay_binding=True,
)
```

The omitted-posterior bound combines the retained prior mass, the exact retained
average likelihood, and a conservative upper bound on the likelihood of any
omitted path. It therefore can be stricter or less conservative than a prior-mass
threshold alone while remaining independent of held-out observations.

When a configured gate fails, inference raises
`MultiContactInferenceRejectedError` unless the caller supplies
`static_fallback_bank`. The fallback must contain exactly one rollout, retain
unit mass, use the same dimensions and contact identifiers, and keep every
contact regime constant over time. The returned posterior then uses that static
trajectory exactly and records the rejected schedule, rollout, mass bound, and
fallback identities in recursively immutable metadata.

The default policy leaves the development API permissive. A claim-bearing
protocol must freeze nontrivial thresholds and an exact fallback before target
access.

## Intervention-conditioned uncertainty and intervals

Conditional variance accumulates uncertainty for every contact transition, not
merely for a collapsed global state:

```text
V_k(t) = V_k(0)
       + q_switch * cumulative_number_of_contact_switches_k(t)
       + q_command * cumulative_sum_g delta(a_g,t)^2
       + q_ood * cumulative_ood_energy(t).
```

Thus simultaneous or staggered bimanual contact changes contribute separately.
A global OOD distance with shape `(T,)` is counted once, while a contact-specific
array with shape `(G, T)` contributes the sum of squared per-contact distances.
Setting all inflation coefficients to zero preserves the supplied conditional
variance exactly.

Marginal credible intervals are computed as quantiles of the conditional
Gaussian path mixture. They are not formed as `mean +/- z * standard deviation`,
which can place interval mass in the low-density gap between separated contact
modes. The posterior still reports the law-of-total-variance moment for summary
and calibration diagnostics.

## Current limitations

This implementation deliberately does not claim calibrated real-data contact
prediction. The contact chains are independent in the prior and currently use
the single-contact Markov transition parameterization. It does not yet provide
source-fitted duration distributions, cross-gripper transition coupling, tactile
label construction, contact-point migration, calibrated force-transmission
parameters, or an official PhysTwin/Warp implementation of the scheduled replay
protocol.

The next evidence-bearing steps are therefore:

1. fit transition, duration, and force-transmission parameters on source
   interactions only;
2. implement finite-area moving contact in a Bayesian-PhysTwin provider that
   executes one continuous rollout per schedule and returns the verified result
   contract without silently omitting failed paths;
3. freeze support thresholds and the exact static fallback before target access;
4. evaluate contact onset, offset, calibration, retained support mass, and
   held-out trajectory prediction on a prospectively reserved cohort; and
5. retain exact fallback to the frozen static operator when support, replay
   binding, or calibration gates fail.
