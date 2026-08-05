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

## Continuous simulation boundary

`MultiContactPathBank` associates each complete joint schedule with one
trajectory of shape `(T, N, C)`. The trajectory must be produced by one
continuous simulator execution for that schedule. It must not be assembled by
splicing independently simulated contact segments, because that would generally
break state, velocity, and internal-stress continuity.

Use `MultiContactPathBank.from_prior` after a simulator has executed every
retained schedule:

```python
from causal4d.multi_contact import (
    MultiContactEnumerationConfig,
    MultiContactPathBank,
    enumerate_multi_contact_paths,
)

prior = enumerate_multi_contact_paths(
    command_activation,             # shape (G, T)
    contact_ids=("left", "right"),
    transition_configs=(left_config, right_config),
    config=MultiContactEnumerationConfig(maximum_joint_paths=128),
)

bank = MultiContactPathBank.from_prior(
    prior,
    trajectories_m,                 # shape (K, T, N, 3)
    base_variance_m2=base_variance,
)
```

The bank and prior expose the same deterministic `schedule_identity`. The
identity covers contact names, path identifiers, complete regime schedules,
normalized prior weights, retained mass, and an explicit schema version. A
future BayesianPhysTwin provider contract can therefore bind a replay result to
the exact schedule support without depending on Causal4D internals.

## Prefix-only posterior inference

`infer_multi_contact_posterior` performs robust Student-t reweighting using only
frames before `prefix_frame_count`. The known future activation sequence is part
of the intervention query and may be used for prior generation and uncertainty
propagation. Future object observations are never read.

The posterior returns:

- joint path weights and predictive trajectory moments;
- per-contact, per-frame probabilities for inactive, sticking, slipping, and
  detached regimes;
- per-contact switch probabilities;
- the probability that any contact switches at each frame;
- an active-contact probability for each contact and frame;
- the exact schedule identity and retained prior mass in metadata.

Changing held-out observations leaves weights and predictive moments byte-exact.
The test suite also checks contact-label permutation symmetry when the complete
joint support is retained.

## Intervention-conditioned uncertainty

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

## Current limitations

This implementation deliberately does not claim calibrated real-data contact
prediction. The contact chains are independent in the prior and currently use
the single-contact Markov transition parameterization. It does not yet provide
source-fitted duration distributions, cross-gripper transition coupling, tactile
label construction, or a BayesianPhysTwin dynamic-schedule replay capability.

The next evidence-bearing steps are therefore:

1. fit transition and duration parameters on source interactions only;
2. add an additive BayesianPhysTwin provider capability that executes one
   continuous rollout per schedule identity;
3. evaluate contact onset, offset, calibration, retained support mass, and
   held-out trajectory prediction on a prospectively reserved Deform360 cohort;
4. retain exact fallback to the frozen static operator when support or
   calibration gates fail.
