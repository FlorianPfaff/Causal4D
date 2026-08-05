# Dynamic Contact-Path Inference

## Status and scope

This is a development extension of Causal4D. It does not modify the frozen
`v0.3.0-causal4d-aip` milestone and is not promoted as real-data evidence. Its
purpose is to close one explicit model gap: the mathematical formulation treats
contact as a path-valued variable `kappa_t`, whereas the first PhysTwin rollout
bank associates one static contact summary with each complete rollout.

The extension is motivated by the Deform360 `001-rope` target, where a gripper
is inactive at the final prefix frame but activates at the first held-out frame.
A static prefix-conditioned contact state necessarily misses that onset. The
new code evaluates whether an action-conditioned distribution over future
contact paths can represent such cases without consuming future observations.

## Contact regimes

Each frame has one discrete regime:

- `inactive`: no realized force transmission;
- `sticking`: active contact with no modeled sliding;
- `slipping`: active contact with reduced or redistributed transmission;
- `detached`: a previously active contact has separated.

The transition matrix is conditioned on a normalized command-activation signal
and its frame-to-frame change. The default model permits activation,
stick-to-slip transitions, release, recovery, and reattachment. The transition
configuration is explicit and serializable.

`enumerate_contact_paths` uses deterministic beam pruning. It returns the
retained prior mass in addition to normalized retained weights, so aggressive
pruning cannot be silently interpreted as complete support.

## Backend boundary

`DynamicContactPathBank` is simulator-neutral. A backend supplies:

- one complete regime path per component;
- one physical/readout trajectory per path;
- a prior weight;
- scalar, node-wise, component-wise, or time-varying conditional variance.

The module does not splice static trajectories at a regime switch. Such
splicing would generally violate simulator state continuity. PhysTwin/Warp or
another simulator must generate the path-conditioned trajectory, including
correct restart state and contact mechanics.

## Prefix-only inference

For a bank of trajectories `X^(k)` and observations through frame `tau`, the
posterior is

```text
w_k^+ proportional to w_k^- L(O_0:tau | X_0:tau^(k)).
```

The likelihood is a robust Student-t score over position and, optionally,
frame differences. It consumes only frames before `prefix_frame_count`.
Changing any held-out observation leaves posterior weights and predictive
moments unchanged; this is covered by a byte-exact unit test.

The known future command is allowed because it is part of the intervention
query. Future object observations are not allowed.

## Intervention-conditioned uncertainty

The per-component conditional variance evolves as

```text
V_k(t) = V_k(0)
       + q_switch * cumulative_switches_k(t)
       + q_command * cumulative_squared_command_changes(t)
       + q_ood * cumulative_squared_action_ood(t).
```

The predictive variance then includes both path-mixture dispersion and the
weighted conditional variance. This preserves the successful persistent mean
while allowing uncertainty to widen after predicted contact changes, abrupt
commands, or action extrapolation.

Setting all three inflation coefficients to zero recovers the supplied static
conditional variance exactly. Setting all transition hazards to zero produces
one constant-regime path with unit weight.

## Controlled delayed-onset benchmark

Run:

```bash
causal4d benchmark dynamic-contact \
  --require-gates \
  --output-json runs/dynamic-contact-delayed-onset-v1.json
```

The benchmark fixes an inactive observed prefix and a known command activation
at the first held-out frame. It compares static prefix persistence with the
dynamic path mixture and checks:

1. at least 50% RMSE improvement over static prefix persistence;
2. posterior expected onset within one frame of the controlled onset;
3. zero future observations consumed by inference.

This is a software and controlled-model test. It is not evidence that the
transition probabilities are calibrated for Deform360 or real PhysTwin data.
Those probabilities must be frozen on source interactions and evaluated under
the locked same-object multi-action protocol.

## PhysTwin integration path

A real integration should:

1. generate a source-frozen contact path beam from command, registration, and
   optional tactile/contact evidence;
2. run official Warp continuously for each retained path and Bayesian-PhysTwin
   particle;
3. build `DynamicContactPathBank` from those trajectories;
4. reweight with the permitted response prefix;
5. report contact-onset, regime, trajectory, and calibration metrics by horizon
   and graph region;
6. retain the static Causal4D operator as the exact zero-hazard control.

The extension should not delay the registered 36-execution acquisition. It is a
specific candidate for the already observed contact-onset failure, not a reason
to reopen unrestricted architecture search on exhausted cases.

## Multiple contact channels

The single-contact API remains unchanged. Independently changing left/right or
support contacts are represented by the factorized joint path support in
[`causal4d_multi_contact_paths.md`](causal4d_multi_contact_paths.md). That
extension preserves one-contact behavior, tracks joint retained mass, and
requires one continuous simulator trajectory for every retained joint schedule.
