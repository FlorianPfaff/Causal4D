# Structured conditional uncertainty and functional support certification

This prospective extension closes two limitations of the latent-contact-v2 path
without changing the registered estimator, the frozen physical protocol, or any
existing artifact identity.

## Structured conditional uncertainty

`causal4d.conditional_uncertainty_v2` augments a
`ContactPatchRolloutBankV2` with two conditional uncertainty terms:

- componentwise independent residual variance, broadcast to the state/parameter
  rollout components and added to the bank's existing noise floor; and
- shared or component-specific Gaussian low-rank trajectory modes.

Every uncertainty object requires upstream artifact IDs and receives a
content-derived ID. Low-rank modes are projected through the existing
`LinearContactObservationGroup` operators before contact-evidence scoring. This
preserves correlations for positions, increments, or any other admitted linear
query instead of converting them prematurely to a diagonal variance.

```python
from causal4d.conditional_uncertainty_v2 import (
    ConditionalPredictiveUncertaintyV2,
    joint_predictive_moments_with_conditional_uncertainty_v2,
    posterior_weights_with_conditional_uncertainty_v2,
    predictive_distribution_with_conditional_uncertainty_v2,
)

uncertainty = ConditionalPredictiveUncertaintyV2(
    source_artifact_ids=(prob4d_belief_id, bpt_covariance_id),
    independent_variance_m2=residual_variance_m2,
    low_rank_factors_m=graph_mode_factors_m,
)

joint_weights, diagnostics = (
    posterior_weights_with_conditional_uncertainty_v2(
        bank,
        evidence,
        uncertainty,
        prefix_frame_count=prefix_frame_count,
    )
)
prediction = predictive_distribution_with_conditional_uncertainty_v2(
    bank,
    uncertainty,
    joint_weights,
)
```

The marginal predictive distribution uses exact one-dimensional
Gaussian-mixture quantiles. For bounded diagnostic windows,
`joint_predictive_moments_with_conditional_uncertainty_v2` materializes the full
trajectory covariance according to

\[
\operatorname{Cov}(Y)=
\mathbb{E}[\operatorname{Cov}(Y\mid Z,\Theta)] +
\operatorname{Cov}(\mathbb{E}[Y\mid Z,\Theta]).
\]

The dense joint covariance is intended for NEES, calibration, and covariance
consistency checks. It is deliberately not produced by default because its
storage is quadratic in the number of queried trajectory coordinates. The
function checks `maximum_joint_dimension` before entering either dense `einsum`
allocation; the default limit is 2,048 coordinates. Raising the limit is an
explicit operator decision. The rejection reports the requested dimension and
the minimum byte count of the dense covariance so an unexpectedly long query
cannot silently exhaust a workstation or CI runner.

## Functional support certificate

`causal4d.functional_support_v1` checks a finite parameter/support reduction in
rollout space on a frozen source-action library. It does not treat reassigned
probability mass or parameter-space moment preservation as sufficient evidence
for nonlinear predictive equivalence.

For every source action, the certificate compares:

- predictive mean RMSE normalized by the full predictive scale;
- predictive variance-trace error;
- maximum marginal interval-endpoint error; and
- weighted energy distance using trajectory RMS as the ground metric.

The energy-distance expectation uses Gram-matrix blocks rather than allocating a
`full_component × reduced_component × trajectory_dimension` difference tensor.
This preserves the exact weighted metric while bounding pairwise working memory
to fixed component blocks.

All thresholds are mandatory constructor arguments and should be frozen from
source-only data before target evaluation. The certificate fails if any source
action fails any threshold and binds the policy, per-action metrics, input
content hashes, and source artifact IDs into one deterministic certificate ID.

```python
from causal4d.functional_support_v1 import (
    FunctionalSupportActionV1,
    FunctionalSupportPolicyV1,
    certify_functional_support_v1,
)

source_case = FunctionalSupportActionV1(
    action_id="source-object/session-03/action-02",
    full_trajectories_m=full_components,
    full_weights=full_weights,
    reduced_trajectories_m=reduced_components,
    reduced_weights=reduced_weights,
)
policy = FunctionalSupportPolicyV1(
    maximum_normalized_mean_error=0.05,
    maximum_variance_trace_relative_error=0.10,
    maximum_interval_endpoint_error_m=0.002,
    maximum_energy_distance_m=0.002,
    minimum_action_count=12,
)
certificate = certify_functional_support_v1(
    source_actions,
    policy=policy,
    source_artifact_ids=(source_freeze_id, simulator_build_id),
)
```

An accepted certificate supports only the frozen source-action query family and
thresholds. It does not establish target-domain accuracy, physical confirmation,
or promotion of latent-contact v2. Those require the separately registered
prospective evaluation and physical experiment.
