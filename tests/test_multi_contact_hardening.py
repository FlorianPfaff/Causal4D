from __future__ import annotations

import numpy as np
import pytest

from causal4d.dynamic_contact import DynamicContactInferenceConfig
from causal4d.multi_contact import (
    MultiContactInferencePolicy,
    MultiContactInferenceRejectedError,
    MultiContactPathBank,
    infer_multi_contact_posterior,
)


def _config(*, confidence_level: float = 0.90) -> DynamicContactInferenceConfig:
    return DynamicContactInferenceConfig(
        observation_scale_m=0.1,
        dynamic_likelihood_weight=0.0,
        switch_variance_m2=0.0,
        command_change_variance_m2=0.0,
        ood_variance_m2=0.0,
        confidence_level=confidence_level,
    )


def _one_path_bank(
    *,
    value: float = 0.0,
    retained_prior_mass: float = 1.0,
    replay_result_identity: str | None = None,
    frame_times_s: np.ndarray | None = None,
) -> MultiContactPathBank:
    return MultiContactPathBank(
        contact_ids=("left",),
        path_ids=("static",),
        regime_paths=np.zeros((1, 1, 3), dtype=np.int8),
        trajectories_m=np.full((1, 3, 1, 2), value, dtype=float),
        prior_weights=np.ones(1),
        base_variance_m2=1e-6,
        retained_prior_mass=retained_prior_mass,
        replay_result_identity=replay_result_identity,
        frame_times_s=frame_times_s,
    )


def test_heteroscedastic_likelihood_penalizes_variance_inflation() -> None:
    bank = MultiContactPathBank(
        contact_ids=("left",),
        path_ids=("precise", "inflated"),
        regime_paths=np.zeros((2, 1, 3), dtype=np.int8),
        trajectories_m=np.zeros((2, 3, 1, 2), dtype=float),
        prior_weights=np.full(2, 0.5),
        base_variance_m2=np.asarray(
            [
                [[0.0, 0.0]],
                [[1.0, 1.0]],
            ]
        ),
    )
    posterior = infer_multi_contact_posterior(
        bank,
        np.zeros((3, 1, 2)),
        prefix_frame_count=2,
        command_activation=np.zeros((1, 3)),
        config=_config(),
    )
    assert posterior.weights[0] > 0.90
    assert posterior.weights[0] > posterior.weights[1]


def test_rollout_identity_binds_trajectory_variance_replay_and_timebase() -> None:
    times = np.asarray([0.0, 0.1, 0.2])
    baseline = _one_path_bank(
        replay_result_identity="provider-result-a",
        frame_times_s=times,
    )
    fortran_order = MultiContactPathBank(
        contact_ids=baseline.contact_ids,
        path_ids=baseline.path_ids,
        regime_paths=np.asfortranarray(baseline.regime_paths),
        trajectories_m=np.asfortranarray(baseline.trajectories_m),
        prior_weights=baseline.prior_weights,
        base_variance_m2=baseline.base_variance_m2,
        replay_result_identity="provider-result-a",
        frame_times_s=np.asfortranarray(times),
    )
    changed_trajectory = _one_path_bank(
        value=0.01,
        replay_result_identity="provider-result-a",
        frame_times_s=times,
    )
    changed_replay = _one_path_bank(
        replay_result_identity="provider-result-b",
        frame_times_s=times,
    )
    changed_variance = MultiContactPathBank(
        contact_ids=baseline.contact_ids,
        path_ids=baseline.path_ids,
        regime_paths=baseline.regime_paths,
        trajectories_m=baseline.trajectories_m,
        prior_weights=baseline.prior_weights,
        base_variance_m2=2e-6,
        replay_result_identity="provider-result-a",
        frame_times_s=times,
    )
    assert baseline.replay_bound
    assert baseline.schedule_identity == changed_trajectory.schedule_identity
    assert baseline.rollout_identity == fortran_order.rollout_identity
    assert baseline.rollout_identity != changed_trajectory.rollout_identity
    assert baseline.rollout_identity != changed_replay.rollout_identity
    assert baseline.rollout_identity != changed_variance.rollout_identity


def test_rollout_timebase_and_external_identity_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _one_path_bank(
            replay_result_identity="provider-result",
            frame_times_s=np.asarray([0.0, 0.2, 0.1]),
        )
    with pytest.raises(ValueError, match="nonempty string"):
        _one_path_bank(
            replay_result_identity="",
            frame_times_s=np.asarray([0.0, 0.1, 0.2]),
        )


def test_retained_support_gate_raises_or_uses_exact_static_fallback() -> None:
    candidate = _one_path_bank(value=9.0, retained_prior_mass=0.4)
    fallback = _one_path_bank(value=2.0)
    policy = MultiContactInferencePolicy(minimum_retained_prior_mass=0.9)
    observations = np.zeros((3, 1, 2))
    activation = np.zeros((1, 3))

    with pytest.raises(
        MultiContactInferenceRejectedError,
        match="retained_prior_mass_below_minimum",
    ):
        infer_multi_contact_posterior(
            candidate,
            observations,
            prefix_frame_count=2,
            command_activation=activation,
            config=_config(),
            policy=policy,
        )

    posterior = infer_multi_contact_posterior(
        candidate,
        observations,
        prefix_frame_count=2,
        command_activation=activation,
        config=_config(),
        policy=policy,
        static_fallback_bank=fallback,
    )
    assert np.array_equal(posterior.weights, np.ones(1))
    assert np.array_equal(posterior.mean_m, fallback.trajectories_m[0])
    assert posterior.metadata["fallback_used"] is True
    assert posterior.metadata["support_gate_passed"] is False
    assert posterior.metadata["support_gate_violations"] == [
        "retained_prior_mass_below_minimum"
    ]
    assert posterior.metadata["rejected_rollout_identity"] == (
        candidate.rollout_identity
    )


def test_omitted_posterior_bound_is_admission_relevant() -> None:
    candidate = _one_path_bank(retained_prior_mass=0.5)
    observations = np.zeros((3, 1, 2))
    activation = np.zeros((1, 3))
    posterior = infer_multi_contact_posterior(
        candidate,
        observations,
        prefix_frame_count=2,
        command_activation=activation,
        config=_config(),
    )
    omitted_bound = float(posterior.metadata["omitted_posterior_mass_upper_bound"])
    assert 0.5 <= omitted_bound < 0.501
    with pytest.raises(
        MultiContactInferenceRejectedError,
        match="omitted_posterior_mass_above_maximum",
    ):
        infer_multi_contact_posterior(
            candidate,
            observations,
            prefix_frame_count=2,
            command_activation=activation,
            config=_config(),
            policy=MultiContactInferencePolicy(maximum_omitted_posterior_mass=0.5),
        )


def test_replay_binding_can_be_required_for_admission() -> None:
    observations = np.zeros((3, 1, 2))
    activation = np.zeros((1, 3))
    policy = MultiContactInferencePolicy(require_replay_binding=True)
    with pytest.raises(
        MultiContactInferenceRejectedError,
        match="replay_binding_required",
    ):
        infer_multi_contact_posterior(
            _one_path_bank(),
            observations,
            prefix_frame_count=2,
            command_activation=activation,
            config=_config(),
            policy=policy,
        )
    admitted = infer_multi_contact_posterior(
        _one_path_bank(
            replay_result_identity="provider-result",
            frame_times_s=np.asarray([0.0, 0.1, 0.2]),
        ),
        observations,
        prefix_frame_count=2,
        command_activation=activation,
        config=_config(),
        policy=policy,
    )
    assert admitted.metadata["support_gate_passed"] is True
    assert admitted.metadata["replay_bound"] is True


def test_intervals_are_marginal_gaussian_mixture_quantiles() -> None:
    trajectories = np.zeros((2, 3, 1, 2), dtype=float)
    trajectories[0, 1:, 0, 0] = -1.0
    trajectories[1, 1:, 0, 0] = 1.0
    bank = MultiContactPathBank(
        contact_ids=("left",),
        path_ids=("negative", "positive"),
        regime_paths=np.zeros((2, 1, 3), dtype=np.int8),
        trajectories_m=trajectories,
        prior_weights=np.full(2, 0.5),
        base_variance_m2=1e-6,
    )
    posterior = infer_multi_contact_posterior(
        bank,
        np.zeros((3, 1, 2)),
        prefix_frame_count=1,
        command_activation=np.zeros((1, 3)),
        config=_config(),
    )
    lower = float(posterior.interval_lower_m[2, 0, 0])
    upper = float(posterior.interval_upper_m[2, 0, 0])
    assert -1.01 < lower < -0.99
    assert 0.99 < upper < 1.01
    assert upper - lower < 2.1
    assert posterior.metadata["interval_method"] == (
        "conditional_gaussian_mixture_quantiles"
    )


def test_posterior_metadata_is_recursively_immutable() -> None:
    posterior = infer_multi_contact_posterior(
        _one_path_bank(),
        np.zeros((3, 1, 2)),
        prefix_frame_count=2,
        command_activation=np.zeros((1, 3)),
        config=_config(),
    )
    with pytest.raises(TypeError, match="immutable"):
        posterior.metadata["config"]["observation_scale_m"] = 4.0
    with pytest.raises(TypeError, match="immutable"):
        posterior.metadata["support_gate_violations"].append("tampered")


def test_policy_numeric_contracts_reject_boolean_coercion() -> None:
    with pytest.raises(ValueError, match="real probability"):
        MultiContactInferencePolicy(
            minimum_retained_prior_mass=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="Boolean"):
        MultiContactInferencePolicy(
            require_replay_binding=np.bool_(True),  # type: ignore[arg-type]
        )
