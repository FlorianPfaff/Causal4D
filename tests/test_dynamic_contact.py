from __future__ import annotations

import numpy as np

from causal4d.cli.dynamic_contact_benchmark import delayed_contact_case
from causal4d.dynamic_contact import (
    ContactRegime,
    ContactTransitionConfig,
    DynamicContactInferenceConfig,
    DynamicContactPathBank,
    contact_conditioned_variance,
    contact_transition_matrix,
    enumerate_contact_paths,
    infer_dynamic_contact_posterior,
)


def _simple_bank() -> tuple[DynamicContactPathBank, np.ndarray]:
    activation = np.asarray([0.0, 0.0, 1.0, 1.0, 1.0])
    paths = np.asarray(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=np.int8,
    )
    trajectories = np.zeros((2, 5, 1, 3), dtype=float)
    trajectories[1, 2:, 0, 0] = [0.01, 0.02, 0.03]
    return (
        DynamicContactPathBank(
            path_ids=("inactive", "onset-2"),
            regime_paths=paths,
            trajectories_m=trajectories,
            prior_weights=np.asarray([0.25, 0.75]),
            base_variance_m2=1e-6,
        ),
        activation,
    )


def test_transition_matrix_is_row_stochastic() -> None:
    matrix = contact_transition_matrix(0.8, 0.2)
    assert np.all(matrix >= 0.0)
    assert np.allclose(np.sum(matrix, axis=1), 1.0)
    assert matrix[ContactRegime.INACTIVE, ContactRegime.STICKING] > 0.5
    assert matrix[ContactRegime.STICKING, ContactRegime.SLIPPING] > 0.0


def test_zero_hazard_path_is_exactly_static() -> None:
    zero = ContactTransitionConfig(
        activation_floor=0.0,
        activation_gain=0.0,
        slip_floor=0.0,
        slip_change_gain=0.0,
        release_floor=0.0,
        release_gain=0.0,
        slip_recovery_probability=0.0,
        reattachment_gain=0.0,
    )
    prior = enumerate_contact_paths(np.ones(6), config=zero)
    assert prior.path_ids == ("inactive:0-5",)
    assert np.array_equal(prior.regime_paths, np.zeros((1, 6), dtype=np.int8))
    assert np.array_equal(prior.weights, np.ones(1))
    assert prior.retained_prior_mass == 1.0


def test_prefix_inference_does_not_read_future_observations() -> None:
    bank, activation = _simple_bank()
    observations = np.zeros((5, 1, 3), dtype=float)
    first = infer_dynamic_contact_posterior(
        bank,
        observations,
        prefix_frame_count=2,
        command_activation=activation,
    )
    perturbed = observations.copy()
    perturbed[2:] = 1e6
    second = infer_dynamic_contact_posterior(
        bank,
        perturbed,
        prefix_frame_count=2,
        command_activation=activation,
    )
    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first.mean_m, second.mean_m)
    assert first.metadata["future_observations_read"] == 0


def test_zero_inflation_preserves_static_conditional_variance() -> None:
    bank, activation = _simple_bank()
    config = DynamicContactInferenceConfig(
        switch_variance_m2=0.0,
        command_change_variance_m2=0.0,
        ood_variance_m2=0.0,
    )
    variance = contact_conditioned_variance(bank, activation, config=config)
    assert np.array_equal(variance, np.full(bank.trajectories_m.shape, 1e-6))


def test_switch_variance_grows_after_contact_onset() -> None:
    bank, activation = _simple_bank()
    config = DynamicContactInferenceConfig(
        switch_variance_m2=4e-6,
        command_change_variance_m2=0.0,
        ood_variance_m2=0.0,
    )
    variance = contact_conditioned_variance(bank, activation, config=config)
    assert np.array_equal(variance[0], np.full((5, 1, 3), 1e-6))
    assert np.all(variance[1, :2] == 1e-6)
    assert np.allclose(variance[1, 2:], 5e-6)


def test_delayed_contact_benchmark_closes_static_failure() -> None:
    result = delayed_contact_case(seed=3)
    assert result["gates"]["dynamic_beats_static_by_50_percent"]
    assert result["gates"]["onset_error_at_most_one_frame"]
    assert result["gates"]["prefix_only_inference"]
    assert result["dynamic_contact_rmse_m"] < result[
        "static_prefix_persistence_rmse_m"
    ]
