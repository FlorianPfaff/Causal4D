from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from causal4d.dynamic_contact import (
    ContactRegime,
    ContactTransitionConfig,
    DynamicContactInferenceConfig,
    enumerate_contact_paths,
)
from causal4d.multi_contact import (
    MultiContactEnumerationConfig,
    MultiContactPathBank,
    enumerate_multi_contact_paths,
    infer_multi_contact_posterior,
    multi_contact_conditioned_variance,
)


def _zero_hazard_config() -> ContactTransitionConfig:
    return ContactTransitionConfig(
        activation_floor=0.0,
        activation_gain=0.0,
        slip_floor=0.0,
        slip_change_gain=0.0,
        release_floor=0.0,
        release_gain=0.0,
        slip_recovery_probability=0.0,
        reattachment_gain=0.0,
    )


def _sorted_marginal(activation: np.ndarray, config: ContactTransitionConfig):
    prior = enumerate_contact_paths(activation, config=config)
    order = sorted(
        range(len(prior.weights)),
        key=lambda index: (-float(prior.weights[index]), prior.path_ids[index]),
    )
    return prior, order


def test_single_contact_matches_existing_enumerator() -> None:
    activation = np.asarray([0.0, 0.2, 0.9, 0.9, 0.1])
    transition = ContactTransitionConfig(maximum_paths=32)
    marginal, order = _sorted_marginal(activation, transition)
    joint = enumerate_multi_contact_paths(
        activation,
        contact_ids=("left",),
        transition_configs=transition,
        config=MultiContactEnumerationConfig(maximum_joint_paths=32),
    )
    assert joint.contact_ids == ("left",)
    assert np.array_equal(joint.regime_paths[:, 0], marginal.regime_paths[order])
    assert np.allclose(joint.weights, marginal.weights[order])
    assert np.isclose(joint.retained_prior_mass, marginal.retained_prior_mass)
    assert joint.full_cartesian_path_count == len(marginal.weights)


def test_contact_identifiers_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="nonempty unique strings"):
        enumerate_multi_contact_paths(
            np.zeros((2, 3)),
            contact_ids=("left", 1),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="positive integer"):
        MultiContactEnumerationConfig(
            maximum_joint_paths=True,  # type: ignore[arg-type]
        )


def test_zero_hazard_multi_contact_prior_is_exactly_static() -> None:
    activation = np.ones((3, 6), dtype=float)
    prior = enumerate_multi_contact_paths(
        activation,
        contact_ids=("left", "right", "support"),
        transition_configs=_zero_hazard_config(),
    )
    assert prior.path_ids == (
        "left=inactive:0-5;right=inactive:0-5;support=inactive:0-5",
    )
    assert np.array_equal(prior.regime_paths, np.zeros((1, 3, 6), dtype=np.int8))
    assert np.array_equal(prior.weights, np.ones(1))
    assert prior.retained_prior_mass == 1.0
    assert np.array_equal(prior.marginal_retained_prior_mass, np.ones(3))


def test_joint_top_k_matches_brute_force_and_accounts_for_retained_mass() -> None:
    activation = np.asarray(
        [
            [0.0, 0.7, 0.9, 0.2],
            [0.0, 0.1, 0.8, 0.8],
        ]
    )
    transition = ContactTransitionConfig(maximum_paths=8)
    marginals = [_sorted_marginal(row, transition) for row in activation]
    joint = enumerate_multi_contact_paths(
        activation,
        transition_configs=transition,
        config=MultiContactEnumerationConfig(maximum_joint_paths=5),
    )
    products = []
    for indices in product(*(range(len(prior.weights)) for prior, _ in marginals)):
        probability = 1.0
        sorted_indices = []
        for contact_index, source_index in enumerate(indices):
            prior, order = marginals[contact_index]
            probability *= float(prior.weights[order[source_index]])
            sorted_indices.append(source_index)
        products.append((probability, tuple(sorted_indices)))
    products.sort(key=lambda item: (-item[0], item[1]))
    expected = products[:5]
    expected_mass = float(
        np.prod([prior.retained_prior_mass for prior, _ in marginals])
        * sum(probability for probability, _ in expected)
    )
    expected_weights = np.asarray([value for value, _ in expected], dtype=float)
    expected_weights /= np.sum(expected_weights)
    assert np.array_equal(
        joint.marginal_path_indices,
        np.asarray([indices for _, indices in expected], dtype=np.int64),
    )
    assert np.allclose(joint.weights, expected_weights)
    assert np.isclose(joint.retained_prior_mass, expected_mass)


def test_retained_mass_is_monotone_in_joint_beam_width() -> None:
    activation = np.asarray(
        [
            [0.0, 0.8, 0.8, 0.1],
            [0.0, 0.3, 0.9, 0.9],
        ]
    )
    transition = ContactTransitionConfig(maximum_paths=12)
    narrow = enumerate_multi_contact_paths(
        activation,
        transition_configs=transition,
        config=MultiContactEnumerationConfig(maximum_joint_paths=3),
    )
    wide = enumerate_multi_contact_paths(
        activation,
        transition_configs=transition,
        config=MultiContactEnumerationConfig(maximum_joint_paths=20),
    )
    assert narrow.retained_prior_mass <= wide.retained_prior_mass
    assert len(narrow.weights) == 3
    assert len(wide.weights) == 20
    assert wide.full_cartesian_path_count > len(wide.weights)


def test_contact_permutation_preserves_full_joint_distribution() -> None:
    activation = np.asarray(
        [
            [0.0, 0.8, 0.9],
            [0.0, 0.2, 0.7],
        ]
    )
    transition = ContactTransitionConfig(maximum_paths=8)
    first = enumerate_multi_contact_paths(
        activation,
        contact_ids=("left", "right"),
        transition_configs=transition,
        config=MultiContactEnumerationConfig(maximum_joint_paths=64),
    )
    second = enumerate_multi_contact_paths(
        activation[::-1],
        contact_ids=("right", "left"),
        transition_configs=transition,
        config=MultiContactEnumerationConfig(maximum_joint_paths=64),
    )

    def distribution(prior, reverse: bool):
        result = {}
        paths = prior.regime_paths[:, ::-1] if reverse else prior.regime_paths
        for path, weight in zip(paths, prior.weights, strict=True):
            result[tuple(int(value) for value in path.ravel())] = float(weight)
        return result

    assert distribution(first, False).keys() == distribution(second, True).keys()
    for key, value in distribution(first, False).items():
        assert np.isclose(value, distribution(second, True)[key])
    assert np.isclose(first.retained_prior_mass, second.retained_prior_mass)


def test_schedule_identity_is_stable_and_bank_preserves_it() -> None:
    activation = np.asarray(
        [
            [0.0, 0.8, 0.8],
            [0.0, 0.2, 0.9],
        ]
    )
    prior = enumerate_multi_contact_paths(
        activation,
        contact_ids=("left", "right"),
        config=MultiContactEnumerationConfig(maximum_joint_paths=12),
    )
    trajectories = np.zeros((len(prior.weights), 3, 1, 3), dtype=float)
    bank = MultiContactPathBank.from_prior(
        prior,
        trajectories,
        base_variance_m2=1e-6,
    )
    copied = enumerate_multi_contact_paths(
        np.asfortranarray(activation),
        contact_ids=("left", "right"),
        config=MultiContactEnumerationConfig(maximum_joint_paths=12),
    )
    assert prior.schedule_identity == copied.schedule_identity
    assert prior.schedule_identity == bank.schedule_identity
    changed_paths = prior.regime_paths.copy()
    changed_paths[0, 0, -1] = ContactRegime.DETACHED
    changed = MultiContactPathBank(
        contact_ids=prior.contact_ids,
        path_ids=prior.path_ids,
        regime_paths=changed_paths,
        trajectories_m=trajectories,
        prior_weights=prior.weights,
        base_variance_m2=1e-6,
        retained_prior_mass=prior.retained_prior_mass,
    )
    assert changed.schedule_identity != prior.schedule_identity


def _posterior_bank() -> tuple[MultiContactPathBank, np.ndarray, np.ndarray]:
    frame_count = 5
    paths = np.asarray(
        [
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 1, 1, 1], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 1, 1, 1]],
            [[0, 0, 1, 1, 1], [0, 0, 1, 1, 1]],
        ],
        dtype=np.int8,
    )
    trajectories = np.zeros((4, frame_count, 1, 3), dtype=float)
    trajectories[1, 2:, 0, 0] = [0.01, 0.02, 0.03]
    trajectories[2, 2:, 0, 1] = [0.01, 0.02, 0.03]
    trajectories[3] = trajectories[1] + trajectories[2]
    bank = MultiContactPathBank(
        contact_ids=("left", "right"),
        path_ids=("none", "left", "right", "both"),
        regime_paths=paths,
        trajectories_m=trajectories,
        prior_weights=np.full(4, 0.25),
        base_variance_m2=1e-8,
    )
    activation = np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
        ]
    )
    observations = trajectories[3].copy()
    return bank, activation, observations


def test_prefix_inference_is_future_invariant_and_returns_contact_marginals() -> None:
    bank, activation, observations = _posterior_bank()
    config = DynamicContactInferenceConfig(
        observation_scale_m=1e-4,
        dynamic_likelihood_weight=0.0,
        switch_variance_m2=0.0,
        command_change_variance_m2=0.0,
        ood_variance_m2=0.0,
    )
    first = infer_multi_contact_posterior(
        bank,
        observations,
        prefix_frame_count=4,
        command_activation=activation,
        config=config,
    )
    perturbed = observations.copy()
    perturbed[4] = 1e6
    second = infer_multi_contact_posterior(
        bank,
        perturbed,
        prefix_frame_count=4,
        command_activation=activation,
        config=config,
    )
    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first.mean_m, second.mean_m)
    assert first.map_path_id == "both"
    assert first.metadata["future_observations_read"] == 0
    assert first.regime_probabilities.shape == (2, 5, 4)
    assert np.allclose(np.sum(first.regime_probabilities, axis=2), 1.0)
    assert np.all(first.active_probability[:, 2:] > 0.95)
    assert np.all(first.switch_probability[:, 2] > 0.95)
    assert first.any_switch_probability[2] > 0.99


def test_variance_counts_each_contact_switch() -> None:
    paths = np.asarray(
        [[[[0], [1], [1]], [[0], [0], [1]]]],
        dtype=np.int8,
    ).reshape(1, 2, 3)
    trajectories = np.zeros((1, 3, 1, 2), dtype=float)
    bank = MultiContactPathBank(
        contact_ids=("left", "right"),
        path_ids=("staggered",),
        regime_paths=paths,
        trajectories_m=trajectories,
        prior_weights=np.ones(1),
        base_variance_m2=2.0,
    )
    config = DynamicContactInferenceConfig(
        switch_variance_m2=1.0,
        command_change_variance_m2=0.0,
        ood_variance_m2=0.0,
    )
    variance = multi_contact_conditioned_variance(
        bank,
        np.zeros((2, 3)),
        config=config,
    )
    expected = np.asarray([2.0, 3.0, 4.0])
    assert np.array_equal(variance[0, :, 0, 0], expected)
    assert np.array_equal(variance[0, :, 0, 1], expected)


def test_numeric_contracts_reject_boolean_and_string_coercion() -> None:
    with pytest.raises(ValueError, match="real numbers"):
        enumerate_multi_contact_paths(
            np.asarray([[False, True]]),
            contact_ids=("left",),
        )
    with pytest.raises(ValueError, match="real numbers"):
        enumerate_multi_contact_paths(
            np.asarray([["0", "1"]]),
            contact_ids=("left",),
        )
    with pytest.raises(ValueError, match="integers"):
        MultiContactPathBank(
            contact_ids=("left",),
            path_ids=("path",),
            regime_paths=np.asarray([[[0.0, 1.0]]]),
            trajectories_m=np.zeros((1, 2, 1, 2)),
            prior_weights=np.ones(1),
            base_variance_m2=1.0,
        )
    with pytest.raises(ValueError, match="outside int8 range"):
        MultiContactPathBank(
            contact_ids=("left",),
            path_ids=("path",),
            regime_paths=np.asarray([[[0, 256]]]),
            trajectories_m=np.zeros((1, 2, 1, 2)),
            prior_weights=np.ones(1),
            base_variance_m2=1.0,
        )


def test_global_ood_distance_is_not_multiplied_by_contact_count() -> None:
    paths = np.zeros((1, 2, 3), dtype=np.int8)
    bank = MultiContactPathBank(
        contact_ids=("left", "right"),
        path_ids=("static",),
        regime_paths=paths,
        trajectories_m=np.zeros((1, 3, 1, 2)),
        prior_weights=np.ones(1),
        base_variance_m2=2.0,
    )
    config = DynamicContactInferenceConfig(
        switch_variance_m2=0.0,
        command_change_variance_m2=0.0,
        ood_variance_m2=1.0,
    )
    global_distance = multi_contact_conditioned_variance(
        bank,
        np.zeros((2, 3)),
        ood_distance=np.asarray([1.0, 2.0, 0.0]),
        config=config,
    )
    per_contact_distance = multi_contact_conditioned_variance(
        bank,
        np.zeros((2, 3)),
        ood_distance=np.asarray([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]]),
        config=config,
    )
    assert np.array_equal(global_distance, per_contact_distance)
    assert np.array_equal(
        global_distance[0, :, 0, 0],
        np.asarray([3.0, 7.0, 7.0]),
    )


def test_prefix_frame_count_and_mask_are_fail_closed() -> None:
    bank, activation, observations = _posterior_bank()
    with pytest.raises(ValueError, match="integer leaving held-out"):
        infer_multi_contact_posterior(
            bank,
            observations,
            prefix_frame_count=True,  # type: ignore[arg-type]
            command_activation=activation,
        )
    with pytest.raises(ValueError, match="mask must contain booleans"):
        infer_multi_contact_posterior(
            bank,
            observations,
            prefix_frame_count=3,
            command_activation=activation,
            mask=np.ones(observations.shape, dtype=int),
        )
