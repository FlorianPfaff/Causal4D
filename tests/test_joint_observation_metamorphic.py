from __future__ import annotations

import numpy as np

from causal4d.joint_observation import (
    LinearJointObservationEvidence,
    joint_component_log_likelihoods,
    posterior_weights_from_joint_observation,
)


RNG_SEED = 20_260_809


def _positive_definite(
    rng: np.random.Generator,
    leading_shape: tuple[int, ...],
    dimension: int,
) -> np.ndarray:
    roots = rng.normal(size=(*leading_shape, dimension, dimension))
    covariance = np.einsum("...ik,...jk->...ij", roots, roots)
    covariance += np.eye(dimension) * 0.05
    return covariance


def _random_evidence(
    rng: np.random.Generator,
    *,
    observation_count: int,
    node_count: int,
    coordinate_count: int,
) -> LinearJointObservationEvidence:
    return LinearJointObservationEvidence(
        evidence_id="metamorphic-joint-observation",
        values_m=rng.normal(size=observation_count),
        row_indices=np.arange(observation_count),
        frame_indices=rng.integers(1, 3, size=observation_count),
        node_indices=rng.integers(0, node_count, size=observation_count),
        coordinate_indices=rng.integers(
            0,
            coordinate_count,
            size=observation_count,
        ),
        coefficients=rng.choice((-1.0, 1.0), size=observation_count),
        base_covariance_m2=_positive_definite(rng, (), observation_count),
        shared_covariance_factor_m=rng.normal(
            scale=0.05,
            size=(observation_count, 2),
        ),
        source_id="metamorphic-test",
    )


def test_component_permutation_preserves_joint_posterior_with_uncertainty() -> None:
    rng = np.random.default_rng(RNG_SEED)
    component_count = 6
    observation_count = 5
    components = rng.normal(size=(component_count, 4, 5, 3))
    evidence = _random_evidence(
        rng,
        observation_count=observation_count,
        node_count=5,
        coordinate_count=3,
    )
    prior = rng.dirichlet(np.ones(component_count))
    prior[[1, 4]] = 0.0
    prior /= np.sum(prior)
    independent_variance = rng.uniform(0.0, 0.1, size=components.shape)
    joint_covariance = _positive_definite(
        rng,
        (component_count,),
        observation_count,
    )
    joint_factor = rng.normal(
        scale=0.03,
        size=(component_count, observation_count, 1),
    )

    posterior, diagnostics = posterior_weights_from_joint_observation(
        prior,
        components,
        evidence,
        prefix_frame_count=3,
        component_independent_variance_m2=independent_variance,
        component_joint_covariance_m2=joint_covariance,
        component_joint_covariance_factor_m=joint_factor,
    )
    permutation = rng.permutation(component_count)
    inverse = np.argsort(permutation)
    permuted_posterior, permuted_diagnostics = posterior_weights_from_joint_observation(
        prior[permutation],
        components[permutation],
        evidence,
        prefix_frame_count=3,
        component_independent_variance_m2=independent_variance[permutation],
        component_joint_covariance_m2=joint_covariance[permutation],
        component_joint_covariance_factor_m=joint_factor[permutation],
    )

    np.testing.assert_allclose(
        posterior,
        permuted_posterior[inverse],
        rtol=1e-12,
        atol=1e-12,
    )
    assert diagnostics == permuted_diagnostics
    assert posterior[1] == 0.0
    assert posterior[4] == 0.0


def test_graph_node_relabelling_preserves_joint_scores() -> None:
    rng = np.random.default_rng(RNG_SEED + 1)
    node_count = 7
    components = rng.normal(size=(5, 4, node_count, 3))
    evidence = _random_evidence(
        rng,
        observation_count=8,
        node_count=node_count,
        coordinate_count=3,
    )
    scores, _ = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=3,
    )

    node_permutation = rng.permutation(node_count)
    inverse = np.argsort(node_permutation)
    relabelled_evidence = LinearJointObservationEvidence(
        evidence_id="metamorphic-node-relabelled",
        values_m=evidence.values_m,
        row_indices=evidence.row_indices,
        frame_indices=evidence.frame_indices,
        node_indices=inverse[evidence.node_indices],
        coordinate_indices=evidence.coordinate_indices,
        coefficients=evidence.coefficients,
        base_covariance_m2=evidence.base_covariance_m2,
        shared_covariance_factor_m=evidence.shared_covariance_factor_m,
        source_id=evidence.source_id,
        metadata=evidence.metadata,
    )
    relabelled_scores, _ = joint_component_log_likelihoods(
        components[:, :, node_permutation, :],
        relabelled_evidence,
        prefix_frame_count=3,
    )

    np.testing.assert_allclose(scores, relabelled_scores, rtol=1e-12, atol=1e-12)


def test_future_suffix_changes_cannot_affect_prefix_only_joint_update() -> None:
    rng = np.random.default_rng(RNG_SEED + 2)
    components = rng.normal(size=(4, 7, 3, 2))
    evidence = _random_evidence(
        rng,
        observation_count=5,
        node_count=3,
        coordinate_count=2,
    )
    prior = rng.dirichlet(np.ones(len(components)))
    variance = rng.uniform(0.0, 0.1, size=components.shape)
    posterior, _ = posterior_weights_from_joint_observation(
        prior,
        components,
        evidence,
        prefix_frame_count=3,
        component_independent_variance_m2=variance,
    )

    changed_components = components.copy()
    changed_variance = variance.copy()
    changed_components[:, 3:] = rng.normal(
        loc=10_000.0,
        scale=100.0,
        size=changed_components[:, 3:].shape,
    )
    changed_variance[:, 3:] = rng.uniform(
        10_000.0,
        20_000.0,
        size=changed_variance[:, 3:].shape,
    )
    changed_posterior, _ = posterior_weights_from_joint_observation(
        prior,
        changed_components,
        evidence,
        prefix_frame_count=3,
        component_independent_variance_m2=changed_variance,
    )

    np.testing.assert_array_equal(posterior, changed_posterior)


def test_sparse_term_order_preserves_scores_and_variance_propagation() -> None:
    rng = np.random.default_rng(RNG_SEED + 3)
    observation_count = 4
    term_count = 10
    row_indices = np.asarray((0, 0, 1, 1, 1, 2, 2, 3, 3, 3))
    frame_indices = rng.integers(1, 3, size=term_count)
    node_indices = rng.integers(0, 4, size=term_count)
    coordinate_indices = rng.integers(0, 3, size=term_count)
    coefficients = rng.normal(size=term_count)
    components = rng.normal(size=(5, 4, 4, 3))
    variance = rng.uniform(0.0, 0.1, size=components.shape)
    covariance = _positive_definite(rng, (), observation_count)
    evidence = LinearJointObservationEvidence(
        evidence_id="metamorphic-term-order",
        values_m=rng.normal(size=observation_count),
        row_indices=row_indices,
        frame_indices=frame_indices,
        node_indices=node_indices,
        coordinate_indices=coordinate_indices,
        coefficients=coefficients,
        base_covariance_m2=covariance,
    )
    scores, _ = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=3,
        component_independent_variance_m2=variance,
    )
    propagated = evidence.apply_independent_covariance(variance)

    permutation = rng.permutation(term_count)
    permuted = LinearJointObservationEvidence(
        evidence_id="metamorphic-term-order-permuted",
        values_m=evidence.values_m,
        row_indices=row_indices[permutation],
        frame_indices=frame_indices[permutation],
        node_indices=node_indices[permutation],
        coordinate_indices=coordinate_indices[permutation],
        coefficients=coefficients[permutation],
        base_covariance_m2=covariance,
    )
    permuted_scores, _ = joint_component_log_likelihoods(
        components,
        permuted,
        prefix_frame_count=3,
        component_independent_variance_m2=variance,
    )
    permuted_propagated = permuted.apply_independent_covariance(variance)

    np.testing.assert_allclose(scores, permuted_scores, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        propagated,
        permuted_propagated,
        rtol=1e-12,
        atol=1e-12,
    )


def test_endpoint_displacement_rows_are_global_translation_invariant() -> None:
    evidence = LinearJointObservationEvidence(
        evidence_id="metamorphic-translation",
        values_m=np.asarray((0.2, -0.1, 0.05)),
        row_indices=np.repeat(np.arange(3), 2),
        frame_indices=np.tile((0, 1), 3),
        node_indices=np.repeat((0, 1, 2), 2),
        coordinate_indices=np.repeat(np.arange(3), 2),
        coefficients=np.tile((-1.0, 1.0), 3),
        base_covariance_m2=np.eye(3) * 0.1,
    )
    rng = np.random.default_rng(RNG_SEED + 4)
    components = rng.normal(size=(4, 3, 3, 3))
    scores, _ = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=2,
    )
    translation = np.asarray((100.0, -250.0, 75.0))
    translated_scores, _ = joint_component_log_likelihoods(
        components + translation,
        evidence,
        prefix_frame_count=2,
    )

    np.testing.assert_allclose(scores, translated_scores, rtol=1e-12, atol=1e-12)
