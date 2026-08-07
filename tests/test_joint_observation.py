from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from causal4d.joint_observation import (
    LinearJointObservationEvidence,
    block_diagonalize_covariance,
    joint_component_log_likelihoods,
    posterior_weights_from_joint_observation,
)


def _evidence(*, factor: np.ndarray | None = None) -> LinearJointObservationEvidence:
    return LinearJointObservationEvidence(
        evidence_id="joint-test",
        values_m=np.array([0.1, -0.2, 0.05]),
        row_indices=np.array([0, 1, 2]),
        frame_indices=np.array([1, 1, 2]),
        node_indices=np.array([0, 1, 0]),
        coordinate_indices=np.array([0, 0, 1]),
        coefficients=np.ones(3),
        base_covariance_m2=np.array(
            [
                [0.04, 0.01, 0.0],
                [0.01, 0.09, 0.015],
                [0.0, 0.015, 0.06],
            ]
        ),
        shared_covariance_factor_m=factor,
        source_id="prob4d",
        metadata={"covariance_semantics": "full-joint"},
    )


def _components() -> np.ndarray:
    values = np.zeros((4, 3, 2, 2), dtype=float)
    values[0, 1, 0, 0] = 0.12
    values[0, 1, 1, 0] = -0.18
    values[0, 2, 0, 1] = 0.04
    values[1, 1, 0, 0] = -0.05
    values[1, 1, 1, 0] = -0.25
    values[1, 2, 0, 1] = 0.2
    values[2, 1, 0, 0] = 0.3
    values[2, 1, 1, 0] = 0.1
    values[2, 2, 0, 1] = -0.1
    values[3, 1, 0, 0] = 0.1
    values[3, 1, 1, 0] = -0.2
    values[3, 2, 0, 1] = 0.05
    return values


def _direct_score(residual: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0.0
    solved = np.linalg.solve(covariance, residual[..., None])[..., 0]
    quadratic = np.einsum("...i,...i->...", residual, solved)
    dimension = residual.shape[-1]
    return -0.5 * (dimension * np.log(2.0 * np.pi) + logdet + quadratic)


def test_dense_and_low_rank_joint_covariance_are_equivalent() -> None:
    factor = np.array([[0.05, 0.0], [0.02, 0.03], [0.01, -0.02]])
    structured = _evidence(factor=factor)
    dense = replace(
        structured,
        base_covariance_m2=(structured.base_covariance_m2 + factor @ factor.T),
        shared_covariance_factor_m=None,
    )
    components = _components()

    structured_score, structured_diagnostics = joint_component_log_likelihoods(
        components,
        structured,
        prefix_frame_count=3,
    )
    dense_score, dense_diagnostics = joint_component_log_likelihoods(
        components,
        dense,
        prefix_frame_count=3,
    )

    np.testing.assert_allclose(structured_score, dense_score, rtol=1e-12, atol=1e-12)
    assert structured_diagnostics.used_low_rank_path is True
    assert dense_diagnostics.used_low_rank_path is False


def test_score_matches_direct_full_covariance_calculation() -> None:
    factor = np.array([[0.05], [0.02], [-0.01]])
    evidence = _evidence(factor=factor)
    components = _components()
    component_factor = np.broadcast_to(
        np.array([[0.01], [-0.02], [0.015]]),
        (4, 3, 1),
    )
    component_dense = np.broadcast_to(np.eye(3) * 0.003, (4, 3, 3))

    score, diagnostics = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=3,
        component_joint_covariance_m2=component_dense,
        component_joint_covariance_factor_m=component_factor,
    )
    residual = evidence.apply(components) - evidence.values_m
    expected = []
    for index in range(len(components)):
        combined_factor = np.concatenate((factor, component_factor[index]), axis=1)
        covariance = (
            evidence.base_covariance_m2
            + component_dense[index]
            + combined_factor @ combined_factor.T
        )
        expected.append(_direct_score(residual[index], covariance))
    np.testing.assert_allclose(score, np.asarray(expected), rtol=1e-12, atol=1e-12)
    assert diagnostics.evidence_shared_rank == 1
    assert diagnostics.component_shared_rank == 1
    assert diagnostics.used_component_covariance is True


def test_row_permutation_preserves_scores_and_posterior() -> None:
    factor = np.array([[0.05], [0.02], [-0.01]])
    evidence = _evidence(factor=factor)
    components = _components()
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    permutation = np.array([2, 0, 1])
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))
    permuted = LinearJointObservationEvidence(
        evidence_id="joint-test-permuted",
        values_m=evidence.values_m[permutation],
        row_indices=inverse[evidence.row_indices],
        frame_indices=evidence.frame_indices,
        node_indices=evidence.node_indices,
        coordinate_indices=evidence.coordinate_indices,
        coefficients=evidence.coefficients,
        base_covariance_m2=evidence.base_covariance_m2[np.ix_(permutation, permutation)],
        shared_covariance_factor_m=factor[permutation],
        source_id=evidence.source_id,
        metadata=evidence.metadata,
    )

    score, _ = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=3,
    )
    permuted_score, _ = joint_component_log_likelihoods(
        components,
        permuted,
        prefix_frame_count=3,
    )
    posterior, _ = posterior_weights_from_joint_observation(
        prior,
        components,
        evidence,
        prefix_frame_count=3,
    )
    permuted_posterior, _ = posterior_weights_from_joint_observation(
        prior,
        components,
        permuted,
        prefix_frame_count=3,
    )

    np.testing.assert_allclose(score, permuted_score, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        posterior,
        permuted_posterior,
        rtol=1e-12,
        atol=1e-12,
    )


def test_unit_scaling_preserves_posterior() -> None:
    evidence = _evidence(factor=np.array([[0.05], [0.02], [-0.01]]))
    components = _components()
    prior = np.array([0.1, 0.2, 0.3, 0.4])
    posterior, _ = posterior_weights_from_joint_observation(
        prior,
        components,
        evidence,
        prefix_frame_count=3,
    )
    scale = 1000.0
    scaled = replace(
        evidence,
        values_m=evidence.values_m * scale,
        base_covariance_m2=evidence.base_covariance_m2 * scale**2,
        shared_covariance_factor_m=(
            None
            if evidence.shared_covariance_factor_m is None
            else evidence.shared_covariance_factor_m * scale
        ),
    )
    scaled_posterior, _ = posterior_weights_from_joint_observation(
        prior,
        components * scale,
        scaled,
        prefix_frame_count=3,
    )
    np.testing.assert_allclose(
        posterior,
        scaled_posterior,
        rtol=1e-12,
        atol=1e-12,
    )


def test_independent_variance_propagation_keeps_shared_selector_covariance() -> None:
    evidence = LinearJointObservationEvidence(
        evidence_id="shared-selector",
        values_m=np.zeros(2),
        row_indices=np.array([0, 1]),
        frame_indices=np.array([1, 1]),
        node_indices=np.array([0, 0]),
        coordinate_indices=np.array([0, 0]),
        coefficients=np.array([2.0, -3.0]),
        base_covariance_m2=np.eye(2),
    )
    variance = np.zeros((4, 2, 2))
    variance[1, 0, 0] = 0.5

    covariance = evidence.apply_independent_covariance(variance)

    np.testing.assert_allclose(
        covariance,
        np.array([[2.0, -3.0], [-3.0, 4.5]]),
    )


def test_block_diagonal_ablation_and_full_covariance_differ() -> None:
    covariance = np.array(
        [
            [1.0, 0.8, 0.0],
            [0.8, 1.0, 0.5],
            [0.0, 0.5, 1.0],
        ]
    )
    block = block_diagonalize_covariance(covariance, ["a", "a", "b"])
    np.testing.assert_allclose(
        block,
        np.array(
            [
                [1.0, 0.8, 0.0],
                [0.8, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    assert not np.allclose(block, covariance)


def test_exact_zero_prior_support_remains_zero() -> None:
    evidence = _evidence()
    posterior, _ = posterior_weights_from_joint_observation(
        np.array([0.5, 0.0, 0.5, 0.0]),
        _components(),
        evidence,
        prefix_frame_count=3,
    )
    assert posterior[1] == 0.0
    assert posterior[3] == 0.0


def test_low_rank_path_does_not_use_dense_slogdet(monkeypatch) -> None:
    evidence = _evidence(factor=np.array([[0.05], [0.02], [-0.01]]))

    def forbidden(*args, **kwargs):
        raise AssertionError("dense determinant path used")

    monkeypatch.setattr(np.linalg, "slogdet", forbidden)
    score, diagnostics = joint_component_log_likelihoods(
        _components(),
        evidence,
        prefix_frame_count=3,
    )
    assert np.all(np.isfinite(score))
    assert diagnostics.used_low_rank_path is True


def test_block_diagonal_base_matches_materialized_dense_covariance() -> None:
    dense = _evidence(factor=np.array([[0.05], [0.02], [-0.01]]))
    blocks = np.stack(
        (
            dense.base_covariance_m2[:1, :1],
            dense.base_covariance_m2[1:2, 1:2],
            dense.base_covariance_m2[2:3, 2:3],
        )
    )
    block_evidence = replace(dense, base_covariance_m2=blocks)
    materialized = replace(
        dense,
        base_covariance_m2=np.diag(np.diag(dense.base_covariance_m2)),
    )

    block_score, block_diagnostics = joint_component_log_likelihoods(
        _components(),
        block_evidence,
        prefix_frame_count=3,
    )
    dense_score, _ = joint_component_log_likelihoods(
        _components(),
        materialized,
        prefix_frame_count=3,
    )

    np.testing.assert_allclose(block_score, dense_score, rtol=1e-12, atol=1e-12)
    assert block_diagnostics.base_covariance_representation == "block_diagonal"
    assert block_diagnostics.base_block_count == 3
    assert block_diagnostics.base_block_size == 1


def test_block_base_propagates_independent_component_variance() -> None:
    dense = _evidence(factor=None)
    blocks = np.stack(
        (
            dense.base_covariance_m2[:1, :1],
            dense.base_covariance_m2[1:2, 1:2],
            dense.base_covariance_m2[2:3, 2:3],
        )
    )
    evidence = replace(dense, base_covariance_m2=blocks)
    variance = np.zeros_like(_components())
    variance[:, 1, 0, 0] = np.array([0.01, 0.02, 0.03, 0.04])

    score, diagnostics = joint_component_log_likelihoods(
        _components(),
        evidence,
        prefix_frame_count=3,
        component_independent_variance_m2=variance,
    )

    assert np.all(np.isfinite(score))
    assert diagnostics.used_component_independent_covariance is True


def test_block_base_rejects_cross_block_selector_reuse() -> None:
    evidence = LinearJointObservationEvidence(
        evidence_id="cross-block-reuse",
        values_m=np.zeros(2),
        row_indices=np.array([0, 1]),
        frame_indices=np.array([1, 1]),
        node_indices=np.array([0, 0]),
        coordinate_indices=np.array([0, 0]),
        coefficients=np.ones(2),
        base_covariance_m2=np.ones((2, 1, 1)),
    )
    variance = np.ones((2, 2, 1))

    with pytest.raises(ValueError, match="off-block covariance"):
        evidence.apply_independent_covariance_blocks(variance)


def test_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        replace(_evidence(), base_covariance_m2=np.ones((3, 3)))
    with pytest.raises(ValueError, match="zero-sum contrast"):
        LinearJointObservationEvidence(
            evidence_id="bad-endpoint",
            values_m=np.zeros(1),
            row_indices=np.array([0]),
            frame_indices=np.array([0]),
            node_indices=np.array([0]),
            coordinate_indices=np.array([0]),
            coefficients=np.array([1.0]),
            base_covariance_m2=np.eye(1),
        )
    with pytest.raises(ValueError, match="crosses the declared prefix"):
        joint_component_log_likelihoods(
            _components(),
            _evidence(),
            prefix_frame_count=2,
        )
    with pytest.raises(ValueError, match="sum to one"):
        posterior_weights_from_joint_observation(
            np.ones(4),
            _components(),
            _evidence(),
            prefix_frame_count=3,
        )
