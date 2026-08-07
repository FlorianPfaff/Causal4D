from __future__ import annotations

import numpy as np
import pytest

from causal4d.joint_observation import joint_component_log_likelihoods
from causal4d.prob4d_joint_observation import joint_observation_from_prob4d


def _descriptor() -> dict[str, object]:
    return {
        "case_id": "case-1",
        "source_revision": "a" * 40,
        "source_artifact_sha256": "b" * 64,
    }


def _arrays() -> dict[str, np.ndarray]:
    return {
        "mean_xyz_m": np.array(
            [
                [0.1, 0.2, 0.3],
                [-0.1, 0.0, 0.4],
            ]
        ),
        "local_covariance_m2": np.array(
            [
                np.eye(3) * 0.04,
                np.array(
                    [
                        [0.09, 0.01, 0.0],
                        [0.01, 0.07, 0.005],
                        [0.0, 0.005, 0.08],
                    ]
                ),
            ]
        ),
        "low_rank_factor_m": np.array(
            [
                [[0.02], [0.01], [0.0]],
                [[-0.01], [0.015], [0.005]],
            ]
        ),
        "frame_ids": np.array([10, 11], dtype=np.int64),
        "entity_ids": np.array([7, 8], dtype=np.int64),
        "factor_group_ids": np.zeros(2, dtype=np.int64),
        "association_probability": np.ones(2),
        "prior_reliability": np.array([0.9, 0.8]),
        "group_prior_nominal_probability": np.ones(2),
        "group_composite_weight": np.array([1.0, 0.5]),
    }


def test_prob4d_adapter_preserves_blocks_factor_and_explicit_mappings(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "causal4d.prob4d_joint_observation.validate_prob4d_causal_observation_metadata",
        lambda descriptor, arrays: {"validated": True},
    )
    evidence, diagnostics = joint_observation_from_prob4d(
        _descriptor(),
        _arrays(),
        rollout_frame_ids=(9, 10, 11),
        entity_to_node={7: 1, 8: 0},
        reliability_policy="record_only",
    )

    assert evidence.base_covariance_representation == "block_diagonal"
    assert evidence.base_covariance_m2.shape == (2, 3, 3)
    assert evidence.shared_covariance_factor_m.shape == (6, 1)
    np.testing.assert_array_equal(evidence.frame_indices, [1, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(evidence.node_indices, [1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(evidence.coordinate_indices, [0, 1, 2, 0, 1, 2])
    assert diagnostics.factor_rank == 1
    assert diagnostics.nonneutral_prior_reliability_count == 2
    assert diagnostics.nonunit_group_composite_weight_count == 1


def test_prob4d_adapter_joint_score_matches_materialized_covariance(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "causal4d.prob4d_joint_observation.validate_prob4d_causal_observation_metadata",
        lambda descriptor, arrays: {"validated": True},
    )
    evidence, _ = joint_observation_from_prob4d(
        _descriptor(),
        _arrays(),
        rollout_frame_ids=(9, 10, 11),
        entity_to_node={7: 1, 8: 0},
        reliability_policy="record_only",
    )
    components = np.zeros((2, 3, 2, 3))
    components[0, 1, 1] = [0.12, 0.18, 0.31]
    components[0, 2, 0] = [-0.12, 0.01, 0.38]
    components[1, 1, 1] = [0.4, -0.2, 0.1]
    components[1, 2, 0] = [0.0, 0.2, 0.7]

    score, diagnostics = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=3,
    )

    base = np.zeros((6, 6))
    for index, block in enumerate(evidence.base_covariance_m2):
        base[3 * index : 3 * index + 3, 3 * index : 3 * index + 3] = block
    factor = evidence.shared_covariance_factor_m
    covariance = base + factor @ factor.T
    residual = evidence.apply(components) - evidence.values_m
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0.0
    solved = np.linalg.solve(covariance, residual[..., None])[..., 0]
    expected = -0.5 * (
        6 * np.log(2.0 * np.pi)
        + logdet
        + np.einsum("...i,...i->...", residual, solved)
    )

    np.testing.assert_allclose(score, expected, rtol=1e-12, atol=1e-12)
    assert diagnostics.base_covariance_representation == "block_diagonal"
    assert diagnostics.used_low_rank_path is True


def test_prob4d_adapter_requires_explicit_reliability_policy(monkeypatch) -> None:
    monkeypatch.setattr(
        "causal4d.prob4d_joint_observation.validate_prob4d_causal_observation_metadata",
        lambda descriptor, arrays: {"validated": True},
    )
    with pytest.raises(ValueError, match="record_only"):
        joint_observation_from_prob4d(
            _descriptor(),
            _arrays(),
            rollout_frame_ids=(9, 10, 11),
            entity_to_node={7: 1, 8: 0},
        )


def test_prob4d_adapter_accepts_neutral_provider_evidence_by_default(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "causal4d.prob4d_joint_observation.validate_prob4d_causal_observation_metadata",
        lambda descriptor, arrays: {"validated": True},
    )
    arrays = _arrays()
    arrays["prior_reliability"] = np.ones(2)
    arrays["group_composite_weight"] = np.ones(2)

    evidence, diagnostics = joint_observation_from_prob4d(
        _descriptor(),
        arrays,
        rollout_frame_ids=(9, 10, 11),
        entity_to_node={7: 1, 8: 0},
    )

    assert evidence.observation_count == 6
    assert diagnostics.reliability_policy == "require_neutral"


def test_prob4d_adapter_fails_on_missing_frame_or_entity_mapping(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "causal4d.prob4d_joint_observation.validate_prob4d_causal_observation_metadata",
        lambda descriptor, arrays: {"validated": True},
    )
    with pytest.raises(ValueError, match="rollout frames not supplied"):
        joint_observation_from_prob4d(
            _descriptor(),
            _arrays(),
            rollout_frame_ids=(9, 10),
            entity_to_node={7: 1, 8: 0},
            reliability_policy="record_only",
        )
    with pytest.raises(ValueError, match="unmapped entities"):
        joint_observation_from_prob4d(
            _descriptor(),
            _arrays(),
            rollout_frame_ids=(9, 10, 11),
            entity_to_node={7: 1},
            reliability_policy="record_only",
        )


def test_prob4d_adapter_rejects_multiple_factor_groups(monkeypatch) -> None:
    monkeypatch.setattr(
        "causal4d.prob4d_joint_observation.validate_prob4d_causal_observation_metadata",
        lambda descriptor, arrays: {"validated": True},
    )
    arrays = _arrays()
    arrays["factor_group_ids"] = np.array([0, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="one shared factor group"):
        joint_observation_from_prob4d(
            _descriptor(),
            arrays,
            rollout_frame_ids=(9, 10, 11),
            entity_to_node={7: 1, 8: 0},
            reliability_policy="record_only",
        )
