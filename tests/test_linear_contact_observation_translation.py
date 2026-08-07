from __future__ import annotations

import numpy as np
import pytest

from causal4d.latent_contact_v2 import LinearContactObservationGroup


def _group(
    *,
    frame_indices: list[int],
    node_indices: list[int],
    coordinate_indices: list[int],
    coefficients: list[float],
) -> LinearContactObservationGroup:
    term_count = len(frame_indices)
    assert len(node_indices) == term_count
    assert len(coordinate_indices) == term_count
    assert len(coefficients) == term_count
    return LinearContactObservationGroup(
        group_id="translation-neutrality-test",
        values_m=np.zeros(1),
        row_indices=np.zeros(term_count, dtype=int),
        frame_indices=np.asarray(frame_indices, dtype=int),
        node_indices=np.asarray(node_indices, dtype=int),
        coordinate_indices=np.asarray(coordinate_indices, dtype=int),
        coefficients=np.asarray(coefficients, dtype=float),
        covariance_m2=np.eye(1),
        contributor_ids=("registered-endpoint", "registered-response"),
        source_id="adversarial-contract-test",
    )


def test_mixed_coordinate_endpoint_cancellation_is_rejected() -> None:
    with pytest.raises(ValueError, match="coordinate-wise zero-sum contrast"):
        _group(
            frame_indices=[0, 1],
            node_indices=[0, 0],
            coordinate_indices=[0, 1],
            coefficients=[1.0, -1.0],
        )


def test_global_zero_sum_cannot_hide_coordinate_translation_mass() -> None:
    with pytest.raises(ValueError, match="coordinate-wise zero-sum contrast"):
        _group(
            frame_indices=[0, 1, 1],
            node_indices=[0, 0, 0],
            coordinate_indices=[0, 0, 1],
            coefficients=[-1.0, 0.5, 0.5],
        )


def test_same_coordinate_response_difference_remains_valid() -> None:
    group = _group(
        frame_indices=[0, 1],
        node_indices=[0, 0],
        coordinate_indices=[0, 0],
        coefficients=[-1.0, 1.0],
    )
    trajectory = np.zeros((2, 1, 2), dtype=float)
    trajectory[0, 0, 0] = 1.25
    trajectory[1, 0, 0] = 3.75

    np.testing.assert_allclose(group.apply(trajectory), [2.5])


def test_multi_node_same_coordinate_contrast_remains_valid() -> None:
    group = _group(
        frame_indices=[0, 1, 1],
        node_indices=[0, 0, 1],
        coordinate_indices=[0, 0, 0],
        coefficients=[-1.0, 0.5, 0.5],
    )
    trajectory = np.zeros((2, 2, 2), dtype=float)
    trajectory[0, 0, 0] = 1.0
    trajectory[1, 0, 0] = 3.0
    trajectory[1, 1, 0] = 5.0

    np.testing.assert_allclose(group.apply(trajectory), [3.0])


def test_each_coordinate_may_have_its_own_valid_difference() -> None:
    group = _group(
        frame_indices=[0, 1, 0, 1],
        node_indices=[0, 0, 0, 0],
        coordinate_indices=[0, 0, 1, 1],
        coefficients=[-1.0, 1.0, -2.0, 2.0],
    )
    trajectory = np.zeros((2, 1, 2), dtype=float)
    trajectory[0, 0] = [1.0, 2.0]
    trajectory[1, 0] = [4.0, 5.0]

    np.testing.assert_allclose(group.apply(trajectory), [9.0])
