from __future__ import annotations

import numpy as np
import pytest

from causal4d.latent_contact_v2 import (
    ContactObservationEvidenceV2,
    LinearContactObservationGroup,
    contact_component_log_likelihoods_v2,
)


def _group(
    *,
    group_id: str,
    values_m: np.ndarray,
    row_indices: np.ndarray,
    coordinate_indices: np.ndarray,
    covariance_m2: np.ndarray,
) -> LinearContactObservationGroup:
    return LinearContactObservationGroup(
        group_id=group_id,
        values_m=values_m,
        row_indices=row_indices,
        frame_indices=np.array([0, 1, 0, 1]),
        node_indices=np.array([0, 0, 1, 1]),
        coordinate_indices=coordinate_indices,
        coefficients=np.array([-1.0, 1.0, -1.0, 1.0]),
        covariance_m2=covariance_m2,
        contributor_ids=("camera",),
        prior_nominal_probability=0.999,
        source_id="invariance-test",
    )


def test_endpoint_contrast_cannot_cancel_across_coordinates() -> None:
    with pytest.raises(ValueError, match="translation-neutral.*per coordinate"):
        LinearContactObservationGroup(
            group_id="mixed-coordinate-endpoint",
            values_m=np.zeros(1),
            row_indices=np.array([0, 0]),
            frame_indices=np.array([0, 1]),
            node_indices=np.array([0, 0]),
            coordinate_indices=np.array([0, 1]),
            coefficients=np.array([1.0, -1.0]),
            covariance_m2=np.eye(1),
            contributor_ids=("camera",),
            source_id="invariance-test",
        )


def test_same_coordinate_multi_node_endpoint_contrast_remains_valid() -> None:
    group = LinearContactObservationGroup(
        group_id="multi-node-x-displacement",
        values_m=np.zeros(1),
        row_indices=np.zeros(4, dtype=int),
        frame_indices=np.array([0, 0, 1, 1]),
        node_indices=np.array([0, 1, 0, 1]),
        coordinate_indices=np.zeros(4, dtype=int),
        coefficients=np.array([-0.25, -0.75, 0.4, 0.6]),
        covariance_m2=np.eye(1),
        contributor_ids=("camera",),
        source_id="invariance-test",
    )

    translated = np.zeros((2, 2, 2), dtype=float)
    translated[..., 0] = 12.5
    assert group.apply(translated)[0] == pytest.approx(0.0)


def test_valid_endpoint_rows_preserve_row_permutation_and_unit_scaling() -> None:
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    group = _group(
        group_id="metres",
        values_m=np.array([0.1, -0.2]),
        row_indices=np.array([0, 0, 1, 1]),
        coordinate_indices=np.array([0, 0, 1, 1]),
        covariance_m2=covariance,
    )
    permuted = _group(
        group_id="permuted",
        values_m=np.array([-0.2, 0.1]),
        row_indices=np.array([1, 1, 0, 0]),
        coordinate_indices=np.array([0, 0, 1, 1]),
        covariance_m2=covariance[np.ix_([1, 0], [1, 0])],
    )
    scaled = _group(
        group_id="millimetres",
        values_m=1000.0 * group.values_m,
        row_indices=group.row_indices,
        coordinate_indices=group.coordinate_indices,
        covariance_m2=1_000_000.0 * group.covariance_m2,
    )

    components = np.zeros((3, 2, 2, 2), dtype=float)
    components[0, 1, 0, 0] = 0.05
    components[0, 1, 1, 1] = -0.10
    components[1, 1, 0, 0] = 0.20
    components[1, 1, 1, 1] = -0.35
    components[2, 1, 0, 0] = -0.10
    components[2, 1, 1, 1] = 0.15

    scores, _ = contact_component_log_likelihoods_v2(
        components,
        ContactObservationEvidenceV2((group,)),
        prefix_frame_count=2,
    )
    permuted_scores, _ = contact_component_log_likelihoods_v2(
        components,
        ContactObservationEvidenceV2((permuted,)),
        prefix_frame_count=2,
    )
    scaled_scores, _ = contact_component_log_likelihoods_v2(
        1000.0 * components,
        ContactObservationEvidenceV2((scaled,)),
        prefix_frame_count=2,
    )

    assert np.allclose(scores, permuted_scores, atol=1e-12, rtol=1e-12)
    assert np.allclose(
        scores - scores[0],
        scaled_scores - scaled_scores[0],
        atol=1e-12,
        rtol=1e-12,
    )
