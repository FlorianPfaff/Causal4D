from __future__ import annotations

import numpy as np
import pytest

from causal4d.joint_observation import (
    LinearJointObservationEvidence,
    posterior_weights_from_joint_observation,
)


def test_frame_zero_contrast_cannot_cancel_across_coordinates() -> None:
    # A global coefficient sum of zero is not enough: +x_0 - y_1 changes under
    # an ordinary translation and therefore reuses the endpoint as an absolute
    # observation. Translation neutrality must hold per coordinate (or through
    # an equivalently explicit registered projection contract).
    with pytest.raises(ValueError, match="contrast|translation|coordinate"):
        LinearJointObservationEvidence(
            evidence_id="mixed-coordinate-endpoint",
            values_m=np.zeros(1),
            row_indices=np.array([0, 0]),
            frame_indices=np.array([0, 1]),
            node_indices=np.array([0, 0]),
            coordinate_indices=np.array([0, 1]),
            coefficients=np.array([1.0, -1.0]),
            base_covariance_m2=np.eye(1),
            source_id="adversarial-test",
        )


def test_all_negative_infinite_component_scores_fail_closed() -> None:
    evidence = LinearJointObservationEvidence(
        evidence_id="overflowing-residuals",
        values_m=np.zeros(1),
        row_indices=np.array([0]),
        frame_indices=np.array([1]),
        node_indices=np.array([0]),
        coordinate_indices=np.array([0]),
        coefficients=np.array([1.0]),
        base_covariance_m2=np.eye(1),
        source_id="adversarial-test",
    )
    components = np.zeros((2, 2, 1, 1), dtype=float)
    components[:, 1, 0, 0] = 1e308

    # Finite inputs can still overflow the quadratic form. Returning NaN weights
    # would silently destroy the exact finite-support posterior contract.
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="finite|normalizer|likelihood"):
            posterior_weights_from_joint_observation(
                np.array([0.5, 0.5]),
                components,
                evidence,
                prefix_frame_count=2,
            )
