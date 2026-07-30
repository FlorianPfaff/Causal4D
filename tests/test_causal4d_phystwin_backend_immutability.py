"""Regression tests for immutable PhysTwin backend artifacts."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip(
    "bayesian_phystwin",
    reason="provider-backed backend tests require Bayesian-PhysTwin",
)

from causal4d.phystwin_backend import (
    BayesianPhysTwinParticles,
    PhysTwinActionProposal,
)


def _assert_read_only(array: np.ndarray) -> None:
    assert not array.flags.writeable
    with pytest.raises(ValueError):
        array.flat[0] = 0


def test_bayesian_phystwin_particles_own_read_only_arrays() -> None:
    log_scales = np.asarray([[0.0, 0.1], [0.2, -0.1]], dtype=float)
    weights = np.asarray([0.6, 0.4], dtype=float)
    grid_indices = np.asarray([[0, 0], [1, 0]], dtype=int)
    expected_log_scales = log_scales.copy()
    expected_weights = weights.copy()
    expected_grid_indices = grid_indices.copy()

    particles = BayesianPhysTwinParticles(
        log_scales=log_scales,
        weights=weights,
        grid_indices=grid_indices,
        source_weight_key="posterior_weights",
        retained_probability_mass=1.0,
    )

    log_scales[:] = 99.0
    weights[:] = 0.5
    grid_indices[:] = 99

    np.testing.assert_array_equal(particles.log_scales, expected_log_scales)
    np.testing.assert_array_equal(particles.weights, expected_weights)
    np.testing.assert_array_equal(particles.grid_indices, expected_grid_indices)
    _assert_read_only(particles.log_scales)
    _assert_read_only(particles.weights)
    _assert_read_only(particles.grid_indices)


def test_phystwin_action_proposal_owns_read_only_controller_points() -> None:
    controller_points = np.arange(24, dtype=float).reshape(4, 2, 3)
    expected = controller_points.copy()
    proposal = PhysTwinActionProposal(
        proposal_id="unit",
        controller_points_m=controller_points,
        prior_weight=1.0,
        future_action_observed=False,
        provenance="unit test",
    )

    controller_points[:] = -1.0

    np.testing.assert_array_equal(proposal.controller_points_m, expected)
    _assert_read_only(proposal.controller_points_m)
