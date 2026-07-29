import numpy as np
import pytest

from causal4d.graph_temporal_discrepancy import (
    canonicalize_graph_eigenbasis,
    fit_graph_temporal_discrepancy,
    forecast_graph_temporal_discrepancy,
    graph_laplacian_basis,
    project_graph_coefficients,
)


pytest.importorskip("scipy")


def _path_graph(node_count: int) -> np.ndarray:
    return np.column_stack((np.arange(node_count - 1), np.arange(1, node_count)))


def _residual_sequence() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_count = 24
    basis, eigenvalues = graph_laplacian_basis(
        node_count,
        _path_graph(node_count),
        rank=6,
    )
    coefficients = np.zeros((30, 6, 3), dtype=float)
    coefficients[0, 0, 0] = 0.02
    coefficients[0, 1, 1] = -0.01
    for frame in range(1, len(coefficients)):
        coefficients[frame] = 0.92 * coefficients[frame - 1]
    residual = np.einsum("nr,trc->tnc", basis, coefficients)
    valid = np.ones(residual.shape[:2], dtype=bool)
    return residual, valid, basis, eigenvalues


def test_graph_basis_and_projection_recover_smooth_residuals() -> None:
    residual, valid, basis, _ = _residual_sequence()
    coefficients = project_graph_coefficients(
        residual,
        valid,
        basis,
        ridge=1e-10,
    )
    reconstructed = np.einsum("nr,trc->tnc", basis, coefficients)
    np.testing.assert_allclose(reconstructed, residual, atol=1e-8)


def test_projection_supports_permuted_noncontiguous_nodes() -> None:
    residual, _, basis, _ = _residual_sequence()
    node_indices = np.asarray([23, 1, 7, 19, 4, 12, 0, 15, 9, 20, 5, 17, 3, 14])
    observed = residual[:, node_indices]
    valid = np.ones(observed.shape[:2], dtype=bool)

    coefficients = project_graph_coefficients(
        observed,
        valid,
        basis,
        ridge=1e-10,
        node_indices=node_indices,
    )
    reconstructed = np.einsum("nr,trc->tnc", basis[node_indices], coefficients)

    np.testing.assert_allclose(reconstructed, observed, atol=1e-8)


@pytest.mark.parametrize(
    "node_indices, message",
    [
        ([0, 1, 1, 3, 4, 5], "unique"),
        ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "integers"),
        ([0, 1, 2, 3, 4, 24], "exceed"),
    ],
)
def test_projection_rejects_invalid_node_indices(
    node_indices: list[int] | list[float],
    message: str,
) -> None:
    residual, valid, basis, _ = _residual_sequence()
    with pytest.raises(ValueError, match=message):
        project_graph_coefficients(
            residual[:, :6],
            valid[:, :6],
            basis,
            ridge=1e-8,
            node_indices=node_indices,
        )


def test_degenerate_eigenbasis_is_rotation_invariant() -> None:
    rng = np.random.default_rng(4)
    basis, _ = np.linalg.qr(rng.normal(size=(12, 4)))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = basis.copy()
    rotated[:, :3] = basis[:, :3] @ rotation
    eigenvalues = np.asarray([0.25, 0.25, 0.25, 0.8])

    canonical = canonicalize_graph_eigenbasis(eigenvalues, basis)
    rotated_canonical = canonicalize_graph_eigenbasis(eigenvalues, rotated)

    np.testing.assert_allclose(rotated_canonical, canonical, atol=1e-12)
    np.testing.assert_allclose(canonical.T @ canonical, np.eye(4), atol=1e-12)


def test_graph_temporal_model_is_stable_and_forecasts_correlated_variance() -> None:
    residual, valid, basis, eigenvalues = _residual_sequence()
    model = fit_graph_temporal_discrepancy(
        residual[:24],
        valid[:24],
        basis,
        eigenvalues,
        rank_candidates=(2, 4, 6),
        projection_ridge=1e-8,
        dynamics_ridge=1e-8,
    )
    assert model.selected_rank in {2, 4, 6}
    assert model.spectral_radius <= 0.995 + 1e-12
    assert len(model.candidate_validation_rmse_m) == 3

    mean, variance = forecast_graph_temporal_discrepancy(
        model,
        residual[20:24],
        valid[20:24],
        total_frame_count=10,
    )
    assert mean.shape == (10, 24, 3)
    assert variance.shape == mean.shape
    assert np.all(variance >= 0.0)
    assert np.linalg.norm(mean[4]) < np.linalg.norm(mean[3])


def test_fit_and_forecast_accept_permuted_node_subsets() -> None:
    residual, valid, basis, eigenvalues = _residual_sequence()
    node_indices = np.asarray([23, 1, 7, 19, 4, 12, 0, 15, 9, 20, 5, 17, 3, 14])
    observed = residual[:, node_indices]
    observed_valid = valid[:, node_indices]
    model = fit_graph_temporal_discrepancy(
        observed[:24],
        observed_valid[:24],
        basis,
        eigenvalues,
        rank_candidates=(2, 4, 6),
        projection_ridge=1e-8,
        dynamics_ridge=1e-8,
        node_indices=node_indices,
    )

    mean, variance = forecast_graph_temporal_discrepancy(
        model,
        observed[20:24],
        observed_valid[20:24],
        total_frame_count=10,
        node_indices=node_indices,
    )

    assert mean.shape == (10, 24, 3)
    assert variance.shape == mean.shape
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(variance))


def test_nonfinite_masked_residual_does_not_poison_projection_variance() -> None:
    residual, valid, basis, eigenvalues = _residual_sequence()
    residual = residual.copy()
    residual[3, 5] = np.nan
    model = fit_graph_temporal_discrepancy(
        residual[:24],
        valid[:24],
        basis,
        eigenvalues,
        rank_candidates=(2, 4),
        projection_ridge=1e-8,
        dynamics_ridge=1e-8,
    )

    assert np.all(np.isfinite(model.projection_variance_m2))


def test_transition_fit_does_not_form_an_explicit_inverse(monkeypatch) -> None:
    residual, valid, basis, eigenvalues = _residual_sequence()

    def fail_inverse(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("matrix inverse must not be formed")

    monkeypatch.setattr(np.linalg, "inv", fail_inverse)
    model = fit_graph_temporal_discrepancy(
        residual[:24],
        valid[:24],
        basis,
        eigenvalues,
        rank_candidates=(2,),
        projection_ridge=1e-8,
        dynamics_ridge=1e-8,
    )

    assert model.transition.shape == (2, 2)


def test_persistence_and_learned_dynamics_are_distinct() -> None:
    residual, valid, basis, eigenvalues = _residual_sequence()
    model = fit_graph_temporal_discrepancy(
        residual[:24],
        valid[:24],
        basis,
        eigenvalues,
        rank_candidates=(2,),
        projection_ridge=1e-8,
        dynamics_ridge=1e-8,
    )
    learned, _ = forecast_graph_temporal_discrepancy(
        model,
        residual[20:24],
        valid[20:24],
        total_frame_count=8,
        dynamics="learned",
    )
    persistent, _ = forecast_graph_temporal_discrepancy(
        model,
        residual[20:24],
        valid[20:24],
        total_frame_count=8,
        dynamics="persistence",
    )
    assert not np.allclose(learned[4:], persistent[4:])
    np.testing.assert_allclose(persistent[4], persistent[3], atol=1e-7)
