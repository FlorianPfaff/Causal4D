import numpy as np

from causal4d.action_conditioned_discrepancy import (
    fit_action_conditioned_graph_discrepancy,
    forecast_action_conditioned_graph_discrepancy,
)


def test_action_conditioned_forecast_beats_persistence_on_driven_modes() -> None:
    rng = np.random.default_rng(2)
    node_count = 8
    rank = 2
    basis, _ = np.linalg.qr(rng.normal(size=(node_count, rank)))
    frame_count = 20
    features = np.zeros((frame_count - 1, 2), dtype=float)
    features[:, 0] = np.tile([-1.0, 1.0], 10)[: frame_count - 1]
    features[:, 1] = np.linspace(-1.0, 1.0, frame_count - 1)
    true_input = np.asarray(
        [
            [[0.004, 0.001], [0.0, 0.002], [-0.001, 0.0]],
            [[-0.002, 0.0], [0.003, -0.001], [0.0, 0.001]],
        ]
    )
    coefficients = np.zeros((frame_count, rank, 3), dtype=float)
    for frame in range(frame_count - 1):
        coefficients[frame + 1] = coefficients[frame] + np.einsum(
            "rcf,f->rc",
            true_input,
            features[frame],
        )
    residual = np.einsum("nr,trc->tnc", basis, coefficients)
    valid = np.ones((frame_count, node_count), dtype=bool)
    model = fit_action_conditioned_graph_discrepancy(
        residual[:14],
        valid[:14],
        basis,
        features[:13],
        feature_names=("signed_speed", "hold_phase"),
        dynamics_ridge=1e-8,
    )
    mean, variance = forecast_action_conditioned_graph_discrepancy(
        model,
        residual[:5],
        valid[:5],
        features,
        total_frame_count=frame_count,
    )
    persistence = np.repeat(residual[4:5], frame_count - 5, axis=0)
    action_rmse = np.sqrt(np.mean(np.square(mean[5:] - residual[5:])))
    persistence_rmse = np.sqrt(
        np.mean(np.square(persistence - residual[5:]))
    )
    assert action_rmse < 0.05 * persistence_rmse
    assert np.all(variance >= 0.0)
    assert model.feature_names == ("signed_speed", "hold_phase")
