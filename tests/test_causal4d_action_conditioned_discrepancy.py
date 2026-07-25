import numpy as np

from causal4d.action_conditioned_discrepancy import (
    ActionConditionedDiscrepancyFeatures,
    ActionConditionedDiscrepancyModel,
    build_action_conditioned_features,
    forecast_action_conditioned_persistence,
)
from causal4d.contracts import array_sha256
from causal4d.discrepancy_belief import (
    GraphDiscrepancyBelief,
    load_graph_discrepancy_belief,
    write_graph_discrepancy_belief,
)


def _belief(basis: np.ndarray) -> GraphDiscrepancyBelief:
    return GraphDiscrepancyBelief(
        basis_sha256=array_sha256(basis),
        component_ids=("component-0",),
        coefficient_mean_m=np.asarray([[[0.01, 0.0, 0.0], [0.0, -0.02, 0.0]]]),
        coefficient_covariance_m2=np.zeros((1, 3, 2, 2)),
        projection_variance_m2=np.asarray([1e-6, 2e-6, 3e-6]),
        transition_model_id="persistence",
        innovation_model_id="unit-test",
        metadata={"future_frames_read": 0},
    )


def test_graph_discrepancy_belief_round_trip(tmp_path) -> None:
    basis = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
    belief = _belief(basis)
    manifest = tmp_path / "belief.json"
    record = write_graph_discrepancy_belief(manifest, belief)
    loaded = load_graph_discrepancy_belief(manifest)
    assert record["artifact_id"] == belief.artifact_id == loaded.artifact_id
    np.testing.assert_array_equal(loaded.coefficient_mean_m, belief.coefficient_mean_m)
    np.testing.assert_array_equal(
        loaded.coefficient_covariance_m2,
        belief.coefficient_covariance_m2,
    )


def test_zero_feature_weights_reproduce_base_persistence_exactly() -> None:
    basis = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
    belief = _belief(basis)
    features = ActionConditionedDiscrepancyFeatures(
        names=("speed",),
        values=np.asarray([[0.0], [3.0], [7.0]]),
    )
    base = np.asarray([[2e-6, 5e-7], [5e-7, 1e-6]])
    model = ActionConditionedDiscrepancyModel(
        feature_names=features.names,
        base_innovation_covariance_m2=base,
        feature_directions=np.asarray([[1.0, 0.0]]),
        feature_weights=np.zeros((1, 1)),
    )
    forecast = forecast_action_conditioned_persistence(belief, model, features, basis)
    for step in range(features.horizon + 1):
        for coordinate in range(3):
            np.testing.assert_allclose(
                forecast.coefficient_covariance_m2[0, step, coordinate],
                step * base,
                atol=1e-15,
            )
    np.testing.assert_allclose(
        forecast.coefficient_mean_m,
        np.broadcast_to(
            belief.coefficient_mean_m[:, None],
            forecast.coefficient_mean_m.shape,
        ),
    )


def test_action_and_realized_intervention_increase_uncertainty_only() -> None:
    basis = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
    belief = _belief(basis)
    controls = np.zeros((3, 1, 3), dtype=float)
    quiet = build_action_conditioned_features(
        controls,
        np.zeros((1, 3)),
        frame_dt_s=0.1,
        phi_names=("gain_multiplier", "delay_steps", "rotation_degrees"),
        phi=np.asarray([1.0, 0.0, 0.0]),
        kappa_names=("attachment_shift_hand_0", "slip_fraction"),
        kappa=np.asarray([0.0, 0.0]),
    )
    moving = controls.copy()
    moving[:, 0, 0] = np.asarray([0.01, 0.03, 0.06])
    aggressive = build_action_conditioned_features(
        moving,
        np.zeros((1, 3)),
        frame_dt_s=0.1,
        phi_names=("gain_multiplier", "delay_steps", "rotation_degrees"),
        phi=np.asarray([1.15, 2.0, 3.0]),
        kappa_names=("attachment_shift_hand_0", "slip_fraction"),
        kappa=np.asarray([2.0, 0.2]),
        contact_policy="new_contact",
    )
    weights = np.zeros((2, len(quiet.names)))
    weights[0, quiet.names.index("control_speed_mps")] = 0.01
    weights[0, quiet.names.index("gain_abs_deviation")] = 0.2
    weights[1, quiet.names.index("slip_fraction")] = 0.2
    weights[1, quiet.names.index("new_contact")] = 0.2
    model = ActionConditionedDiscrepancyModel(
        feature_names=quiet.names,
        base_innovation_covariance_m2=np.zeros((2, 2)),
        feature_directions=np.eye(2),
        feature_weights=weights,
    )
    quiet_forecast = forecast_action_conditioned_persistence(
        belief,
        model,
        quiet,
        basis,
    )
    aggressive_forecast = forecast_action_conditioned_persistence(
        belief,
        model,
        aggressive,
        basis,
    )
    np.testing.assert_array_equal(
        quiet_forecast.coefficient_mean_m,
        aggressive_forecast.coefficient_mean_m,
    )
    quiet_trace = np.trace(
        quiet_forecast.coefficient_covariance_m2[0, -1],
        axis1=1,
        axis2=2,
    ).sum()
    aggressive_trace = np.trace(
        aggressive_forecast.coefficient_covariance_m2[0, -1],
        axis1=1,
        axis2=2,
    ).sum()
    assert aggressive_trace > quiet_trace
    assert np.all(np.diff(aggressive_forecast.readout_variance_m2[0], axis=0) >= -1e-15)
