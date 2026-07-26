import numpy as np

from causal4d.action_conditioned_discrepancy import (
    ActionConditionedDiscrepancyFeatures,
    ActionConditionedDiscrepancyModel,
    build_action_conditioned_features,
    forecast_action_conditioned_persistence,
)
from causal4d.contracts import array_sha256
from causal4d.discrepancy_belief import GraphDiscrepancyBelief
from causal4d.stable_discrepancy_dynamics import (
    StableDiscrepancyTransitionModel,
    forecast_action_conditioned_dynamics,
)


def _features(*, gain: float, rotation_degrees: float, schema: str):
    controls = np.zeros((3, 1, 3), dtype=float)
    controls[:, 0, 0] = [0.01, 0.02, 0.01]
    return build_action_conditioned_features(
        controls,
        np.zeros((1, 3)),
        frame_dt_s=0.1,
        phi_names=("gain_multiplier", "delay_steps", "rotation_degrees"),
        phi=np.asarray([gain, 0.0, rotation_degrees]),
        kappa_names=("attachment_shift_hand_0", "slip_fraction"),
        kappa=np.asarray([-2.0, 0.1]),
        feature_schema=schema,
    )


def test_signed_feature_schema_preserves_realization_direction() -> None:
    magnitude_low = _features(gain=0.85, rotation_degrees=-3.0, schema="magnitude_v1")
    magnitude_high = _features(gain=1.15, rotation_degrees=3.0, schema="magnitude_v1")
    np.testing.assert_allclose(magnitude_low.values, magnitude_high.values)

    signed_low = _features(gain=0.85, rotation_degrees=-3.0, schema="signed_v2")
    signed_high = _features(gain=1.15, rotation_degrees=3.0, schema="signed_v2")
    gain_index = signed_low.names.index("gain_signed_deviation")
    rotation_index = signed_low.names.index("rotation_sin")
    shift_index = signed_low.names.index("attachment_shift_mean")
    assert np.all(signed_low.values[:, gain_index] < 0.0)
    assert np.all(signed_high.values[:, gain_index] > 0.0)
    assert np.all(signed_low.values[:, rotation_index] < 0.0)
    assert np.all(signed_high.values[:, rotation_index] > 0.0)
    assert np.all(signed_low.values[:, shift_index] < 0.0)
    assert signed_low.step_duration_s == 0.1


def _belief(basis: np.ndarray) -> GraphDiscrepancyBelief:
    return GraphDiscrepancyBelief(
        basis_sha256=array_sha256(basis),
        component_ids=("component-0",),
        coefficient_mean_m=np.asarray(
            [[[0.02, 0.0, 0.0], [0.0, -0.01, 0.0]]]
        ),
        coefficient_covariance_m2=np.broadcast_to(
            np.asarray([[4e-6, 1e-6], [1e-6, 3e-6]])[None, None],
            (1, 3, 2, 2),
        ).copy(),
        projection_variance_m2=np.asarray([1e-6, 1e-6, 1e-6]),
        transition_model_id="persistence",
        innovation_model_id="unit",
    )


def _constant_features(
    horizon: int,
    duration: float,
) -> ActionConditionedDiscrepancyFeatures:
    return ActionConditionedDiscrepancyFeatures(
        names=("drive",),
        values=np.ones((horizon, 1), dtype=float),
        step_duration_s=duration,
        schema_id="unit",
    )


def test_continuous_persistence_covariance_is_frame_rate_invariant() -> None:
    basis = np.eye(2)
    belief = _belief(basis)
    model = ActionConditionedDiscrepancyModel(
        feature_names=("drive",),
        base_innovation_covariance_m2=np.asarray(
            [[2e-6, 5e-7], [5e-7, 1e-6]]
        ),
        feature_directions=np.zeros((0, 2)),
        feature_weights=np.zeros((0, 1)),
        time_parameterization="per_second",
    )
    coarse = forecast_action_conditioned_persistence(
        belief,
        model,
        _constant_features(2, 0.5),
        basis,
    )
    fine = forecast_action_conditioned_persistence(
        belief,
        model,
        _constant_features(10, 0.1),
        basis,
    )
    np.testing.assert_allclose(
        coarse.coefficient_covariance_m2[:, -1],
        fine.coefficient_covariance_m2[:, -1],
        atol=1e-15,
        rtol=1e-12,
    )


def test_continuous_affine_dynamics_are_frame_rate_invariant() -> None:
    basis = np.eye(2)
    belief = _belief(basis)
    covariance = ActionConditionedDiscrepancyModel(
        feature_names=("drive",),
        base_innovation_covariance_m2=np.asarray(
            [[2e-6, 4e-7], [4e-7, 1e-6]]
        ),
        feature_directions=np.zeros((0, 2)),
        feature_weights=np.zeros((0, 1)),
        time_parameterization="per_second",
    )
    transition = StableDiscrepancyTransitionModel(
        feature_names=("drive",),
        rank=2,
        skew_generators=np.asarray([[[0.0, -1.0], [1.0, 0.0]]]),
        skew_feature_weights=np.asarray([[0.3]]),
        contraction_directions=np.asarray([[1.0, 0.0]]),
        contraction_feature_weights=np.asarray([[0.4]]),
        drift_directions_m=np.asarray(
            [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        ),
        drift_feature_weights=np.asarray([[0.005]]),
        time_parameterization="per_second",
    )
    coarse = forecast_action_conditioned_dynamics(
        belief,
        covariance,
        transition,
        _constant_features(2, 0.5),
        basis,
    )
    fine = forecast_action_conditioned_dynamics(
        belief,
        covariance,
        transition,
        _constant_features(10, 0.1),
        basis,
    )
    np.testing.assert_allclose(
        coarse.coefficient_mean_m[:, -1],
        fine.coefficient_mean_m[:, -1],
        atol=1e-13,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        coarse.coefficient_covariance_m2[:, -1],
        fine.coefficient_covariance_m2[:, -1],
        atol=1e-13,
        rtol=1e-11,
    )
