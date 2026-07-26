import numpy as np
import pytest

from causal4d.action_conditioned_discrepancy import (
    ActionConditionedDiscrepancyFeatures,
    ActionConditionedDiscrepancyModel,
    forecast_action_conditioned_persistence,
)
from causal4d.contracts import array_sha256
from causal4d.discrepancy_belief import GraphDiscrepancyBelief
from causal4d.discrepancy_mean_transition import (
    ActionConditionedMeanTransitionModel,
    forecast_action_conditioned_movement,
)


def _belief(basis: np.ndarray, *, zero_mean: bool = False) -> GraphDiscrepancyBelief:
    mean = np.zeros((1, 2, 3), dtype=float)
    if not zero_mean:
        mean[0, 0, 0] = 1.0
    covariance = np.zeros((1, 3, 2, 2), dtype=float)
    covariance[0, 0] = np.diag([4e-6, 1e-6])
    return GraphDiscrepancyBelief(
        basis_sha256=array_sha256(basis),
        component_ids=("component-0",),
        coefficient_mean_m=mean,
        coefficient_covariance_m2=covariance,
        projection_variance_m2=np.asarray([1e-6, 2e-6, 3e-6]),
        transition_model_id="persistence",
        innovation_model_id="unit-test",
        metadata={"future_frames_read": 0},
    )


def _covariance_model(
    feature_names: tuple[str, ...],
    *,
    base: np.ndarray | None = None,
) -> ActionConditionedDiscrepancyModel:
    return ActionConditionedDiscrepancyModel(
        feature_names=feature_names,
        base_innovation_covariance_m2=(
            np.zeros((2, 2), dtype=float) if base is None else base
        ),
        feature_directions=np.zeros((0, 2), dtype=float),
        feature_weights=np.zeros((0, len(feature_names)), dtype=float),
    )


def test_persistence_model_reproduces_existing_forecast_exactly() -> None:
    basis = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
    belief = _belief(basis)
    features = ActionConditionedDiscrepancyFeatures(
        names=("speed",),
        values=np.asarray([[0.0], [3.0], [7.0]]),
    )
    covariance_model = _covariance_model(
        features.names,
        base=np.asarray([[2e-6, 5e-7], [5e-7, 1e-6]]),
    )
    persistence = ActionConditionedMeanTransitionModel.persistence(
        features.names,
        belief.rank,
    )

    expected = forecast_action_conditioned_persistence(
        belief,
        covariance_model,
        features,
        basis,
    )
    actual = forecast_action_conditioned_movement(
        belief,
        persistence,
        covariance_model,
        features,
        basis,
    )

    assert actual.model_id == expected.model_id
    for field in (
        "coefficient_mean_m",
        "coefficient_covariance_m2",
        "readout_mean_m",
        "readout_variance_m2",
    ):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


def test_transition_contracts_and_rotates_between_graph_modes() -> None:
    basis = np.eye(2)
    belief = _belief(basis)
    features = ActionConditionedDiscrepancyFeatures(
        names=("movement",),
        values=np.ones((1, 1), dtype=float),
    )
    model = ActionConditionedMeanTransitionModel(
        feature_names=features.names,
        contraction_directions=np.asarray([[1.0, 0.0]]),
        contraction_weights=np.asarray([[np.sqrt(np.log(2.0))]]),
        rotation_generators=np.asarray([[[0.0, -1.0], [1.0, 0.0]]]),
        rotation_weights=np.asarray([[np.pi / 2.0]]),
        forcing_weights_m=np.zeros((2, 3, 1), dtype=float),
    )

    transition, forcing = model.transition_and_forcing(np.ones(1))
    singular_values = np.linalg.svd(transition, compute_uv=False)
    assert singular_values[0] <= 1.0 + 1e-12
    np.testing.assert_array_equal(forcing, np.zeros((2, 3)))

    forecast = forecast_action_conditioned_movement(
        belief,
        model,
        _covariance_model(features.names),
        features,
        basis,
    )
    np.testing.assert_allclose(
        forecast.coefficient_mean_m[0, 1, :, 0],
        np.asarray([0.0, 0.5]),
        atol=1e-12,
    )
    assert np.trace(forecast.coefficient_covariance_m2[0, 1, 0]) < np.trace(
        belief.coefficient_covariance_m2[0, 0]
    )


def test_action_forcing_is_bounded_in_graph_coefficient_space() -> None:
    basis = np.eye(2)
    belief = _belief(basis, zero_mean=True)
    features = ActionConditionedDiscrepancyFeatures(
        names=("movement",),
        values=np.ones((1, 1), dtype=float),
    )
    forcing_weights = np.zeros((2, 3, 1), dtype=float)
    forcing_weights[0, 0, 0] = 10.0
    model = ActionConditionedMeanTransitionModel(
        feature_names=features.names,
        contraction_directions=np.zeros((0, 2), dtype=float),
        contraction_weights=np.zeros((0, 1), dtype=float),
        rotation_generators=np.zeros((0, 2, 2), dtype=float),
        rotation_weights=np.zeros((0, 1), dtype=float),
        forcing_weights_m=forcing_weights,
        maximum_forcing_norm_m=0.01,
    )

    _, forcing = model.transition_and_forcing(np.ones(1))
    np.testing.assert_allclose(np.linalg.norm(forcing), 0.01, atol=1e-15)
    forecast = forecast_action_conditioned_movement(
        belief,
        model,
        _covariance_model(features.names),
        features,
        basis,
    )
    np.testing.assert_allclose(
        np.linalg.norm(forecast.coefficient_mean_m[0, 1]),
        0.01,
        atol=1e-15,
    )


def test_rotation_generators_must_be_skew_symmetric() -> None:
    with pytest.raises(ValueError, match="skew-symmetric"):
        ActionConditionedMeanTransitionModel(
            feature_names=("movement",),
            contraction_directions=np.zeros((0, 2), dtype=float),
            contraction_weights=np.zeros((0, 1), dtype=float),
            rotation_generators=np.asarray([[[0.0, 1.0], [0.0, 0.0]]]),
            rotation_weights=np.ones((1, 1), dtype=float),
            forcing_weights_m=np.zeros((2, 3, 1), dtype=float),
        )
