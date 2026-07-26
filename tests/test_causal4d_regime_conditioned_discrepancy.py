import numpy as np
import pytest

from causal4d.action_conditioned_discrepancy import (
    ActionConditionedDiscrepancyFeatures,
    ActionConditionedDiscrepancyModel,
)
from causal4d.contracts import array_sha256
from causal4d.discrepancy_belief import GraphDiscrepancyBelief
from causal4d.dynamic_contact import ContactRegime
from causal4d.regime_conditioned_discrepancy import (
    RegimeConditionedDiscrepancyTransitionModel,
    forecast_regime_conditioned_discrepancy,
)


def _belief(basis: np.ndarray) -> GraphDiscrepancyBelief:
    return GraphDiscrepancyBelief(
        basis_sha256=array_sha256(basis),
        component_ids=("component-a", "component-b"),
        coefficient_mean_m=np.asarray(
            [
                [[0.01, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.01, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ]
        ),
        coefficient_covariance_m2=np.broadcast_to(
            np.asarray([[1e-4, 2e-5], [2e-5, 7e-5]])[None, None],
            (2, 3, 2, 2),
        ).copy(),
        projection_variance_m2=np.asarray([1e-6] * 3),
        transition_model_id="persistence",
        innovation_model_id="unit-test",
    )


def _model(
    *,
    base_rates: np.ndarray | None = None,
    feature_weights: np.ndarray | None = None,
) -> RegimeConditionedDiscrepancyTransitionModel:
    targets = np.repeat(np.eye(2)[None], 4, axis=0)
    targets[ContactRegime.STICKING] = np.asarray([[0.0, -0.8], [0.8, 0.0]])
    return RegimeConditionedDiscrepancyTransitionModel(
        feature_names=("speed",),
        target_matrices=targets,
        base_activation_rates=(
            np.zeros(4) if base_rates is None else base_rates
        ),
        feature_weights=(
            np.zeros((4, 1)) if feature_weights is None else feature_weights
        ),
    )


def test_zero_activation_is_exact_persistence() -> None:
    basis = np.eye(2)
    belief = _belief(basis)
    features = ActionConditionedDiscrepancyFeatures(
        names=("speed",),
        values=np.asarray([[0.0], [3.0]]),
    )
    forecast = forecast_regime_conditioned_discrepancy(
        belief,
        _model(),
        features,
        np.asarray([ContactRegime.INACTIVE, ContactRegime.STICKING]),
        basis,
    )
    np.testing.assert_array_equal(
        forecast.coefficient_mean_m,
        np.broadcast_to(
            belief.coefficient_mean_m[:, None],
            forecast.coefficient_mean_m.shape,
        ),
    )
    np.testing.assert_array_equal(
        forecast.coefficient_covariance_m2,
        np.broadcast_to(
            belief.coefficient_covariance_m2[:, None],
            forecast.coefficient_covariance_m2.shape,
        ),
    )


def test_sticking_transition_contracts_and_mixes_modes() -> None:
    basis = np.eye(2)
    belief = _belief(basis)
    features = ActionConditionedDiscrepancyFeatures(
        names=("speed",),
        values=np.asarray([[1.0]]),
    )
    base_rates = np.zeros(4)
    base_rates[ContactRegime.STICKING] = np.log(2.0)
    forecast = forecast_regime_conditioned_discrepancy(
        belief,
        _model(base_rates=base_rates),
        features,
        np.asarray([ContactRegime.STICKING]),
        basis,
    )
    expected_transition = 0.5 * np.eye(2) + 0.5 * np.asarray(
        [[0.0, -0.8], [0.8, 0.0]]
    )
    np.testing.assert_allclose(
        forecast.transition_matrices[0, 0],
        expected_transition,
    )
    np.testing.assert_allclose(
        forecast.coefficient_mean_m[0, 1, :, 0],
        expected_transition @ np.asarray([0.01, 0.0]),
    )
    assert np.linalg.norm(forecast.coefficient_mean_m[0, 1, :, 0]) < 0.01
    assert forecast.coefficient_mean_m[0, 1, 1, 0] > 0.0


def test_component_specific_regimes_produce_different_means() -> None:
    basis = np.eye(2)
    belief = _belief(basis)
    features = ActionConditionedDiscrepancyFeatures(
        names=("speed",),
        values=np.asarray([[1.0]]),
    )
    base_rates = np.zeros(4)
    base_rates[ContactRegime.STICKING] = 2.0
    forecast = forecast_regime_conditioned_discrepancy(
        belief,
        _model(base_rates=base_rates),
        features,
        np.asarray(
            [[ContactRegime.INACTIVE], [ContactRegime.STICKING]]
        ),
        basis,
    )
    np.testing.assert_array_equal(
        forecast.coefficient_mean_m[0, 1],
        belief.coefficient_mean_m[0],
    )
    assert not np.array_equal(
        forecast.coefficient_mean_m[1, 1],
        belief.coefficient_mean_m[1],
    )


def test_covariance_uses_transition_and_conditioned_innovation() -> None:
    basis = np.eye(2)
    belief = _belief(basis)
    features = ActionConditionedDiscrepancyFeatures(
        names=("speed",),
        values=np.asarray([[2.0]]),
    )
    base_rates = np.zeros(4)
    base_rates[ContactRegime.STICKING] = 1.0
    innovation_model = ActionConditionedDiscrepancyModel(
        feature_names=("speed",),
        base_innovation_covariance_m2=np.eye(2) * 1e-6,
        feature_directions=np.asarray([[1.0, 0.0]]),
        feature_weights=np.asarray([[0.01]]),
    )
    forecast = forecast_regime_conditioned_discrepancy(
        belief,
        _model(base_rates=base_rates),
        features,
        np.asarray([ContactRegime.STICKING]),
        basis,
        innovation_model=innovation_model,
    )
    assert np.all(
        np.linalg.eigvalsh(forecast.coefficient_covariance_m2) >= -1e-12
    )
    assert forecast.innovation_model_id == innovation_model.model_id


def test_expansive_target_is_rejected() -> None:
    targets = np.repeat(np.eye(2)[None], 4, axis=0)
    targets[0, 0, 0] = 1.01
    with pytest.raises(ValueError, match="non-expansive"):
        RegimeConditionedDiscrepancyTransitionModel(
            feature_names=("speed",),
            target_matrices=targets,
            base_activation_rates=np.zeros(4),
            feature_weights=np.zeros((4, 1)),
        )
