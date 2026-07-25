import numpy as np

from causal4d.action_conditioned_discrepancy import (
    ActionConditionedDiscrepancyModel,
)


def test_increment_cap_never_shrinks_base_covariance() -> None:
    base = np.diag([4.0, 2.0])
    model = ActionConditionedDiscrepancyModel(
        feature_names=("action",),
        base_innovation_covariance_m2=base,
        feature_directions=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        feature_weights=np.asarray([[10.0], [10.0]]),
        maximum_increment_trace_m2=1.0,
    )

    covariance = model.innovation_covariance_m2(np.asarray([1.0]))
    increment = covariance - base

    assert np.trace(increment) <= 1.0 + 1e-12
    assert np.min(np.linalg.eigvalsh(increment)) >= -1e-12
    assert np.min(np.linalg.eigvalsh(covariance - base)) >= -1e-12


def test_zero_action_increment_preserves_base_exactly_with_cap() -> None:
    base = np.asarray([[2.0, 0.25], [0.25, 1.0]])
    model = ActionConditionedDiscrepancyModel(
        feature_names=("action",),
        base_innovation_covariance_m2=base,
        feature_directions=np.asarray([[1.0, 0.0]]),
        feature_weights=np.asarray([[5.0]]),
        maximum_increment_trace_m2=0.1,
    )

    np.testing.assert_allclose(
        model.innovation_covariance_m2(np.asarray([0.0])),
        base,
        atol=1e-14,
        rtol=1e-14,
    )
