import numpy as np

from causal4d.identifiability import (
    IdentifiabilityConfig,
    evaluate_response_identifiability,
)


def test_orthogonal_response_subspaces_pass() -> None:
    intervention = np.asarray([[1.0], [0.0], [0.0]])
    nuisance = np.asarray([[0.0], [1.0], [0.0]])
    result = evaluate_response_identifiability(intervention, nuisance)
    assert result.passed
    assert np.isclose(result.minimum_principal_angle_degrees, 90.0)
    assert np.isclose(result.projection_fraction, 0.0)


def test_collinear_response_subspaces_fail_closed() -> None:
    intervention = np.asarray([[1.0], [2.0], [3.0]])
    nuisance = 2.0 * intervention
    result = evaluate_response_identifiability(intervention, nuisance)
    assert not result.passed
    assert "principal_angle" in result.failed_checks
    assert "conditional_singular_value" in result.failed_checks


def test_whitening_can_remove_untrusted_rows() -> None:
    intervention = np.asarray([[1.0], [0.0], [1.0]])
    nuisance = np.asarray([[0.0], [1.0], [1.0]])
    strict = IdentifiabilityConfig(minimum_principal_angle_degrees=80.0)
    unweighted = evaluate_response_identifiability(
        intervention,
        nuisance,
        config=strict,
    )
    weighted = evaluate_response_identifiability(
        intervention,
        nuisance,
        whitening=np.asarray([1.0, 1.0, 0.0]),
        config=strict,
    )
    assert not unweighted.passed
    assert weighted.passed
