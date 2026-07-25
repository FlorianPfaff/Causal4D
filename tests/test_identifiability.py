import numpy as np

from causal4d.identifiability import (
    IdentifiabilityConfig,
    assess_intervention_identifiability,
    finite_response_sensitivity,
)


def test_camera_only_common_mode_is_unidentifiable() -> None:
    intervention = np.asarray([[1.0], [1.0]])
    nuisance = np.asarray([[1.0], [1.0]])
    result = assess_intervention_identifiability(intervention, nuisance)
    assert not result.identifiable
    assert result.effective_rank == 0
    assert "intervention_response_absorbed_by_nuisance" in result.failure_reasons


def test_independent_anchor_restores_identifiability() -> None:
    intervention = np.asarray([[1.0], [1.0], [1.0]])
    nuisance = np.asarray([[1.0], [1.0], [0.0]])
    result = assess_intervention_identifiability(
        intervention,
        nuisance,
        config=IdentifiabilityConfig(
            minimum_information_eigenvalue=0.1,
            minimum_residualized_response_fraction=0.1,
            maximum_subspace_cosine=0.95,
        ),
    )
    assert result.identifiable
    assert result.effective_rank == 1
    assert result.residualized_response_fraction > 0.1


def test_finite_response_sensitivity_uses_declared_steps() -> None:
    reference = np.zeros((2, 1))
    perturbed = np.asarray([[[2.0], [4.0]], [[-3.0], [6.0]]])
    matrix = finite_response_sensitivity(reference, perturbed, [2.0, 3.0])
    assert np.array_equal(matrix, np.asarray([[1.0, -1.0], [2.0, 2.0]]))
