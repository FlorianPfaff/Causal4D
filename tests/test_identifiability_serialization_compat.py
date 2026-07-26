import numpy as np

from causal4d.identifiability import assess_intervention_identifiability


def test_legacy_identifiability_serialization_is_unchanged() -> None:
    result = assess_intervention_identifiability(np.asarray([[1.0], [0.0]]))
    assert tuple(result.as_dict()) == (
        "effective_rank",
        "parameter_count",
        "minimum_eigenvalue",
        "condition_number",
        "residualized_response_fraction",
        "maximum_subspace_cosine",
        "identifiable",
        "failure_reasons",
        "eigenvalues",
    )
    assert "identified_basis" not in result.as_dict()
