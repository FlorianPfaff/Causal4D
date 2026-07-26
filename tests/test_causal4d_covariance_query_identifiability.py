import numpy as np
import pytest

from causal4d.contracts import array_sha256
from causal4d.discrepancy_belief import (
    GraphDiscrepancyBelief,
    graph_discrepancy_group_covariances,
)
from causal4d.grouped_likelihood import (
    group_log_likelihood,
    grouped_component_log_likelihoods,
)
from causal4d.identifiability import (
    IdentifiabilityConfig,
    assess_intervention_identifiability,
    project_identifiable_intervention_update,
)
from causal4d.observation_evidence import (
    GroupedObservationEvidence,
    ObservationGroup,
)


def _correlated_group() -> ObservationGroup:
    return ObservationGroup(
        group_id="correlated",
        values_m=np.zeros(2),
        frame_indices=np.asarray([1, 1]),
        node_indices=np.asarray([0, 0]),
        coordinate_indices=np.asarray([0, 1]),
        covariance_m2=np.eye(2) * 0.01,
        contributor_ids=("unit:frame:1",),
        prior_nominal_probability=0.99,
        outlier_scale_multiplier=100.0,
        degrees_of_freedom=4.0,
        source_id="unit",
    )


def test_full_covariance_distinguishes_common_and_difference_modes() -> None:
    predictions = np.asarray([[1.0, 1.0], [1.0, -1.0]])
    covariance = np.asarray([[1.0, 0.99], [0.99, 1.0]])
    score, _ = group_log_likelihood(
        predictions,
        _correlated_group(),
        additive_covariance_m2=covariance,
    )
    assert score[0] > score[1]


def test_group_covariance_broadcasts_over_component_axes() -> None:
    components = np.zeros((2, 3, 3, 1, 2))
    components[0, :, 1, 0] = [1.0, 1.0]
    components[1, :, 1, 0] = [1.0, -1.0]
    evidence = GroupedObservationEvidence(groups=(_correlated_group(),))
    covariance = np.broadcast_to(
        np.asarray([[1.0, 0.99], [0.99, 1.0]]),
        (3, 2, 2),
    )
    score, diagnostics = grouped_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=2,
        component_group_covariance_m2={"correlated": covariance},
    )
    assert score.shape == (2, 3)
    assert np.all(score[0] > score[1])
    assert diagnostics.full_covariance_group_ids == ("correlated",)


def test_non_psd_component_covariance_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        group_log_likelihood(
            np.asarray([[0.0, 0.0]]),
            _correlated_group(),
            additive_covariance_m2=np.asarray([[1.0, 2.0], [2.0, 1.0]]),
        )


def test_graph_covariance_preserves_time_correlation_and_component_order() -> None:
    basis = np.asarray([[1.0], [2.0]])
    belief = GraphDiscrepancyBelief(
        basis_sha256=array_sha256(basis),
        component_ids=("a", "b"),
        coefficient_mean_m=np.zeros((2, 1, 3)),
        coefficient_covariance_m2=np.asarray(
            [
                [[[4.0]], [[9.0]], [[16.0]]],
                [[[1.0]], [[1.0]], [[1.0]]],
            ]
        ),
        projection_variance_m2=np.asarray([0.5, 0.25, 0.1]),
        transition_model_id="persistence",
        innovation_model_id="unit",
    )
    group = ObservationGroup(
        group_id="graph",
        values_m=np.zeros(3),
        frame_indices=np.asarray([1, 2, 2]),
        node_indices=np.asarray([0, 0, 1]),
        coordinate_indices=np.asarray([0, 0, 1]),
        covariance_m2=np.eye(3),
        contributor_ids=("unit:graph",),
        source_id="unit",
    )
    covariance = graph_discrepancy_group_covariances(
        belief,
        basis,
        GroupedObservationEvidence(groups=(group,)),
        component_ids=("b", "a"),
    )["graph"]
    assert covariance.shape == (2, 3, 3)
    assert np.isclose(covariance[0, 0, 1], 1.0)
    assert np.isclose(covariance[1, 0, 1], 4.0)
    assert covariance[0, 0, 2] == 0.0
    assert np.isclose(covariance[1, 2, 2], 4.0 * 9.0 + 0.25)


def test_query_can_be_identifiable_under_partial_parameter_identification() -> None:
    intervention = np.asarray([[1.0, 1.0], [0.0, 1.0]])
    nuisance = np.asarray([[0.0], [1.0]])
    result = assess_intervention_identifiability(
        intervention,
        nuisance,
        query_sensitivity=np.asarray([[1.0, 1.0]]),
        config=IdentifiabilityConfig(
            minimum_information_eigenvalue=1e-8,
            minimum_residualized_response_fraction=0.01,
            maximum_subspace_cosine=1.0,
            maximum_query_null_response_fraction=1e-8,
        ),
    )
    assert not result.identifiable
    assert result.effective_rank == 1
    assert result.query_identifiable
    assert result.query_null_response_fraction < 1e-12
    assert np.allclose(
        project_identifiable_intervention_update([1.0, -1.0], result),
        0.0,
    )
    assert np.allclose(
        project_identifiable_intervention_update([1.0, 1.0], result),
        [1.0, 1.0],
    )


def test_query_gate_rejects_unresolved_direction() -> None:
    intervention = np.asarray([[1.0, 1.0], [0.0, 1.0]])
    nuisance = np.asarray([[0.0], [1.0]])
    result = assess_intervention_identifiability(
        intervention,
        nuisance,
        query_sensitivity=np.asarray([[1.0, -1.0]]),
        config=IdentifiabilityConfig(maximum_subspace_cosine=1.0),
    )
    assert result.query_identifiable is False
    assert result.query_null_response_fraction > 0.99


def test_parameter_standardization_is_unit_invariant() -> None:
    base = assess_intervention_identifiability(
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        parameter_scales=np.asarray([2.0, 3.0]),
    )
    changed_units = assess_intervention_identifiability(
        np.asarray([[1.0, 0.0], [0.0, 0.001]]),
        parameter_scales=np.asarray([2.0, 3000.0]),
    )
    assert np.allclose(
        base.conditional_information,
        changed_units.conditional_information,
    )
    assert np.allclose(base.eigenvalues, changed_units.eigenvalues)
