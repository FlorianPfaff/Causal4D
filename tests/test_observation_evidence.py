import numpy as np

from causal4d.grouped_likelihood import posterior_weights_from_grouped_evidence
from causal4d.observation_evidence import GroupedObservationEvidence, ObservationGroup


def _group(group_id: str, contributor: str) -> ObservationGroup:
    return ObservationGroup(
        group_id=group_id,
        values_m=np.asarray([1.0]),
        frame_indices=np.asarray([1]),
        node_indices=np.asarray([0]),
        coordinate_indices=np.asarray([0]),
        covariance_m2=np.asarray([[0.01]]),
        contributor_ids=(contributor,),
        prior_nominal_probability=0.9,
        outlier_scale_multiplier=100.0,
        degrees_of_freedom=4.0,
        source_id="unit",
    )


def test_duplicate_contributor_does_not_sharpen_posterior() -> None:
    components = np.zeros((2, 3, 1, 1), dtype=float)
    components[1, 1, 0, 0] = 1.0
    prior = np.asarray([0.5, 0.5])
    single = GroupedObservationEvidence(groups=(_group("g1", "same"),))
    duplicated = GroupedObservationEvidence(
        groups=(_group("g1", "same"), _group("g2", "same"))
    )
    first, _ = posterior_weights_from_grouped_evidence(
        prior, components, single, prefix_frame_count=2
    )
    second, _ = posterior_weights_from_grouped_evidence(
        prior, components, duplicated, prefix_frame_count=2
    )
    assert np.allclose(first, second, atol=1e-12, rtol=1e-12)
    assert first[1] > first[0]


def test_dense_prefix_excludes_future_frames() -> None:
    observations = np.zeros((5, 2, 3), dtype=float)
    evidence = GroupedObservationEvidence.from_dense_prefix(
        observations,
        prefix_frame_count=3,
        scale_m=0.01,
    )
    assert {int(group.frame_indices[0]) for group in evidence.groups} == {1, 2}
