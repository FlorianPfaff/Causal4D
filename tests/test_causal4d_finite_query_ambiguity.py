import numpy as np

from causal4d.finite_query_ambiguity import (
    FiniteQueryAmbiguityConfig,
    assess_finite_query_ambiguity,
)


_CONFIG = FiniteQueryAmbiguityConfig(
    maximum_prefix_rms_mahalanobis=0.25,
    minimum_query_rms_distance=1.0,
    maximum_ambiguous_pair_mass=0.20,
    maximum_weighted_query_distance=0.20,
)


def test_query_ambiguity_rejects_prefix_indistinguishable_futures() -> None:
    result = assess_finite_query_ambiguity(
        np.asarray([[0.0, 0.0], [0.05, 0.0]]),
        np.asarray([[0.0], [2.0]]),
        np.asarray([0.5, 0.5]),
        prefix_covariance=1.0,
        query_scale=1.0,
        config=_CONFIG,
    )
    assert not result.admissible
    assert result.ambiguous_pair_mass == 0.5
    assert result.weighted_query_distance == 1.0
    assert result.pair_indices.tolist() == [[0, 1]]


def test_query_ambiguity_is_permutation_invariant() -> None:
    prefix = np.asarray([[0.0, 0.0], [0.05, 0.0], [3.0, 1.0]])
    query = np.asarray([[0.0], [2.0], [0.0]])
    weights = np.asarray([0.3, 0.4, 0.3])
    reference = assess_finite_query_ambiguity(
        prefix,
        query,
        weights,
        config=_CONFIG,
    )
    permutation = np.asarray([2, 0, 1])
    changed = assess_finite_query_ambiguity(
        prefix[permutation],
        query[permutation],
        weights[permutation],
        config=_CONFIG,
    )
    assert reference.admissible == changed.admissible
    assert np.isclose(reference.ambiguous_pair_mass, changed.ambiguous_pair_mass)
    assert np.isclose(
        reference.weighted_query_distance,
        changed.weighted_query_distance,
    )


def test_query_ambiguity_is_invariant_to_exact_support_cloning() -> None:
    reference = assess_finite_query_ambiguity(
        np.asarray([[0.0], [0.0]]),
        np.asarray([[0.0], [2.0]]),
        np.asarray([0.4, 0.6]),
        config=_CONFIG,
    )
    cloned = assess_finite_query_ambiguity(
        np.asarray([[0.0], [0.0], [0.0]]),
        np.asarray([[0.0], [0.0], [2.0]]),
        np.asarray([0.1, 0.3, 0.6]),
        config=_CONFIG,
    )
    assert np.isclose(reference.ambiguous_pair_mass, cloned.ambiguous_pair_mass)
    assert np.isclose(
        reference.weighted_query_distance,
        cloned.weighted_query_distance,
    )


def test_identical_query_predictions_are_not_ambiguous() -> None:
    result = assess_finite_query_ambiguity(
        np.zeros((3, 2)),
        np.ones((3, 4)),
        np.asarray([0.2, 0.3, 0.5]),
        config=_CONFIG,
    )
    assert result.admissible
    assert result.ambiguous_pair_mass == 0.0
    assert len(result.pair_indices) == 0
