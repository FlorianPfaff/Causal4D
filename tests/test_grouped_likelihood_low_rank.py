import numpy as np
import pytest

from causal4d.grouped_likelihood import (
    group_log_likelihood,
    grouped_component_log_likelihoods,
    posterior_weights_from_grouped_evidence,
)
from causal4d.observation_evidence import (
    GroupedObservationEvidence,
    ObservationGroup,
)


def _group(dimension: int, *, group_id: str = "structured") -> ObservationGroup:
    generator = np.random.default_rng(4)
    matrix = generator.normal(size=(dimension, dimension))
    covariance = matrix @ matrix.T + 0.25 * np.eye(dimension)
    return ObservationGroup(
        group_id=group_id,
        values_m=generator.normal(scale=0.1, size=dimension),
        frame_indices=np.ones(dimension, dtype=np.int64),
        node_indices=np.zeros(dimension, dtype=np.int64),
        coordinate_indices=np.arange(dimension, dtype=np.int64),
        covariance_m2=covariance,
        contributor_ids=(f"unit:{group_id}",),
        prior_nominal_probability=0.91,
        outlier_scale_multiplier=30.0,
        degrees_of_freedom=5.0,
        source_id="unit",
    )


def test_low_rank_factor_matches_dense_covariance_update() -> None:
    generator = np.random.default_rng(9)
    dimension = 7
    predictions = generator.normal(size=(2, 3, dimension))
    factor = generator.normal(scale=0.15, size=(2, 3, dimension, 2))
    dense = np.einsum("...ir,...jr->...ij", factor, factor)
    additive_variance = generator.uniform(0.01, 0.05, size=predictions.shape)
    group = _group(dimension)

    dense_score, dense_responsibility = group_log_likelihood(
        predictions,
        group,
        additive_variance_m2=additive_variance,
        additive_covariance_m2=dense,
    )
    factor_score, factor_responsibility = group_log_likelihood(
        predictions,
        group,
        additive_variance_m2=additive_variance,
        additive_covariance_factor_m=factor,
    )

    np.testing.assert_allclose(factor_score, dense_score, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        factor_responsibility,
        dense_responsibility,
        rtol=1e-11,
        atol=1e-11,
    )


def test_low_rank_path_does_not_use_dense_slogdet(monkeypatch) -> None:
    def forbidden_slogdet(*_args, **_kwargs):
        raise AssertionError("low-rank scoring must not use a dense determinant")

    monkeypatch.setattr(np.linalg, "slogdet", forbidden_slogdet)
    predictions = np.asarray([[0.2, -0.1, 0.3]])
    factor = np.asarray([[0.1], [0.2], [-0.1]])

    score, responsibility = group_log_likelihood(
        predictions,
        _group(3),
        additive_covariance_factor_m=factor,
    )

    assert np.all(np.isfinite(score))
    assert np.all((responsibility > 0.0) & (responsibility < 1.0))


def test_grouped_factor_broadcasts_and_records_structured_groups() -> None:
    group = ObservationGroup(
        group_id="factor",
        values_m=np.zeros(2),
        frame_indices=np.asarray([1, 1]),
        node_indices=np.asarray([0, 0]),
        coordinate_indices=np.asarray([0, 1]),
        covariance_m2=np.eye(2) * 0.2,
        contributor_ids=("unit:factor",),
        source_id="unit",
    )
    evidence = GroupedObservationEvidence(groups=(group,))
    components = np.zeros((2, 3, 3, 1, 2))
    components[0, :, 1, 0] = [0.5, 0.5]
    components[1, :, 1, 0] = [0.5, -0.5]
    factor = np.broadcast_to(
        np.asarray([[0.4], [0.4]]),
        (3, 2, 1),
    )

    score, diagnostics = grouped_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=2,
        component_group_covariance_factor_m={"factor": factor},
    )

    assert score.shape == (2, 3)
    assert np.all(score[0] > score[1])
    assert diagnostics.full_covariance_group_ids == ()
    assert diagnostics.low_rank_covariance_group_ids == ("factor",)


def test_structured_and_dense_paths_produce_identical_posteriors() -> None:
    generator = np.random.default_rng(12)
    group = _group(3)
    evidence = GroupedObservationEvidence(groups=(group,))
    components = generator.normal(size=(4, 3, 1, 3))
    factor = generator.normal(scale=0.1, size=(4, 3, 2))
    dense = np.einsum("...ir,...jr->...ij", factor, factor)
    prior = np.asarray([0.1, 0.2, 0.3, 0.4])

    dense_posterior, _ = posterior_weights_from_grouped_evidence(
        prior,
        components,
        evidence,
        prefix_frame_count=2,
        component_group_covariance_m2={"structured": dense},
    )
    factor_posterior, diagnostics = posterior_weights_from_grouped_evidence(
        prior,
        components,
        evidence,
        prefix_frame_count=2,
        component_group_covariance_factor_m={"structured": factor},
    )

    np.testing.assert_allclose(
        factor_posterior,
        dense_posterior,
        rtol=1e-11,
        atol=1e-11,
    )
    assert np.isclose(np.sum(factor_posterior), 1.0)
    assert diagnostics.low_rank_covariance_group_ids == ("structured",)


def test_invalid_low_rank_factors_fail_closed() -> None:
    group = _group(3)
    predictions = np.zeros((2, 3))

    with pytest.raises(ValueError, match="positive_rank"):
        group_log_likelihood(
            predictions,
            group,
            additive_covariance_factor_m=np.empty((3, 0)),
        )
    invalid = np.ones((3, 1))
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="factor must be finite"):
        group_log_likelihood(
            predictions,
            group,
            additive_covariance_factor_m=invalid,
        )

    evidence = GroupedObservationEvidence(groups=(group,))
    components = np.zeros((2, 3, 1, 3))
    with pytest.raises(ValueError, match="unknown groups"):
        grouped_component_log_likelihoods(
            components,
            evidence,
            prefix_frame_count=2,
            component_group_covariance_factor_m={
                "unknown": np.ones((3, 1))
            },
        )
