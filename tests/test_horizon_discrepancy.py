from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from causal4d.contracts import TwinBelief, build_causal_context
from causal4d.horizon_discrepancy import (
    HorizonDiscrepancyBankV1,
    build_horizon_discrepancy_bank,
)

provider_api = pytest.importorskip(
    "bayesian_phystwin.causal4d_belief_provider_v2"
)
if not hasattr(provider_api, "HorizonDiscrepancyCalibrationV1"):
    pytest.skip(
        "installed Bayesian-PhysTwin lacks horizon discrepancy provider v2",
        allow_module_level=True,
    )


def _belief() -> TwinBelief:
    observations = np.zeros((5, 2, 3), dtype=float)
    observed_actions = np.zeros((5, 1, 3), dtype=float)
    counterfactual_actions = np.ones((5, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="horizon-discrepancy-test-v1",
        case_id="case-001",
        observations=observations,
        observed_actions=observed_actions,
        counterfactual_actions=counterfactual_actions,
        intervention_frame=3,
    )
    return TwinBelief(
        context=context,
        endpoint_frame=2,
        particle_ids=("particle-a", "particle-b"),
        theta_names=("spring_log_scale",),
        endpoint_position_m=np.zeros((2, 3, 3), dtype=float),
        endpoint_velocity_mps=np.zeros((2, 3, 3), dtype=float),
        theta=np.asarray(((0.0,), (0.2,)), dtype=float),
        discrepancy_mean_m=np.zeros((2, 3, 3), dtype=float),
        discrepancy_variance_m2=np.full((2, 3, 3), 1e-6, dtype=float),
        weights=np.asarray((0.6, 0.4), dtype=float),
        metadata={"test_fixture": True},
    )


def _posteriors():
    residual_a = np.zeros((4, 2, 3), dtype=float)
    residual_a[:, 0, 0] = (0.001, 0.002, 0.003, 0.004)
    residual_a[:, 1, 1] = (0.0, -0.001, -0.002, -0.003)
    residual_b = 1.5 * residual_a
    valid = np.ones((4, 2), dtype=bool)
    return (
        provider_api.infer_model_averaged_bayesian_anchor_endpoint(
            residual_a,
            valid,
            end_frame=4,
        ),
        provider_api.infer_model_averaged_bayesian_anchor_endpoint(
            residual_b,
            valid,
            end_frame=4,
        ),
    )


def _calibration():
    return provider_api.HorizonDiscrepancyCalibrationV1(
        source_group_ids=("source-a", "source-b", "source-c"),
        source_summary_sha256="a" * 64,
        horizon_steps=(1, 2, 4),
        mean_reversion_half_life_steps=2.0,
        minimum_mean_retention=0.1,
        stationary_std_m=np.asarray((0.01, 0.02, 0.03)),
        additional_process_std_m_per_sqrt_step=np.asarray(
            (0.001, 0.002, 0.003)
        ),
        component_process_variance_scale=1.0,
        metadata={"split": "source-only"},
    )


def _build(
    *,
    horizons=(3, 0, 1),
    maximum_discrepancy_m=None,
) -> HorizonDiscrepancyBankV1:
    return build_horizon_discrepancy_bank(
        _belief(),
        _posteriors(),
        _calibration(),
        horizon_steps=horizons,
        lift_indices=np.asarray(((0, 1),), dtype=np.int64),
        lift_weights=np.asarray(((0.25, 0.75),), dtype=float),
        maximum_discrepancy_m=maximum_discrepancy_m,
        provider_revision="b" * 40,
        metadata={"registered_role": "development-only"},
    )


def test_builds_source_bound_horizon_bank_with_endpoint_parity() -> None:
    belief = _belief()
    posteriors = _posteriors()
    bank = build_horizon_discrepancy_bank(
        belief,
        posteriors,
        _calibration(),
        horizon_steps=(3, 0, 1),
        lift_indices=np.asarray(((0, 1),), dtype=np.int64),
        lift_weights=np.asarray(((0.25, 0.75),), dtype=float),
        provider_revision="b" * 40,
    )

    np.testing.assert_array_equal(bank.horizon_steps, (0, 1, 3))
    assert bank.mean_m.shape == (2, 3, 3, 3)
    assert bank.covariance_m2.shape == (2, 3, 3, 3, 3)
    assert bank.twin_belief_id == belief.artifact_id
    assert bank.particle_ids == belief.particle_ids
    np.testing.assert_allclose(bank.particle_weights, belief.weights)
    assert bank.calibration_id == _calibration().artifact_id
    assert bank.provider_revision == "b" * 40
    assert len(bank.artifact_id) == 64
    assert bank.metadata["future_observations_read"] == 0
    assert bank.metadata["calibration_source_group_count"] == 3
    assert bank.metadata["calibration_target_outcomes_used"] is False
    assert not bank.mean_m.flags.writeable
    assert not bank.covariance_m2.flags.writeable

    for particle_index, posterior in enumerate(posteriors):
        np.testing.assert_allclose(
            bank.mean_m[particle_index, 0, :2],
            posterior.mean_m,
        )
        np.testing.assert_allclose(
            bank.covariance_m2[particle_index, 0, :2],
            posterior.covariance_m2,
        )
        expected_extra_mean = (
            0.25 * posterior.mean_m[0] + 0.75 * posterior.mean_m[1]
        )
        expected_extra_covariance = (
            0.25**2 * posterior.covariance_m2[0]
            + 0.75**2 * posterior.covariance_m2[1]
        )
        np.testing.assert_allclose(
            bank.mean_m[particle_index, 0, 2],
            expected_extra_mean,
        )
        np.testing.assert_allclose(
            bank.covariance_m2[particle_index, 0, 2],
            expected_extra_covariance,
        )

    assert bank.mean_retention[0] == 1.0
    np.testing.assert_array_equal(bank.additional_axis_variance_m2[0], 0.0)
    assert bank.mean_retention[-1] < bank.mean_retention[1] < 1.0
    assert np.all(bank.additional_axis_variance_m2[-1] > 0.0)


def test_moments_accessor_returns_registered_immutable_view() -> None:
    bank = _build()

    mean, covariance = bank.moments_at_horizon(1)

    np.testing.assert_array_equal(mean, bank.mean_m[:, 1])
    np.testing.assert_array_equal(covariance, bank.covariance_m2[:, 1])
    assert not mean.flags.writeable
    assert not covariance.flags.writeable
    with pytest.raises(KeyError, match="not registered"):
        bank.moments_at_horizon(2)


def test_mean_cap_changes_means_without_rewriting_covariance() -> None:
    uncapped = _build()
    capped = _build(maximum_discrepancy_m=5e-4)

    assert np.max(np.linalg.norm(capped.mean_m, axis=-1)) <= 5e-4 + 1e-15
    np.testing.assert_array_equal(capped.covariance_m2, uncapped.covariance_m2)
    assert capped.artifact_id != uncapped.artifact_id


@pytest.mark.parametrize(
    "horizons",
    [
        (1, 2),
        (0, 1, 1),
        (0, -1),
        (0, 1.5),
    ],
)
def test_rejects_noncanonical_horizon_sets(horizons) -> None:
    with pytest.raises(ValueError, match="horizon_steps"):
        _build(horizons=horizons)


def test_rejects_invalid_or_repeated_lift_nodes() -> None:
    belief = _belief()
    posteriors = _posteriors()
    calibration = _calibration()

    with pytest.raises(ValueError, match="unavailable tracked node"):
        build_horizon_discrepancy_bank(
            belief,
            posteriors,
            calibration,
            horizon_steps=(0, 1),
            lift_indices=np.asarray(((0, 2),), dtype=np.int64),
            lift_weights=np.asarray(((0.5, 0.5),), dtype=float),
            provider_revision="b" * 40,
        )
    with pytest.raises(ValueError, match="must not repeat"):
        build_horizon_discrepancy_bank(
            belief,
            posteriors,
            calibration,
            horizon_steps=(0, 1),
            lift_indices=np.asarray(((0, 0),), dtype=np.int64),
            lift_weights=np.asarray(((0.5, 0.5),), dtype=float),
            provider_revision="b" * 40,
        )


def test_rejects_particle_or_calibration_type_mismatch() -> None:
    belief = _belief()

    with pytest.raises(ValueError, match="every TwinBelief particle"):
        build_horizon_discrepancy_bank(
            belief,
            _posteriors()[:1],
            _calibration(),
            horizon_steps=(0, 1),
            lift_indices=np.asarray(((0, 1),), dtype=np.int64),
            lift_weights=np.asarray(((0.5, 0.5),), dtype=float),
            provider_revision="b" * 40,
        )
    with pytest.raises(TypeError, match="calibration"):
        build_horizon_discrepancy_bank(
            belief,
            _posteriors(),
            object(),
            horizon_steps=(0, 1),
            lift_indices=np.asarray(((0, 1),), dtype=np.int64),
            lift_weights=np.asarray(((0.5, 0.5),), dtype=float),
            provider_revision="b" * 40,
        )


def test_artifact_identity_binds_provider_and_numerical_values() -> None:
    bank = _build()
    changed_provider = replace(bank, provider_revision="c" * 40)
    changed_mean_values = np.asarray(bank.mean_m).copy()
    changed_mean_values[0, -1, 0, 0] += 1e-6
    changed_mean = replace(bank, mean_m=changed_mean_values)

    assert changed_provider.artifact_id != bank.artifact_id
    assert changed_mean.artifact_id != bank.artifact_id
