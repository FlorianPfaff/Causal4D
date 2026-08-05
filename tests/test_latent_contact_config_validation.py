from __future__ import annotations

import pytest

from causal4d.contact_inference import LatentContactConfig


@pytest.mark.parametrize(
    "field",
    (
        "observation_fraction",
        "observation_noise_std_m",
        "node_prior_smoothing",
        "categorical_prior_smoothing",
        "gain_prior_bandwidth",
        "slip_prior_bandwidth",
        "rotation_prior_bandwidth_deg",
        "confidence_level",
        "variance_scale_min",
        "variance_scale_max",
        "gate_gap_closure",
        "gate_matched_degradation",
        "gate_coverage_tolerance",
        "gate_node_accuracy",
        "gate_node_credible_coverage",
        "gate_node_calibration_error",
        "gate_gain_mae",
        "gate_gain_coverage",
        "gate_delay_mae_steps",
        "gate_delay_map_accuracy",
        "gate_delay_coverage",
        "gate_minimum_topology_gap_closure",
    ),
)
@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_non_finite_scalar_controls_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        LatentContactConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("likelihood_scales_m", ()),
        ("dynamic_likelihood_weights", ()),
        ("likelihood_powers", ()),
        ("posterior_temperatures", ()),
        ("gain_values", ()),
        ("delay_values", ()),
        ("slip_values", ()),
        ("rotation_values_deg", ()),
        ("likelihood_scales_m", (0.0015, float("nan"))),
        ("dynamic_likelihood_weights", (0.0, float("inf"))),
        ("likelihood_powers", (0.2, float("nan"))),
        ("posterior_temperatures", (1.0, float("inf"))),
        ("gain_values", (0.7, float("nan"))),
        ("slip_values", (0.0, float("inf"))),
        ("rotation_values_deg", (0.0, float("nan"))),
        ("likelihood_scales_m", (0.0015, 0.0015)),
        ("dynamic_likelihood_weights", (0.0, 0.0)),
        ("likelihood_powers", (0.2, 0.2)),
        ("posterior_temperatures", (1.0, 1.0)),
        ("gain_values", (0.7, 0.7)),
        ("delay_values", (0, 0)),
        ("slip_values", (0.0, 0.0)),
        ("rotation_values_deg", (0.0, 0.0)),
        ("delay_values", (0, True)),
        ("delay_values", (0, 1.5)),
    ),
)
def test_invalid_candidate_grids_are_rejected(
    field: str,
    value: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError, match=field):
        LatentContactConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("gain_prior_bandwidth", 0.0),
        ("slip_prior_bandwidth", 0.0),
        ("rotation_prior_bandwidth_deg", 0.0),
        ("node_prior_smoothing", 0.0),
        ("categorical_prior_smoothing", 0.0),
        ("gate_gain_mae", -0.01),
        ("gate_delay_mae_steps", -0.01),
    ),
)
def test_positive_and_nonnegative_scalar_bounds_are_enforced(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        LatentContactConfig(**{field: value})


@pytest.mark.parametrize(
    "field",
    (
        "gate_gap_closure",
        "gate_matched_degradation",
        "gate_coverage_tolerance",
        "gate_node_accuracy",
        "gate_node_credible_coverage",
        "gate_node_calibration_error",
        "gate_gain_coverage",
        "gate_delay_map_accuracy",
        "gate_delay_coverage",
        "gate_minimum_topology_gap_closure",
    ),
)
@pytest.mark.parametrize("value", (-0.01, 1.01))
def test_probability_like_gate_bounds_are_enforced(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        LatentContactConfig(**{field: value})


@pytest.mark.parametrize("value", (True, 0, 1.5))
def test_parameter_particle_count_requires_a_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="parameter_particle_count"):
        LatentContactConfig(parameter_particle_count=value)


@pytest.mark.parametrize("frame_count", (True, 3, 3.5))
def test_prefix_frame_count_rejects_invalid_totals(frame_count: object) -> None:
    with pytest.raises(ValueError, match="frame_count"):
        LatentContactConfig().prefix_frame_count(frame_count)


def test_minimum_valid_frame_count_preserves_one_future_frame() -> None:
    assert LatentContactConfig().prefix_frame_count(4) == 3
