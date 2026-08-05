"""Adversarial regressions for fail-closed PhysTwin hypothesis boundaries."""

import numpy as np
import pytest

import causal4d.phystwin_backend as backend_module
import causal4d.rollout_bank_io as rollout_bank_io
from causal4d.phystwin_backend import (
    PhysTwinActionProposal,
    PhysTwinContactState,
    PhysTwinHypothesisConfig,
    build_contact_states,
    hidden_action_proposals,
    transform_controller_trajectory,
)


def _valid_action(**overrides: object) -> PhysTwinActionProposal:
    values: dict[str, object] = {
        "proposal_id": "proposal",
        "controller_points_m": np.zeros((4, 1, 3), dtype=float),
        "prior_weight": 1.0,
        "future_action_observed": False,
        "provenance": "unit test",
    }
    values.update(overrides)
    return PhysTwinActionProposal(**values)  # type: ignore[arg-type]


def _valid_contact(**overrides: object) -> PhysTwinContactState:
    values: dict[str, object] = {
        "attachment_shifts": (0,),
        "gain_multiplier": 1.0,
        "delay_steps": 0,
        "slip_fraction": 0.0,
        "rotation_degrees": 0.0,
        "prior_weight": 1.0,
    }
    values.update(overrides)
    return PhysTwinContactState(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, True, "1.0"],
)
def test_action_prior_weight_fails_closed(value: object) -> None:
    with pytest.raises(ValueError):
        _valid_action(prior_weight=value)


@pytest.mark.parametrize("value", [0, 1, np.bool_(True), "false"])
def test_action_observation_flag_requires_exact_bool(value: object) -> None:
    with pytest.raises(ValueError):
        _valid_action(future_action_observed=value)


@pytest.mark.parametrize("field", ["proposal_id", "provenance"])
def test_action_text_fields_require_nonempty_strings(field: str) -> None:
    with pytest.raises(ValueError):
        _valid_action(**{field: ""})
    with pytest.raises(ValueError):
        _valid_action(**{field: 1})


@pytest.mark.parametrize(
    "controls",
    [
        np.zeros((0, 1, 3), dtype=float),
        np.zeros((1, 0, 3), dtype=float),
        np.full((1, 1, 3), np.nan),
    ],
)
def test_action_controls_must_be_nonempty_and_finite(
    controls: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        _valid_action(controller_points_m=controls)


def test_real_numpy_scalars_are_normalized_without_string_coercion() -> None:
    action = _valid_action(prior_weight=np.float64(0.5))
    contact = _valid_contact(
        gain_multiplier=np.float64(1.0),
        slip_fraction=np.float64(0.25),
        rotation_degrees=np.float64(3.0),
        prior_weight=np.float64(0.5),
    )
    assert type(action.prior_weight) is float
    assert type(contact.gain_multiplier) is float
    assert type(contact.slip_fraction) is float
    assert type(contact.rotation_degrees) is float
    assert type(contact.prior_weight) is float


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gain_multiplier", float("nan")),
        ("gain_multiplier", float("inf")),
        ("gain_multiplier", 0.0),
        ("delay_steps", 1.0),
        ("delay_steps", True),
        ("slip_fraction", float("nan")),
        ("slip_fraction", 1.0),
        ("rotation_degrees", float("nan")),
        ("prior_weight", float("nan")),
        ("prior_weight", 0.0),
    ],
)
def test_contact_state_rejects_nonfinite_or_coercible_scalars(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        _valid_contact(**{field: value})


@pytest.mark.parametrize(
    "attachment_shifts",
    [(0.0,), (True,), (np.int64(0),), np.asarray([0], dtype=int)],
)
def test_contact_shifts_require_exact_integer_tuple(
    attachment_shifts: object,
) -> None:
    with pytest.raises(ValueError):
        _valid_contact(attachment_shifts=attachment_shifts)


@pytest.mark.parametrize(
    "config",
    [
        {"attachment_shift_values": (-1, 0, 0, 1)},
        {"gain_values": (1.0, float("nan"))},
        {"gain_values": (1.0, 1.0)},
        {"delay_values": (0, 1.0)},
        {"delay_values": (0, True)},
        {"slip_values": (0.0, float("inf"))},
        {"rotation_values_degrees": (0.0, float("nan"))},
        {"maximum_contact_states": True},
    ],
)
def test_hypothesis_config_rejects_invalid_grid_values(
    config: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        PhysTwinHypothesisConfig(**config)  # type: ignore[arg-type]


@pytest.mark.parametrize("hand_count", [True, 1.0, 0])
def test_contact_state_builder_requires_exact_positive_hand_count(
    hand_count: object,
) -> None:
    with pytest.raises(ValueError):
        build_contact_states(hand_count)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "labels",
    [
        np.asarray([0.0]),
        np.asarray([True]),
        np.asarray([-1]),
        np.asarray([1]),
    ],
)
def test_controller_groups_are_exact_nonnegative_contiguous_labels(
    labels: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        transform_controller_trajectory(
            np.zeros((4, 1, 3), dtype=float),
            labels,
            _valid_contact(),
            start_frame=2,
        )


@pytest.mark.parametrize("start_frame", [True, 2.0])
def test_controller_transform_start_frame_requires_exact_integer(
    start_frame: object,
) -> None:
    with pytest.raises(ValueError):
        transform_controller_trajectory(
            np.zeros((4, 1, 3), dtype=float),
            np.asarray([0]),
            _valid_contact(),
            start_frame=start_frame,  # type: ignore[arg-type]
        )


def test_hidden_action_parameters_fail_closed() -> None:
    controls = np.zeros((6, 1, 3), dtype=float)
    with pytest.raises(ValueError):
        hidden_action_proposals(controls, start_frame=True)
    with pytest.raises(ValueError):
        hidden_action_proposals(controls, start_frame=3, history_frames=2.0)
    with pytest.raises(ValueError):
        hidden_action_proposals(controls, start_frame=3, damping=float("nan"))


def test_backend_reexports_strict_rollout_bank_io() -> None:
    assert backend_module.save_rollout_bank is rollout_bank_io.save_rollout_bank
    assert backend_module.load_rollout_bank is rollout_bank_io.load_rollout_bank
