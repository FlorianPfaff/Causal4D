import numpy as np
import pytest

from causal4d.contact_inference import ContactState
from causal4d.contact_metrics import (
    aggregate_contact_recovery,
    contact_recovery_metrics,
)


def _state(node: int) -> ContactState:
    return ContactState(
        contact_nodes=(node,),
        gain_multiplier=1.0,
        delay_steps=0,
        slip_fraction=0.0,
        rotation_radians=0.0,
    )


@pytest.mark.parametrize(
    "weights",
    (
        np.asarray([1.1, -0.1]),
        np.asarray([np.nan, np.nan]),
        np.asarray([np.inf, -np.inf]),
    ),
)
def test_contact_recovery_rejects_invalid_probability_support(
    weights: np.ndarray,
) -> None:
    states = (_state(0), _state(1))

    with pytest.raises(ValueError, match="finite and nonnegative"):
        contact_recovery_metrics(
            states,
            weights,
            states[0],
            confidence_level=0.9,
        )


def test_contact_recovery_rejects_invalid_confidence_level() -> None:
    states = (_state(0), _state(1))

    for confidence_level in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="confidence_level"):
            contact_recovery_metrics(
                states,
                np.asarray([0.5, 0.5]),
                states[0],
                confidence_level=confidence_level,
            )


def test_contact_recovery_reports_tie_aware_diagnostics() -> None:
    states = (_state(0), _state(1))
    result = contact_recovery_metrics(
        states,
        np.asarray([0.5, 0.5]),
        states[1],
        confidence_level=0.5,
    )

    assert result["node_map"] == "0"
    assert not result["node_correct"]
    assert result["node_map_set"] == "0|1"
    assert result["node_map_set_size"] == 2
    assert result["node_truth_in_map_set"]
    assert result["node_credible_set_size"] == 1
    assert not result["node_credible_covered"]
    assert result["node_tie_closed_credible_set_size"] == 2
    assert result["node_tie_closed_credible_covered"]
    assert result["node_normalized_entropy"] == pytest.approx(1.0)

    aggregate = aggregate_contact_recovery(
        [
            {
                **result,
                "object": "symmetric_graph",
                "setting": "online_adaptation",
                "world_condition": "shifted_contact",
            }
        ]
    )[0]
    assert aggregate["node_map_set_coverage"] == 1.0
    assert aggregate["node_tie_closed_credible_coverage"] == 1.0
    assert aggregate["mean_node_normalized_entropy"] == pytest.approx(1.0)
