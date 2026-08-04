"""Recovery and calibration metrics for latent contact posteriors."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Hashable, Sequence, TypeVar

import numpy as np

from causal4d.contact_inference import ContactState


Value = TypeVar("Value", bound=Hashable)


def _group_probabilities(
    states: Sequence[ContactState],
    weights: np.ndarray,
    getter: Callable[[ContactState], Value],
) -> dict[Value, float]:
    probabilities: dict[Value, float] = defaultdict(float)
    for state, weight in zip(states, weights, strict=True):
        probabilities[getter(state)] += float(weight)
    return dict(probabilities)


def _weighted_quantile(
    probabilities: dict[float | int, float], probability: float
) -> float:
    values = np.asarray(sorted(probabilities), dtype=float)
    weights = np.asarray([probabilities[value] for value in sorted(probabilities)])
    cumulative = np.cumsum(weights)
    index = min(
        int(np.searchsorted(cumulative, probability, side="left")), values.size - 1
    )
    return float(values[index])


def _weighted_crps(probabilities: dict[float | int, float], truth: float) -> float:
    values = np.asarray(list(probabilities), dtype=float)
    weights = np.asarray(list(probabilities.values()), dtype=float)
    first = float(np.sum(weights * np.abs(values - truth)))
    pairwise = np.abs(values[:, None] - values[None, :])
    second = 0.5 * float(np.sum(weights[:, None] * weights[None, :] * pairwise))
    return first - second


def _credible_set(
    probabilities: dict[Value, float],
    confidence_level: float,
    *,
    close_boundary_ties: bool = False,
) -> set[Value]:
    ordered = sorted(
        probabilities.items(), key=lambda item: item[1], reverse=True
    )
    cumulative = 0.0
    selected: set[Value] = set()
    boundary_probability: float | None = None
    for value, probability in ordered:
        selected.add(value)
        cumulative += probability
        if cumulative >= confidence_level:
            boundary_probability = probability
            break
    if close_boundary_ties and boundary_probability is not None:
        for value, probability in ordered[len(selected) :]:
            if not np.isclose(
                probability,
                boundary_probability,
                rtol=1e-12,
                atol=1e-15,
            ):
                break
            selected.add(value)
    return selected


def _credible_set_coverage(
    probabilities: dict[Value, float],
    truth: Value,
    confidence_level: float,
) -> bool:
    return truth in _credible_set(probabilities, confidence_level)


def _categorical_entropy(probabilities: dict[Value, float]) -> tuple[float, float]:
    positive = np.asarray(
        [probability for probability in probabilities.values() if probability > 0.0]
    )
    entropy = -float(np.sum(positive * np.log(positive)))
    normalized = (
        entropy / np.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    )
    return entropy, float(normalized)


def _continuous_metrics(
    probabilities: dict[float | int, float],
    truth: float,
    *,
    confidence_level: float,
    prefix: str,
) -> dict[str, Any]:
    mean = float(
        sum(float(value) * probability for value, probability in probabilities.items())
    )
    tail = 0.5 * (1.0 - confidence_level)
    lower = _weighted_quantile(probabilities, tail)
    upper = _weighted_quantile(probabilities, 1.0 - tail)
    return {
        f"{prefix}_truth": truth,
        f"{prefix}_posterior_mean": mean,
        f"{prefix}_absolute_error": abs(mean - truth),
        f"{prefix}_interval_lower": lower,
        f"{prefix}_interval_upper": upper,
        f"{prefix}_covered": bool(lower <= truth <= upper),
        f"{prefix}_crps": _weighted_crps(probabilities, truth),
    }


def contact_recovery_metrics(
    states: Sequence[ContactState],
    weights: np.ndarray,
    truth: ContactState,
    *,
    confidence_level: float,
) -> dict[str, Any]:
    """Score node, gain, delay, slip, and frame-bias posterior recovery."""

    states = tuple(states)
    weights = np.asarray(weights, dtype=float)
    if not states or weights.shape != (len(states),):
        raise ValueError("contact weights must align with a nonempty state support")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("contact weights must be finite and nonnegative")
    if not np.isclose(np.sum(weights), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("contact weights must sum to one")
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1)")

    node_probabilities = _group_probabilities(
        states, weights, lambda state: state.contact_nodes
    )
    gain_probabilities = _group_probabilities(
        states, weights, lambda state: state.gain_multiplier
    )
    delay_probabilities = _group_probabilities(
        states, weights, lambda state: state.delay_steps
    )
    slip_probabilities = _group_probabilities(
        states, weights, lambda state: state.slip_fraction
    )
    rotation_probabilities = _group_probabilities(
        states, weights, lambda state: float(np.rad2deg(state.rotation_radians))
    )

    node_map, node_confidence = max(
        node_probabilities.items(), key=lambda item: item[1]
    )
    node_map_set = tuple(
        nodes
        for nodes, probability in sorted(node_probabilities.items())
        if np.isclose(
            probability,
            node_confidence,
            rtol=1e-12,
            atol=1e-15,
        )
    )
    node_credible_set = _credible_set(node_probabilities, confidence_level)
    node_tie_closed_credible_set = _credible_set(
        node_probabilities,
        confidence_level,
        close_boundary_ties=True,
    )
    node_entropy, node_normalized_entropy = _categorical_entropy(node_probabilities)
    node_brier = float(
        sum(
            (probability - float(nodes == truth.contact_nodes)) ** 2
            for nodes, probability in node_probabilities.items()
        )
    )
    delay_map, delay_confidence = max(
        delay_probabilities.items(), key=lambda item: item[1]
    )
    positive = weights > 0.0
    entropy = -float(np.sum(weights[positive] * np.log(weights[positive])))

    result: dict[str, Any] = {
        "node_truth": ";".join(map(str, truth.contact_nodes)),
        "node_map": ";".join(map(str, node_map)),
        "node_correct": bool(node_map == truth.contact_nodes),
        "node_confidence": float(node_confidence),
        "node_map_set": "|".join(
            ";".join(map(str, nodes)) for nodes in node_map_set
        ),
        "node_map_set_size": len(node_map_set),
        "node_truth_in_map_set": truth.contact_nodes in node_map_set,
        "node_support_size": len(node_probabilities),
        "node_entropy": node_entropy,
        "node_normalized_entropy": node_normalized_entropy,
        "node_truth_probability": float(
            node_probabilities.get(truth.contact_nodes, 0.0)
        ),
        "node_brier": node_brier,
        "node_credible_covered": truth.contact_nodes in node_credible_set,
        "node_credible_set_size": len(node_credible_set),
        "node_tie_closed_credible_covered": (
            truth.contact_nodes in node_tie_closed_credible_set
        ),
        "node_tie_closed_credible_set_size": len(node_tie_closed_credible_set),
        "delay_map": int(delay_map),
        "delay_map_correct": bool(delay_map == truth.delay_steps),
        "delay_map_confidence": float(delay_confidence),
        "joint_positive_support_size": int(np.count_nonzero(positive)),
        "joint_effective_sample_size": float(1.0 / np.sum(np.square(weights))),
        "joint_normalized_entropy": float(
            entropy / np.log(weights.size) if weights.size > 1 else 0.0
        ),
    }
    result.update(
        _continuous_metrics(
            gain_probabilities,
            truth.gain_multiplier,
            confidence_level=confidence_level,
            prefix="gain",
        )
    )
    result.update(
        _continuous_metrics(
            delay_probabilities,
            float(truth.delay_steps),
            confidence_level=confidence_level,
            prefix="delay",
        )
    )
    result.update(
        _continuous_metrics(
            slip_probabilities,
            truth.slip_fraction,
            confidence_level=confidence_level,
            prefix="slip",
        )
    )
    result.update(
        _continuous_metrics(
            rotation_probabilities,
            float(np.rad2deg(truth.rotation_radians)),
            confidence_level=confidence_level,
            prefix="rotation_deg",
        )
    )
    return result


def aggregate_contact_recovery(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate recovery and calibration by inference setting and world."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["setting"]), str(row["world_condition"]))].append(row)
    output: list[dict[str, Any]] = []
    for (setting, world), selected in sorted(groups.items()):
        node_accuracy = float(np.mean([float(row["node_correct"]) for row in selected]))
        node_confidence = float(np.mean([row["node_confidence"] for row in selected]))
        aggregate: dict[str, Any] = {
            "setting": setting,
            "world_condition": world,
            "case_count": len(selected),
            "object_count": len({row["object"] for row in selected}),
            "node_accuracy": node_accuracy,
            "mean_node_confidence": node_confidence,
            "node_calibration_error": abs(node_confidence - node_accuracy),
            "mean_node_truth_probability": float(
                np.mean([row["node_truth_probability"] for row in selected])
            ),
            "mean_node_brier": float(
                np.mean([row["node_brier"] for row in selected])
            ),
            "node_credible_coverage": float(
                np.mean([float(row["node_credible_covered"]) for row in selected])
            ),
            "mean_gain_absolute_error": float(
                np.mean([row["gain_absolute_error"] for row in selected])
            ),
            "gain_coverage": float(
                np.mean([float(row["gain_covered"]) for row in selected])
            ),
            "mean_gain_crps": float(
                np.mean([row["gain_crps"] for row in selected])
            ),
            "delay_map_accuracy": float(
                np.mean([float(row["delay_map_correct"]) for row in selected])
            ),
            "mean_delay_absolute_error": float(
                np.mean([row["delay_absolute_error"] for row in selected])
            ),
            "delay_coverage": float(
                np.mean([float(row["delay_covered"]) for row in selected])
            ),
            "mean_delay_crps": float(
                np.mean([row["delay_crps"] for row in selected])
            ),
            "mean_slip_absolute_error": float(
                np.mean([row["slip_absolute_error"] for row in selected])
            ),
            "slip_coverage": float(
                np.mean([float(row["slip_covered"]) for row in selected])
            ),
            "mean_rotation_absolute_error_deg": float(
                np.mean([row["rotation_deg_absolute_error"] for row in selected])
            ),
            "rotation_coverage": float(
                np.mean([float(row["rotation_deg_covered"]) for row in selected])
            ),
            "mean_joint_effective_sample_size": float(
                np.mean([row["joint_effective_sample_size"] for row in selected])
            ),
        }
        diagnostic_fields = {
            "node_map_set_coverage": "node_truth_in_map_set",
            "mean_node_map_set_size": "node_map_set_size",
            "mean_node_credible_set_size": "node_credible_set_size",
            "node_tie_closed_credible_coverage": (
                "node_tie_closed_credible_covered"
            ),
            "mean_node_tie_closed_credible_set_size": (
                "node_tie_closed_credible_set_size"
            ),
            "mean_node_normalized_entropy": "node_normalized_entropy",
            "mean_joint_positive_support_size": "joint_positive_support_size",
        }
        for output_name, row_name in diagnostic_fields.items():
            if all(row_name in row for row in selected):
                aggregate[output_name] = float(
                    np.mean([float(row[row_name]) for row in selected])
                )
        output.append(aggregate)
    return output
