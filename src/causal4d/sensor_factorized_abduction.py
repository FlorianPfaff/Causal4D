"""Independent-sensor factors for realized-intervention abduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from causal4d.contracts import FactualIntervention
from causal4d.sensor_evidence import ActuatorEvidence, ContactWrenchEvidence
from causal4d.weighting import log_weights_from_probabilities


@dataclass(frozen=True)
class IndependentSensorAbductionConfig:
    """Robust factor settings with capped effective sample counts."""

    degrees_of_freedom: float = 4.0
    actuator_likelihood_power: float = 1.0
    wrench_likelihood_power: float = 1.0
    actuator_effective_sample_cap: float = 32.0
    wrench_effective_sample_cap: float = 16.0
    minimum_variance: float = 1.0e-12
    uninformative_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if not np.isfinite(self.degrees_of_freedom) or self.degrees_of_freedom <= 0:
            raise ValueError("degrees_of_freedom must be finite and positive")
        for name, value in (
            ("actuator_likelihood_power", self.actuator_likelihood_power),
            ("wrench_likelihood_power", self.wrench_likelihood_power),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (
            ("actuator_effective_sample_cap", self.actuator_effective_sample_cap),
            ("wrench_effective_sample_cap", self.wrench_effective_sample_cap),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.minimum_variance) or self.minimum_variance <= 0.0:
            raise ValueError("minimum_variance must be finite and positive")
        if (
            not np.isfinite(self.uninformative_tolerance)
            or self.uninformative_tolerance < 0.0
        ):
            raise ValueError("uninformative_tolerance must be finite and nonnegative")


def _predicted_variance(
    values: np.ndarray | None,
    expected_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    if values is None:
        return np.zeros(expected_shape, dtype=float)
    supplied = np.asarray(values, dtype=float)
    try:
        result = np.broadcast_to(supplied, expected_shape).copy()
    except ValueError as error:
        raise ValueError(f"{name} must broadcast to {expected_shape}") from error
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _student_t_component_log_likelihood(
    observed: np.ndarray,
    observation_variance: np.ndarray,
    valid_mask: np.ndarray,
    predicted: np.ndarray,
    predicted_variance: np.ndarray | None,
    *,
    degrees_of_freedom: float,
    effective_sample_cap: float,
    minimum_variance: float,
    name: str,
) -> tuple[np.ndarray, int]:
    if not np.all(np.isfinite(predicted)):
        raise ValueError(f"{name} predictions must be finite")
    component_variance = _predicted_variance(
        predicted_variance,
        predicted.shape,
        f"{name}_predicted_variance",
    )
    valid_count = int(np.sum(valid_mask))
    if valid_count == 0:
        return np.zeros(predicted.shape[0], dtype=float), 0
    total_variance = observation_variance[None] + component_variance
    total_variance = total_variance + float(minimum_variance)
    residual = predicted - observed[None]
    nu = float(degrees_of_freedom)
    score = -0.5 * np.log(total_variance) - 0.5 * (nu + 1.0) * np.log1p(
        np.square(residual) / (nu * total_variance)
    )
    score = np.where(valid_mask[None], score, 0.0)
    axes = tuple(range(1, score.ndim))
    log_likelihood = np.sum(score, axis=axes)
    effective_count = min(float(valid_count), float(effective_sample_cap))
    log_likelihood *= effective_count / float(valid_count)
    return log_likelihood, valid_count


def _validate_prediction_shape(
    values: np.ndarray,
    component_count: int,
    observation_shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    predicted = np.asarray(values, dtype=float)
    expected = (component_count,) + observation_shape
    if predicted.shape != expected:
        raise ValueError(f"{name} must have shape {expected}")
    return predicted


def _effective_sample_size(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(np.square(np.asarray(weights, dtype=float))))


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    maximum = float(np.max(log_weights))
    if not np.isfinite(maximum):
        raise RuntimeError("independent-sensor posterior normalization failed")
    weights = np.exp(log_weights - maximum)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("independent-sensor posterior normalization failed")
    return weights / total


def _validate_evidence_binding(
    factual: FactualIntervention,
    evidence: ActuatorEvidence | ContactWrenchEvidence,
    *,
    name: str,
) -> None:
    expected = (
        factual.context.protocol_id,
        factual.context.case_id,
        factual.context.u_obs.action_id,
    )
    supplied = (
        evidence.protocol_id,
        evidence.case_id,
        evidence.observed_action_id,
    )
    if supplied != expected:
        raise ValueError(
            f"{name} evidence identifies a different protocol, case, or observed action"
        )
    if evidence.evidence_frame_stop > factual.evidence_frame_stop:
        raise ValueError(f"{name} evidence extends beyond the factual prefix")


def reweight_factual_intervention_with_independent_sensors(
    factual: FactualIntervention,
    *,
    actuator_evidence: ActuatorEvidence | None = None,
    predicted_actuator_positions_m: np.ndarray | None = None,
    predicted_actuator_variance_m2: np.ndarray | None = None,
    wrench_evidence: ContactWrenchEvidence | None = None,
    predicted_contact_wrench: np.ndarray | None = None,
    predicted_wrench_variance: np.ndarray | None = None,
    config: IndependentSensorAbductionConfig | None = None,
) -> FactualIntervention:
    """Multiply an existing factual posterior by independent sensor factors.

    The input posterior is assumed to have consumed the permitted object prefix.
    This API deliberately has no object-observation argument. With absent,
    invalid, zero-powered, or component-invariant factors it returns ``factual``
    exactly.
    """

    settings = config or IndependentSensorAbductionConfig()
    component_count = len(factual.weights)
    if actuator_evidence is None and predicted_actuator_positions_m is not None:
        raise ValueError("actuator predictions require actuator_evidence")
    if actuator_evidence is not None and predicted_actuator_positions_m is None:
        raise ValueError("actuator_evidence requires actuator predictions")
    if wrench_evidence is None and predicted_contact_wrench is not None:
        raise ValueError("wrench predictions require wrench_evidence")
    if wrench_evidence is not None and predicted_contact_wrench is None:
        raise ValueError("wrench_evidence requires wrench predictions")
    if actuator_evidence is None and wrench_evidence is None:
        return factual
    if (
        actuator_evidence is not None
        and wrench_evidence is not None
        and actuator_evidence.clock_id != wrench_evidence.clock_id
    ):
        raise ValueError("actuator and wrench evidence must use the same clock")

    total_log_factor = np.zeros(component_count, dtype=float)
    factor_summaries: list[dict[str, Any]] = []

    if actuator_evidence is not None:
        _validate_evidence_binding(factual, actuator_evidence, name="actuator")
        predicted = _validate_prediction_shape(
            predicted_actuator_positions_m,
            component_count,
            actuator_evidence.positions_m.shape,
            "predicted_actuator_positions_m",
        )
        log_likelihood, valid_count = _student_t_component_log_likelihood(
            actuator_evidence.positions_m,
            actuator_evidence.variance_m2,
            actuator_evidence.valid_mask,
            predicted,
            predicted_actuator_variance_m2,
            degrees_of_freedom=settings.degrees_of_freedom,
            effective_sample_cap=settings.actuator_effective_sample_cap,
            minimum_variance=settings.minimum_variance,
            name="actuator",
        )
        informative = bool(
            valid_count > 0
            and settings.actuator_likelihood_power > 0.0
            and np.ptp(log_likelihood) > settings.uninformative_tolerance
        )
        if informative:
            total_log_factor += settings.actuator_likelihood_power * log_likelihood
        factor_summaries.append(
            {
                "kind": "actuator",
                "evidence_id": actuator_evidence.artifact_id,
                "clock_id": actuator_evidence.clock_id,
                "valid_scalar_count": valid_count,
                "likelihood_power": settings.actuator_likelihood_power,
                "informative": informative,
                "log_likelihood_range": float(np.ptp(log_likelihood)),
            }
        )

    if wrench_evidence is not None:
        _validate_evidence_binding(factual, wrench_evidence, name="wrench")
        predicted = _validate_prediction_shape(
            predicted_contact_wrench,
            component_count,
            wrench_evidence.wrench.shape,
            "predicted_contact_wrench",
        )
        log_likelihood, valid_count = _student_t_component_log_likelihood(
            wrench_evidence.wrench,
            wrench_evidence.variance,
            wrench_evidence.valid_mask,
            predicted,
            predicted_wrench_variance,
            degrees_of_freedom=settings.degrees_of_freedom,
            effective_sample_cap=settings.wrench_effective_sample_cap,
            minimum_variance=settings.minimum_variance,
            name="wrench",
        )
        informative = bool(
            valid_count > 0
            and settings.wrench_likelihood_power > 0.0
            and np.ptp(log_likelihood) > settings.uninformative_tolerance
        )
        if informative:
            total_log_factor += settings.wrench_likelihood_power * log_likelihood
        factor_summaries.append(
            {
                "kind": "contact_wrench",
                "evidence_id": wrench_evidence.artifact_id,
                "clock_id": wrench_evidence.clock_id,
                "quantity_names": list(wrench_evidence.quantity_names),
                "valid_scalar_count": valid_count,
                "likelihood_power": settings.wrench_likelihood_power,
                "informative": informative,
                "log_likelihood_range": float(np.ptp(log_likelihood)),
            }
        )

    if not any(summary["informative"] for summary in factor_summaries):
        return factual

    prior = np.asarray(factual.weights, dtype=float)
    posterior = _normalize_log_weights(
        log_weights_from_probabilities(prior, name="factual weights")
        + total_log_factor
    )
    metadata = dict(factual.metadata)
    metadata["independent_sensor_abduction"] = {
        "source_factual_intervention_id": factual.artifact_id,
        "factors": factor_summaries,
        "config": asdict(settings),
        "object_observation_likelihood_reused": False,
        "future_object_frames_read": 0,
        "effective_sample_size_before": _effective_sample_size(prior),
        "effective_sample_size_after": _effective_sample_size(posterior),
    }
    return FactualIntervention(
        context=factual.context,
        component_ids=factual.component_ids,
        phi_names=factual.phi_names,
        kappa_names=factual.kappa_names,
        phi=factual.phi,
        kappa_obs=factual.kappa_obs,
        hypothesis_indices=factual.hypothesis_indices,
        twin_particle_indices=factual.twin_particle_indices,
        weights=posterior,
        evidence_frame_stop=factual.evidence_frame_stop,
        source_twin_belief_id=factual.source_twin_belief_id,
        metadata=metadata,
    )


def predict_affine_actuator_realizations(
    commanded_positions_m: np.ndarray,
    phi_names: tuple[str, ...],
    phi: np.ndarray,
    *,
    rotation_axis: np.ndarray | tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> np.ndarray:
    """Map command trajectories to component-wise actuator realizations."""

    commands = np.asarray(commanded_positions_m, dtype=float)
    values = np.asarray(phi, dtype=float)
    if commands.ndim != 3 or commands.shape[2] != 3:
        raise ValueError(
            "commanded_positions_m must have shape (frame, controller, 3)"
        )
    if not np.all(np.isfinite(commands)):
        raise ValueError("commanded_positions_m must be finite")
    if values.ndim != 2 or values.shape[1] != len(phi_names):
        raise ValueError("phi must have shape (component, len(phi_names))")
    if not np.all(np.isfinite(values)):
        raise ValueError("phi must be finite")
    required = {"gain_multiplier", "delay_steps", "rotation_degrees"}
    if not required.issubset(phi_names):
        raise ValueError(
            "phi_names must include gain_multiplier, delay_steps, and "
            "rotation_degrees"
        )
    axis = np.asarray(rotation_axis, dtype=float).reshape(-1)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("rotation_axis must be a finite three-vector")
    norm = float(np.linalg.norm(axis))
    if norm <= np.finfo(float).eps:
        raise ValueError("rotation_axis must be nonzero")
    axis = axis / norm
    gain_index = phi_names.index("gain_multiplier")
    delay_index = phi_names.index("delay_steps")
    rotation_index = phi_names.index("rotation_degrees")
    result = np.empty((len(values),) + commands.shape, dtype=float)
    anchor = commands[0]
    frame_indices = np.arange(len(commands))
    for component, row in enumerate(values):
        delay_value = float(row[delay_index])
        delay = int(np.rint(delay_value))
        if delay < 0 or not np.isclose(delay_value, delay, atol=1.0e-8):
            raise ValueError("delay_steps must contain nonnegative integers")
        source = np.maximum(frame_indices - delay, 0)
        displacement = commands[source] - anchor
        angle = np.deg2rad(float(row[rotation_index]))
        cross = np.cross(axis, displacement)
        projection = np.sum(displacement * axis, axis=-1, keepdims=True) * axis
        rotated = (
            displacement * np.cos(angle)
            + cross * np.sin(angle)
            + projection * (1.0 - np.cos(angle))
        )
        result[component] = anchor + float(row[gain_index]) * rotated
    return result
