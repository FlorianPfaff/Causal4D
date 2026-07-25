"""Action-conditioned covariance growth for graph-persistent discrepancy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from causal4d.contracts import array_sha256
from causal4d.discrepancy_belief import GraphDiscrepancyBelief


CONTACT_POLICIES = ("same_grasp", "new_contact")


def _readonly(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).copy()
    array.setflags(write=False)
    return array


def _parameter_vector(
    names: Sequence[str],
    values: np.ndarray | Sequence[float],
    *,
    label: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    declared_names = tuple(map(str, names))
    supplied_values = np.asarray(values, dtype=float)
    if supplied_values.shape != (len(declared_names),):
        raise ValueError(f"{label} values must match {label}_names")
    if not np.all(np.isfinite(supplied_values)):
        raise ValueError(f"{label} values must be finite")
    if len(set(declared_names)) != len(declared_names):
        raise ValueError(f"{label}_names must be unique")
    return declared_names, supplied_values


def _named_value(
    names: tuple[str, ...],
    values: np.ndarray,
    name: str,
    default: float,
) -> float:
    try:
        index = names.index(name)
    except ValueError:
        return float(default)
    return float(values[index])


def _positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


@dataclass(frozen=True)
class ActionConditionedDiscrepancyFeatures:
    """Shared or component-specific features for each forecast transition."""

    names: tuple[str, ...]
    values: np.ndarray
    component_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        names = tuple(map(str, self.names))
        values = _readonly(self.values)
        if not names or len(set(names)) != len(names):
            raise ValueError("feature names must be nonempty and unique")
        if values.ndim not in {2, 3} or values.shape[-1] != len(names):
            raise ValueError("feature values must have shape (H, F) or (K, H, F)")
        if values.shape[-2] < 1 or not np.all(np.isfinite(values)):
            raise ValueError("feature values must contain a finite forecast horizon")
        if values.ndim == 3:
            if self.component_ids is None or len(self.component_ids) != values.shape[0]:
                raise ValueError(
                    "component-specific features require matching component_ids"
                )
            if len(set(self.component_ids)) != len(self.component_ids):
                raise ValueError("feature component_ids must be unique")
        elif self.component_ids is not None:
            raise ValueError("shared features must not declare component_ids")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "values", values)

    @property
    def horizon(self) -> int:
        return int(self.values.shape[-2])


def build_action_conditioned_features(
    controller_points_m: np.ndarray,
    control_anchor_m: np.ndarray,
    *,
    frame_dt_s: float,
    phi_names: Sequence[str] = (),
    phi: np.ndarray | Sequence[float] = (),
    kappa_names: Sequence[str] = (),
    kappa: np.ndarray | Sequence[float] = (),
    contact_policy: Literal["same_grasp", "new_contact"] = "same_grasp",
) -> ActionConditionedDiscrepancyFeatures:
    """Build interpretable transition features without reading future outcomes."""

    controls = np.asarray(controller_points_m, dtype=float)
    anchor = np.asarray(control_anchor_m, dtype=float)
    if controls.ndim != 3 or controls.shape[2] != 3:
        raise ValueError("controller_points_m must have shape (H, C, 3)")
    if controls.shape[0] < 1:
        raise ValueError("controller_points_m must contain a forecast horizon")
    if anchor.shape != controls.shape[1:]:
        raise ValueError("control_anchor_m must have shape (C, 3)")
    if not np.all(np.isfinite(controls)) or not np.all(np.isfinite(anchor)):
        raise ValueError("controller trajectories must be finite")
    if not np.isfinite(frame_dt_s) or frame_dt_s <= 0.0:
        raise ValueError("frame_dt_s must be finite and positive")
    if contact_policy not in CONTACT_POLICIES:
        raise ValueError("contact_policy must be same_grasp or new_contact")

    persistent_names, persistent_values = _parameter_vector(
        phi_names,
        phi,
        label="phi",
    )
    event_names, event_values = _parameter_vector(
        kappa_names,
        kappa,
        label="kappa",
    )

    full = np.concatenate((anchor[None], controls), axis=0)
    displacement = np.diff(full, axis=0)
    velocity = displacement / frame_dt_s
    mean_velocity = np.mean(velocity, axis=1)
    speed = np.mean(np.linalg.norm(velocity, axis=2), axis=1)
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1]) / frame_dt_s
    acceleration_norm = np.mean(np.linalg.norm(acceleration, axis=2), axis=1)
    direction_norm = np.linalg.norm(mean_velocity, axis=1, keepdims=True)
    direction = mean_velocity / np.maximum(direction_norm, np.finfo(float).eps)

    gain = _named_value(
        persistent_names,
        persistent_values,
        "gain_multiplier",
        1.0,
    )
    delay_steps = _named_value(
        persistent_names,
        persistent_values,
        "delay_steps",
        0.0,
    )
    rotation_degrees = _named_value(
        persistent_names,
        persistent_values,
        "rotation_degrees",
        0.0,
    )
    slip = _named_value(event_names, event_values, "slip_fraction", 0.0)
    shifts = [
        float(event_values[index])
        for index, name in enumerate(event_names)
        if name.startswith("attachment_shift_hand_")
    ]
    attachment_shift_rms = (
        float(np.sqrt(np.mean(np.square(shifts)))) if shifts else 0.0
    )
    horizon = len(controls)

    def repeated(value: float) -> np.ndarray:
        return np.full(horizon, float(value), dtype=float)

    values = np.column_stack(
        (
            speed,
            acceleration_norm,
            direction,
            repeated(abs(gain - 1.0)),
            repeated(abs(delay_steps) * frame_dt_s),
            repeated(abs(np.deg2rad(rotation_degrees))),
            repeated(max(slip, 0.0)),
            repeated(attachment_shift_rms),
            repeated(float(contact_policy == "new_contact")),
        )
    )
    names = (
        "control_speed_mps",
        "control_acceleration_mps2",
        "direction_x",
        "direction_y",
        "direction_z",
        "gain_abs_deviation",
        "delay_s",
        "rotation_abs_rad",
        "slip_fraction",
        "attachment_shift_rms",
        "new_contact",
    )
    return ActionConditionedDiscrepancyFeatures(names=names, values=values)


@dataclass(frozen=True)
class ActionConditionedDiscrepancyModel:
    """Positive-semidefinite feature-conditioned innovation covariance.

    Each feature direction contributes ``(w_j^T f)^2 v_j v_j^T``. Zero feature
    weights reproduce the base covariance exactly. ``maximum_increment_trace_m2``
    caps only the action-dependent addition, never the declared base uncertainty.
    """

    feature_names: tuple[str, ...]
    base_innovation_covariance_m2: np.ndarray
    feature_directions: np.ndarray
    feature_weights: np.ndarray
    model_id: str = "action-conditioned-graph-persistence-v1"
    maximum_increment_trace_m2: float | None = None

    def __post_init__(self) -> None:
        feature_names = tuple(map(str, self.feature_names))
        base = _readonly(self.base_innovation_covariance_m2)
        directions = _readonly(self.feature_directions)
        weights = _readonly(self.feature_weights)
        if not self.model_id or not feature_names:
            raise ValueError("model id and feature names must be nonempty")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("feature names must be unique")
        if base.ndim != 2 or base.shape[0] != base.shape[1] or base.shape[0] < 1:
            raise ValueError("base covariance must have shape (rank, rank)")
        rank = base.shape[0]
        if directions.ndim != 2 or directions.shape[1] != rank:
            raise ValueError("feature directions must have shape (J, rank)")
        if weights.shape != (directions.shape[0], len(feature_names)):
            raise ValueError("feature weights must have shape (J, feature_count)")
        if not all(np.all(np.isfinite(value)) for value in (base, directions, weights)):
            raise ValueError("action-conditioned model arrays must be finite")
        if not np.allclose(base, base.T, atol=1e-10, rtol=1e-10):
            raise ValueError("base covariance must be symmetric")
        if float(np.min(np.linalg.eigvalsh(base), initial=0.0)) < -1e-10:
            raise ValueError("base covariance must be positive semidefinite")
        base = _readonly(_positive_semidefinite(base))
        if len(directions):
            norms = np.linalg.norm(directions, axis=1)
            if np.any(norms <= 0.0):
                raise ValueError("feature directions must be nonzero")
            directions = _readonly(directions / norms[:, None])
        if self.maximum_increment_trace_m2 is not None and (
            not np.isfinite(self.maximum_increment_trace_m2)
            or self.maximum_increment_trace_m2 <= 0.0
        ):
            raise ValueError("maximum_increment_trace_m2 must be positive")
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "base_innovation_covariance_m2", base)
        object.__setattr__(self, "feature_directions", directions)
        object.__setattr__(self, "feature_weights", weights)

    @property
    def rank(self) -> int:
        return int(self.base_innovation_covariance_m2.shape[0])

    def innovation_covariance_m2(self, feature_vector: np.ndarray) -> np.ndarray:
        """Return the base covariance plus a capped PSD action increment."""

        features = np.asarray(feature_vector, dtype=float)
        if features.shape != (len(self.feature_names),) or not np.all(
            np.isfinite(features)
        ):
            raise ValueError("feature vector does not match the model")
        increment = np.zeros_like(self.base_innovation_covariance_m2)
        if len(self.feature_directions):
            rates = np.square(self.feature_weights @ features)
            increment = np.einsum(
                "j,ji,jk->ik",
                rates,
                self.feature_directions,
                self.feature_directions,
            )
            increment = _positive_semidefinite(increment)
        if self.maximum_increment_trace_m2 is not None:
            trace = float(np.trace(increment))
            if trace > self.maximum_increment_trace_m2:
                increment *= self.maximum_increment_trace_m2 / trace
        return _positive_semidefinite(
            self.base_innovation_covariance_m2 + increment
        )


@dataclass(frozen=True)
class ActionConditionedDiscrepancyForecast:
    """Coefficient and graph-readout moments including the prefix endpoint."""

    coefficient_mean_m: np.ndarray
    coefficient_covariance_m2: np.ndarray
    readout_mean_m: np.ndarray
    readout_variance_m2: np.ndarray
    model_id: str


def forecast_action_conditioned_persistence(
    belief: GraphDiscrepancyBelief,
    model: ActionConditionedDiscrepancyModel,
    features: ActionConditionedDiscrepancyFeatures,
    basis: np.ndarray,
) -> ActionConditionedDiscrepancyForecast:
    """Persist the discrepancy mean while growing covariance by action regime."""

    graph_basis = np.asarray(basis, dtype=float)
    if graph_basis.ndim != 2 or graph_basis.shape[1] != belief.rank:
        raise ValueError("basis must have shape (node, belief.rank)")
    if not np.all(np.isfinite(graph_basis)):
        raise ValueError("basis must be finite")
    if array_sha256(graph_basis) != belief.basis_sha256:
        raise ValueError("basis hash differs from the graph-discrepancy belief")
    if model.rank != belief.rank or model.feature_names != features.names:
        raise ValueError(
            "model rank or feature schema differs from the belief/features"
        )

    component_count = len(belief.component_ids)
    if features.values.ndim == 2:
        feature_values = np.broadcast_to(
            features.values[None],
            (component_count, *features.values.shape),
        )
    else:
        if features.component_ids != belief.component_ids:
            raise ValueError("component-specific features differ from belief support")
        feature_values = features.values
    horizon = features.horizon
    mean = np.broadcast_to(
        belief.coefficient_mean_m[:, None],
        (component_count, horizon + 1, belief.rank, 3),
    ).copy()
    covariance = np.empty(
        (component_count, horizon + 1, 3, belief.rank, belief.rank),
        dtype=float,
    )
    covariance[:, 0] = belief.coefficient_covariance_m2
    for step in range(horizon):
        for component in range(component_count):
            increment = model.innovation_covariance_m2(
                feature_values[component, step]
            )
            for coordinate in range(3):
                covariance[component, step + 1, coordinate] = (
                    _positive_semidefinite(
                        covariance[component, step, coordinate] + increment
                    )
                )

    readout_mean = np.einsum("nr,khrc->khnc", graph_basis, mean)
    readout_variance = np.empty_like(readout_mean)
    for component in range(component_count):
        for step in range(horizon + 1):
            for coordinate in range(3):
                readout_variance[component, step, :, coordinate] = (
                    np.einsum(
                        "ni,ij,nj->n",
                        graph_basis,
                        covariance[component, step, coordinate],
                        graph_basis,
                    )
                    + belief.projection_variance_m2[coordinate]
                )
    return ActionConditionedDiscrepancyForecast(
        coefficient_mean_m=mean,
        coefficient_covariance_m2=covariance,
        readout_mean_m=readout_mean,
        readout_variance_m2=np.maximum(readout_variance, 0.0),
        model_id=model.model_id,
    )
