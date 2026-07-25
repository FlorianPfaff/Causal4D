"""Grouped robust observation factors for finite physical rollout banks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from causal4d.rollout_bank import JointRolloutBank


@dataclass(frozen=True)
class ObservationGroup:
    """One effective, possibly correlated observation likelihood factor.

    A group observes one rollout node at one frame. When ``reference_frame_index``
    is provided, ``values_m`` is interpreted as a displacement from that reference
    frame. This keeps position and increment evidence explicit instead of silently
    counting finite differences as independent position measurements.
    """

    frame_index: int
    node_index: int
    values_m: np.ndarray
    covariance_m2: np.ndarray
    nominal_probability: float = 0.95
    outlier_scale_multiplier: float = 25.0
    composite_weight: float = 1.0
    reference_frame_index: int | None = None
    source_id: str = "physical_observation"
    view_id: str | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values_m, dtype=float)
        covariance = np.asarray(self.covariance_m2, dtype=float)
        if values.ndim != 1 or not len(values):
            raise ValueError("values_m must be a nonempty coordinate vector")
        if covariance.shape != (len(values), len(values)):
            raise ValueError("covariance_m2 must match values_m")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("covariance_m2 must be finite")
        covariance = 0.5 * (covariance + covariance.T)
        if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
            raise ValueError("covariance_m2 must be positive definite")
        if self.frame_index < 0 or self.node_index < 0:
            raise ValueError("frame and node indices must be nonnegative")
        if self.reference_frame_index is not None:
            if self.reference_frame_index < 0:
                raise ValueError("reference_frame_index must be nonnegative")
            if self.reference_frame_index == self.frame_index:
                raise ValueError("reference and observed frames must differ")
        if not 0.0 <= self.nominal_probability <= 1.0:
            raise ValueError("nominal_probability must lie in [0, 1]")
        if self.outlier_scale_multiplier <= 1.0:
            raise ValueError("outlier_scale_multiplier must exceed one")
        if self.composite_weight <= 0.0 or not np.isfinite(self.composite_weight):
            raise ValueError("composite_weight must be finite and positive")
        if not self.source_id:
            raise ValueError("source_id must be nonempty")
        object.__setattr__(self, "values_m", values)
        object.__setattr__(self, "covariance_m2", covariance)


def _normalize_joint_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=float)
    if values.ndim != 2 or not np.any(np.isfinite(values)):
        raise ValueError("joint log weights must contain finite support")
    maximum = float(np.max(values[np.isfinite(values)]))
    weights = np.exp(np.where(np.isfinite(values), values - maximum, -np.inf))
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("joint posterior normalization failed")
    return weights / total


def _base_log_weights(
    bank: "JointRolloutBank", base_weights: np.ndarray | None
) -> np.ndarray:
    weights = (
        bank.prior_joint_weights
        if base_weights is None
        else np.asarray(base_weights, dtype=float)
    )
    if weights.shape != bank.prior_joint_weights.shape:
        raise ValueError("base_weights must match the joint rollout support")
    if np.any(weights < 0.0) or not np.isclose(np.sum(weights), 1.0):
        raise ValueError("base_weights must be nonnegative and sum to one")
    return np.log(np.maximum(weights, 1e-300))


def _particle_field_at_frame(
    values: np.ndarray | None,
    *,
    particle_count: int,
    frame_count: int,
    node_count: int,
    coordinate_count: int,
    frame_index: int,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    field = np.asarray(values, dtype=float)
    static_shape = (particle_count, node_count, coordinate_count)
    temporal_shape = (particle_count, frame_count, node_count, coordinate_count)
    if field.shape == static_shape:
        selected = field
    elif field.shape == temporal_shape:
        selected = field[:, frame_index]
    else:
        raise ValueError(f"{name} must have shape {static_shape} or {temporal_shape}")
    if not np.all(np.isfinite(selected)):
        raise ValueError(f"{name} must be finite")
    return selected


def _multivariate_student_t_logpdf(
    residual: np.ndarray,
    covariance_m2: np.ndarray,
    *,
    degrees_of_freedom: float,
) -> np.ndarray:
    """Return normalized multivariate Student-t log densities.

    ``covariance_m2`` is the desired covariance, not the Student-t scale matrix.
    Therefore ``degrees_of_freedom`` must exceed two.
    """

    if degrees_of_freedom <= 2.0:
        raise ValueError("grouped Student-t degrees_of_freedom must exceed two")
    vectors = np.asarray(residual, dtype=float)
    covariance = np.asarray(covariance_m2, dtype=float)
    dimension = vectors.shape[-1]
    if covariance.shape[-2:] != (dimension, dimension):
        raise ValueError("covariance does not match residual dimension")
    scale = covariance * ((degrees_of_freedom - 2.0) / degrees_of_freedom)
    sign, logdet = np.linalg.slogdet(scale)
    if np.any(sign <= 0.0):
        raise ValueError("Student-t scale must be positive definite")
    solved = np.linalg.solve(scale, vectors[..., None])[..., 0]
    mahalanobis = np.sum(vectors * solved, axis=-1)
    from math import lgamma, log, pi

    constant = (
        lgamma((degrees_of_freedom + dimension) / 2.0)
        - lgamma(degrees_of_freedom / 2.0)
        - 0.5 * dimension * log(degrees_of_freedom * pi)
    )
    return (
        constant
        - 0.5 * logdet
        - 0.5
        * (degrees_of_freedom + dimension)
        * np.log1p(mahalanobis / degrees_of_freedom)
    )


def _mixture_log_score(
    residual: np.ndarray,
    covariance_m2: np.ndarray,
    *,
    degrees_of_freedom: float,
    nominal_probability: float,
    outlier_scale_multiplier: float,
) -> np.ndarray:
    nominal = _multivariate_student_t_logpdf(
        residual,
        covariance_m2,
        degrees_of_freedom=degrees_of_freedom,
    )
    if nominal_probability == 1.0:
        return nominal
    outlier = _multivariate_student_t_logpdf(
        residual,
        covariance_m2 * outlier_scale_multiplier,
        degrees_of_freedom=degrees_of_freedom,
    )
    if nominal_probability == 0.0:
        return outlier
    return np.logaddexp(
        np.log(nominal_probability) + nominal,
        np.log1p(-nominal_probability) + outlier,
    )


def update_from_grouped_observations(
    bank: "JointRolloutBank",
    groups: Sequence[ObservationGroup],
    *,
    degrees_of_freedom: float = 4.0,
    base_weights: np.ndarray | None = None,
    particle_discrepancy_m: np.ndarray | None = None,
    particle_discrepancy_variance_m2: np.ndarray | None = None,
) -> np.ndarray:
    """Update a rollout bank with robust effective observation groups."""

    if degrees_of_freedom <= 2.0:
        raise ValueError("degrees_of_freedom must exceed two for covariance semantics")
    factors = tuple(groups)
    if not factors:
        raise ValueError("at least one observation group is required")
    particle_count = len(bank.parameter_weights)
    log_weights = _base_log_weights(bank, base_weights)
    for group in factors:
        if group.frame_index >= bank.frame_count or group.node_index >= bank.node_count:
            raise ValueError("observation group exceeds rollout support")
        if len(group.values_m) != bank.coordinate_count:
            raise ValueError(
                "observation group coordinate count differs from rollout bank"
            )
        predicted = bank.trajectories[
            :, :, group.frame_index, group.node_index
        ].astype(float)
        mean_field = _particle_field_at_frame(
            particle_discrepancy_m,
            particle_count=particle_count,
            frame_count=bank.frame_count,
            node_count=bank.node_count,
            coordinate_count=bank.coordinate_count,
            frame_index=group.frame_index,
            name="particle_discrepancy_m",
        )
        if mean_field is not None:
            predicted += mean_field[None, :, group.node_index]
        if group.reference_frame_index is not None:
            if group.reference_frame_index >= bank.frame_count:
                raise ValueError("reference frame exceeds rollout support")
            reference = bank.trajectories[
                :, :, group.reference_frame_index, group.node_index
            ].astype(float)
            reference_field = _particle_field_at_frame(
                particle_discrepancy_m,
                particle_count=particle_count,
                frame_count=bank.frame_count,
                node_count=bank.node_count,
                coordinate_count=bank.coordinate_count,
                frame_index=group.reference_frame_index,
                name="particle_discrepancy_m",
            )
            if reference_field is not None:
                reference += reference_field[None, :, group.node_index]
            predicted -= reference
        valid = np.isfinite(group.values_m)
        if not np.any(valid):
            continue
        residual = predicted[..., valid] - group.values_m[valid]
        covariance = group.covariance_m2[np.ix_(valid, valid)]
        static_variance_shape = (
            particle_count,
            bank.node_count,
            bank.coordinate_count,
        )
        static_increment_variance_cancels = (
            group.reference_frame_index is not None
            and particle_discrepancy_variance_m2 is not None
            and np.asarray(particle_discrepancy_variance_m2).shape
            == static_variance_shape
        )
        variance_field = (
            None
            if static_increment_variance_cancels
            else _particle_field_at_frame(
                particle_discrepancy_variance_m2,
                particle_count=particle_count,
                frame_count=bank.frame_count,
                node_count=bank.node_count,
                coordinate_count=bank.coordinate_count,
                frame_index=group.frame_index,
                name="particle_discrepancy_variance_m2",
            )
        )
        if variance_field is None:
            covariance_by_particle = np.broadcast_to(
                covariance,
                (particle_count, *covariance.shape),
            ).copy()
        else:
            selected_variance = variance_field[:, group.node_index][:, valid]
            if np.any(selected_variance < 0.0):
                raise ValueError("particle discrepancy variance must be nonnegative")
            covariance_by_particle = np.broadcast_to(
                covariance,
                (particle_count, *covariance.shape),
            ).copy()
            diagonal = np.arange(np.sum(valid))
            covariance_by_particle[:, diagonal, diagonal] += selected_variance
        if (
            group.reference_frame_index is not None
            and not static_increment_variance_cancels
        ):
            reference_variance = _particle_field_at_frame(
                particle_discrepancy_variance_m2,
                particle_count=particle_count,
                frame_count=bank.frame_count,
                node_count=bank.node_count,
                coordinate_count=bank.coordinate_count,
                frame_index=group.reference_frame_index,
                name="particle_discrepancy_variance_m2",
            )
            if reference_variance is not None:
                selected_reference_variance = reference_variance[
                    :, group.node_index
                ][:, valid]
                if np.any(selected_reference_variance < 0.0):
                    raise ValueError(
                        "particle discrepancy variance must be nonnegative"
                    )
                diagonal = np.arange(np.sum(valid))
                covariance_by_particle[
                    :, diagonal, diagonal
                ] += selected_reference_variance
        score = _mixture_log_score(
            residual,
            covariance_by_particle[None],
            degrees_of_freedom=degrees_of_freedom,
            nominal_probability=group.nominal_probability,
            outlier_scale_multiplier=group.outlier_scale_multiplier,
        )
        log_weights += group.composite_weight * score
    return _normalize_joint_log_weights(log_weights)


def dense_prefix_observation_groups(
    observations_m: np.ndarray,
    *,
    prefix_frame_count: int,
    observation_scale_m: float,
    likelihood_power: float,
    mask: np.ndarray | None = None,
    nominal_probability: float = 0.95,
    outlier_scale_multiplier: float = 25.0,
    increment_likelihood_weight: float = 0.0,
    source_id: str = "dense_prefix",
    view_id: str | None = None,
) -> tuple[ObservationGroup, ...]:
    """Convert a dense causal prefix into conservatively weighted 3-D groups."""

    observations = np.asarray(observations_m, dtype=float)
    if observations.ndim != 3 or observations.shape[2] not in {2, 3}:
        raise ValueError("observations_m must have shape (T, N, 2|3)")
    if not 2 <= prefix_frame_count <= len(observations):
        raise ValueError(
            "prefix_frame_count must reveal at least one post-endpoint frame"
        )
    if observation_scale_m <= 0.0 or likelihood_power <= 0.0:
        raise ValueError("observation scale and likelihood power must be positive")
    if increment_likelihood_weight < 0.0:
        raise ValueError("increment_likelihood_weight must be nonnegative")
    valid = np.isfinite(observations)
    if mask is not None:
        supplied = np.asarray(mask, dtype=bool)
        if supplied.shape == observations.shape[:2]:
            supplied = np.repeat(
                supplied[:, :, None], observations.shape[2], axis=2
            )
        if supplied.shape != observations.shape:
            raise ValueError("mask must have shape (T, N) or (T, N, C)")
        valid &= supplied
    positions = []
    for frame in range(1, prefix_frame_count):
        for node in range(observations.shape[1]):
            if np.any(valid[frame, node]):
                values = observations[frame, node].copy()
                values[~valid[frame, node]] = np.nan
                positions.append((frame, node, values))
    if not positions:
        raise ValueError("prefix contains no valid observation groups")
    position_coordinate_count = sum(
        np.sum(np.isfinite(values)) for _, _, values in positions
    )
    position_weight = likelihood_power / float(position_coordinate_count)
    covariance = np.eye(observations.shape[2]) * observation_scale_m**2
    groups = [
        ObservationGroup(
            frame_index=frame,
            node_index=node,
            values_m=values,
            covariance_m2=covariance,
            nominal_probability=nominal_probability,
            outlier_scale_multiplier=outlier_scale_multiplier,
            composite_weight=position_weight,
            source_id=source_id,
            view_id=view_id,
        )
        for frame, node, values in positions
    ]
    if increment_likelihood_weight > 0.0 and prefix_frame_count >= 3:
        increments = []
        for frame in range(2, prefix_frame_count):
            for node in range(observations.shape[1]):
                pair_valid = valid[frame, node] & valid[frame - 1, node]
                if np.any(pair_valid):
                    values = observations[frame, node] - observations[frame - 1, node]
                    values = values.copy()
                    values[~pair_valid] = np.nan
                    increments.append((frame, node, values))
        if increments:
            increment_coordinate_count = sum(
                np.sum(np.isfinite(values)) for _, _, values in increments
            )
            increment_weight = (
                likelihood_power
                * increment_likelihood_weight
                / float(increment_coordinate_count)
            )
            increment_covariance = 2.0 * covariance
            groups.extend(
                ObservationGroup(
                    frame_index=frame,
                    reference_frame_index=frame - 1,
                    node_index=node,
                    values_m=values,
                    covariance_m2=increment_covariance,
                    nominal_probability=nominal_probability,
                    outlier_scale_multiplier=outlier_scale_multiplier,
                    composite_weight=increment_weight,
                    source_id=f"{source_id}:increment",
                    view_id=view_id,
                )
                for frame, node, values in increments
            )
    return tuple(groups)
