"""Action-conditioned graph discrepancy dynamics around persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from causal4d.graph_temporal_discrepancy import project_graph_coefficients


@dataclass(frozen=True)
class ActionConditionedGraphDiscrepancyModel:
    """Low-rank discrepancy increments driven by measured causal features.

    The model retains graph persistence as its null and adds coordinate-specific
    low-rank increments,

    ``c[t+1] = c[t] + d + B psi[t] + w[t]``.
    """

    basis: np.ndarray
    drift: np.ndarray
    input_matrix: np.ndarray
    innovation_covariance: np.ndarray
    projection_variance_m2: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_names: tuple[str, ...]
    fit_frame_count: int
    projection_ridge: float
    dynamics_ridge: float

    def __post_init__(self) -> None:
        basis = np.asarray(self.basis, dtype=float)
        drift = np.asarray(self.drift, dtype=float)
        input_matrix = np.asarray(self.input_matrix, dtype=float)
        innovation = np.asarray(self.innovation_covariance, dtype=float)
        projection = np.asarray(self.projection_variance_m2, dtype=float)
        feature_mean = np.asarray(self.feature_mean, dtype=float)
        feature_scale = np.asarray(self.feature_scale, dtype=float)
        if basis.ndim != 2:
            raise ValueError("basis must have shape (node, rank)")
        rank = basis.shape[1]
        if drift.shape != (rank, 3):
            raise ValueError("drift must have shape (rank, 3)")
        if input_matrix.ndim != 3 or input_matrix.shape[:2] != (rank, 3):
            raise ValueError("input_matrix must have shape (rank, 3, feature)")
        feature_count = input_matrix.shape[2]
        if innovation.shape != (3, rank, rank):
            raise ValueError("innovation_covariance must have shape (3, rank, rank)")
        if projection.shape != (3,):
            raise ValueError("projection_variance_m2 must have three coordinates")
        if feature_mean.shape != (feature_count,) or feature_scale.shape != (
            feature_count,
        ):
            raise ValueError("feature normalization must match input_matrix")
        if len(self.feature_names) != feature_count:
            raise ValueError("feature_names must match input_matrix")
        if np.any(feature_scale <= 0.0):
            raise ValueError("feature_scale must be positive")
        if np.any(projection < 0.0):
            raise ValueError("projection variance must be nonnegative")
        for coordinate in range(3):
            if np.min(np.linalg.eigvalsh(innovation[coordinate])) < -1e-10:
                raise ValueError("innovation covariance must be nonnegative")
        if not all(
            np.all(np.isfinite(value))
            for value in (
                basis,
                drift,
                input_matrix,
                innovation,
                projection,
                feature_mean,
                feature_scale,
            )
        ):
            raise ValueError("action-conditioned model arrays must be finite")
        if self.fit_frame_count < 3:
            raise ValueError("fit_frame_count must be at least three")
        if self.projection_ridge <= 0.0 or self.dynamics_ridge <= 0.0:
            raise ValueError("ridge parameters must be positive")
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "drift", drift)
        object.__setattr__(self, "input_matrix", input_matrix)
        object.__setattr__(self, "innovation_covariance", innovation)
        object.__setattr__(self, "projection_variance_m2", projection)
        object.__setattr__(self, "feature_mean", feature_mean)
        object.__setattr__(self, "feature_scale", feature_scale)


def fit_action_conditioned_graph_discrepancy(
    residual_m: np.ndarray,
    valid: np.ndarray,
    basis: np.ndarray,
    action_features: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    projection_ridge: float = 1e-5,
    dynamics_ridge: float = 1e-4,
) -> ActionConditionedGraphDiscrepancyModel:
    """Fit ``c[t+1] = c[t] + d + B psi[t] + w[t]`` on source-only data."""

    residual = np.asarray(residual_m, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    modes = np.asarray(basis, dtype=float)
    features = np.asarray(action_features, dtype=float)
    if residual.ndim != 3 or residual.shape[2] != 3 or len(residual) < 3:
        raise ValueError("residual_m must have shape (T>=3, node, 3)")
    if mask.shape != residual.shape[:2]:
        raise ValueError("valid must have shape (T, node)")
    if features.ndim != 2 or features.shape[0] != len(residual) - 1:
        raise ValueError("action_features must have shape (T-1, feature)")
    if modes.ndim != 2 or modes.shape[0] < residual.shape[1]:
        raise ValueError("basis does not cover observed residual nodes")
    if not np.all(np.isfinite(features)):
        raise ValueError("action_features must be finite")
    if projection_ridge <= 0.0 or dynamics_ridge <= 0.0:
        raise ValueError("ridge parameters must be positive")
    names = (
        tuple(f"feature_{index}" for index in range(features.shape[1]))
        if feature_names is None
        else tuple(feature_names)
    )
    if len(names) != features.shape[1] or len(set(names)) != len(names):
        raise ValueError("feature_names must be unique and match action_features")
    coefficients = project_graph_coefficients(
        residual,
        mask,
        modes,
        ridge=projection_ridge,
    )
    feature_mean = np.mean(features, axis=0)
    feature_scale = np.std(features, axis=0)
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    normalized = (features - feature_mean) / feature_scale
    design = np.vstack((np.ones((1, len(normalized))), normalized.T))
    penalty = np.eye(features.shape[1] + 1) * dynamics_ridge
    penalty[0, 0] = 0.0
    inverse_precision = np.linalg.inv(design @ design.T + penalty)
    deltas = np.diff(coefficients, axis=0)
    drift = np.zeros((modes.shape[1], 3), dtype=float)
    input_matrix = np.zeros(
        (modes.shape[1], 3, features.shape[1]),
        dtype=float,
    )
    innovation_covariance = np.zeros((3, modes.shape[1], modes.shape[1]))
    for coordinate in range(3):
        target = deltas[:, :, coordinate].T
        coefficient_map = target @ design.T @ inverse_precision
        drift[:, coordinate] = coefficient_map[:, 0]
        input_matrix[:, coordinate] = coefficient_map[:, 1:]
        innovation = target - coefficient_map @ design
        covariance = innovation @ innovation.T / innovation.shape[1]
        innovation_covariance[coordinate] = (
            0.5 * (covariance + covariance.T)
            + 1e-12 * np.eye(modes.shape[1])
        )
    reconstructed = np.einsum(
        "nr,trc->tnc",
        modes[: residual.shape[1]],
        coefficients,
    )
    projection_error = reconstructed - residual
    projection_variance = np.asarray(
        [
            np.mean(np.square(projection_error[:, :, coordinate][mask]))
            for coordinate in range(3)
        ]
    )
    return ActionConditionedGraphDiscrepancyModel(
        basis=modes,
        drift=drift,
        input_matrix=input_matrix,
        innovation_covariance=innovation_covariance,
        projection_variance_m2=projection_variance,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        feature_names=names,
        fit_frame_count=len(residual),
        projection_ridge=projection_ridge,
        dynamics_ridge=dynamics_ridge,
    )


def forecast_action_conditioned_graph_discrepancy(
    model: ActionConditionedGraphDiscrepancyModel,
    prefix_residual_m: np.ndarray,
    prefix_valid: np.ndarray,
    action_features: np.ndarray,
    *,
    total_frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast discrepancy with source-fitted action-conditioned increments."""

    prefix = np.asarray(prefix_residual_m, dtype=float)
    valid = np.asarray(prefix_valid, dtype=bool)
    features = np.asarray(action_features, dtype=float)
    if not 2 <= len(prefix) < total_frame_count:
        raise ValueError("prefix must reveal evidence and leave future frames")
    if prefix.ndim != 3 or prefix.shape[2] != 3:
        raise ValueError("prefix_residual_m must have shape (T, node, 3)")
    if valid.shape != prefix.shape[:2]:
        raise ValueError("prefix_valid must have shape (T, node)")
    if features.shape != (total_frame_count - 1, len(model.feature_names)):
        raise ValueError("action_features must cover every rollout transition")
    if not np.all(np.isfinite(features)):
        raise ValueError("action_features must be finite")
    coefficients = project_graph_coefficients(
        prefix,
        valid,
        model.basis,
        ridge=model.projection_ridge,
    )
    mean = np.zeros((total_frame_count, model.basis.shape[0], 3), dtype=float)
    variance = np.zeros_like(mean)
    mean[: len(prefix)] = np.einsum(
        "nr,trc->tnc",
        model.basis,
        coefficients,
    )
    variance[: len(prefix)] = model.projection_variance_m2[None, None]
    current = coefficients[-1].copy()
    covariance = np.zeros_like(model.innovation_covariance)
    normalized = (features - model.feature_mean) / model.feature_scale
    for frame in range(len(prefix), total_frame_count):
        increment = model.drift + np.einsum(
            "rcf,f->rc",
            model.input_matrix,
            normalized[frame - 1],
        )
        current = current + increment
        covariance = covariance + model.innovation_covariance
        mean[frame] = model.basis @ current
        for coordinate in range(3):
            marginal = np.einsum(
                "ni,ij,nj->n",
                model.basis,
                covariance[coordinate],
                model.basis,
            )
            variance[frame, :, coordinate] = (
                marginal + model.projection_variance_m2[coordinate]
            )
    return mean, variance
