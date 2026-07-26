"""Stable contact-regime-conditioned graph-discrepancy mean transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal4d.action_conditioned_discrepancy import (
    ActionConditionedDiscrepancyFeatures,
    ActionConditionedDiscrepancyModel,
)
from causal4d.contracts import array_sha256
from causal4d.discrepancy_belief import GraphDiscrepancyBelief
from causal4d.dynamic_contact import CONTACT_REGIME_NAMES


def _readonly(values: np.ndarray, *, dtype: type | None = float) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


@dataclass(frozen=True)
class RegimeConditionedDiscrepancyTransitionModel:
    """Stable feature-activated transition toward one operator per regime.

    The transition is a convex combination of identity and a declared
    non-expansive target operator. Therefore it cannot increase the Euclidean
    coefficient norm. Zero base rates and zero feature weights recover identity
    exactly.
    """

    feature_names: tuple[str, ...]
    target_matrices: np.ndarray
    base_activation_rates: np.ndarray
    feature_weights: np.ndarray
    regime_names: tuple[str, ...] = CONTACT_REGIME_NAMES
    model_id: str = "regime-conditioned-graph-transition-v1"

    def __post_init__(self) -> None:
        feature_names = tuple(map(str, self.feature_names))
        regime_names = tuple(map(str, self.regime_names))
        targets = _readonly(self.target_matrices)
        base_rates = _readonly(self.base_activation_rates)
        weights = _readonly(self.feature_weights)
        if not self.model_id or not feature_names:
            raise ValueError("model_id and feature_names must be nonempty")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names must be unique")
        if not regime_names or len(set(regime_names)) != len(regime_names):
            raise ValueError("regime_names must be nonempty and unique")
        if targets.ndim != 3 or targets.shape[0] != len(regime_names):
            raise ValueError(
                "target_matrices must have shape (regime_count, rank, rank)"
            )
        if targets.shape[1] < 1 or targets.shape[1] != targets.shape[2]:
            raise ValueError("target_matrices must be square with positive rank")
        if base_rates.shape != (len(regime_names),):
            raise ValueError("base_activation_rates must match regime_names")
        if weights.shape != (len(regime_names), len(feature_names)):
            raise ValueError("feature_weights must have shape (regime_count, F)")
        if not all(
            np.all(np.isfinite(value)) for value in (targets, base_rates, weights)
        ):
            raise ValueError("transition arrays must be finite")
        if np.any(base_rates < 0.0):
            raise ValueError("base activation rates must be nonnegative")
        spectral_norms = np.linalg.svd(targets, compute_uv=False)[:, 0]
        if np.any(spectral_norms > 1.0 + 1e-10):
            raise ValueError("target transition matrices must be non-expansive")
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "regime_names", regime_names)
        object.__setattr__(self, "target_matrices", targets)
        object.__setattr__(self, "base_activation_rates", base_rates)
        object.__setattr__(self, "feature_weights", weights)

    @property
    def rank(self) -> int:
        return int(self.target_matrices.shape[1])

    def transition_matrix(
        self,
        regime: int,
        feature_vector: np.ndarray,
    ) -> np.ndarray:
        """Return a stable transition, with exact identity at zero activation."""

        regime_index = int(regime)
        if not 0 <= regime_index < len(self.regime_names):
            raise ValueError("regime index is outside the declared support")
        features = np.asarray(feature_vector, dtype=float)
        if features.shape != (len(self.feature_names),) or not np.all(
            np.isfinite(features)
        ):
            raise ValueError("feature vector does not match the transition model")
        projected = float(self.feature_weights[regime_index] @ features)
        rate = self.base_activation_rates[regime_index] + projected * projected
        if not np.isfinite(rate):
            raise ValueError("feature activation produced a non-finite rate")
        identity = np.eye(self.rank)
        if rate == 0.0:
            return identity
        activation = -np.expm1(-rate)
        return (
            (1.0 - activation) * identity
            + activation * self.target_matrices[regime_index]
        )


@dataclass(frozen=True)
class RegimeConditionedDiscrepancyForecast:
    """Coefficient and graph-readout moments for a declared contact path."""

    coefficient_mean_m: np.ndarray
    coefficient_covariance_m2: np.ndarray
    readout_mean_m: np.ndarray
    readout_variance_m2: np.ndarray
    transition_matrices: np.ndarray
    regime_paths: np.ndarray
    transition_model_id: str
    innovation_model_id: str | None


def _component_features(
    features: ActionConditionedDiscrepancyFeatures,
    component_ids: tuple[str, ...],
) -> np.ndarray:
    component_count = len(component_ids)
    values = np.asarray(features.values, dtype=float)
    if values.ndim == 2:
        return np.broadcast_to(
            values[None],
            (component_count, *values.shape),
        )
    if values.ndim != 3:
        raise ValueError("feature values must have shape (H, F) or (K, H, F)")
    if features.component_ids != component_ids:
        raise ValueError("component-specific feature IDs differ from the belief")
    return values


def _component_regime_paths(
    regime_paths: np.ndarray,
    *,
    component_count: int,
    horizon: int,
    regime_count: int,
) -> np.ndarray:
    supplied = np.asarray(regime_paths)
    if supplied.ndim == 1:
        if supplied.shape != (horizon,):
            raise ValueError("shared regime path must have shape (H,)")
        supplied = np.broadcast_to(supplied[None], (component_count, horizon))
    elif supplied.shape != (component_count, horizon):
        raise ValueError("regime_paths must have shape (H,) or (K, H)")
    if not np.issubdtype(supplied.dtype, np.integer):
        if not np.all(np.equal(supplied, np.floor(supplied))):
            raise ValueError("regime paths must contain integer indices")
    paths = np.asarray(supplied, dtype=np.int64)
    if np.any(paths < 0) or np.any(paths >= regime_count):
        raise ValueError("regime paths contain an unknown regime")
    return paths


def forecast_regime_conditioned_discrepancy(
    belief: GraphDiscrepancyBelief,
    transition_model: RegimeConditionedDiscrepancyTransitionModel,
    features: ActionConditionedDiscrepancyFeatures,
    regime_paths: np.ndarray,
    basis: np.ndarray,
    *,
    innovation_model: ActionConditionedDiscrepancyModel | None = None,
) -> RegimeConditionedDiscrepancyForecast:
    """Propagate discrepancy moments under stable contact-conditioned dynamics."""

    graph_basis = np.asarray(basis, dtype=float)
    if graph_basis.ndim != 2 or graph_basis.shape[1] != belief.rank:
        raise ValueError("basis must have shape (node_count, belief.rank)")
    if not np.all(np.isfinite(graph_basis)):
        raise ValueError("basis must be finite")
    if array_sha256(graph_basis) != belief.basis_sha256:
        raise ValueError("basis hash differs from the graph-discrepancy belief")
    if transition_model.rank != belief.rank:
        raise ValueError("transition rank differs from the discrepancy belief")
    if transition_model.feature_names != features.names:
        raise ValueError("transition feature schema differs from supplied features")
    if innovation_model is not None and (
        innovation_model.rank != belief.rank
        or innovation_model.feature_names != features.names
    ):
        raise ValueError("innovation model differs from the belief or features")

    component_ids = tuple(belief.component_ids)
    component_count = len(component_ids)
    horizon = features.horizon
    feature_values = _component_features(features, component_ids)
    paths = _component_regime_paths(
        regime_paths,
        component_count=component_count,
        horizon=horizon,
        regime_count=len(transition_model.regime_names),
    )

    mean = np.empty(
        (component_count, horizon + 1, belief.rank, 3),
        dtype=float,
    )
    covariance = np.empty(
        (component_count, horizon + 1, 3, belief.rank, belief.rank),
        dtype=float,
    )
    transitions = np.empty(
        (component_count, horizon, belief.rank, belief.rank),
        dtype=float,
    )
    mean[:, 0] = belief.coefficient_mean_m
    covariance[:, 0] = belief.coefficient_covariance_m2
    identity = np.eye(belief.rank)
    zero_innovation = np.zeros((belief.rank, belief.rank), dtype=float)

    for component in range(component_count):
        for step in range(horizon):
            feature = feature_values[component, step]
            transition = transition_model.transition_matrix(
                int(paths[component, step]),
                feature,
            )
            transitions[component, step] = transition
            innovation = (
                zero_innovation
                if innovation_model is None
                else innovation_model.innovation_covariance_m2(feature)
            )
            innovation = _positive_semidefinite(innovation)
            if np.array_equal(transition, identity):
                mean[component, step + 1] = mean[component, step]
                if np.array_equal(innovation, zero_innovation):
                    covariance[component, step + 1] = covariance[component, step]
                else:
                    for coordinate in range(3):
                        covariance[component, step + 1, coordinate] = (
                            _positive_semidefinite(
                                covariance[component, step, coordinate] + innovation
                            )
                        )
                continue
            for coordinate in range(3):
                mean[component, step + 1, :, coordinate] = (
                    transition @ mean[component, step, :, coordinate]
                )
                covariance[component, step + 1, coordinate] = (
                    _positive_semidefinite(
                        transition
                        @ covariance[component, step, coordinate]
                        @ transition.T
                        + innovation
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
    return RegimeConditionedDiscrepancyForecast(
        coefficient_mean_m=mean,
        coefficient_covariance_m2=covariance,
        readout_mean_m=readout_mean,
        readout_variance_m2=np.maximum(readout_variance, 0.0),
        transition_matrices=transitions,
        regime_paths=paths,
        transition_model_id=transition_model.model_id,
        innovation_model_id=(
            None if innovation_model is None else innovation_model.model_id
        ),
    )
