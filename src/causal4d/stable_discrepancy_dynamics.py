"""Stable action-conditioned mean dynamics for graph discrepancy beliefs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from causal4d.action_conditioned_discrepancy import (
    ActionConditionedDiscrepancyFeatures,
    ActionConditionedDiscrepancyForecast,
    ActionConditionedDiscrepancyModel,
)
from causal4d.contracts import array_sha256
from causal4d.discrepancy_belief import GraphDiscrepancyBelief


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result.setflags(write=False)
    return result


def _positive_semidefinite(values: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (values + values.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


def _normalized_rows(values: np.ndarray, *, width: int, name: str) -> np.ndarray:
    supplied = np.asarray(values, dtype=float)
    if supplied.ndim != 2 or supplied.shape[1] != width:
        raise ValueError(f"{name} must have shape (J, {width})")
    if not np.all(np.isfinite(supplied)):
        raise ValueError(f"{name} must be finite")
    if not len(supplied):
        return supplied
    norms = np.linalg.norm(supplied, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError(f"{name} rows must be nonzero")
    return supplied / norms[:, None]


@dataclass(frozen=True)
class StableDiscrepancyTransitionModel:
    """Dissipative graph-mode transport with exact identity fallback.

    The generator is ``G(f)=S(f)-C(f)``. ``S`` is skew-symmetric and ``C`` is
    positive semidefinite, so ``A(f)=expm(G(f))`` is non-expansive while allowing
    graph-mode rotation. Empty arrays produce ``A=I`` and zero drift exactly.
    """

    feature_names: tuple[str, ...]
    rank: int
    skew_generators: np.ndarray
    skew_feature_weights: np.ndarray
    contraction_directions: np.ndarray
    contraction_feature_weights: np.ndarray
    drift_directions_m: np.ndarray
    drift_feature_weights: np.ndarray
    model_id: str = "stable-action-conditioned-discrepancy-v1"
    maximum_drift_norm_m: float | None = None

    def __post_init__(self) -> None:
        names = tuple(map(str, self.feature_names))
        if not names or len(set(names)) != len(names):
            raise ValueError("feature_names must be nonempty and unique")
        if self.rank < 1 or not self.model_id:
            raise ValueError("rank and model_id must be valid")
        feature_count = len(names)

        raw_skew = np.asarray(self.skew_generators, dtype=float)
        if raw_skew.ndim != 3 or raw_skew.shape[1:] != (self.rank, self.rank):
            raise ValueError("skew_generators must have shape (J, rank, rank)")
        if not np.all(np.isfinite(raw_skew)):
            raise ValueError("skew_generators must be finite")
        skew = 0.5 * (raw_skew - raw_skew.swapaxes(1, 2))
        if len(skew):
            norms = np.sqrt(np.sum(np.square(skew), axis=(1, 2)))
            if np.any(norms <= 0.0):
                raise ValueError("skew_generators must be nonzero")
            skew = skew / norms[:, None, None]
        skew_weights = np.asarray(self.skew_feature_weights, dtype=float)
        if skew_weights.shape != (len(skew), feature_count):
            raise ValueError(
                "skew_feature_weights must have shape (J, feature_count)"
            )

        contraction = _normalized_rows(
            self.contraction_directions,
            width=self.rank,
            name="contraction_directions",
        )
        contraction_weights = np.asarray(
            self.contraction_feature_weights,
            dtype=float,
        )
        if contraction_weights.shape != (len(contraction), feature_count):
            raise ValueError(
                "contraction_feature_weights must have shape (J, feature_count)"
            )

        drift = np.asarray(self.drift_directions_m, dtype=float)
        if drift.ndim != 3 or drift.shape[1:] != (self.rank, 3):
            raise ValueError("drift_directions_m must have shape (J, rank, 3)")
        if not np.all(np.isfinite(drift)):
            raise ValueError("drift_directions_m must be finite")
        if len(drift):
            norms = np.sqrt(np.sum(np.square(drift), axis=(1, 2)))
            if np.any(norms <= 0.0):
                raise ValueError("drift_directions_m entries must be nonzero")
            drift = drift / norms[:, None, None]
        drift_weights = np.asarray(self.drift_feature_weights, dtype=float)
        if drift_weights.shape != (len(drift), feature_count):
            raise ValueError(
                "drift_feature_weights must have shape (J, feature_count)"
            )
        if not all(
            np.all(np.isfinite(value))
            for value in (skew_weights, contraction_weights, drift_weights)
        ):
            raise ValueError("transition feature weights must be finite")
        if self.maximum_drift_norm_m is not None and (
            not np.isfinite(self.maximum_drift_norm_m)
            or self.maximum_drift_norm_m <= 0.0
        ):
            raise ValueError("maximum_drift_norm_m must be positive")

        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "skew_generators", _readonly(skew))
        object.__setattr__(self, "skew_feature_weights", _readonly(skew_weights))
        object.__setattr__(
            self,
            "contraction_directions",
            _readonly(contraction),
        )
        object.__setattr__(
            self,
            "contraction_feature_weights",
            _readonly(contraction_weights),
        )
        object.__setattr__(self, "drift_directions_m", _readonly(drift))
        object.__setattr__(self, "drift_feature_weights", _readonly(drift_weights))

    @classmethod
    def identity(
        cls,
        *,
        feature_names: tuple[str, ...],
        rank: int,
    ) -> "StableDiscrepancyTransitionModel":
        """Construct the exact graph-persistence transition."""

        feature_count = len(feature_names)
        return cls(
            feature_names=feature_names,
            rank=rank,
            skew_generators=np.zeros((0, rank, rank)),
            skew_feature_weights=np.zeros((0, feature_count)),
            contraction_directions=np.zeros((0, rank)),
            contraction_feature_weights=np.zeros((0, feature_count)),
            drift_directions_m=np.zeros((0, rank, 3)),
            drift_feature_weights=np.zeros((0, feature_count)),
            model_id="exact-graph-persistence",
        )

    def _features(self, feature_vector: np.ndarray) -> np.ndarray:
        values = np.asarray(feature_vector, dtype=float)
        if values.shape != (len(self.feature_names),) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError("feature vector does not match transition model")
        return values

    def transition_operator(self, feature_vector: np.ndarray) -> np.ndarray:
        """Return the non-expansive transition matrix for one forecast step."""

        features = self._features(feature_vector)
        generator = np.zeros((self.rank, self.rank), dtype=float)
        if len(self.skew_generators):
            generator += np.einsum(
                "j,jab->ab",
                self.skew_feature_weights @ features,
                self.skew_generators,
            )
        if len(self.contraction_directions):
            rates = np.square(self.contraction_feature_weights @ features)
            generator -= np.einsum(
                "j,ja,jb->ab",
                rates,
                self.contraction_directions,
                self.contraction_directions,
            )
        if not np.any(generator):
            return np.eye(self.rank)
        transition = np.asarray(expm(generator), dtype=float)
        if not np.all(np.isfinite(transition)):
            raise RuntimeError("discrepancy transition is non-finite")
        return transition

    def drift_increment_m(self, feature_vector: np.ndarray) -> np.ndarray:
        """Return a bounded coefficient-space mean increment."""

        features = self._features(feature_vector)
        if not len(self.drift_directions_m):
            return np.zeros((self.rank, 3), dtype=float)
        drift = np.einsum(
            "j,jrc->rc",
            self.drift_feature_weights @ features,
            self.drift_directions_m,
        )
        if self.maximum_drift_norm_m is not None:
            norm = float(np.linalg.norm(drift))
            if norm > self.maximum_drift_norm_m:
                drift *= self.maximum_drift_norm_m / norm
        return drift


def forecast_action_conditioned_dynamics(
    belief: GraphDiscrepancyBelief,
    innovation_model: ActionConditionedDiscrepancyModel,
    transition_model: StableDiscrepancyTransitionModel,
    features: ActionConditionedDiscrepancyFeatures,
    basis: np.ndarray,
) -> ActionConditionedDiscrepancyForecast:
    """Propagate mean and covariance under one shared action feature sequence."""

    graph_basis = np.asarray(basis, dtype=float)
    if graph_basis.ndim != 2 or graph_basis.shape[1] != belief.rank:
        raise ValueError("basis must have shape (node, belief.rank)")
    if not np.all(np.isfinite(graph_basis)):
        raise ValueError("basis must be finite")
    if array_sha256(graph_basis) != belief.basis_sha256:
        raise ValueError("basis hash differs from graph-discrepancy belief")
    if innovation_model.rank != belief.rank or transition_model.rank != belief.rank:
        raise ValueError("discrepancy model ranks differ from the belief")
    if not (
        innovation_model.feature_names
        == transition_model.feature_names
        == features.names
    ):
        raise ValueError("discrepancy feature schemas differ")

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
    mean = np.empty((component_count, horizon + 1, belief.rank, 3), dtype=float)
    covariance = np.empty(
        (component_count, horizon + 1, 3, belief.rank, belief.rank),
        dtype=float,
    )
    mean[:, 0] = belief.coefficient_mean_m
    covariance[:, 0] = belief.coefficient_covariance_m2
    for component in range(component_count):
        for step in range(horizon):
            feature_vector = feature_values[component, step]
            transition = transition_model.transition_operator(feature_vector)
            mean[component, step + 1] = (
                transition @ mean[component, step]
                + transition_model.drift_increment_m(feature_vector)
            )
            innovation = innovation_model.innovation_covariance_m2(feature_vector)
            for coordinate in range(3):
                previous = covariance[component, step, coordinate]
                covariance[component, step + 1, coordinate] = (
                    _positive_semidefinite(
                        transition @ previous @ transition.T + innovation
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
        model_id=f"{transition_model.model_id}+{innovation_model.model_id}",
    )
