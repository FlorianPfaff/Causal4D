"""Stable action-conditioned propagation of low-rank discrepancy means."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from causal4d.action_conditioned_discrepancy import (
    ActionConditionedDiscrepancyFeatures,
    ActionConditionedDiscrepancyForecast,
    ActionConditionedDiscrepancyModel,
    forecast_action_conditioned_persistence,
)
from causal4d.contracts import array_sha256
from causal4d.discrepancy_belief import GraphDiscrepancyBelief


def _readonly(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).copy()
    array.setflags(write=False)
    return array


def _positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


def _bounded_frobenius(matrix: np.ndarray, maximum: float | None) -> np.ndarray:
    if maximum is None:
        return matrix
    norm = float(np.linalg.norm(matrix))
    if norm <= maximum or norm == 0.0:
        return matrix
    return matrix * (maximum / norm)


@dataclass(frozen=True)
class ActionConditionedMeanTransitionModel:
    """Stable graph-mode movement for a discrepancy coefficient mean.

    A feature vector produces a rotation generator ``Omega(f)``, a positive
    semidefinite contraction generator ``D(f)``, and a bounded forcing field
    ``b(f)``. The transition is

    ``A(f) = exp(Omega(f)) @ exp(-D(f))``.

    The first factor is orthogonal and the second is non-expansive. Zero
    transition and forcing weights reproduce exact graph persistence.
    """

    feature_names: tuple[str, ...]
    contraction_directions: np.ndarray
    contraction_weights: np.ndarray
    rotation_generators: np.ndarray
    rotation_weights: np.ndarray
    forcing_weights_m: np.ndarray
    model_id: str = "action-conditioned-discrepancy-mean-v1"
    maximum_generator_norm: float | None = None
    maximum_forcing_norm_m: float | None = None

    def __post_init__(self) -> None:
        names = tuple(map(str, self.feature_names))
        contraction_directions = _readonly(self.contraction_directions)
        contraction_weights = _readonly(self.contraction_weights)
        rotation_generators = _readonly(self.rotation_generators)
        rotation_weights = _readonly(self.rotation_weights)
        forcing_weights = _readonly(self.forcing_weights_m)
        if not self.model_id or not names or len(set(names)) != len(names):
            raise ValueError("model id and unique feature names must be nonempty")
        if forcing_weights.ndim != 3 or forcing_weights.shape[1:] != (3, len(names)):
            raise ValueError(
                "forcing_weights_m must have shape (rank, 3, feature_count)"
            )
        rank = forcing_weights.shape[0]
        if rank < 1:
            raise ValueError("transition rank must be positive")
        if (
            contraction_directions.ndim != 2
            or contraction_directions.shape[1] != rank
        ):
            raise ValueError("contraction_directions must have shape (Jc, rank)")
        if contraction_weights.shape != (len(contraction_directions), len(names)):
            raise ValueError(
                "contraction_weights must have shape (Jc, feature_count)"
            )
        if rotation_generators.shape != (len(rotation_generators), rank, rank):
            raise ValueError("rotation_generators must have shape (Jr, rank, rank)")
        if rotation_weights.shape != (len(rotation_generators), len(names)):
            raise ValueError("rotation_weights must have shape (Jr, feature_count)")
        arrays = (
            contraction_directions,
            contraction_weights,
            rotation_generators,
            rotation_weights,
            forcing_weights,
        )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("mean-transition arrays must be finite")
        if len(contraction_directions):
            norms = np.linalg.norm(contraction_directions, axis=1)
            if np.any(norms <= 0.0):
                raise ValueError("contraction directions must be nonzero")
            contraction_directions = _readonly(
                contraction_directions / norms[:, None]
            )
        if not np.allclose(
            rotation_generators,
            -rotation_generators.swapaxes(-1, -2),
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError("rotation generators must be skew-symmetric")
        for value, name in (
            (self.maximum_generator_norm, "maximum_generator_norm"),
            (self.maximum_forcing_norm_m, "maximum_forcing_norm_m"),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "contraction_directions", contraction_directions)
        object.__setattr__(self, "contraction_weights", contraction_weights)
        object.__setattr__(self, "rotation_generators", rotation_generators)
        object.__setattr__(self, "rotation_weights", rotation_weights)
        object.__setattr__(self, "forcing_weights_m", forcing_weights)

    @classmethod
    def persistence(
        cls,
        feature_names: tuple[str, ...],
        rank: int,
    ) -> "ActionConditionedMeanTransitionModel":
        """Construct the exact graph-persistence fallback."""

        if rank < 1:
            raise ValueError("rank must be positive")
        feature_count = len(feature_names)
        return cls(
            feature_names=feature_names,
            contraction_directions=np.zeros((0, rank)),
            contraction_weights=np.zeros((0, feature_count)),
            rotation_generators=np.zeros((0, rank, rank)),
            rotation_weights=np.zeros((0, feature_count)),
            forcing_weights_m=np.zeros((rank, 3, feature_count)),
            model_id="graph-persistence-mean",
        )

    @property
    def rank(self) -> int:
        return int(self.forcing_weights_m.shape[0])

    @property
    def is_exact_persistence(self) -> bool:
        return (
            not np.any(self.contraction_weights)
            and not np.any(self.rotation_weights)
            and not np.any(self.forcing_weights_m)
        )

    def transition_and_forcing(
        self,
        feature_vector: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a non-expansive mode transition and bounded additive movement."""

        features = np.asarray(feature_vector, dtype=float)
        if features.shape != (len(self.feature_names),) or not np.all(
            np.isfinite(features)
        ):
            raise ValueError(
                "feature vector does not match the mean-transition model"
            )

        contraction = np.zeros((self.rank, self.rank), dtype=float)
        if len(self.contraction_directions):
            amplitudes = self.contraction_weights @ features
            contraction = np.einsum(
                "j,ji,jk->ik",
                np.square(amplitudes),
                self.contraction_directions,
                self.contraction_directions,
            )
        rotation = np.zeros((self.rank, self.rank), dtype=float)
        if len(self.rotation_generators):
            rates = self.rotation_weights @ features
            rotation = np.einsum("j,jik->ik", rates, self.rotation_generators)

        contraction = _bounded_frobenius(
            contraction,
            self.maximum_generator_norm,
        )
        rotation = _bounded_frobenius(rotation, self.maximum_generator_norm)
        rotation_transition = expm(rotation) if np.any(rotation) else np.eye(self.rank)
        contraction_transition = (
            expm(-contraction) if np.any(contraction) else np.eye(self.rank)
        )
        transition = rotation_transition @ contraction_transition

        forcing = np.einsum("rcf,f->rc", self.forcing_weights_m, features)
        forcing = _bounded_frobenius(forcing, self.maximum_forcing_norm_m)
        return transition, forcing


def _component_features(
    belief: GraphDiscrepancyBelief,
    features: ActionConditionedDiscrepancyFeatures,
) -> np.ndarray:
    component_count = len(belief.component_ids)
    if features.values.ndim == 2:
        return np.broadcast_to(
            features.values[None],
            (component_count, *features.values.shape),
        )
    if features.component_ids != belief.component_ids:
        raise ValueError("component-specific features differ from belief support")
    return features.values


def forecast_action_conditioned_movement(
    belief: GraphDiscrepancyBelief,
    mean_model: ActionConditionedMeanTransitionModel,
    covariance_model: ActionConditionedDiscrepancyModel,
    features: ActionConditionedDiscrepancyFeatures,
    basis: np.ndarray,
) -> ActionConditionedDiscrepancyForecast:
    """Propagate graph-mode movement and covariance without state injection.

    When ``mean_model`` is the persistence fallback, this function delegates to
    ``forecast_action_conditioned_persistence`` and returns that result unchanged.
    """

    if mean_model.is_exact_persistence:
        return forecast_action_conditioned_persistence(
            belief,
            covariance_model,
            features,
            basis,
        )

    graph_basis = np.asarray(basis, dtype=float)
    if graph_basis.ndim != 2 or graph_basis.shape[1] != belief.rank:
        raise ValueError("basis must have shape (node, belief.rank)")
    if not np.all(np.isfinite(graph_basis)):
        raise ValueError("basis must be finite")
    if array_sha256(graph_basis) != belief.basis_sha256:
        raise ValueError("basis hash differs from the graph-discrepancy belief")
    if (
        mean_model.rank != belief.rank
        or covariance_model.rank != belief.rank
        or mean_model.feature_names != features.names
        or covariance_model.feature_names != features.names
    ):
        raise ValueError(
            "model rank or feature schema differs from belief/features"
        )

    feature_values = _component_features(belief, features)
    component_count = len(belief.component_ids)
    horizon = features.horizon
    mean = np.empty(
        (component_count, horizon + 1, belief.rank, 3),
        dtype=float,
    )
    covariance = np.empty(
        (component_count, horizon + 1, 3, belief.rank, belief.rank),
        dtype=float,
    )
    mean[:, 0] = belief.coefficient_mean_m
    covariance[:, 0] = belief.coefficient_covariance_m2

    for step in range(horizon):
        for component in range(component_count):
            feature_vector = feature_values[component, step]
            transition, forcing = mean_model.transition_and_forcing(feature_vector)
            mean[component, step + 1] = (
                transition @ mean[component, step] + forcing
            )
            innovation = covariance_model.innovation_covariance_m2(feature_vector)
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
        model_id=f"{mean_model.model_id}+{covariance_model.model_id}",
    )
