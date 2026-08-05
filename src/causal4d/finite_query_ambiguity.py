"""Finite-support identifiability diagnostics for a specific future query."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causal4d.immutable_array import readonly_array


def _readonly(values: np.ndarray, *, dtype: type = float) -> np.ndarray:
    return readonly_array(values, dtype=dtype)


def _normalized_weights(values: np.ndarray) -> np.ndarray:
    weights = np.asarray(values, dtype=float).reshape(-1)
    if not len(weights) or not np.all(np.isfinite(weights)):
        raise ValueError("prior_weights must be a finite nonempty vector")
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError("prior_weights must be nonnegative with positive mass")
    return weights / np.sum(weights)


def _whitened_components(
    values: np.ndarray,
    covariance: np.ndarray | float | None,
) -> np.ndarray:
    components = np.asarray(values, dtype=float)
    if components.ndim < 2 or not np.all(np.isfinite(components)):
        raise ValueError("responses must have shape (component, ...) and be finite")
    flattened = components.reshape(len(components), -1)
    dimension = flattened.shape[1]
    if dimension == 0:
        raise ValueError("responses must contain at least one coordinate")
    if covariance is None:
        return flattened
    noise = np.asarray(covariance, dtype=float)
    if noise.ndim == 0:
        if not np.isfinite(noise) or float(noise) <= 0.0:
            raise ValueError("scalar covariance must be finite and positive")
        return flattened / np.sqrt(float(noise))
    if noise.shape == (dimension,):
        if not np.all(np.isfinite(noise)) or np.any(noise <= 0.0):
            raise ValueError("diagonal covariance must be finite and positive")
        return flattened / np.sqrt(noise)[None]
    if noise.shape != (dimension, dimension):
        raise ValueError("covariance must be scalar, diagonal, or full response size")
    if not np.all(np.isfinite(noise)) or not np.allclose(
        noise,
        noise.T,
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError("full covariance must be finite and symmetric")
    try:
        factor = np.linalg.cholesky(noise)
    except np.linalg.LinAlgError as error:
        raise ValueError("full covariance must be positive definite") from error
    return np.linalg.solve(factor, flattened.T).T


def _scaled_query_components(
    values: np.ndarray,
    scale: np.ndarray | float | None,
) -> np.ndarray:
    components = np.asarray(values, dtype=float)
    if components.ndim < 2 or not np.all(np.isfinite(components)):
        raise ValueError(
            "query responses must have shape (component, ...) and be finite"
        )
    flattened = components.reshape(len(components), -1)
    if scale is None:
        return flattened
    supplied = np.asarray(scale, dtype=float)
    if supplied.ndim == 0:
        if not np.isfinite(supplied) or float(supplied) <= 0.0:
            raise ValueError("query_scale must be finite and positive")
        return flattened / float(supplied)
    if supplied.shape != (flattened.shape[1],):
        raise ValueError("query_scale must be scalar or match query coordinates")
    if not np.all(np.isfinite(supplied)) or np.any(supplied <= 0.0):
        raise ValueError("query_scale must be finite and positive")
    return flattened / supplied[None]


@dataclass(frozen=True)
class FiniteQueryAmbiguityConfig:
    """Thresholds for indistinguishable-prefix, divergent-query support pairs."""

    maximum_prefix_rms_mahalanobis: float = 1.0
    minimum_query_rms_distance: float = 1.0
    maximum_ambiguous_pair_mass: float = 0.05
    maximum_weighted_query_distance: float = 0.05

    def __post_init__(self) -> None:
        if not np.isfinite(self.maximum_prefix_rms_mahalanobis) or (
            self.maximum_prefix_rms_mahalanobis < 0.0
        ):
            raise ValueError("maximum_prefix_rms_mahalanobis must be nonnegative")
        if not np.isfinite(self.minimum_query_rms_distance) or (
            self.minimum_query_rms_distance < 0.0
        ):
            raise ValueError("minimum_query_rms_distance must be nonnegative")
        for name in (
            "maximum_ambiguous_pair_mass",
            "maximum_weighted_query_distance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class FiniteQueryAmbiguityResult:
    """Global ambiguity of a finite posterior support for one future query."""

    component_count: int
    pair_indices: np.ndarray
    prefix_rms_mahalanobis: np.ndarray
    query_rms_distance: np.ndarray
    pair_probability_mass: np.ndarray
    ambiguous_pair_mass: float
    weighted_query_distance: float
    maximum_query_distance: float
    admissible: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        pairs = _readonly(self.pair_indices, dtype=np.int64)
        prefix = _readonly(self.prefix_rms_mahalanobis)
        query = _readonly(self.query_rms_distance)
        mass = _readonly(self.pair_probability_mass)
        count = len(pairs)
        if pairs.shape != (count, 2):
            raise ValueError("pair_indices must have shape (pair, 2)")
        if (
            prefix.shape != (count,)
            or query.shape != (count,)
            or mass.shape != (count,)
        ):
            raise ValueError("pair diagnostics must align")
        if np.any(pairs < 0) or np.any(pairs >= self.component_count):
            raise ValueError("pair indices exceed component support")
        if np.any(prefix < 0.0) or np.any(query < 0.0) or np.any(mass < 0.0):
            raise ValueError("pair diagnostics must be nonnegative")
        for value, name in (
            (self.ambiguous_pair_mass, "ambiguous_pair_mass"),
            (self.weighted_query_distance, "weighted_query_distance"),
            (self.maximum_query_distance, "maximum_query_distance"),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        object.__setattr__(self, "pair_indices", pairs)
        object.__setattr__(self, "prefix_rms_mahalanobis", prefix)
        object.__setattr__(self, "query_rms_distance", query)
        object.__setattr__(self, "pair_probability_mass", mass)
        object.__setattr__(self, "failure_reasons", tuple(self.failure_reasons))

    def as_dict(self) -> dict[str, object]:
        return {
            "component_count": self.component_count,
            "ambiguous_pair_count": len(self.pair_indices),
            "ambiguous_pair_mass": self.ambiguous_pair_mass,
            "weighted_query_distance": self.weighted_query_distance,
            "maximum_query_distance": self.maximum_query_distance,
            "admissible": self.admissible,
            "failure_reasons": list(self.failure_reasons),
            "pair_indices": self.pair_indices.tolist(),
            "prefix_rms_mahalanobis": self.prefix_rms_mahalanobis.tolist(),
            "query_rms_distance": self.query_rms_distance.tolist(),
            "pair_probability_mass": self.pair_probability_mass.tolist(),
        }


def assess_finite_query_ambiguity(
    prefix_responses: np.ndarray,
    query_responses: np.ndarray,
    prior_weights: np.ndarray,
    *,
    prefix_covariance: np.ndarray | float | None = None,
    query_scale: np.ndarray | float | None = None,
    config: FiniteQueryAmbiguityConfig | None = None,
) -> FiniteQueryAmbiguityResult:
    """Detect support pairs hidden by the prefix but divergent for the query.

    The diagnostic is invariant to support permutation and to splitting one
    component into exact clones whose weights sum to the original mass. Pair mass
    is the probability of drawing the two distinct components in either order.
    """

    settings = config or FiniteQueryAmbiguityConfig()
    prefix = _whitened_components(prefix_responses, prefix_covariance)
    query = _scaled_query_components(query_responses, query_scale)
    weights = _normalized_weights(prior_weights)
    if len(prefix) != len(query) or len(prefix) != len(weights):
        raise ValueError("prefix, query, and prior support must align")
    component_count = len(weights)
    ambiguous_pairs: list[tuple[int, int]] = []
    prefix_distances: list[float] = []
    query_distances: list[float] = []
    pair_masses: list[float] = []
    for first in range(component_count):
        if weights[first] <= 0.0:
            continue
        for second in range(first + 1, component_count):
            if weights[second] <= 0.0:
                continue
            prefix_distance = float(
                np.sqrt(np.mean(np.square(prefix[first] - prefix[second])))
            )
            query_distance = float(
                np.sqrt(np.mean(np.square(query[first] - query[second])))
            )
            if (
                prefix_distance <= settings.maximum_prefix_rms_mahalanobis
                and query_distance >= settings.minimum_query_rms_distance
            ):
                ambiguous_pairs.append((first, second))
                prefix_distances.append(prefix_distance)
                query_distances.append(query_distance)
                pair_masses.append(2.0 * weights[first] * weights[second])
    masses = np.asarray(pair_masses, dtype=float)
    distances = np.asarray(query_distances, dtype=float)
    ambiguous_mass = float(np.sum(masses))
    weighted_distance = float(np.sum(masses * distances))
    maximum_distance = float(np.max(distances)) if len(distances) else 0.0
    reasons = []
    if ambiguous_mass > settings.maximum_ambiguous_pair_mass:
        reasons.append("ambiguous_support_mass_exceeds_threshold")
    if weighted_distance > settings.maximum_weighted_query_distance:
        reasons.append("weighted_query_divergence_exceeds_threshold")
    return FiniteQueryAmbiguityResult(
        component_count=component_count,
        pair_indices=np.asarray(ambiguous_pairs, dtype=np.int64).reshape(-1, 2),
        prefix_rms_mahalanobis=np.asarray(prefix_distances, dtype=float),
        query_rms_distance=distances,
        pair_probability_mass=masses,
        ambiguous_pair_mass=ambiguous_mass,
        weighted_query_distance=weighted_distance,
        maximum_query_distance=maximum_distance,
        admissible=not reasons,
        failure_reasons=tuple(reasons),
    )
