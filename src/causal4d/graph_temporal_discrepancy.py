"""Low-rank graph and temporal model for deformable-object discrepancy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from causal4d.immutable_array import readonly_array


@dataclass(frozen=True)
class GraphTemporalDiscrepancyModel:
    """Graph basis with stable linear coefficient dynamics."""

    basis: np.ndarray
    eigenvalues: np.ndarray
    transition: np.ndarray
    innovation_covariance: np.ndarray
    projection_variance_m2: np.ndarray
    selected_rank: int
    candidate_validation_rmse_m: tuple[tuple[int, float], ...]
    spectral_radius_before_clipping: float
    spectral_radius: float
    fit_frame_count: int
    projection_ridge: float
    dynamics_ridge: float

    def __post_init__(self) -> None:
        basis = readonly_array(self.basis, dtype=float)
        eigenvalues = readonly_array(self.eigenvalues, dtype=float)
        transition = readonly_array(self.transition, dtype=float)
        innovation = readonly_array(self.innovation_covariance, dtype=float)
        projection = readonly_array(self.projection_variance_m2, dtype=float)
        rank = self.selected_rank
        if basis.ndim != 2 or basis.shape[1] != rank:
            raise ValueError("basis must have shape (node, selected_rank)")
        if eigenvalues.shape != (rank,):
            raise ValueError("eigenvalues must match selected_rank")
        if transition.shape != (rank, rank) or innovation.shape != (rank, rank):
            raise ValueError("coefficient dynamics must match selected_rank")
        if projection.shape != (3,):
            raise ValueError("projection variance must have three coordinates")
        if not all(
            np.all(np.isfinite(value))
            for value in (basis, eigenvalues, transition, innovation, projection)
        ):
            raise ValueError("graph-temporal model arrays must be finite")
        if np.any(projection < 0.0) or np.any(np.linalg.eigvalsh(innovation) < -1e-10):
            raise ValueError("graph-temporal variances must be nonnegative")
        if self.spectral_radius > 1.0 + 1e-10:
            raise ValueError("graph-temporal transition must be stable")
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "innovation_covariance", innovation)
        object.__setattr__(self, "projection_variance_m2", projection)


def _canonicalize_eigenspace(cluster_basis: np.ndarray) -> np.ndarray:
    """Choose a node-order-deterministic basis for one repeated eigenspace."""

    orthonormal, _ = np.linalg.qr(np.asarray(cluster_basis, dtype=float))
    width = orthonormal.shape[1]
    residual_coefficients = orthonormal.copy()
    canonical = np.empty_like(orthonormal)
    tolerance = (
        128.0 * np.finfo(float).eps * max(1, orthonormal.shape[0], orthonormal.shape[1])
    )
    for column in range(width):
        leverage = np.einsum(
            "ij,ij->i",
            residual_coefficients,
            residual_coefficients,
        )
        pivot = int(np.argmax(leverage))
        pivot_leverage = float(leverage[pivot])
        if pivot_leverage <= tolerance:
            raise RuntimeError("degenerate eigenspace canonicalization lost rank")
        coefficient = residual_coefficients[pivot] / np.sqrt(pivot_leverage)
        vector = orthonormal @ coefficient
        if vector[pivot] < 0.0:
            coefficient = -coefficient
            vector = -vector
        canonical[:, column] = vector
        residual_coefficients -= np.outer(
            residual_coefficients @ coefficient,
            coefficient,
        )
    return canonical


def canonicalize_graph_eigenbasis(
    eigenvalues: np.ndarray,
    basis: np.ndarray,
    *,
    degeneracy_atol: float = 1e-10,
    degeneracy_rtol: float = 1e-10,
) -> np.ndarray:
    """Canonicalize signs and numerically repeated graph eigenspaces.

    Sign fixing alone is insufficient when an eigensolver may return an arbitrary
    rotation within a repeated eigenspace. Repeated clusters are therefore rebuilt
    from their invariant subspace using graph-node order as the deterministic
    pivot rule. Singleton modes retain the historical largest-entry sign rule.
    """

    values = np.asarray(eigenvalues, dtype=float)
    vectors = np.asarray(basis, dtype=float)
    if values.ndim != 1 or vectors.ndim != 2 or vectors.shape[1] != len(values):
        raise ValueError("basis columns must match the eigenvalue vector")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
        raise ValueError("eigenvalues and basis must be finite")
    if (
        not np.isfinite(degeneracy_atol)
        or degeneracy_atol < 0.0
        or not np.isfinite(degeneracy_rtol)
        or degeneracy_rtol < 0.0
    ):
        raise ValueError("degeneracy tolerances must be finite and nonnegative")
    if len(values) > 1 and np.any(np.diff(values) < -degeneracy_atol):
        raise ValueError("eigenvalues must be sorted in nondecreasing order")

    canonical = vectors.copy()
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and np.isclose(
            values[stop],
            values[start],
            atol=degeneracy_atol,
            rtol=degeneracy_rtol,
        ):
            stop += 1
        if stop - start == 1:
            pivot = int(np.argmax(np.abs(canonical[:, start])))
            if canonical[pivot, start] < 0.0:
                canonical[:, start] *= -1.0
        else:
            canonical[:, start:stop] = _canonicalize_eigenspace(
                canonical[:, start:stop]
            )
        start = stop
    return canonical


def graph_laplacian_basis(
    node_count: int,
    springs: np.ndarray,
    *,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return low-frequency eigenvectors of the symmetric normalized Laplacian."""

    try:
        from scipy import sparse
        from scipy.sparse.linalg import eigsh
    except (ImportError, OSError) as error:
        raise RuntimeError("graph-temporal discrepancy requires scipy") from error
    edges = np.asarray(springs, dtype=np.int64)
    if node_count < 2 or not 1 <= rank < node_count:
        raise ValueError("rank must lie in [1, node_count)")
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) == 0:
        raise ValueError("springs must have nonempty shape (edge, 2)")
    if np.any(edges < 0) or np.any(edges >= node_count):
        raise ValueError("spring endpoint exceeds node_count")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("self springs are not supported")
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    columns = np.concatenate((edges[:, 1], edges[:, 0]))
    adjacency = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, columns)),
        shape=(node_count, node_count),
    ).tocsr()
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse_root = np.zeros_like(degree)
    active = degree > 0.0
    inverse_root[active] = 1.0 / np.sqrt(degree[active])
    scaling = sparse.diags(inverse_root)
    laplacian = sparse.eye(node_count, format="csr") - scaling @ adjacency @ scaling
    if node_count <= 256:
        eigenvalues, basis = np.linalg.eigh(laplacian.toarray())
        eigenvalues = eigenvalues[:rank]
        basis = basis[:, :rank]
    else:
        eigenvalues, basis = eigsh(
            laplacian,
            k=rank,
            which="SM",
            tol=1e-7,
            v0=np.linspace(1.0, 2.0, node_count),
        )
        order = np.argsort(eigenvalues, kind="mergesort")
        eigenvalues = eigenvalues[order]
        basis = basis[:, order]
    basis = canonicalize_graph_eigenbasis(eigenvalues, basis)
    return basis, np.maximum(eigenvalues, 0.0)


def _validated_node_indices(
    values: np.ndarray | Sequence[int] | None,
    *,
    observed_node_count: int,
    basis_node_count: int,
) -> np.ndarray:
    if observed_node_count > basis_node_count:
        raise ValueError("basis does not cover the observed residual nodes")
    if values is None:
        return np.arange(observed_node_count, dtype=np.int64)
    supplied = np.asarray(values)
    if supplied.shape != (observed_node_count,):
        raise ValueError("node_indices must identify every observed residual node")
    if supplied.dtype.kind not in {"i", "u"}:
        raise ValueError("node_indices must contain integers")
    indices = np.asarray(supplied, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= basis_node_count):
        raise ValueError("node_indices exceed the graph basis")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("node_indices must be unique")
    return indices


def project_graph_coefficients(
    residual_m: np.ndarray,
    valid: np.ndarray,
    basis: np.ndarray,
    *,
    ridge: float,
    node_indices: np.ndarray | Sequence[int] | None = None,
) -> np.ndarray:
    """Project partially observed residual fields onto fixed graph modes.

    ``node_indices`` binds each residual column to an explicit graph node. Omitting
    it retains the historical convention that observed columns are nodes
    ``0, ..., observed_node_count - 1``.
    """

    residual = np.asarray(residual_m, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    modes = np.asarray(basis, dtype=float)
    if residual.ndim != 3 or residual.shape[2] != 3:
        raise ValueError("residual_m must have shape (T, observed_node, 3)")
    if mask.shape != residual.shape[:2]:
        raise ValueError("valid must have shape (T, observed_node)")
    if modes.ndim != 2:
        raise ValueError("basis must have shape (node, rank)")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("projection ridge must be finite and positive")
    indices = _validated_node_indices(
        node_indices,
        observed_node_count=residual.shape[1],
        basis_node_count=modes.shape[0],
    )
    rank = modes.shape[1]
    coefficients = np.zeros((len(residual), rank, 3), dtype=float)
    observed_basis = modes[indices]
    identity = np.eye(rank)
    for frame in range(len(residual)):
        selected = mask[frame] & np.all(np.isfinite(residual[frame]), axis=1)
        if np.sum(selected) < rank:
            raise ValueError("too few valid graph nodes to estimate discrepancy modes")
        design = observed_basis[selected]
        precision = design.T @ design + ridge * identity
        coefficients[frame] = np.linalg.solve(
            precision,
            design.T @ residual[frame, selected],
        )
    return coefficients


def _fit_transition(
    coefficients: np.ndarray,
    *,
    ridge: float,
    maximum_spectral_radius: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 3 or values.shape[2] != 3 or len(values) < 3:
        raise ValueError("coefficients must have shape (T>=3, rank, 3)")
    if not np.all(np.isfinite(values)):
        raise ValueError("coefficients must be finite")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("dynamics ridge must be finite and positive")
    if not 0.0 < maximum_spectral_radius <= 1.0:
        raise ValueError("maximum_spectral_radius must lie in (0, 1]")
    rank = values.shape[1]
    source = values[:-1].transpose(1, 0, 2).reshape(rank, -1)
    target = values[1:].transpose(1, 0, 2).reshape(rank, -1)
    gram = source @ source.T + ridge * np.eye(rank)
    transition = np.linalg.solve(gram, source @ target.T).T
    radius_before = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if radius_before > maximum_spectral_radius:
        transition *= maximum_spectral_radius / radius_before
    radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    innovation = target - transition @ source
    covariance = innovation @ innovation.T / innovation.shape[1]
    covariance = 0.5 * (covariance + covariance.T) + 1e-12 * np.eye(rank)
    return transition, covariance, radius_before, radius


def fit_graph_temporal_discrepancy(
    residual_m: np.ndarray,
    valid: np.ndarray,
    full_basis: np.ndarray,
    full_eigenvalues: np.ndarray,
    *,
    rank_candidates: Sequence[int] = (4, 8, 16, 32),
    validation_fraction: float = 0.25,
    projection_ridge: float = 1e-5,
    dynamics_ridge: float = 1e-4,
    maximum_spectral_radius: float = 0.995,
    node_indices: np.ndarray | Sequence[int] | None = None,
) -> GraphTemporalDiscrepancyModel:
    """Select rank on an O-minus suffix and refit stable coefficient dynamics."""

    residual = np.asarray(residual_m, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    basis = np.asarray(full_basis, dtype=float)
    eigenvalues = np.asarray(full_eigenvalues, dtype=float)
    if residual.ndim != 3 or residual.shape[2] != 3:
        raise ValueError("residual_m must have shape (T, observed_node, 3)")
    if mask.shape != residual.shape[:2]:
        raise ValueError("valid must have shape (T, observed_node)")
    if basis.ndim != 2:
        raise ValueError("full_basis must have shape (node, rank)")
    indices = _validated_node_indices(
        node_indices,
        observed_node_count=residual.shape[1],
        basis_node_count=basis.shape[0],
    )
    candidates = tuple(sorted(set(map(int, rank_candidates))))
    if not candidates or candidates[0] < 1 or candidates[-1] > basis.shape[1]:
        raise ValueError("rank candidates must be covered by full_basis")
    if eigenvalues.shape != (basis.shape[1],):
        raise ValueError("full eigenvalues must match full_basis")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie in (0, 0.5)")
    if not np.isfinite(projection_ridge) or projection_ridge <= 0.0:
        raise ValueError("projection_ridge must be finite and positive")
    if not np.isfinite(dynamics_ridge) or dynamics_ridge <= 0.0:
        raise ValueError("dynamics_ridge must be finite and positive")
    if not 0.0 < maximum_spectral_radius <= 1.0:
        raise ValueError("maximum_spectral_radius must lie in (0, 1]")
    split = max(3, int(np.floor(len(residual) * (1.0 - validation_fraction))))
    if len(residual) - split < 2:
        raise ValueError("discrepancy fit needs at least two validation frames")

    validation_scores = []
    coefficient_cache = {}
    for rank in candidates:
        selected_basis = basis[:, :rank]
        coefficients = project_graph_coefficients(
            residual,
            mask,
            selected_basis,
            ridge=projection_ridge,
            node_indices=indices,
        )
        coefficient_cache[rank] = coefficients
        transition, _, _, _ = _fit_transition(
            coefficients[:split],
            ridge=dynamics_ridge,
            maximum_spectral_radius=maximum_spectral_radius,
        )
        predicted_coefficients = np.einsum(
            "ij,tjc->tic",
            transition,
            coefficients[split - 1 : -1],
        )
        predicted = np.einsum(
            "nr,trc->tnc",
            selected_basis[indices],
            predicted_coefficients,
        )
        target = residual[split:]
        selected_mask = mask[split:] & np.all(np.isfinite(target), axis=2)
        if not np.any(selected_mask):
            raise ValueError("validation suffix contains no finite residual vectors")
        score = float(np.sqrt(np.mean(np.square((predicted - target)[selected_mask]))))
        validation_scores.append((rank, score))
    selected_rank = min(validation_scores, key=lambda value: (value[1], value[0]))[0]
    selected_basis = basis[:, :selected_rank]
    coefficients = coefficient_cache[selected_rank]
    transition, innovation, radius_before, radius = _fit_transition(
        coefficients,
        ridge=dynamics_ridge,
        maximum_spectral_radius=maximum_spectral_radius,
    )
    reconstructed = np.einsum(
        "nr,trc->tnc",
        selected_basis[indices],
        coefficients,
    )
    projection_error = reconstructed - residual
    projection_mask = mask & np.all(np.isfinite(residual), axis=2)
    if not np.any(projection_mask):
        raise ValueError("discrepancy fit contains no finite residual vectors")
    projection_variance = np.asarray(
        [
            np.mean(np.square(projection_error[:, :, coordinate][projection_mask]))
            for coordinate in range(3)
        ]
    )
    return GraphTemporalDiscrepancyModel(
        basis=selected_basis,
        eigenvalues=eigenvalues[:selected_rank],
        transition=transition,
        innovation_covariance=innovation,
        projection_variance_m2=projection_variance,
        selected_rank=selected_rank,
        candidate_validation_rmse_m=tuple(validation_scores),
        spectral_radius_before_clipping=radius_before,
        spectral_radius=radius,
        fit_frame_count=len(residual),
        projection_ridge=projection_ridge,
        dynamics_ridge=dynamics_ridge,
    )


def forecast_graph_temporal_discrepancy(
    model: GraphTemporalDiscrepancyModel,
    prefix_residual_m: np.ndarray,
    prefix_valid: np.ndarray,
    *,
    total_frame_count: int,
    dynamics: Literal["learned", "persistence"] = "learned",
    node_indices: np.ndarray | Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast graph discrepancy using only the supplied prefix residuals."""

    prefix = np.asarray(prefix_residual_m, dtype=float)
    valid = np.asarray(prefix_valid, dtype=bool)
    if not 2 <= len(prefix) < total_frame_count:
        raise ValueError("prefix must reveal evidence and leave future frames")
    coefficients = project_graph_coefficients(
        prefix,
        valid,
        model.basis,
        ridge=model.projection_ridge,
        node_indices=node_indices,
    )
    node_count = model.basis.shape[0]
    mean = np.zeros((total_frame_count, node_count, 3), dtype=float)
    variance = np.zeros_like(mean)
    mean[: len(prefix)] = np.einsum(
        "nr,trc->tnc",
        model.basis,
        coefficients,
    )
    transition = (
        model.transition if dynamics == "learned" else np.eye(model.selected_rank)
    )
    current = coefficients[-1].copy()
    covariance = np.zeros((model.selected_rank, model.selected_rank), dtype=float)
    for frame in range(len(prefix), total_frame_count):
        current = transition @ current
        covariance = (
            transition @ covariance @ transition.T + model.innovation_covariance
        )
        mean[frame] = model.basis @ current
        marginal = np.einsum(
            "ni,ij,nj->n",
            model.basis,
            covariance,
            model.basis,
        )
        variance[frame] = marginal[:, None] + model.projection_variance_m2[None]
    variance[: len(prefix)] = model.projection_variance_m2[None, None]
    return mean, np.maximum(variance, 0.0)
