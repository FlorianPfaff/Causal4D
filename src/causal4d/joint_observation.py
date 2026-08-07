"""Full-joint Gaussian observation updates for finite Causal4D support.

The grouped robust likelihood deliberately caps repeated evidence group by group.
This module provides the complementary exact Gaussian path for producers such as
Prob4D that export one covariance over all selected observation rows. Dense and
low-rank covariance contributions are both supported without forming the
low-rank update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array, readonly_integer_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.weighting import log_weights_from_probabilities


JOINT_OBSERVATION_SCHEMA_VERSION = 1


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_positive_integer(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        plain_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_covariance(
    values: np.ndarray,
    *,
    dimension: int,
    name: str,
    leading_shape: tuple[int, ...] = (),
) -> np.ndarray:
    covariance = np.asarray(values, dtype=float)
    try:
        covariance = np.broadcast_to(
            covariance,
            (*leading_shape, dimension, dimension),
        )
    except ValueError as error:
        raise ValueError(
            f"{name} must broadcast to {leading_shape + (dimension, dimension)}"
        ) from error
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(
        covariance,
        covariance.swapaxes(-1, -2),
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError(f"{name} must be symmetric")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return covariance


def _validated_factor(
    values: np.ndarray,
    *,
    dimension: int,
    name: str,
    leading_shape: tuple[int, ...] = (),
) -> np.ndarray:
    factor = np.asarray(values, dtype=float)
    if factor.ndim < 2 or factor.shape[-2] != dimension:
        raise ValueError(f"{name} must end in (observation, rank)")
    rank = factor.shape[-1]
    if rank < 1:
        raise ValueError(f"{name} rank must be positive")
    try:
        factor = np.broadcast_to(factor, (*leading_shape, dimension, rank))
    except ValueError as error:
        raise ValueError(
            f"{name} must broadcast to component leading dimensions"
        ) from error
    if not np.all(np.isfinite(factor)):
        raise ValueError(f"{name} must be finite")
    return factor


@dataclass(frozen=True)
class LinearJointObservationEvidence:
    """One linear observation vector with one covariance over every row.

    Parallel sparse term vectors encode a linear map from trajectories ending in
    ``(frame, node, coordinate)`` to the observation rows. ``base_covariance_m2``
    is positive definite. ``shared_covariance_factor_m`` optionally adds a
    positive-semidefinite cross-row contribution ``U U.T``.
    """

    evidence_id: str
    values_m: np.ndarray
    row_indices: np.ndarray
    frame_indices: np.ndarray
    node_indices: np.ndarray
    coordinate_indices: np.ndarray
    coefficients: np.ndarray
    base_covariance_m2: np.ndarray
    shared_covariance_factor_m: np.ndarray | None = None
    source_id: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        evidence_id = _require_nonempty_string(self.evidence_id, name="evidence_id")
        source_id = _require_nonempty_string(self.source_id, name="source_id")
        values = readonly_array(self.values_m, dtype=float)
        rows = readonly_integer_array(self.row_indices, name="row_indices")
        frames = readonly_integer_array(self.frame_indices, name="frame_indices")
        nodes = readonly_integer_array(self.node_indices, name="node_indices")
        coordinates = readonly_integer_array(
            self.coordinate_indices,
            name="coordinate_indices",
        )
        coefficients = readonly_array(self.coefficients, dtype=float)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("values_m must be a nonempty vector")
        term_count = len(rows)
        if term_count == 0 or any(
            vector.shape != (term_count,)
            for vector in (frames, nodes, coordinates, coefficients)
        ):
            raise ValueError("joint observation term vectors must be aligned")
        if (
            np.any(rows < 0)
            or np.any(rows >= len(values))
            or np.any(frames < 0)
            or np.any(nodes < 0)
            or np.any(coordinates < 0)
        ):
            raise ValueError("joint observation indices are out of range")
        if set(map(int, rows)) != set(range(len(values))):
            raise ValueError("every joint observation row must contain a term")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(coefficients)):
            raise ValueError("joint observation values and coefficients must be finite")
        if np.any(coefficients == 0.0):
            raise ValueError("zero joint observation coefficients are not allowed")
        covariance = readonly_array(
            _validated_covariance(
                self.base_covariance_m2,
                dimension=len(values),
                name="base_covariance_m2",
            ),
            dtype=float,
        )
        factor = None
        if self.shared_covariance_factor_m is not None:
            factor = readonly_array(
                _validated_factor(
                    self.shared_covariance_factor_m,
                    dimension=len(values),
                    name="shared_covariance_factor_m",
                ),
                dtype=float,
            )
        for row in np.unique(rows[frames == 0]):
            selected = rows == row
            if not np.isclose(
                float(np.sum(coefficients[selected])),
                0.0,
                atol=1e-12,
                rtol=1e-12,
            ):
                raise ValueError(
                    "endpoint frame zero may appear only in a zero-sum contrast"
                )
            if not np.any(frames[selected] > 0):
                raise ValueError("endpoint contrasts require a response frame")
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "values_m", values)
        object.__setattr__(self, "row_indices", rows)
        object.__setattr__(self, "frame_indices", frames)
        object.__setattr__(self, "node_indices", nodes)
        object.__setattr__(self, "coordinate_indices", coordinates)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "base_covariance_m2", covariance)
        object.__setattr__(self, "shared_covariance_factor_m", factor)
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message=(
                    "joint observation metadata must contain finite JSON data"
                ),
            ),
        )

    @property
    def observation_count(self) -> int:
        return len(self.values_m)

    @property
    def shared_rank(self) -> int:
        factor = self.shared_covariance_factor_m
        return 0 if factor is None else factor.shape[-1]

    @property
    def artifact_id(self) -> str:
        factor = self.shared_covariance_factor_m
        return _canonical_sha256(
            {
                "schema_version": JOINT_OBSERVATION_SCHEMA_VERSION,
                "evidence_id": self.evidence_id,
                "source_id": self.source_id,
                "values_sha256": array_sha256(self.values_m),
                "row_indices_sha256": array_sha256(self.row_indices),
                "frame_indices_sha256": array_sha256(self.frame_indices),
                "node_indices_sha256": array_sha256(self.node_indices),
                "coordinate_indices_sha256": array_sha256(
                    self.coordinate_indices
                ),
                "coefficients_sha256": array_sha256(self.coefficients),
                "base_covariance_sha256": array_sha256(
                    self.base_covariance_m2
                ),
                "shared_covariance_factor_sha256": (
                    None if factor is None else array_sha256(factor)
                ),
                "metadata": plain_json(self.metadata),
            }
        )

    def validate_prefix(
        self,
        *,
        prefix_frame_count: int,
        rollout_shape: Sequence[int],
    ) -> None:
        prefix = _require_positive_integer(
            prefix_frame_count,
            name="prefix_frame_count",
        )
        if prefix < 2:
            raise ValueError("prefix_frame_count must reveal a response frame")
        if len(rollout_shape) != 3:
            raise ValueError("rollout shape must be (frame, node, coordinate)")
        frame_count, node_count, coordinate_count = (
            _require_positive_integer(
                int(value),
                name=f"rollout_shape[{index}]",
            )
            for index, value in enumerate(rollout_shape)
        )
        if prefix > frame_count:
            raise ValueError("prefix_frame_count exceeds the rollout")
        if np.any(self.frame_indices >= prefix):
            raise ValueError("joint observation crosses the declared prefix")
        if np.any(self.node_indices >= node_count):
            raise ValueError("joint observation references an unavailable node")
        if np.any(self.coordinate_indices >= coordinate_count):
            raise ValueError(
                "joint observation references an unavailable coordinate"
            )

    def apply(self, trajectories_m: np.ndarray) -> np.ndarray:
        trajectories = np.asarray(trajectories_m, dtype=float)
        if trajectories.ndim < 3:
            raise ValueError("trajectories_m must end in (frame, node, coordinate)")
        selected = trajectories[
            ...,
            self.frame_indices,
            self.node_indices,
            self.coordinate_indices,
        ]
        output = np.zeros(
            (*trajectories.shape[:-3], self.observation_count),
            dtype=float,
        )
        for term_index, row in enumerate(self.row_indices):
            output[..., int(row)] += (
                self.coefficients[term_index] * selected[..., term_index]
            )
        return output

    def apply_independent_covariance(self, variance_m2: np.ndarray) -> np.ndarray:
        """Propagate diagonal trajectory variance through the sparse operator.

        Reusing one trajectory scalar in multiple observation rows correctly
        creates cross-row covariance instead of silently retaining only a diagonal.
        """

        variances = np.asarray(variance_m2, dtype=float)
        if variances.ndim < 3:
            raise ValueError("variance_m2 must end in (frame, node, coordinate)")
        if not np.all(np.isfinite(variances)) or np.any(variances < 0.0):
            raise ValueError("component variances must be finite and nonnegative")
        selected = variances[
            ...,
            self.frame_indices,
            self.node_indices,
            self.coordinate_indices,
        ]
        output = np.zeros(
            (
                *variances.shape[:-3],
                self.observation_count,
                self.observation_count,
            ),
            dtype=float,
        )
        selectors = tuple(
            zip(
                map(int, self.frame_indices),
                map(int, self.node_indices),
                map(int, self.coordinate_indices),
            )
        )
        for left, left_selector in enumerate(selectors):
            left_row = int(self.row_indices[left])
            for right, right_selector in enumerate(selectors):
                if left_selector != right_selector:
                    continue
                right_row = int(self.row_indices[right])
                output[..., left_row, right_row] += (
                    self.coefficients[left]
                    * self.coefficients[right]
                    * selected[..., left]
                )
        return output


@dataclass(frozen=True)
class JointGaussianLikelihoodDiagnostics:
    """Representation and dimension details for a full-joint update."""

    observation_count: int
    evidence_shared_rank: int
    component_shared_rank: int
    used_component_independent_covariance: bool
    used_component_dense_covariance: bool
    used_low_rank_path: bool


def block_diagonalize_covariance(
    covariance_m2: np.ndarray,
    block_ids: Sequence[Any],
) -> np.ndarray:
    """Return the explicit block-diagonal ablation for labelled observations."""

    covariance = np.asarray(covariance_m2, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance_m2 must be a square matrix")
    labels = tuple(block_ids)
    if len(labels) != covariance.shape[0]:
        raise ValueError("block_ids must match covariance dimension")
    _validated_covariance(
        covariance,
        dimension=covariance.shape[0],
        name="covariance_m2",
    )
    result = covariance.copy()
    for row, row_label in enumerate(labels):
        for column, column_label in enumerate(labels):
            if row_label != column_label:
                result[row, column] = 0.0
    _validated_covariance(
        result,
        dimension=result.shape[0],
        name="block-diagonal covariance",
    )
    return result


def _joint_gaussian_log_density(
    residual: np.ndarray,
    base_covariance_m2: np.ndarray,
    covariance_factor_m: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(residual, dtype=float)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("residual must end in a nonempty observation dimension")
    if not np.all(np.isfinite(values)):
        raise ValueError("residual must be finite")
    dimension = values.shape[-1]
    leading_shape = values.shape[:-1]
    base = _validated_covariance(
        base_covariance_m2,
        dimension=dimension,
        name="base_covariance_m2",
        leading_shape=leading_shape,
    )
    base_cholesky = np.linalg.cholesky(base)
    whitened_residual = np.linalg.solve(
        base_cholesky,
        values[..., None],
    )[..., 0]
    quadratic = np.einsum(
        "...i,...i->...",
        whitened_residual,
        whitened_residual,
    )
    log_determinant = 2.0 * np.sum(
        np.log(np.diagonal(base_cholesky, axis1=-2, axis2=-1)),
        axis=-1,
    )
    if covariance_factor_m is not None:
        factor = _validated_factor(
            covariance_factor_m,
            dimension=dimension,
            name="covariance_factor_m",
            leading_shape=leading_shape,
        )
        whitened_factor = np.linalg.solve(base_cholesky, factor)
        rank = factor.shape[-1]
        low_rank_system = np.eye(rank) + np.einsum(
            "...ir,...is->...rs",
            whitened_factor,
            whitened_factor,
        )
        try:
            low_rank_cholesky = np.linalg.cholesky(low_rank_system)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "low-rank covariance system must be positive definite"
            ) from error
        projection = np.einsum(
            "...ir,...i->...r",
            whitened_factor,
            whitened_residual,
        )
        whitened_projection = np.linalg.solve(
            low_rank_cholesky,
            projection[..., None],
        )[..., 0]
        correction = np.einsum(
            "...r,...r->...",
            whitened_projection,
            whitened_projection,
        )
        quadratic = np.maximum(quadratic - correction, 0.0)
        log_determinant += 2.0 * np.sum(
            np.log(
                np.diagonal(
                    low_rank_cholesky,
                    axis1=-2,
                    axis2=-1,
                )
            ),
            axis=-1,
        )
    return -0.5 * (
        dimension * np.log(2.0 * np.pi) + log_determinant + quadratic
    )


def joint_component_log_likelihoods(
    predicted_components_m: np.ndarray,
    evidence: LinearJointObservationEvidence,
    *,
    prefix_frame_count: int,
    component_independent_variance_m2: np.ndarray | None = None,
    component_joint_covariance_m2: np.ndarray | None = None,
    component_joint_covariance_factor_m: np.ndarray | None = None,
) -> tuple[np.ndarray, JointGaussianLikelihoodDiagnostics]:
    """Score finite trajectory support with the complete joint covariance."""

    components = np.asarray(predicted_components_m, dtype=float)
    if components.ndim < 4:
        raise ValueError(
            "predicted_components_m must end in (frame, node, coordinate)"
        )
    if not np.all(np.isfinite(components)):
        raise ValueError("predicted components must be finite")
    evidence.validate_prefix(
        prefix_frame_count=prefix_frame_count,
        rollout_shape=components.shape[-3:],
    )
    leading_shape = components.shape[:-3]
    predictions = evidence.apply(components)
    residual = predictions - evidence.values_m
    base = np.broadcast_to(
        evidence.base_covariance_m2,
        (*leading_shape, evidence.observation_count, evidence.observation_count),
    ).copy()
    used_independent = component_independent_variance_m2 is not None
    if component_independent_variance_m2 is not None:
        variance = np.broadcast_to(
            np.asarray(component_independent_variance_m2, dtype=float),
            components.shape,
        )
        base += evidence.apply_independent_covariance(variance)
    used_dense = component_joint_covariance_m2 is not None
    if component_joint_covariance_m2 is not None:
        base += _validated_covariance(
            component_joint_covariance_m2,
            dimension=evidence.observation_count,
            name="component_joint_covariance_m2",
            leading_shape=leading_shape,
        )

    factors = []
    if evidence.shared_covariance_factor_m is not None:
        factors.append(
            np.broadcast_to(
                evidence.shared_covariance_factor_m,
                (
                    *leading_shape,
                    evidence.observation_count,
                    evidence.shared_rank,
                ),
            )
        )
    component_rank = 0
    if component_joint_covariance_factor_m is not None:
        component_factor = _validated_factor(
            component_joint_covariance_factor_m,
            dimension=evidence.observation_count,
            name="component_joint_covariance_factor_m",
            leading_shape=leading_shape,
        )
        component_rank = component_factor.shape[-1]
        factors.append(component_factor)
    factor = None if not factors else np.concatenate(factors, axis=-1)
    score = _joint_gaussian_log_density(residual, base, factor)
    diagnostics = JointGaussianLikelihoodDiagnostics(
        observation_count=evidence.observation_count,
        evidence_shared_rank=evidence.shared_rank,
        component_shared_rank=component_rank,
        used_component_independent_covariance=used_independent,
        used_component_dense_covariance=used_dense,
        used_low_rank_path=factor is not None,
    )
    return score, diagnostics


def posterior_weights_from_joint_observation(
    prior_weights: np.ndarray,
    predicted_components_m: np.ndarray,
    evidence: LinearJointObservationEvidence,
    *,
    prefix_frame_count: int,
    component_independent_variance_m2: np.ndarray | None = None,
    component_joint_covariance_m2: np.ndarray | None = None,
    component_joint_covariance_factor_m: np.ndarray | None = None,
) -> tuple[np.ndarray, JointGaussianLikelihoodDiagnostics]:
    """Apply a full-joint Gaussian observation without creating prior support."""

    prior = np.asarray(prior_weights, dtype=float)
    component_shape = np.asarray(predicted_components_m).shape[:-3]
    if prior.shape != component_shape:
        raise ValueError("prior_weights must match component leading dimensions")
    if not np.isclose(np.sum(prior), 1.0):
        raise ValueError("prior_weights must sum to one")
    score, diagnostics = joint_component_log_likelihoods(
        predicted_components_m,
        evidence,
        prefix_frame_count=prefix_frame_count,
        component_independent_variance_m2=component_independent_variance_m2,
        component_joint_covariance_m2=component_joint_covariance_m2,
        component_joint_covariance_factor_m=component_joint_covariance_factor_m,
    )
    log_posterior = log_weights_from_probabilities(
        prior,
        name="prior_weights",
    ) + score
    maximum = float(np.max(log_posterior))
    posterior = np.exp(log_posterior - maximum)
    posterior /= np.sum(posterior)
    return posterior, diagnostics


__all__ = [
    "JOINT_OBSERVATION_SCHEMA_VERSION",
    "JointGaussianLikelihoodDiagnostics",
    "LinearJointObservationEvidence",
    "block_diagonalize_covariance",
    "joint_component_log_likelihoods",
    "posterior_weights_from_joint_observation",
]
