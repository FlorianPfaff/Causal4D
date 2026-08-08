"""Prospective normalized, support-aware contact-patch inference.

This module is deliberately additive.  It does not alter the registered latent-
contact estimator.  The v2 path combines linear covariance-aware observation
operators, dimension-normalized robust likelihoods, deterministic weighted
parameter coresets, sparse contact-patch hypotheses, and exact marginal Gaussian-
mixture intervals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from itertools import product
from math import lgamma
from typing import Any, Generic, Literal, Mapping, Sequence, TypeVar

import numpy as np
from scipy.special import ndtr

from causal4d.baselines import ParameterPosterior, PredictiveDistribution
from causal4d.contact_inference import ContactPrior, LatentContactConfig
from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array, readonly_integer_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.parameter_support import (
    ParameterSupportReduction,
    SupportMethod,
    reduce_parameter_support,
)
from causal4d.simulator import (
    Action,
    GraphObject,
    SimulatorConfig,
    WorldCondition,
    graph_adjacency,
    simulate_particles,
)
from causal4d.weighting import log_weights_from_probabilities


LATENT_CONTACT_V2_SCHEMA_VERSION = 1
ContactEndpoint = Literal[
    "factual_continuation",
    "same_grasp_transfer",
    "new_contact_transfer",
]

BaselineT = TypeVar("BaselineT")
CandidateT = TypeVar("CandidateT")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        plain_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_float(
    value: Any,
    *,
    name: str,
    minimum: Any = None,
    maximum: Any = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _normalized_weights(
    values: np.ndarray | Sequence[float],
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind == "b":
        raise ValueError(f"{name} must contain numbers, not Booleans")
    weights = readonly_array(raw, dtype=float)
    if weights.ndim == 0 or weights.size == 0:
        raise ValueError(f"{name} must be nonempty")
    if expected_shape is not None and weights.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError(f"{name} must contain positive mass")
    return readonly_array(weights / total, dtype=float)


def _validated_covariance(
    values: np.ndarray, *, dimension: int, name: str
) -> np.ndarray:
    covariance = readonly_array(values, dtype=float)
    if covariance.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape ({dimension}, {dimension})")
    if not np.all(np.isfinite(covariance)) or not np.allclose(
        covariance,
        covariance.T,
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError(f"{name} must be finite and symmetric")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return covariance


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _validated_string_tuple(values: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(
        _require_nonempty_string(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty and unique")
    return result


@dataclass(frozen=True)
class LinearContactObservationGroup:
    """One robustly scored linear function of an admitted response prefix.

    The sparse operator is represented by parallel term vectors.  For term ``k``,
    ``coefficients[k]`` multiplies the rollout scalar selected by frame, node, and
    coordinate indices and contributes it to ``row_indices[k]``.  Endpoint frame
    zero may appear only in a translation-neutral, zero-sum contrast for every
    coordinate, which admits endpoint-to-first-response increments without
    treating the endpoint as a fresh absolute observation.
    """

    group_id: str
    values_m: np.ndarray
    row_indices: np.ndarray
    frame_indices: np.ndarray
    node_indices: np.ndarray
    coordinate_indices: np.ndarray
    coefficients: np.ndarray
    covariance_m2: np.ndarray
    contributor_ids: tuple[str, ...]
    prior_nominal_probability: float = 0.95
    outlier_scale_multiplier: float = 100.0
    degrees_of_freedom: float = 4.0
    composite_weight: float = 1.0
    source_id: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        group_id = _require_nonempty_string(self.group_id, name="group_id")
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
        contributors = _validated_string_tuple(
            self.contributor_ids,
            name="contributor_ids",
        )
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("values_m must be a nonempty vector")
        term_count = len(rows)
        if term_count == 0 or any(
            vector.shape != (term_count,)
            for vector in (frames, nodes, coordinates, coefficients)
        ):
            raise ValueError(
                "linear-operator term vectors must be aligned and nonempty"
            )
        if (
            np.any(rows < 0)
            or np.any(rows >= len(values))
            or np.any(frames < 0)
            or np.any(nodes < 0)
            or np.any(coordinates < 0)
        ):
            raise ValueError("linear-operator indices are out of range")
        if set(map(int, rows)) != set(range(len(values))):
            raise ValueError("every observation row must contain at least one term")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(coefficients)):
            raise ValueError("group values and coefficients must be finite")
        if np.any(coefficients == 0.0):
            raise ValueError("zero operator coefficients are not allowed")
        covariance = _validated_covariance(
            self.covariance_m2,
            dimension=len(values),
            name="covariance_m2",
        )
        for row in np.unique(rows[frames == 0]):
            row_terms = rows == row
            for coordinate in np.unique(coordinates[row_terms]):
                coordinate_terms = row_terms & (coordinates == coordinate)
                if not np.isclose(
                    float(np.sum(coefficients[coordinate_terms])),
                    0.0,
                    atol=1e-12,
                    rtol=1e-12,
                ):
                    raise ValueError(
                        "endpoint frame zero may appear only in a "
                        "translation-neutral zero-sum contrast per coordinate"
                    )
            if not np.any(frames[row_terms] > 0):
                raise ValueError("endpoint contrasts require a positive response frame")
        prior = _finite_float(
            self.prior_nominal_probability,
            name="prior_nominal_probability",
            minimum=np.finfo(float).eps,
            maximum=1.0 - np.finfo(float).eps,
        )
        outlier = _finite_float(
            self.outlier_scale_multiplier,
            name="outlier_scale_multiplier",
            minimum=1.0 + np.finfo(float).eps,
        )
        degrees = _finite_float(
            self.degrees_of_freedom,
            name="degrees_of_freedom",
            minimum=2.0 + np.finfo(float).eps,
        )
        composite = _finite_float(
            self.composite_weight,
            name="composite_weight",
            minimum=np.finfo(float).eps,
            maximum=1.0,
        )
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "values_m", values)
        object.__setattr__(self, "row_indices", rows)
        object.__setattr__(self, "frame_indices", frames)
        object.__setattr__(self, "node_indices", nodes)
        object.__setattr__(self, "coordinate_indices", coordinates)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "covariance_m2", covariance)
        object.__setattr__(self, "contributor_ids", contributors)
        object.__setattr__(self, "prior_nominal_probability", prior)
        object.__setattr__(self, "outlier_scale_multiplier", outlier)
        object.__setattr__(self, "degrees_of_freedom", degrees)
        object.__setattr__(self, "composite_weight", composite)
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="group metadata must contain finite JSON data",
            ),
        )

    @property
    def coordinate_count(self) -> int:
        return len(self.values_m)

    @property
    def operator_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": LATENT_CONTACT_V2_SCHEMA_VERSION,
                "group_id": self.group_id,
                "source_id": self.source_id,
                "values_sha256": array_sha256(self.values_m),
                "row_indices_sha256": array_sha256(self.row_indices),
                "frame_indices_sha256": array_sha256(self.frame_indices),
                "node_indices_sha256": array_sha256(self.node_indices),
                "coordinate_indices_sha256": array_sha256(self.coordinate_indices),
                "coefficients_sha256": array_sha256(self.coefficients),
                "covariance_sha256": array_sha256(self.covariance_m2),
                "contributor_ids": list(self.contributor_ids),
                "prior_nominal_probability": self.prior_nominal_probability,
                "outlier_scale_multiplier": self.outlier_scale_multiplier,
                "degrees_of_freedom": self.degrees_of_freedom,
                "composite_weight": self.composite_weight,
                "metadata": plain_json(self.metadata),
            }
        )

    def validate_prefix(
        self,
        *,
        prefix_frame_count: int,
        rollout_shape: Sequence[int],
    ) -> None:
        prefix = _positive_int(prefix_frame_count, name="prefix_frame_count")
        if prefix < 2:
            raise ValueError(
                "prefix_frame_count must reveal at least one response frame"
            )
        if len(rollout_shape) != 3:
            raise ValueError("rollout shape must be (frame, node, coordinate)")
        frame_count, node_count, coordinate_count = map(int, rollout_shape)
        if prefix > frame_count:
            raise ValueError("prefix_frame_count exceeds the rollout")
        if np.any(self.frame_indices >= prefix):
            raise ValueError("observation group crosses the declared prefix")
        if np.any(self.node_indices >= node_count):
            raise ValueError("observation group references an unavailable node")
        if np.any(self.coordinate_indices >= coordinate_count):
            raise ValueError("observation group references an unavailable coordinate")

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
            (*trajectories.shape[:-3], self.coordinate_count), dtype=float
        )
        for term_index, row in enumerate(self.row_indices):
            output[..., int(row)] += (
                self.coefficients[term_index] * selected[..., term_index]
            )
        return output

    def apply_independent_variance(self, variance_m2: np.ndarray) -> np.ndarray:
        variances = np.asarray(variance_m2, dtype=float)
        if variances.ndim < 3:
            raise ValueError("variance_m2 must end in (frame, node, coordinate)")
        if np.any(~np.isfinite(variances)) or np.any(variances < 0.0):
            raise ValueError("component variances must be finite and nonnegative")
        selected = variances[
            ...,
            self.frame_indices,
            self.node_indices,
            self.coordinate_indices,
        ]
        output = np.zeros((*variances.shape[:-3], self.coordinate_count), dtype=float)
        for term_index, row in enumerate(self.row_indices):
            output[..., int(row)] += (
                self.coefficients[term_index] ** 2 * selected[..., term_index]
            )
        return output


@dataclass(frozen=True)
class ContactObservationEvidenceV2:
    """Linear observation groups with multiplicity and dimension power control."""

    groups: tuple[LinearContactObservationGroup, ...]
    evidence_id: str = "contact_observation_evidence_v2"
    dimension_normalization_power: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        if not groups or any(
            type(group) is not LinearContactObservationGroup for group in groups
        ):
            raise ValueError("groups must contain linear contact observation groups")
        identifiers = tuple(group.group_id for group in groups)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("group IDs must be unique")
        evidence_id = _require_nonempty_string(self.evidence_id, name="evidence_id")
        power = _finite_float(
            self.dimension_normalization_power,
            name="dimension_normalization_power",
            minimum=0.0,
            maximum=1.0,
        )
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "dimension_normalization_power", power)
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="evidence metadata must contain finite JSON data",
            ),
        )

    @property
    def contributor_multiplicity(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for group in self.groups:
            for contributor in group.contributor_ids:
                result[contributor] = result.get(contributor, 0) + 1
        return result

    @property
    def effective_group_weights(self) -> tuple[float, ...]:
        multiplicity = self.contributor_multiplicity
        return tuple(
            group.composite_weight
            / max(multiplicity[contributor] for contributor in group.contributor_ids)
            / group.coordinate_count**self.dimension_normalization_power
            for group in self.groups
        )

    @property
    def artifact_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": LATENT_CONTACT_V2_SCHEMA_VERSION,
                "artifact_kind": "Causal4DContactObservationEvidenceV2",
                "evidence_id": self.evidence_id,
                "dimension_normalization_power": self.dimension_normalization_power,
                "group_ids": [group.operator_id for group in self.groups],
                "metadata": plain_json(self.metadata),
            }
        )

    def validate_prefix(
        self,
        *,
        prefix_frame_count: int,
        rollout_shape: Sequence[int],
    ) -> None:
        for group in self.groups:
            group.validate_prefix(
                prefix_frame_count=prefix_frame_count,
                rollout_shape=rollout_shape,
            )

    @classmethod
    def from_dense_prefix(
        cls,
        observations_m: np.ndarray,
        *,
        prefix_frame_count: int,
        position_scale_m: float,
        mask: np.ndarray | None = None,
        coordinate_correlation: np.ndarray | None = None,
        node_correlation: float = 0.0,
        difference_correlation: float = 0.0,
        position_composite_weight: float = 1.0,
        dynamic_composite_weight: float = 1.0,
        include_positions: bool = True,
        include_differences: bool = True,
        prior_nominal_probability: float = 0.95,
        outlier_scale_multiplier: float = 100.0,
        degrees_of_freedom: float = 4.0,
        source_id: str = "dense_contact_prefix",
        dimension_normalization_power: float = 1.0,
    ) -> "ContactObservationEvidenceV2":
        observations = np.asarray(observations_m, dtype=float)
        if observations.ndim != 3 or observations.shape[2] not in {2, 3}:
            raise ValueError("observations_m must have shape (frame, node, 2|3)")
        prefix = _positive_int(prefix_frame_count, name="prefix_frame_count")
        if not 2 <= prefix <= len(observations):
            raise ValueError("prefix_frame_count must reveal an admitted response")
        scale = _finite_float(
            position_scale_m,
            name="position_scale_m",
            minimum=np.finfo(float).eps,
        )
        node_rho = _finite_float(
            node_correlation,
            name="node_correlation",
            minimum=0.0,
            maximum=1.0 - np.finfo(float).eps,
        )
        difference_rho = _finite_float(
            difference_correlation,
            name="difference_correlation",
            minimum=-1.0 + np.finfo(float).eps,
            maximum=1.0 - np.finfo(float).eps,
        )
        if not include_positions and not include_differences:
            raise ValueError("at least one evidence block must be enabled")
        coordinate_count = observations.shape[2]
        coordinate_matrix: np.ndarray
        if coordinate_correlation is None:
            coordinate_matrix = np.eye(coordinate_count, dtype=float)
        else:
            coordinate_matrix = _validated_covariance(
                np.asarray(coordinate_correlation, dtype=float),
                dimension=coordinate_count,
                name="coordinate_correlation",
            )
            diagonal = np.diag(coordinate_matrix)
            if not np.allclose(diagonal, 1.0, atol=1e-12, rtol=1e-12):
                raise ValueError("coordinate_correlation must have a unit diagonal")
        valid = np.all(np.isfinite(observations), axis=2)
        if mask is not None:
            supplied = np.asarray(mask, dtype=bool)
            if supplied.shape == observations.shape:
                supplied = np.all(supplied, axis=2)
            if supplied.shape != observations.shape[:2]:
                raise ValueError("mask must have shape (T, N) or (T, N, C)")
            valid &= supplied

        def frame_covariance(node_count: int) -> np.ndarray:
            node_matrix = (1.0 - node_rho) * np.eye(node_count) + node_rho * np.ones(
                (node_count, node_count), dtype=float
            )
            return scale**2 * np.kron(node_matrix, coordinate_matrix)

        groups: list[LinearContactObservationGroup] = []
        for frame in range(1, prefix):
            if include_positions:
                nodes = np.flatnonzero(valid[frame])
                if len(nodes):
                    count = len(nodes) * coordinate_count
                    groups.append(
                        LinearContactObservationGroup(
                            group_id=f"{source_id}:position:{frame}",
                            values_m=observations[frame, nodes].reshape(-1),
                            row_indices=np.arange(count, dtype=np.int64),
                            frame_indices=np.full(count, frame, dtype=np.int64),
                            node_indices=np.repeat(nodes, coordinate_count),
                            coordinate_indices=np.tile(
                                np.arange(coordinate_count, dtype=np.int64),
                                len(nodes),
                            ),
                            coefficients=np.ones(count, dtype=float),
                            covariance_m2=frame_covariance(len(nodes)),
                            contributor_ids=(f"{source_id}:frame:{frame}",),
                            prior_nominal_probability=prior_nominal_probability,
                            outlier_scale_multiplier=outlier_scale_multiplier,
                            degrees_of_freedom=degrees_of_freedom,
                            composite_weight=position_composite_weight,
                            source_id=source_id,
                            metadata={"block": "position", "frame": frame},
                        )
                    )
            if include_differences:
                nodes = np.flatnonzero(valid[frame] & valid[frame - 1])
                if len(nodes):
                    count = len(nodes) * coordinate_count
                    row_indices: np.ndarray = np.repeat(
                        np.arange(count, dtype=np.int64), 2
                    )
                    frame_indices = np.tile(
                        np.asarray((frame - 1, frame), dtype=np.int64),
                        count,
                    )
                    node_base: np.ndarray = np.repeat(nodes, coordinate_count)
                    node_indices: np.ndarray = np.repeat(node_base, 2)
                    coordinate_base = np.tile(
                        np.arange(coordinate_count, dtype=np.int64),
                        len(nodes),
                    )
                    coordinate_indices: np.ndarray = np.repeat(coordinate_base, 2)
                    coefficients = np.tile(np.asarray((-1.0, 1.0)), count)
                    difference_values = (
                        observations[frame, nodes] - observations[frame - 1, nodes]
                    ).reshape(-1)
                    groups.append(
                        LinearContactObservationGroup(
                            group_id=f"{source_id}:difference:{frame - 1}:{frame}",
                            values_m=difference_values,
                            row_indices=row_indices,
                            frame_indices=frame_indices,
                            node_indices=node_indices,
                            coordinate_indices=coordinate_indices,
                            coefficients=coefficients,
                            covariance_m2=(
                                2.0
                                * (1.0 - difference_rho)
                                * frame_covariance(len(nodes))
                            ),
                            contributor_ids=(
                                f"{source_id}:frame:{frame - 1}",
                                f"{source_id}:frame:{frame}",
                            ),
                            prior_nominal_probability=prior_nominal_probability,
                            outlier_scale_multiplier=outlier_scale_multiplier,
                            degrees_of_freedom=degrees_of_freedom,
                            composite_weight=dynamic_composite_weight,
                            source_id=source_id,
                            metadata={
                                "block": "difference",
                                "frame_start": frame - 1,
                                "frame_stop": frame,
                                "difference_correlation": difference_rho,
                            },
                        )
                    )
        if not groups:
            raise ValueError("dense prefix contains no valid evidence groups")
        return cls(
            groups=tuple(groups),
            evidence_id=f"{source_id}:prefix:{prefix}",
            dimension_normalization_power=dimension_normalization_power,
            metadata={
                "constructor": "from_dense_prefix",
                "prefix_frame_count_including_endpoint": prefix,
                "position_scale_m": scale,
                "node_correlation": node_rho,
                "difference_correlation": difference_rho,
                "include_positions": include_positions,
                "include_differences": include_differences,
            },
        )


def _student_t_log_density(
    residual: np.ndarray,
    covariance_m2: np.ndarray,
    *,
    degrees_of_freedom: float,
    covariance_multiplier: float,
) -> np.ndarray:
    values = np.asarray(residual, dtype=float)
    dimension = values.shape[-1]
    covariance = np.asarray(covariance_m2, dtype=float)
    try:
        covariance = np.broadcast_to(
            covariance,
            (*values.shape[:-1], dimension, dimension),
        )
    except ValueError as error:
        raise ValueError(
            "covariance cannot broadcast to the residual support"
        ) from error
    if not np.all(np.isfinite(covariance)) or not np.allclose(
        covariance,
        covariance.swapaxes(-1, -2),
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError("covariance must be finite and symmetric")
    scale_multiplier = (
        (degrees_of_freedom - 2.0) / degrees_of_freedom * covariance_multiplier
    )
    scale = covariance * scale_multiplier
    try:
        factor = np.linalg.cholesky(scale)
    except np.linalg.LinAlgError as error:
        raise ValueError("Student-t scale must be positive definite") from error
    solved = np.linalg.solve(factor, values[..., None])[..., 0]
    mahalanobis = np.einsum("...i,...i->...", solved, solved)
    log_determinant = 2.0 * np.sum(
        np.log(np.diagonal(factor, axis1=-2, axis2=-1)),
        axis=-1,
    )
    normalization = (
        lgamma(0.5 * (degrees_of_freedom + dimension))
        - lgamma(0.5 * degrees_of_freedom)
        - 0.5 * (dimension * np.log(degrees_of_freedom * np.pi) + log_determinant)
    )
    return normalization - 0.5 * (degrees_of_freedom + dimension) * np.log1p(
        mahalanobis / degrees_of_freedom
    )


def _robust_group_log_likelihood(
    predicted_values_m: np.ndarray,
    group: LinearContactObservationGroup,
    *,
    additive_variance_m2: np.ndarray | None,
    additive_covariance_m2: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(predicted_values_m, dtype=float)
    if predictions.shape[-1] != group.coordinate_count:
        raise ValueError("predicted group values have the wrong dimension")
    residual = predictions - group.values_m
    leading_shape = predictions.shape[:-1]
    covariance = np.broadcast_to(
        group.covariance_m2,
        (*leading_shape, group.coordinate_count, group.coordinate_count),
    ).copy()
    if additive_variance_m2 is not None:
        variance = np.asarray(additive_variance_m2, dtype=float)
        if variance.shape != predictions.shape:
            raise ValueError("additive variance must match predicted group values")
        if np.any(~np.isfinite(variance)) or np.any(variance < 0.0):
            raise ValueError("additive variance must be finite and nonnegative")
        diagonal = np.arange(group.coordinate_count)
        covariance[..., diagonal, diagonal] += variance
    if additive_covariance_m2 is not None:
        additive = np.asarray(additive_covariance_m2, dtype=float)
        try:
            additive = np.broadcast_to(additive, covariance.shape)
        except ValueError as error:
            raise ValueError("additive covariance has an incompatible shape") from error
        if not np.all(np.isfinite(additive)) or not np.allclose(
            additive,
            additive.swapaxes(-1, -2),
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError("additive covariance must be finite and symmetric")
        if float(np.min(np.linalg.eigvalsh(additive), initial=0.0)) < -1e-10:
            raise ValueError("additive covariance must be positive semidefinite")
        covariance += additive
    nominal = _student_t_log_density(
        residual,
        covariance,
        degrees_of_freedom=group.degrees_of_freedom,
        covariance_multiplier=1.0,
    )
    outlier = _student_t_log_density(
        residual,
        covariance,
        degrees_of_freedom=group.degrees_of_freedom,
        covariance_multiplier=group.outlier_scale_multiplier,
    )
    log_nominal = np.log(group.prior_nominal_probability) + nominal
    log_outlier = np.log1p(-group.prior_nominal_probability) + outlier
    mixture = np.logaddexp(log_nominal, log_outlier)
    return mixture, np.exp(log_nominal - mixture)


@dataclass(frozen=True)
class ContactLikelihoodV2Diagnostics:
    group_ids: tuple[str, ...]
    effective_group_weights: tuple[float, ...]
    group_coordinate_counts: tuple[int, ...]
    nominal_responsibilities: np.ndarray
    evidence_id: str

    def __post_init__(self) -> None:
        responsibilities = readonly_array(self.nominal_responsibilities, dtype=float)
        if responsibilities.shape[-1] != len(self.group_ids):
            raise ValueError("nominal responsibilities must identify every group")
        if np.any(~np.isfinite(responsibilities)) or np.any(
            (responsibilities < 0.0) | (responsibilities > 1.0)
        ):
            raise ValueError("nominal responsibilities must lie in [0, 1]")
        object.__setattr__(self, "nominal_responsibilities", responsibilities)


def contact_component_log_likelihoods_v2(
    predicted_components_m: np.ndarray,
    evidence: ContactObservationEvidenceV2,
    *,
    prefix_frame_count: int,
    component_variance_m2: np.ndarray | None = None,
    component_group_covariance_m2: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, ContactLikelihoodV2Diagnostics]:
    components = np.asarray(predicted_components_m, dtype=float)
    if components.ndim < 4:
        raise ValueError("predicted components must end in (frame, node, coordinate)")
    if not np.all(np.isfinite(components)):
        raise ValueError("predicted components must be finite")
    evidence.validate_prefix(
        prefix_frame_count=prefix_frame_count,
        rollout_shape=components.shape[-3:],
    )
    variance = None
    if component_variance_m2 is not None:
        try:
            variance = np.broadcast_to(
                np.asarray(component_variance_m2, dtype=float),
                components.shape,
            )
        except ValueError as error:
            raise ValueError(
                "component variance cannot broadcast to components"
            ) from error
        if np.any(~np.isfinite(variance)) or np.any(variance < 0.0):
            raise ValueError("component variance must be finite and nonnegative")
    covariance_by_group = dict(component_group_covariance_m2 or {})
    known = {group.group_id for group in evidence.groups}
    unknown = set(covariance_by_group) - known
    if unknown:
        raise ValueError(
            f"component covariance references unknown groups: {sorted(unknown)}"
        )
    total = np.zeros(components.shape[:-3], dtype=float)
    responsibilities: list[np.ndarray] = []
    effective = evidence.effective_group_weights
    for group, weight in zip(evidence.groups, effective, strict=True):
        selected = group.apply(components)
        selected_variance = (
            None if variance is None else group.apply_independent_variance(variance)
        )
        log_likelihood, responsibility = _robust_group_log_likelihood(
            selected,
            group,
            additive_variance_m2=selected_variance,
            additive_covariance_m2=covariance_by_group.get(group.group_id),
        )
        total += weight * log_likelihood
        responsibilities.append(responsibility)
    return total, ContactLikelihoodV2Diagnostics(
        group_ids=tuple(group.group_id for group in evidence.groups),
        effective_group_weights=effective,
        group_coordinate_counts=tuple(
            group.coordinate_count for group in evidence.groups
        ),
        nominal_responsibilities=np.stack(responsibilities, axis=-1),
        evidence_id=evidence.artifact_id,
    )


def posterior_weights_from_contact_evidence_v2(
    prior_weights: np.ndarray,
    predicted_components_m: np.ndarray,
    evidence: ContactObservationEvidenceV2,
    *,
    prefix_frame_count: int,
    component_variance_m2: np.ndarray | None = None,
    component_group_covariance_m2: Mapping[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, ContactLikelihoodV2Diagnostics]:
    prior = np.asarray(prior_weights, dtype=float)
    if prior.shape != np.asarray(predicted_components_m).shape[:-3]:
        raise ValueError("prior weights must match component leading dimensions")
    normalized = _normalized_weights(
        prior, name="prior_weights", expected_shape=prior.shape
    )
    log_likelihood, diagnostics = contact_component_log_likelihoods_v2(
        predicted_components_m,
        evidence,
        prefix_frame_count=prefix_frame_count,
        component_variance_m2=component_variance_m2,
        component_group_covariance_m2=component_group_covariance_m2,
    )
    log_weights = (
        log_weights_from_probabilities(
            normalized,
            name="latent-contact-v2 prior weights",
        )
        + log_likelihood
    )
    log_weights -= float(np.max(log_weights))
    posterior = np.exp(log_weights)
    posterior /= np.sum(posterior)
    return posterior, diagnostics


def _graph_distances_from_source(
    adjacency: tuple[tuple[int, ...], ...],
    source: int,
    maximum_distance: int,
) -> dict[int, int]:
    distances = {source: 0}
    frontier = [source]
    for node in frontier:
        distance = distances[node]
        if distance >= maximum_distance:
            continue
        for neighbor in adjacency[node]:
            if neighbor not in distances:
                distances[neighbor] = distance + 1
                frontier.append(neighbor)
    return distances


def _connected_support(
    support: set[int],
    adjacency: tuple[tuple[int, ...], ...],
) -> bool:
    if not support:
        return False
    reached = {next(iter(support))}
    frontier = list(reached)
    for node in frontier:
        for neighbor in adjacency[node]:
            if neighbor in support and neighbor not in reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    return reached == support


@dataclass(frozen=True)
class SparseContactPatch:
    """A channel-wise sparse simplex over material nodes."""

    graph_name: str
    center_nodes: tuple[int, ...]
    channel_node_weights: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        graph_name = _require_nonempty_string(self.graph_name, name="graph_name")
        centers = tuple(self.center_nodes)
        if not centers or any(type(value) is not int or value < 0 for value in centers):
            raise ValueError("center_nodes must be nonnegative integers")
        weights = readonly_array(self.channel_node_weights, dtype=float)
        if (
            weights.ndim != 2
            or weights.shape[0] != len(centers)
            or weights.shape[1] == 0
        ):
            raise ValueError("channel_node_weights must have shape (contact, node)")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("patch weights must be finite and nonnegative")
        if not np.allclose(
            np.sum(weights, axis=1),
            1.0,
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("every contact-channel patch must sum to one")
        if any(center >= weights.shape[1] for center in centers):
            raise ValueError("patch center exceeds the node count")
        object.__setattr__(self, "graph_name", graph_name)
        object.__setattr__(self, "center_nodes", centers)
        object.__setattr__(self, "channel_node_weights", weights)
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="patch metadata must contain finite JSON data",
            ),
        )

    @property
    def contact_count(self) -> int:
        return self.channel_node_weights.shape[0]

    @property
    def node_count(self) -> int:
        return self.channel_node_weights.shape[1]

    @property
    def patch_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": LATENT_CONTACT_V2_SCHEMA_VERSION,
                "artifact_kind": "Causal4DSparseContactPatch",
                "graph_name": self.graph_name,
                "center_nodes": list(self.center_nodes),
                "channel_node_weights_sha256": array_sha256(self.channel_node_weights),
                "metadata": plain_json(self.metadata),
            }
        )

    def expanded_action(self, action: Action) -> Action:
        if len(action.contact_nodes) != self.contact_count:
            raise ValueError("patch and commanded action contact counts differ")
        support = np.flatnonzero(np.any(self.channel_node_weights > 0.0, axis=0))
        if len(support) == 0:
            raise ValueError("patch has no positive support")
        forces = np.stack(
            [
                np.einsum(
                    "q,tqc->tc",
                    self.channel_node_weights[:, node],
                    action.commanded_forces,
                )
                for node in support
            ],
            axis=1,
        )
        if not np.allclose(
            np.sum(forces, axis=1),
            np.sum(action.commanded_forces, axis=1),
            atol=1e-12,
            rtol=1e-12,
        ):
            raise RuntimeError("contact patch failed to conserve commanded force")
        return Action(
            action_id=f"{action.action_id}:patch:{self.patch_id[:12]}",
            split=action.split,
            contact_nodes=tuple(map(int, support)),
            commanded_forces=forces,
        )


@dataclass(frozen=True)
class ContactPatchStateV2:
    patch: SparseContactPatch
    gain_multiplier: float
    delay_steps: int
    rotation_radians: float

    def __post_init__(self) -> None:
        gain = _finite_float(
            self.gain_multiplier,
            name="gain_multiplier",
            minimum=np.finfo(float).eps,
        )
        if type(self.delay_steps) is not int or self.delay_steps < 0:
            raise ValueError("delay_steps must be a nonnegative integer")
        rotation = _finite_float(self.rotation_radians, name="rotation_radians")
        object.__setattr__(self, "gain_multiplier", gain)
        object.__setattr__(self, "rotation_radians", rotation)

    @property
    def state_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": LATENT_CONTACT_V2_SCHEMA_VERSION,
                "artifact_kind": "Causal4DContactPatchStateV2",
                "patch_id": self.patch.patch_id,
                "gain_multiplier": self.gain_multiplier,
                "delay_steps": self.delay_steps,
                "rotation_radians": self.rotation_radians,
            }
        )

    def expanded_action(self, action: Action) -> Action:
        return self.patch.expanded_action(action)

    def condition(self) -> WorldCondition:
        return WorldCondition(
            name="contact_patch_v2",
            contact_gain_multiplier=self.gain_multiplier,
            contact_delay_steps=self.delay_steps,
            contact_spread=0.0,
            control_rotation_radians=self.rotation_radians,
        )


@dataclass(frozen=True)
class ContactPatchHypothesisSupportV2:
    states: tuple[ContactPatchStateV2, ...]
    prior_weights: np.ndarray
    retained_patch_prior_mass: float
    full_patch_count: int
    retained_patch_count: int

    def __post_init__(self) -> None:
        states = tuple(self.states)
        if not states or len({state.state_id for state in states}) != len(states):
            raise ValueError("contact-patch states must be nonempty and unique")
        weights = _normalized_weights(
            self.prior_weights,
            name="contact-patch prior weights",
            expected_shape=(len(states),),
        )
        retained = _finite_float(
            self.retained_patch_prior_mass,
            name="retained_patch_prior_mass",
            minimum=np.finfo(float).eps,
            maximum=1.0,
        )
        if (
            self.full_patch_count < self.retained_patch_count
            or self.retained_patch_count < 1
        ):
            raise ValueError("patch support counts are inconsistent")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "prior_weights", weights)
        object.__setattr__(self, "retained_patch_prior_mass", retained)


@dataclass(frozen=True)
class GraphContactPatchModelV2:
    """Generate graph-local sparse contact patches and physical effect states."""

    prior: ContactPrior
    config: LatentContactConfig
    patch_spreads: tuple[float, ...] = (0.0, 0.25, 0.50)
    patch_spread_probabilities: tuple[float, ...] = (0.50, 0.30, 0.20)
    center_radius_hops: int = 1
    maximum_joint_patches: int = 256

    def __post_init__(self) -> None:
        spreads = tuple(
            _finite_float(value, name="patch_spread", minimum=0.0, maximum=1.0)
            for value in self.patch_spreads
        )
        if not spreads or len(set(spreads)) != len(spreads):
            raise ValueError("patch_spreads must be nonempty and unique")
        probabilities = _normalized_weights(
            self.patch_spread_probabilities,
            name="patch_spread_probabilities",
            expected_shape=(len(spreads),),
        )
        radius = _positive_int(self.center_radius_hops, name="center_radius_hops")
        maximum = _positive_int(
            self.maximum_joint_patches, name="maximum_joint_patches"
        )
        object.__setattr__(self, "patch_spreads", spreads)
        object.__setattr__(
            self,
            "patch_spread_probabilities",
            tuple(map(float, probabilities)),
        )
        object.__setattr__(self, "center_radius_hops", radius)
        object.__setattr__(self, "maximum_joint_patches", maximum)

    def _channel_candidates(
        self,
        graph: GraphObject,
        nominal_node: int,
    ) -> tuple[tuple[int, np.ndarray, float, float], ...]:
        adjacency = graph_adjacency(graph)
        distances = _graph_distances_from_source(
            adjacency,
            nominal_node,
            self.center_radius_hops,
        )
        centers = tuple(sorted(distances, key=lambda node: (distances[node], node)))
        shifted = tuple(node for node in centers if node != nominal_node)
        center_probabilities = (
            {nominal_node: 1.0}
            if not shifted
            else {
                nominal_node: 1.0 - self.prior.shift_probability,
                **{
                    node: self.prior.shift_probability / len(shifted)
                    for node in shifted
                },
            }
        )
        candidates: dict[str, tuple[int, np.ndarray, float, float]] = {}
        for center in centers:
            neighbors = adjacency[center]
            for spread, spread_probability in zip(
                self.patch_spreads,
                self.patch_spread_probabilities,
                strict=True,
            ):
                weights = np.zeros(graph.node_count, dtype=float)
                if spread == 0.0 or not neighbors:
                    weights[center] = 1.0
                else:
                    weights[center] = 1.0 - spread
                    neighbor_scores = np.asarray(
                        [
                            np.exp(
                                -float(
                                    distances.get(neighbor, self.center_radius_hops + 1)
                                )
                            )
                            for neighbor in neighbors
                        ],
                        dtype=float,
                    )
                    neighbor_scores /= np.sum(neighbor_scores)
                    weights[np.asarray(neighbors, dtype=int)] += (
                        spread * neighbor_scores
                    )
                key = array_sha256(weights)
                probability = center_probabilities[center] * spread_probability
                existing = candidates.get(key)
                if existing is None:
                    candidates[key] = (center, weights, probability, spread)
                else:
                    candidates[key] = (
                        existing[0],
                        existing[1],
                        existing[2] + probability,
                        existing[3],
                    )
        result = tuple(
            sorted(
                candidates.values(),
                key=lambda item: (-item[2], item[0], item[3], array_sha256(item[1])),
            )
        )
        for _, weights, _, _ in result:
            support = set(map(int, np.flatnonzero(weights > 0.0)))
            if not _connected_support(support, adjacency):
                raise RuntimeError("generated patch support is not graph-connected")
        return result

    def hypotheses(
        self,
        graph: GraphObject,
        action: Action,
    ) -> ContactPatchHypothesisSupportV2:
        if any(node < 0 or node >= graph.node_count for node in action.contact_nodes):
            raise ValueError("commanded action contains an invalid contact node")
        per_channel = tuple(
            self._channel_candidates(graph, node) for node in action.contact_nodes
        )
        joint_patches: list[tuple[SparseContactPatch, float]] = []
        for combination in product(*per_channel):
            centers = tuple(item[0] for item in combination)
            weights = np.stack([item[1] for item in combination])
            probability = float(np.prod([item[2] for item in combination]))
            spreads = tuple(float(item[3]) for item in combination)
            patch = SparseContactPatch(
                graph_name=graph.name,
                center_nodes=centers,
                channel_node_weights=weights,
                metadata={"channel_spreads": list(spreads)},
            )
            joint_patches.append((patch, probability))
        joint_patches.sort(key=lambda item: (-item[1], item[0].patch_id))
        full_patch_count = len(joint_patches)
        retained = joint_patches[: self.maximum_joint_patches]
        retained_patch_mass = float(sum(probability for _, probability in retained))
        if retained_patch_mass <= 0.0:
            raise ValueError("retained patch support has zero prior mass")

        states: list[ContactPatchStateV2] = []
        state_weights: list[float] = []
        for patch, patch_probability in retained:
            for gain_index, delay_index, rotation_index in product(
                range(len(self.config.gain_values)),
                range(len(self.config.delay_values)),
                range(len(self.config.rotation_values_radians)),
            ):
                state = ContactPatchStateV2(
                    patch=patch,
                    gain_multiplier=self.config.gain_values[gain_index],
                    delay_steps=self.config.delay_values[delay_index],
                    rotation_radians=self.config.rotation_values_radians[
                        rotation_index
                    ],
                )
                states.append(state)
                state_weights.append(
                    patch_probability
                    * self.prior.gain_probabilities[gain_index]
                    * self.prior.delay_probabilities[delay_index]
                    * self.prior.rotation_probabilities[rotation_index]
                )
        return ContactPatchHypothesisSupportV2(
            states=tuple(states),
            prior_weights=np.asarray(state_weights, dtype=float),
            retained_patch_prior_mass=retained_patch_mass,
            full_patch_count=full_patch_count,
            retained_patch_count=len(retained),
        )


@dataclass(frozen=True)
class ContactV2SupportPolicy:
    parameter_support_method: SupportMethod = "weighted_coreset"
    maximum_parameter_count: int = 16
    minimum_represented_parameter_mass: float = 0.999
    minimum_retained_patch_prior_mass: float = 0.95
    maximum_parameter_mean_error_l2: float | None = None
    maximum_parameter_covariance_error_frobenius: float | None = None

    def __post_init__(self) -> None:
        if self.parameter_support_method not in {"top_mass", "weighted_coreset"}:
            raise ValueError("unknown parameter support method")
        maximum_count = _positive_int(
            self.maximum_parameter_count,
            name="maximum_parameter_count",
        )
        represented = _finite_float(
            self.minimum_represented_parameter_mass,
            name="minimum_represented_parameter_mass",
            minimum=np.finfo(float).eps,
            maximum=1.0,
        )
        patch_mass = _finite_float(
            self.minimum_retained_patch_prior_mass,
            name="minimum_retained_patch_prior_mass",
            minimum=np.finfo(float).eps,
            maximum=1.0,
        )
        mean_error = self.maximum_parameter_mean_error_l2
        covariance_error = self.maximum_parameter_covariance_error_frobenius
        if mean_error is not None:
            mean_error = _finite_float(
                mean_error,
                name="maximum_parameter_mean_error_l2",
                minimum=0.0,
            )
        if covariance_error is not None:
            covariance_error = _finite_float(
                covariance_error,
                name="maximum_parameter_covariance_error_frobenius",
                minimum=0.0,
            )
        object.__setattr__(self, "maximum_parameter_count", maximum_count)
        object.__setattr__(self, "minimum_represented_parameter_mass", represented)
        object.__setattr__(self, "minimum_retained_patch_prior_mass", patch_mass)
        object.__setattr__(self, "maximum_parameter_mean_error_l2", mean_error)
        object.__setattr__(
            self,
            "maximum_parameter_covariance_error_frobenius",
            covariance_error,
        )


@dataclass(frozen=True)
class ContactV2SupportDecision:
    accepted: bool
    reasons: tuple[str, ...]
    parameter_reduction: ParameterSupportReduction
    retained_patch_prior_mass: float
    policy: ContactV2SupportPolicy

    def __post_init__(self) -> None:
        reasons = tuple(self.reasons)
        if self.accepted and reasons:
            raise ValueError("accepted support decisions cannot have rejection reasons")
        if not self.accepted and not reasons:
            raise ValueError("rejected support decisions require a reason")
        if len(set(reasons)) != len(reasons):
            raise ValueError("support rejection reasons must be unique")
        object.__setattr__(self, "reasons", reasons)

    @property
    def decision_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": LATENT_CONTACT_V2_SCHEMA_VERSION,
                "artifact_kind": "Causal4DContactV2SupportDecision",
                "accepted": self.accepted,
                "reasons": list(self.reasons),
                "parameter_reduction": self.parameter_reduction.as_dict(),
                "retained_patch_prior_mass": self.retained_patch_prior_mass,
                "policy": asdict(self.policy),
            }
        )


def evaluate_contact_v2_support(
    reduction: ParameterSupportReduction,
    *,
    retained_patch_prior_mass: float,
    policy: ContactV2SupportPolicy,
) -> ContactV2SupportDecision:
    retained = _finite_float(
        retained_patch_prior_mass,
        name="retained_patch_prior_mass",
        minimum=np.finfo(float).eps,
        maximum=1.0,
    )
    reasons: list[str] = []
    if (
        reduction.represented_probability_mass
        < policy.minimum_represented_parameter_mass
    ):
        reasons.append("parameter_probability_mass_not_represented")
    if retained < policy.minimum_retained_patch_prior_mass:
        reasons.append("patch_prior_mass_not_retained")
    if (
        policy.maximum_parameter_mean_error_l2 is not None
        and reduction.mean_error_l2 > policy.maximum_parameter_mean_error_l2
    ):
        reasons.append("parameter_mean_error_exceeds_limit")
    if (
        policy.maximum_parameter_covariance_error_frobenius is not None
        and reduction.covariance_error_frobenius
        > policy.maximum_parameter_covariance_error_frobenius
    ):
        reasons.append("parameter_covariance_error_exceeds_limit")
    return ContactV2SupportDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        parameter_reduction=reduction,
        retained_patch_prior_mass=retained,
        policy=policy,
    )


class ContactV2SupportRejectedError(ValueError):
    """Raised before rollout simulation when finite support is inadmissible."""


@dataclass(frozen=True)
class ContactEffectPosteriorV2:
    patch_ids: tuple[str, ...]
    patch_weights: np.ndarray
    channel_node_mass: np.ndarray
    channel_node_covariance: np.ndarray
    expected_channel_support_size: np.ndarray

    def __post_init__(self) -> None:
        patch_ids = tuple(self.patch_ids)
        weights = _normalized_weights(
            self.patch_weights,
            name="patch_weights",
            expected_shape=(len(patch_ids),),
        )
        mass = readonly_array(self.channel_node_mass, dtype=float)
        covariance = readonly_array(self.channel_node_covariance, dtype=float)
        support_size = readonly_array(self.expected_channel_support_size, dtype=float)
        if mass.ndim != 2:
            raise ValueError("channel_node_mass must have shape (contact, node)")
        if covariance.shape != (mass.shape[0], mass.shape[1], mass.shape[1]):
            raise ValueError("channel_node_covariance has the wrong shape")
        if support_size.shape != (mass.shape[0],):
            raise ValueError("expected_channel_support_size has the wrong shape")
        if not np.allclose(np.sum(mass, axis=1), 1.0, atol=1e-12, rtol=1e-12):
            raise ValueError("channel node masses must sum to one")
        object.__setattr__(self, "patch_ids", patch_ids)
        object.__setattr__(self, "patch_weights", weights)
        object.__setattr__(self, "channel_node_mass", mass)
        object.__setattr__(self, "channel_node_covariance", covariance)
        object.__setattr__(self, "expected_channel_support_size", support_size)


def gaussian_mixture_quantiles(
    component_means: np.ndarray,
    component_variances: np.ndarray,
    component_weights: np.ndarray,
    probabilities: Sequence[float],
    *,
    iterations: int = 80,
) -> np.ndarray:
    """Invert marginal Gaussian-mixture CDFs by deterministic vectorized bisection."""

    means = np.asarray(component_means, dtype=float)
    variances = np.asarray(component_variances, dtype=float)
    if means.ndim < 2:
        raise ValueError("component_means must have a component axis and coordinates")
    try:
        variances = np.broadcast_to(variances, means.shape)
    except ValueError as error:
        raise ValueError("component variances cannot broadcast to means") from error
    weights = _normalized_weights(
        component_weights,
        name="component_weights",
        expected_shape=(means.shape[0],),
    )
    if np.any(~np.isfinite(means)) or np.any(~np.isfinite(variances)):
        raise ValueError("mixture components must be finite")
    if np.any(variances <= 0.0):
        raise ValueError("mixture component variances must be positive")
    values = tuple(
        _finite_float(
            probability,
            name="probability",
            minimum=np.finfo(float).eps,
            maximum=1.0 - np.finfo(float).eps,
        )
        for probability in probabilities
    )
    if not values:
        raise ValueError("at least one mixture probability is required")
    iteration_count = _positive_int(iterations, name="iterations")
    flat_means = means.reshape(means.shape[0], -1)
    flat_standard_deviations = np.sqrt(variances.reshape(variances.shape[0], -1))
    lower_seed = np.min(flat_means - 12.0 * flat_standard_deviations, axis=0)
    upper_seed = np.max(flat_means + 12.0 * flat_standard_deviations, axis=0)
    results: list[np.ndarray] = []
    for probability in values:
        lower = lower_seed.copy()
        upper = upper_seed.copy()
        for _ in range(iteration_count):
            midpoint = 0.5 * (lower + upper)
            cdf = np.sum(
                weights[:, None]
                * ndtr((midpoint[None, :] - flat_means) / flat_standard_deviations),
                axis=0,
            )
            lower = np.where(cdf < probability, midpoint, lower)
            upper = np.where(cdf >= probability, midpoint, upper)
        results.append(0.5 * (lower + upper))
    return np.stack(results).reshape((len(values), *means.shape[1:]))


@dataclass(frozen=True)
class ContactPatchRolloutBankV2:
    graph_object: GraphObject
    action: Action
    states: tuple[ContactPatchStateV2, ...]
    state_prior_weights: np.ndarray
    parameter_particles: np.ndarray
    parameter_weights: np.ndarray
    trajectories_m: np.ndarray
    variance_floor_m2: float
    confidence_level: float
    support_decision: ContactV2SupportDecision

    def __post_init__(self) -> None:
        states = tuple(self.states)
        if not states or len({state.state_id for state in states}) != len(states):
            raise ValueError("rollout states must be nonempty and unique")
        state_weights = _normalized_weights(
            self.state_prior_weights,
            name="state_prior_weights",
            expected_shape=(len(states),),
        )
        particles = readonly_array(self.parameter_particles, dtype=float)
        if particles.ndim != 2 or particles.shape[1] != 3:
            raise ValueError("parameter_particles must have shape (particle, 3)")
        parameter_weights = _normalized_weights(
            self.parameter_weights,
            name="parameter_weights",
            expected_shape=(len(particles),),
        )
        trajectories = readonly_array(self.trajectories_m, dtype=float)
        expected = (
            len(states),
            len(particles),
            self.action.frame_count,
            self.graph_object.node_count,
            2,
        )
        if trajectories.shape != expected or not np.all(np.isfinite(trajectories)):
            raise ValueError(f"trajectories_m must be finite with shape {expected}")
        variance_floor = _finite_float(
            self.variance_floor_m2,
            name="variance_floor_m2",
            minimum=np.finfo(float).eps,
        )
        confidence = _finite_float(
            self.confidence_level,
            name="confidence_level",
            minimum=np.finfo(float).eps,
            maximum=1.0 - np.finfo(float).eps,
        )
        if not self.support_decision.accepted:
            raise ValueError("a rollout bank cannot contain rejected finite support")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "state_prior_weights", state_weights)
        object.__setattr__(self, "parameter_particles", particles)
        object.__setattr__(self, "parameter_weights", parameter_weights)
        object.__setattr__(self, "trajectories_m", trajectories)
        object.__setattr__(self, "variance_floor_m2", variance_floor)
        object.__setattr__(self, "confidence_level", confidence)

    @property
    def prior_joint_weights(self) -> np.ndarray:
        return self.state_prior_weights[:, None] * self.parameter_weights[None, :]

    @property
    def bank_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": LATENT_CONTACT_V2_SCHEMA_VERSION,
                "artifact_kind": "Causal4DContactPatchRolloutBankV2",
                "graph_name": self.graph_object.name,
                "action_id": self.action.action_id,
                "state_ids": [state.state_id for state in self.states],
                "state_prior_weights_sha256": array_sha256(self.state_prior_weights),
                "parameter_particles_sha256": array_sha256(self.parameter_particles),
                "parameter_weights_sha256": array_sha256(self.parameter_weights),
                "trajectories_sha256": array_sha256(self.trajectories_m),
                "variance_floor_m2": self.variance_floor_m2,
                "confidence_level": self.confidence_level,
                "support_decision_id": self.support_decision.decision_id,
            }
        )

    def update_weights(
        self,
        evidence: ContactObservationEvidenceV2,
        *,
        prefix_frame_count: int,
        component_group_covariance_m2: Mapping[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, ContactLikelihoodV2Diagnostics]:
        component_variance = np.full_like(
            self.trajectories_m,
            self.variance_floor_m2,
            dtype=float,
        )
        return posterior_weights_from_contact_evidence_v2(
            self.prior_joint_weights,
            self.trajectories_m,
            evidence,
            prefix_frame_count=prefix_frame_count,
            component_variance_m2=component_variance,
            component_group_covariance_m2=component_group_covariance_m2,
        )

    def predictive_distribution(
        self,
        joint_weights: np.ndarray | None = None,
        *,
        method: str = "latent_contact_patch_v2",
        conditional_variance_multiplier: float = 1.0,
        include_intervals: bool = True,
    ) -> PredictiveDistribution:
        weights = _normalized_weights(
            self.prior_joint_weights if joint_weights is None else joint_weights,
            name="joint_weights",
            expected_shape=self.prior_joint_weights.shape,
        )
        multiplier = _finite_float(
            conditional_variance_multiplier,
            name="conditional_variance_multiplier",
            minimum=np.finfo(float).eps,
        )
        flat_weights = weights.reshape(-1)
        components = self.trajectories_m.reshape(
            -1,
            *self.trajectories_m.shape[-3:],
        )
        conditional_variance = np.full_like(
            components,
            multiplier * self.variance_floor_m2,
            dtype=float,
        )
        mean = np.sum(flat_weights[:, None, None, None] * components, axis=0)
        variance = np.sum(
            flat_weights[:, None, None, None]
            * (conditional_variance + np.square(components - mean[None, ...])),
            axis=0,
        )
        if not include_intervals:
            return PredictiveDistribution(method=method, mean=mean, variance=variance)
        tail = 0.5 * (1.0 - self.confidence_level)
        interval = gaussian_mixture_quantiles(
            components,
            conditional_variance,
            flat_weights,
            (tail, 1.0 - tail),
        )
        return PredictiveDistribution(
            method=method,
            mean=mean,
            variance=variance,
            interval_lower=interval[0],
            interval_upper=interval[1],
        )

    def effect_posterior(
        self,
        joint_weights: np.ndarray,
    ) -> ContactEffectPosteriorV2:
        weights = _normalized_weights(
            joint_weights,
            name="joint_weights",
            expected_shape=self.prior_joint_weights.shape,
        )
        state_weights = np.sum(weights, axis=1)
        patch_indices: dict[str, list[int]] = {}
        for index, state in enumerate(self.states):
            patch_indices.setdefault(state.patch.patch_id, []).append(index)
        patch_ids = tuple(sorted(patch_indices))
        patch_weights = np.asarray(
            [np.sum(state_weights[patch_indices[patch_id]]) for patch_id in patch_ids],
            dtype=float,
        )
        representative = [
            self.states[patch_indices[patch_id][0]].patch for patch_id in patch_ids
        ]
        patch_arrays = np.stack(
            [patch.channel_node_weights for patch in representative]
        )
        mean = np.einsum("p,pqn->qn", patch_weights, patch_arrays)
        centered = patch_arrays - mean[None, ...]
        covariance = np.einsum("p,pqi,pqj->qij", patch_weights, centered, centered)
        support_size = np.einsum(
            "p,pq->q",
            patch_weights,
            np.sum(patch_arrays > 0.0, axis=2),
        )
        return ContactEffectPosteriorV2(
            patch_ids=patch_ids,
            patch_weights=patch_weights,
            channel_node_mass=mean,
            channel_node_covariance=covariance,
            expected_channel_support_size=support_size,
        )


def build_contact_patch_rollout_bank_v2(
    graph_object: GraphObject,
    action: Action,
    posterior: ParameterPosterior,
    model: GraphContactPatchModelV2,
    *,
    simulator_config: SimulatorConfig,
    support_policy: ContactV2SupportPolicy,
    variance_floor_m2: float,
    confidence_level: float,
) -> ContactPatchRolloutBankV2:
    """Build v2 finite support after source-blind support admission."""

    reduction = reduce_parameter_support(
        posterior.particles,
        posterior.weights,
        maximum_count=support_policy.maximum_parameter_count,
        method=support_policy.parameter_support_method,
    )
    hypothesis_support = model.hypotheses(graph_object, action)
    decision = evaluate_contact_v2_support(
        reduction,
        retained_patch_prior_mass=hypothesis_support.retained_patch_prior_mass,
        policy=support_policy,
    )
    if not decision.accepted:
        raise ContactV2SupportRejectedError(
            "latent-contact-v2 support rejected: " + ", ".join(decision.reasons)
        )
    particles = posterior.particles[reduction.indices]
    trajectories = np.stack(
        [
            simulate_particles(
                graph_object,
                state.expanded_action(action),
                particles,
                state.condition(),
                simulator_config,
            )
            for state in hypothesis_support.states
        ],
        axis=0,
    )
    return ContactPatchRolloutBankV2(
        graph_object=graph_object,
        action=action,
        states=hypothesis_support.states,
        state_prior_weights=hypothesis_support.prior_weights,
        parameter_particles=particles,
        parameter_weights=reduction.weights,
        trajectories_m=trajectories,
        variance_floor_m2=variance_floor_m2,
        confidence_level=confidence_level,
        support_decision=decision,
    )


@dataclass(frozen=True)
class ContactV2Selection(Generic[BaselineT, CandidateT]):
    accepted: bool
    reasons: tuple[str, ...]
    baseline: BaselineT
    candidate: CandidateT
    deployed: BaselineT | CandidateT

    def __post_init__(self) -> None:
        expected = self.candidate if self.accepted else self.baseline
        if self.deployed is not expected:
            raise ValueError(
                "deployed object does not preserve exact selection identity"
            )
        if self.accepted and self.reasons:
            raise ValueError("accepted selections cannot contain rejection reasons")
        if not self.accepted and not self.reasons:
            raise ValueError("rejected selections require a reason")


def select_contact_v2_candidate(
    support_decision: ContactV2SupportDecision,
    *,
    baseline: BaselineT,
    candidate: CandidateT,
) -> ContactV2Selection[BaselineT, CandidateT]:
    """Apply support admission while preserving exact fallback object identity."""

    accepted = support_decision.accepted
    return ContactV2Selection(
        accepted=accepted,
        reasons=() if accepted else support_decision.reasons,
        baseline=baseline,
        candidate=candidate,
        deployed=candidate if accepted else baseline,
    )


__all__ = [
    "LATENT_CONTACT_V2_SCHEMA_VERSION",
    "ContactEndpoint",
    "ContactEffectPosteriorV2",
    "ContactLikelihoodV2Diagnostics",
    "ContactObservationEvidenceV2",
    "ContactPatchHypothesisSupportV2",
    "ContactPatchRolloutBankV2",
    "ContactPatchStateV2",
    "ContactV2Selection",
    "ContactV2SupportDecision",
    "ContactV2SupportPolicy",
    "ContactV2SupportRejectedError",
    "GraphContactPatchModelV2",
    "LinearContactObservationGroup",
    "SparseContactPatch",
    "build_contact_patch_rollout_bank_v2",
    "contact_component_log_likelihoods_v2",
    "evaluate_contact_v2_support",
    "gaussian_mixture_quantiles",
    "posterior_weights_from_contact_evidence_v2",
    "select_contact_v2_candidate",
]
