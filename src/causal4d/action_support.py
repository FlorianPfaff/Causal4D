"""Source-frozen support admission for action-conditioned counterfactuals.

The calibration consumes only source action/intervention feature trajectories.
Target decisions consume only the proposed query features and posterior component
weights. No object-response continuation is accepted by this module. Rejection
selects the caller-provided baseline object by identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Generic, Mapping, Sequence, TypeVar

import numpy as np

from causal4d.action_conditioned_discrepancy import (
    ActionConditionedDiscrepancyFeatures,
)
from causal4d.artifact_io import load_strict_json_object, read_regular_file
from causal4d.atomic_io import atomic_write_json
from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json


ACTION_SUPPORT_SCHEMA_VERSION = 1
_ACTION_SUPPORT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "calibration_id",
        "feature_names",
        "feature_schema",
        "candidate_model_id",
        "support_feature_names",
        "source_case_ids",
        "source_case_artifact_ids",
        "source_case_summaries",
        "source_feature_lower_bounds",
        "source_feature_upper_bounds",
        "summary_center",
        "summary_scale",
        "maximum_nearest_source_distance",
        "maximum_supported_distance",
        "feature_lower_bounds",
        "feature_upper_bounds",
        "support_margin",
        "scale_floor",
        "minimum_supported_component_mass",
        "source_futures_used",
        "target_futures_used",
    }
)

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
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _contains_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        return value.dtype.kind == "b"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_boolean(item) for item in value)
    return False


def _validated_numeric_array(value: Any, *, name: str) -> np.ndarray:
    if _contains_boolean(value):
        raise ValueError(f"{name} must contain numbers, not Booleans")
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must contain finite numbers")
    result = readonly_array(raw, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _validated_boolean_array(value: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind != "b":
        raise ValueError(f"{name} must contain Booleans")
    return readonly_array(raw, dtype=bool)


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    missing = sorted(expected - set(payload))
    extra = sorted(set(payload) - expected)
    if missing or extra:
        raise ValueError(f"{name} fields changed; missing={missing}, extra={extra}")


def _support_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(feature_names) + (
        "__step_duration_s",
        "__elapsed_time_s",
    )


def _feature_values(
    features: ActionConditionedDiscrepancyFeatures,
) -> np.ndarray:
    if not isinstance(features, ActionConditionedDiscrepancyFeatures):
        raise ValueError("features must be ActionConditionedDiscrepancyFeatures")
    values = np.asarray(features.values, dtype=float)
    if values.ndim == 2:
        values = values[None]
    elif values.ndim != 3:
        raise ValueError("action features must have shape (H, F) or (K, H, F)")
    durations = features.component_step_durations(values.shape[0])
    elapsed = np.cumsum(durations, axis=1)
    return np.concatenate(
        (values, durations[:, :, None], elapsed[:, :, None]),
        axis=2,
    )


def _normalized_component_weights(
    component_count: int,
    values: np.ndarray | Sequence[float] | None,
    *,
    name: str,
) -> np.ndarray:
    if values is None:
        if component_count != 1:
            raise ValueError(f"{name} are required for component-specific features")
        return np.ones(1, dtype=float)
    weights = _validated_numeric_array(values, name=name)
    if weights.shape != (component_count,):
        raise ValueError(f"{name} must have shape ({component_count},)")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not np.isclose(np.sum(weights), 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must sum to one")
    return weights


def _case_summary(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    horizon = values.shape[1]
    weighted = weights[:, None, None]
    mean = np.sum(values * weighted, axis=(0, 1)) / horizon
    rms = np.sqrt(np.sum(np.square(values) * weighted, axis=(0, 1)) / horizon)
    return np.concatenate((mean, rms))


def _component_summaries(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values, axis=1)
    rms = np.sqrt(np.mean(np.square(values), axis=1))
    return np.concatenate((mean, rms), axis=1)


def _positive_weight_bounds(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    supported = values[weights > 0.0]
    if not len(supported):
        raise ValueError("source case has no positive component support")
    flattened = supported.reshape(-1, values.shape[-1])
    return np.min(flattened, axis=0), np.max(flattened, axis=0)


def _derive_calibration(
    source_summaries: np.ndarray,
    source_lower_bounds: np.ndarray,
    source_upper_bounds: np.ndarray,
    *,
    support_margin: float,
    scale_floor: float,
) -> dict[str, np.ndarray | float]:
    center = np.median(source_summaries, axis=0)
    absolute_deviation = np.abs(source_summaries - center)
    mad_scale = 1.4826 * np.median(absolute_deviation, axis=0)
    half_range = 0.5 * np.ptp(source_summaries, axis=0)
    magnitude_floor = scale_floor * np.maximum(np.abs(center), 1.0)
    scale = np.maximum.reduce((mad_scale, half_range, magnitude_floor))

    standardized = (source_summaries[:, None] - source_summaries[None]) / scale
    pairwise = np.sqrt(np.mean(np.square(standardized), axis=2))
    np.fill_diagonal(pairwise, np.inf)
    nearest = np.min(pairwise, axis=1)
    maximum_nearest = max(float(np.max(nearest)), scale_floor)

    raw_lower = np.min(source_lower_bounds, axis=0)
    raw_upper = np.max(source_upper_bounds, axis=0)
    raw_half_width = 0.5 * (raw_upper - raw_lower)
    feature_center = 0.5 * (raw_lower + raw_upper)
    feature_floor = scale_floor * np.maximum(np.abs(feature_center), 1.0)
    expansion = (support_margin - 1.0) * np.maximum(
        raw_half_width,
        feature_floor,
    )
    return {
        "summary_center": center,
        "summary_scale": scale,
        "maximum_nearest_source_distance": maximum_nearest,
        "maximum_supported_distance": maximum_nearest * support_margin,
        "feature_lower_bounds": raw_lower - expansion,
        "feature_upper_bounds": raw_upper + expansion,
    }


def _allclose(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.allclose(left, right, atol=1e-12, rtol=1e-12))


@dataclass(frozen=True)
class ActionSupportSourceCase:
    """One source-only action-feature case used to define support."""

    case_id: str
    features: ActionConditionedDiscrepancyFeatures
    component_weights: np.ndarray | Sequence[float] | None = None

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id:
            raise ValueError("source action-support case_id must be nonempty")
        if not isinstance(self.features, ActionConditionedDiscrepancyFeatures):
            raise ValueError("features must be ActionConditionedDiscrepancyFeatures")
        values = _feature_values(self.features)
        weights = _normalized_component_weights(
            values.shape[0],
            self.component_weights,
            name="component_weights",
        )
        object.__setattr__(
            self,
            "component_weights",
            readonly_array(weights, dtype=float),
        )

    @property
    def artifact_id(self) -> str:
        component_ids = self.features.component_ids
        payload = {
            "schema_version": ACTION_SUPPORT_SCHEMA_VERSION,
            "artifact_kind": "Causal4DActionSupportSourceCase",
            "case_id": self.case_id,
            "feature_names": list(self.features.names),
            "feature_schema": self.features.schema_id,
            "support_feature_names": list(_support_feature_names(self.features.names)),
            "feature_values_sha256": array_sha256(self.features.values),
            "step_duration_sha256": array_sha256(
                np.asarray(self.features.step_duration_s, dtype=float)
            ),
            "component_ids": None if component_ids is None else list(component_ids),
            "component_weights_sha256": array_sha256(
                np.asarray(self.component_weights, dtype=float)
            ),
            "source_futures_used": False,
            "target_futures_used": False,
        }
        return _canonical_sha256(payload)

    def summary_and_bounds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = _feature_values(self.features)
        weights: np.ndarray = np.asarray(self.component_weights, dtype=float)
        lower, upper = _positive_weight_bounds(values, weights)
        return _case_summary(values, weights), lower, upper


@dataclass(frozen=True)
class ActionSupportCalibration:
    """Content-addressed source envelope for query-feature admission."""

    feature_names: tuple[str, ...]
    feature_schema: str
    candidate_model_id: str
    support_feature_names: tuple[str, ...]
    source_case_ids: tuple[str, ...]
    source_case_artifact_ids: tuple[str, ...]
    source_case_summaries: np.ndarray
    source_feature_lower_bounds: np.ndarray
    source_feature_upper_bounds: np.ndarray
    summary_center: np.ndarray
    summary_scale: np.ndarray
    maximum_nearest_source_distance: float
    maximum_supported_distance: float
    feature_lower_bounds: np.ndarray
    feature_upper_bounds: np.ndarray
    support_margin: float
    scale_floor: float
    minimum_supported_component_mass: float
    source_futures_used: bool = False
    target_futures_used: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.feature_names, (str, bytes)):
            raise ValueError("feature_names must be a sequence of strings")
        feature_names = tuple(self.feature_names)
        if not feature_names or len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names must be nonempty and unique")
        if any(type(value) is not str or not value for value in feature_names):
            raise ValueError("feature_names must contain nonempty strings")
        reserved_names = {"__step_duration_s", "__elapsed_time_s"}
        if reserved_names.intersection(feature_names):
            raise ValueError("feature_names use reserved support dimensions")
        if type(self.feature_schema) is not str or not self.feature_schema:
            raise ValueError("feature_schema must be nonempty")
        if type(self.candidate_model_id) is not str or not self.candidate_model_id:
            raise ValueError("candidate_model_id must be nonempty")
        if isinstance(self.support_feature_names, (str, bytes)):
            raise ValueError("support_feature_names must be a sequence of strings")
        support_names = tuple(self.support_feature_names)
        expected_support_names = _support_feature_names(feature_names)
        if support_names != expected_support_names:
            raise ValueError("support_feature_names do not match the support schema")

        if isinstance(self.source_case_ids, (str, bytes)):
            raise ValueError("source_case_ids must be a sequence of strings")
        if isinstance(self.source_case_artifact_ids, (str, bytes)):
            raise ValueError("source_case_artifact_ids must be a sequence of digests")
        case_ids = tuple(self.source_case_ids)
        artifact_ids = tuple(self.source_case_artifact_ids)
        if len(case_ids) < 3 or len(set(case_ids)) != len(case_ids):
            raise ValueError("at least three unique source cases are required")
        if case_ids != tuple(sorted(case_ids)):
            raise ValueError("source_case_ids must use canonical sorted order")
        if any(type(value) is not str or not value for value in case_ids):
            raise ValueError("source_case_ids must contain nonempty strings")
        if len(artifact_ids) != len(case_ids) or len(set(artifact_ids)) != len(
            artifact_ids
        ):
            raise ValueError("source artifact IDs must be aligned and unique")
        for index, value in enumerate(artifact_ids):
            _require_sha256(value, name=f"source_case_artifact_ids[{index}]")

        feature_count = len(support_names)
        case_count = len(case_ids)
        summaries = _validated_numeric_array(
            self.source_case_summaries,
            name="source_case_summaries",
        )
        source_lower = _validated_numeric_array(
            self.source_feature_lower_bounds,
            name="source_feature_lower_bounds",
        )
        source_upper = _validated_numeric_array(
            self.source_feature_upper_bounds,
            name="source_feature_upper_bounds",
        )
        center = _validated_numeric_array(
            self.summary_center,
            name="summary_center",
        )
        scale = _validated_numeric_array(
            self.summary_scale,
            name="summary_scale",
        )
        lower = _validated_numeric_array(
            self.feature_lower_bounds,
            name="feature_lower_bounds",
        )
        upper = _validated_numeric_array(
            self.feature_upper_bounds,
            name="feature_upper_bounds",
        )
        if summaries.shape != (case_count, 2 * feature_count):
            raise ValueError("source_case_summaries have invalid shape")
        if source_lower.shape != (case_count, feature_count) or source_upper.shape != (
            case_count,
            feature_count,
        ):
            raise ValueError("source feature bounds have invalid shape")
        if center.shape != (2 * feature_count,) or scale.shape != (2 * feature_count,):
            raise ValueError("summary center and scale have invalid shape")
        if lower.shape != (feature_count,) or upper.shape != (feature_count,):
            raise ValueError("feature bounds have invalid shape")
        arrays = (summaries, source_lower, source_upper, center, scale, lower, upper)
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("action-support calibration arrays must be finite")
        if np.any(source_lower > source_upper) or np.any(lower > upper):
            raise ValueError("action-support lower bounds must not exceed upper bounds")
        if np.any(scale <= 0.0):
            raise ValueError("summary_scale must be strictly positive")

        support_margin = _finite_float(
            self.support_margin,
            name="support_margin",
            minimum=1.0,
        )
        scale_floor = _finite_float(
            self.scale_floor,
            name="scale_floor",
            minimum=float(np.finfo(float).eps),
        )
        minimum_mass = _finite_float(
            self.minimum_supported_component_mass,
            name="minimum_supported_component_mass",
            minimum=0.0,
            maximum=1.0,
        )
        if minimum_mass <= 0.0:
            raise ValueError("minimum_supported_component_mass must be positive")
        maximum_nearest = _finite_float(
            self.maximum_nearest_source_distance,
            name="maximum_nearest_source_distance",
            minimum=0.0,
        )
        maximum_supported = _finite_float(
            self.maximum_supported_distance,
            name="maximum_supported_distance",
            minimum=0.0,
        )
        if type(self.source_futures_used) is not bool or self.source_futures_used:
            raise ValueError("source_futures_used must be false")
        if type(self.target_futures_used) is not bool or self.target_futures_used:
            raise ValueError("target_futures_used must be false")

        derived = _derive_calibration(
            summaries,
            source_lower,
            source_upper,
            support_margin=support_margin,
            scale_floor=scale_floor,
        )
        for name, supplied in (
            ("summary_center", center),
            ("summary_scale", scale),
            ("feature_lower_bounds", lower),
            ("feature_upper_bounds", upper),
        ):
            if not _allclose(supplied, np.asarray(derived[name])):
                raise ValueError(f"{name} does not match the source-derived value")
        if not np.isclose(
            maximum_nearest,
            float(derived["maximum_nearest_source_distance"]),
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError(
                "maximum_nearest_source_distance does not match source summaries"
            )
        if not np.isclose(
            maximum_supported,
            float(derived["maximum_supported_distance"]),
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("maximum_supported_distance does not match policy")

        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "support_feature_names", support_names)
        object.__setattr__(self, "source_case_ids", case_ids)
        object.__setattr__(self, "source_case_artifact_ids", artifact_ids)
        object.__setattr__(self, "source_case_summaries", summaries)
        object.__setattr__(self, "source_feature_lower_bounds", source_lower)
        object.__setattr__(self, "source_feature_upper_bounds", source_upper)
        object.__setattr__(self, "summary_center", center)
        object.__setattr__(self, "summary_scale", scale)
        object.__setattr__(self, "feature_lower_bounds", lower)
        object.__setattr__(self, "feature_upper_bounds", upper)
        object.__setattr__(self, "support_margin", support_margin)
        object.__setattr__(self, "scale_floor", scale_floor)
        object.__setattr__(self, "minimum_supported_component_mass", minimum_mass)
        object.__setattr__(
            self,
            "maximum_nearest_source_distance",
            maximum_nearest,
        )
        object.__setattr__(self, "maximum_supported_distance", maximum_supported)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": ACTION_SUPPORT_SCHEMA_VERSION,
            "artifact_kind": "Causal4DActionSupportCalibration",
            "feature_names": list(self.feature_names),
            "feature_schema": self.feature_schema,
            "candidate_model_id": self.candidate_model_id,
            "support_feature_names": list(self.support_feature_names),
            "source_case_ids": list(self.source_case_ids),
            "source_case_artifact_ids": list(self.source_case_artifact_ids),
            "source_case_summaries": self.source_case_summaries.tolist(),
            "source_feature_lower_bounds": (self.source_feature_lower_bounds.tolist()),
            "source_feature_upper_bounds": (self.source_feature_upper_bounds.tolist()),
            "summary_center": self.summary_center.tolist(),
            "summary_scale": self.summary_scale.tolist(),
            "maximum_nearest_source_distance": (self.maximum_nearest_source_distance),
            "maximum_supported_distance": self.maximum_supported_distance,
            "feature_lower_bounds": self.feature_lower_bounds.tolist(),
            "feature_upper_bounds": self.feature_upper_bounds.tolist(),
            "support_margin": self.support_margin,
            "scale_floor": self.scale_floor,
            "minimum_supported_component_mass": (self.minimum_supported_component_mass),
            "source_futures_used": self.source_futures_used,
            "target_futures_used": self.target_futures_used,
        }

    @property
    def calibration_id(self) -> str:
        return _canonical_sha256(self._payload())

    def as_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["calibration_id"] = self.calibration_id
        return payload


@dataclass(frozen=True)
class ActionSupportDecision:
    """Target-safe decision over posterior component action support."""

    calibration_id: str
    target_input_id: str
    candidate_model_id: str
    component_ids: tuple[str, ...]
    component_distances: np.ndarray
    maximum_supported_distance: float
    component_within_summary_distance: np.ndarray
    component_within_feature_bounds: np.ndarray
    component_supported: np.ndarray
    component_weights: np.ndarray
    supported_component_mass: float
    minimum_supported_component_mass: float
    accepted: bool
    rejection_reasons: tuple[str, ...]
    future_observation_frames_read: int = 0

    def __post_init__(self) -> None:
        _require_sha256(self.calibration_id, name="calibration_id")
        _require_sha256(self.target_input_id, name="target_input_id")
        if type(self.candidate_model_id) is not str or not self.candidate_model_id:
            raise ValueError("candidate_model_id must be nonempty")
        if isinstance(self.component_ids, (str, bytes)):
            raise ValueError("component_ids must be a sequence of strings")
        component_ids = tuple(self.component_ids)
        if not component_ids or len(set(component_ids)) != len(component_ids):
            raise ValueError("component_ids must be nonempty and unique")
        if any(type(value) is not str or not value for value in component_ids):
            raise ValueError("component_ids must contain nonempty strings")
        count = len(component_ids)
        distances = _validated_numeric_array(
            self.component_distances,
            name="component_distances",
        )
        maximum_distance = _finite_float(
            self.maximum_supported_distance,
            name="maximum_supported_distance",
            minimum=0.0,
        )
        within_distance = _validated_boolean_array(
            self.component_within_summary_distance,
            name="component_within_summary_distance",
        )
        within = _validated_boolean_array(
            self.component_within_feature_bounds,
            name="component_within_feature_bounds",
        )
        supported = _validated_boolean_array(
            self.component_supported,
            name="component_supported",
        )
        weights = _validated_numeric_array(
            self.component_weights,
            name="component_weights",
        )
        arrays = (distances, within_distance, within, supported, weights)
        if any(value.shape != (count,) for value in arrays):
            raise ValueError("component decision arrays must match component_ids")
        if not np.all(np.isfinite(distances)) or np.any(distances < 0.0):
            raise ValueError("component distances must be finite and nonnegative")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("component weights must be finite and nonnegative")
        if not np.isclose(np.sum(weights), 1.0, atol=1e-12, rtol=1e-12):
            raise ValueError("component weights must sum to one")
        expected_within_distance = distances <= maximum_distance + 1e-12
        if not np.array_equal(within_distance, expected_within_distance):
            raise ValueError(
                "component_within_summary_distance does not match distances"
            )
        if not np.array_equal(supported, within & within_distance):
            raise ValueError("component_supported does not match support checks")
        supported_mass = _finite_float(
            self.supported_component_mass,
            name="supported_component_mass",
            minimum=0.0,
            maximum=1.0,
        )
        expected_mass = float(np.sum(weights[supported]))
        if not np.isclose(supported_mass, expected_mass, atol=1e-12, rtol=1e-12):
            raise ValueError("supported_component_mass does not match components")
        minimum_mass = _finite_float(
            self.minimum_supported_component_mass,
            name="minimum_supported_component_mass",
            minimum=0.0,
            maximum=1.0,
        )
        if minimum_mass <= 0.0:
            raise ValueError("minimum_supported_component_mass must be positive")
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be Boolean")
        expected_accepted = supported_mass + 1e-12 >= minimum_mass
        if self.accepted != expected_accepted:
            raise ValueError("accepted does not match supported component mass")
        if isinstance(self.rejection_reasons, (str, bytes)):
            raise ValueError("rejection_reasons must be a sequence of strings")
        reasons = tuple(self.rejection_reasons)
        if self.accepted and reasons:
            raise ValueError("accepted decisions must not contain rejection reasons")
        if not self.accepted and not reasons:
            raise ValueError("rejected decisions must explain their rejection")
        if any(type(value) is not str or not value for value in reasons):
            raise ValueError("rejection reasons must be nonempty strings")
        if len(set(reasons)) != len(reasons):
            raise ValueError("rejection reasons must be unique")
        if type(self.future_observation_frames_read) is not int or (
            self.future_observation_frames_read != 0
        ):
            raise ValueError("future_observation_frames_read must be zero")
        object.__setattr__(self, "component_ids", component_ids)
        object.__setattr__(self, "component_distances", distances)
        object.__setattr__(self, "maximum_supported_distance", maximum_distance)
        object.__setattr__(
            self,
            "component_within_summary_distance",
            within_distance,
        )
        object.__setattr__(self, "component_within_feature_bounds", within)
        object.__setattr__(self, "component_supported", supported)
        object.__setattr__(self, "component_weights", weights)
        object.__setattr__(self, "supported_component_mass", supported_mass)
        object.__setattr__(self, "minimum_supported_component_mass", minimum_mass)
        object.__setattr__(self, "rejection_reasons", reasons)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": ACTION_SUPPORT_SCHEMA_VERSION,
            "artifact_kind": "Causal4DActionSupportDecision",
            "calibration_id": self.calibration_id,
            "target_input_id": self.target_input_id,
            "candidate_model_id": self.candidate_model_id,
            "component_ids": list(self.component_ids),
            "component_distances": self.component_distances.tolist(),
            "maximum_supported_distance": self.maximum_supported_distance,
            "component_within_summary_distance": (
                self.component_within_summary_distance.tolist()
            ),
            "component_within_feature_bounds": (
                self.component_within_feature_bounds.tolist()
            ),
            "component_supported": self.component_supported.tolist(),
            "component_weights": self.component_weights.tolist(),
            "supported_component_mass": self.supported_component_mass,
            "minimum_supported_component_mass": (self.minimum_supported_component_mass),
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "future_observation_frames_read": self.future_observation_frames_read,
        }

    @property
    def decision_id(self) -> str:
        return _canonical_sha256(self._payload())

    def as_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["decision_id"] = self.decision_id
        return payload


@dataclass(frozen=True)
class ActionSupportSelection(Generic[BaselineT, CandidateT]):
    """An accepted candidate or the exact caller-provided baseline object."""

    baseline: BaselineT
    candidate: CandidateT
    decision: ActionSupportDecision
    deployed: BaselineT | CandidateT

    def __post_init__(self) -> None:
        expected = self.candidate if self.decision.accepted else self.baseline
        if self.deployed is not expected:
            raise ValueError("deployed object does not match the support decision")


def fit_action_support_calibration(
    source_cases: Sequence[ActionSupportSourceCase],
    *,
    candidate_model_id: str,
    support_margin: float = 1.10,
    scale_floor: float = 1.0e-9,
    minimum_supported_component_mass: float = 0.95,
) -> ActionSupportCalibration:
    """Fit a deterministic source-only action-support envelope."""

    supplied_cases = tuple(source_cases)
    if any(type(case) is not ActionSupportSourceCase for case in supplied_cases):
        raise ValueError("source_cases must contain ActionSupportSourceCase values")
    cases = tuple(sorted(supplied_cases, key=lambda case: case.case_id))
    if len(cases) < 3:
        raise ValueError("at least three source cases are required")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("source case IDs must be unique")
    feature_names = cases[0].features.names
    feature_schema = cases[0].features.schema_id
    for case in cases[1:]:
        if case.features.names != feature_names:
            raise ValueError("source action-support feature names differ")
        if case.features.schema_id != feature_schema:
            raise ValueError("source action-support feature schemas differ")

    if type(candidate_model_id) is not str or not candidate_model_id:
        raise ValueError("candidate_model_id must be nonempty")
    support_margin = _finite_float(
        support_margin,
        name="support_margin",
        minimum=1.0,
    )
    scale_floor = _finite_float(
        scale_floor,
        name="scale_floor",
        minimum=float(np.finfo(float).eps),
    )
    minimum_mass = _finite_float(
        minimum_supported_component_mass,
        name="minimum_supported_component_mass",
        minimum=0.0,
        maximum=1.0,
    )
    if minimum_mass <= 0.0:
        raise ValueError("minimum_supported_component_mass must be positive")

    summaries = []
    lower_bounds = []
    upper_bounds = []
    for case in cases:
        summary, lower, upper = case.summary_and_bounds()
        summaries.append(summary)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
    summary_array = np.stack(summaries)
    lower_array = np.stack(lower_bounds)
    upper_array = np.stack(upper_bounds)
    derived = _derive_calibration(
        summary_array,
        lower_array,
        upper_array,
        support_margin=support_margin,
        scale_floor=scale_floor,
    )
    return ActionSupportCalibration(
        feature_names=tuple(feature_names),
        feature_schema=feature_schema,
        candidate_model_id=candidate_model_id,
        support_feature_names=_support_feature_names(feature_names),
        source_case_ids=tuple(case.case_id for case in cases),
        source_case_artifact_ids=tuple(case.artifact_id for case in cases),
        source_case_summaries=summary_array,
        source_feature_lower_bounds=lower_array,
        source_feature_upper_bounds=upper_array,
        summary_center=np.asarray(derived["summary_center"]),
        summary_scale=np.asarray(derived["summary_scale"]),
        maximum_nearest_source_distance=float(
            derived["maximum_nearest_source_distance"]
        ),
        maximum_supported_distance=float(derived["maximum_supported_distance"]),
        feature_lower_bounds=np.asarray(derived["feature_lower_bounds"]),
        feature_upper_bounds=np.asarray(derived["feature_upper_bounds"]),
        support_margin=support_margin,
        scale_floor=scale_floor,
        minimum_supported_component_mass=minimum_mass,
    )


def _target_input_id(
    features: ActionConditionedDiscrepancyFeatures,
    component_ids: tuple[str, ...],
    component_weights: np.ndarray,
    candidate_model_id: str,
) -> str:
    payload = {
        "schema_version": ACTION_SUPPORT_SCHEMA_VERSION,
        "artifact_kind": "Causal4DActionSupportInput",
        "feature_names": list(features.names),
        "feature_schema": features.schema_id,
        "candidate_model_id": candidate_model_id,
        "support_feature_names": list(_support_feature_names(features.names)),
        "feature_values_sha256": array_sha256(features.values),
        "step_duration_sha256": array_sha256(
            np.asarray(features.step_duration_s, dtype=float)
        ),
        "component_ids": list(component_ids),
        "component_weights_sha256": array_sha256(component_weights),
        "future_observation_frames_read": 0,
    }
    return _canonical_sha256(payload)


def evaluate_action_support(
    calibration: ActionSupportCalibration,
    features: ActionConditionedDiscrepancyFeatures,
    *,
    candidate_model_id: str,
    component_weights: np.ndarray | Sequence[float] | None = None,
    component_ids: Sequence[str] | None = None,
) -> ActionSupportDecision:
    """Evaluate query support without accepting any object-response future."""

    if type(candidate_model_id) is not str or not candidate_model_id:
        raise ValueError("candidate_model_id must be nonempty")
    if candidate_model_id != calibration.candidate_model_id:
        raise ValueError("candidate model does not match the support calibration")
    if tuple(features.names) != calibration.feature_names:
        raise ValueError("target feature names do not match the calibration")
    if features.schema_id != calibration.feature_schema:
        raise ValueError("target feature schema does not match the calibration")
    values = _feature_values(features)
    count = values.shape[0]
    weights = _normalized_component_weights(
        count,
        component_weights,
        name="component_weights",
    )
    if component_ids is None:
        if features.component_ids is not None:
            identifiers = tuple(features.component_ids)
        else:
            identifiers = tuple(f"component-{index}" for index in range(count))
    else:
        if isinstance(component_ids, (str, bytes)):
            raise ValueError("component_ids must be a sequence of strings")
        identifiers = tuple(component_ids)
    if len(identifiers) != count or len(set(identifiers)) != count:
        raise ValueError("component_ids must uniquely identify every component")
    if (
        features.component_ids is not None
        and tuple(features.component_ids) != identifiers
    ):
        raise ValueError("component_ids disagree with the feature artifact")

    summaries = _component_summaries(values)
    source = calibration.source_case_summaries
    standardized = (summaries[:, None] - source[None]) / calibration.summary_scale
    distances = np.min(
        np.sqrt(np.mean(np.square(standardized), axis=2)),
        axis=1,
    )
    tolerance = 1e-12
    lower = calibration.feature_lower_bounds[None, None]
    upper = calibration.feature_upper_bounds[None, None]
    within_bounds = np.all(
        (values >= lower - tolerance) & (values <= upper + tolerance),
        axis=(1, 2),
    )
    within_distance = distances <= calibration.maximum_supported_distance + tolerance
    supported = within_bounds & within_distance
    supported_mass = float(np.sum(weights[supported]))
    accepted = supported_mass + tolerance >= (
        calibration.minimum_supported_component_mass
    )
    reasons: list[str] = []
    if not accepted:
        reasons.append("insufficient_supported_component_mass")
        if np.any(~within_distance & (weights > 0.0)):
            reasons.append("summary_distance_out_of_support")
        if np.any(~within_bounds & (weights > 0.0)):
            reasons.append("feature_value_out_of_support")
    return ActionSupportDecision(
        calibration_id=calibration.calibration_id,
        target_input_id=_target_input_id(
            features,
            identifiers,
            weights,
            candidate_model_id,
        ),
        candidate_model_id=candidate_model_id,
        component_ids=identifiers,
        component_distances=distances,
        maximum_supported_distance=calibration.maximum_supported_distance,
        component_within_summary_distance=within_distance,
        component_within_feature_bounds=within_bounds,
        component_supported=supported,
        component_weights=weights,
        supported_component_mass=supported_mass,
        minimum_supported_component_mass=(calibration.minimum_supported_component_mass),
        accepted=accepted,
        rejection_reasons=tuple(reasons),
    )


def select_action_supported_candidate(
    calibration: ActionSupportCalibration,
    features: ActionConditionedDiscrepancyFeatures,
    *,
    baseline: BaselineT,
    candidate: CandidateT,
    candidate_model_id: str,
    component_weights: np.ndarray | Sequence[float] | None = None,
    component_ids: Sequence[str] | None = None,
) -> ActionSupportSelection[BaselineT, CandidateT]:
    """Select the candidate or preserve the exact baseline object on rejection."""

    decision = evaluate_action_support(
        calibration,
        features,
        candidate_model_id=candidate_model_id,
        component_weights=component_weights,
        component_ids=component_ids,
    )
    deployed: BaselineT | CandidateT = candidate if decision.accepted else baseline
    return ActionSupportSelection(
        baseline=baseline,
        candidate=candidate,
        decision=decision,
        deployed=deployed,
    )


def write_action_support_calibration(
    path: str | Path,
    calibration: ActionSupportCalibration,
    *,
    overwrite: bool = False,
) -> None:
    """Publish one source-only calibration atomically and verify the result."""

    atomic_write_json(path, calibration.as_dict(), overwrite=overwrite)
    loaded = load_action_support_calibration(path)
    if loaded.calibration_id != calibration.calibration_id:
        raise RuntimeError("published action-support calibration changed identity")


def load_action_support_calibration(
    path: str | Path,
) -> ActionSupportCalibration:
    """Load and independently reconstruct one action-support calibration."""

    snapshot = read_regular_file(path, name="action-support calibration")
    payload = load_strict_json_object(
        snapshot.payload,
        name="action-support calibration",
    )
    _require_exact_fields(
        payload,
        _ACTION_SUPPORT_FIELDS,
        name="action-support calibration",
    )
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != ACTION_SUPPORT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported action-support calibration schema")
    if (
        type(payload["artifact_kind"]) is not str
        or payload["artifact_kind"] != "Causal4DActionSupportCalibration"
    ):
        raise ValueError("unexpected action-support artifact kind")
    expected_id = _require_sha256(
        payload["calibration_id"],
        name="calibration_id",
    )
    calibration = ActionSupportCalibration(
        feature_names=payload["feature_names"],
        feature_schema=payload["feature_schema"],
        candidate_model_id=payload["candidate_model_id"],
        support_feature_names=payload["support_feature_names"],
        source_case_ids=payload["source_case_ids"],
        source_case_artifact_ids=payload["source_case_artifact_ids"],
        source_case_summaries=payload["source_case_summaries"],
        source_feature_lower_bounds=payload["source_feature_lower_bounds"],
        source_feature_upper_bounds=payload["source_feature_upper_bounds"],
        summary_center=payload["summary_center"],
        summary_scale=payload["summary_scale"],
        maximum_nearest_source_distance=payload["maximum_nearest_source_distance"],
        maximum_supported_distance=payload["maximum_supported_distance"],
        feature_lower_bounds=payload["feature_lower_bounds"],
        feature_upper_bounds=payload["feature_upper_bounds"],
        support_margin=payload["support_margin"],
        scale_floor=payload["scale_floor"],
        minimum_supported_component_mass=payload["minimum_supported_component_mass"],
        source_futures_used=payload["source_futures_used"],
        target_futures_used=payload["target_futures_used"],
    )
    if calibration.calibration_id != expected_id:
        raise ValueError("action-support calibration ID does not match its contents")
    return calibration


def load_claim_bearing_action_support_calibration(
    path: str | Path,
    *,
    expected_calibration_id: str,
) -> ActionSupportCalibration:
    """Load a calibration and require an independently frozen identity."""

    expected = _require_sha256(
        expected_calibration_id,
        name="expected_calibration_id",
    )
    calibration = load_action_support_calibration(path)
    if calibration.calibration_id != expected:
        raise ValueError(
            "action-support calibration differs from the frozen expected identity"
        )
    return calibration


__all__ = [
    "ACTION_SUPPORT_SCHEMA_VERSION",
    "ActionSupportCalibration",
    "ActionSupportDecision",
    "ActionSupportSelection",
    "ActionSupportSourceCase",
    "evaluate_action_support",
    "fit_action_support_calibration",
    "load_action_support_calibration",
    "load_claim_bearing_action_support_calibration",
    "select_action_supported_candidate",
    "write_action_support_calibration",
]
