"""Source-calibrated baseline-relative regret control for Causal4D queries.

Source cases may use held-out source outcomes to record candidate-versus-baseline
loss. Target decisions consume only a frozen feature vector, immutable artifact
identities, and prerequisite gate decisions. Rejection preserves the exact
caller-provided baseline object.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Generic, Literal, Mapping, Sequence, TypeVar

import numpy as np

from causal4d.artifact_io import load_strict_json_object, read_regular_file
from causal4d.atomic_io import atomic_write_json
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json


COUNTERFACTUAL_REGRET_SCHEMA_VERSION = 1
COUNTERFACTUAL_REGRET_ENDPOINTS = (
    "factual_continuation",
    "same_grasp_transfer",
    "new_contact_transfer",
)
METRIC_DIRECTIONS = ("lower", "higher")

_FEATURE_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "feature_id",
        "names",
        "values",
    }
)
_PREREQUISITE_FIELDS = frozenset({"name", "decision_id", "accepted"})
_SOURCE_CASE_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "source_case_artifact_id",
        "case_id",
        "session_id",
        "protocol_id",
        "endpoint",
        "baseline_role",
        "candidate_role",
        "metric_id",
        "metric_unit",
        "metric_direction",
        "baseline_artifact_id",
        "candidate_artifact_id",
        "features",
        "baseline_loss",
        "candidate_loss",
        "source_future_outcomes_used",
        "target_future_outcomes_used",
    }
)
_CERTIFICATE_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "certificate_id",
        "protocol_id",
        "endpoint",
        "baseline_role",
        "candidate_role",
        "metric_id",
        "metric_unit",
        "metric_direction",
        "feature_names",
        "source_cases",
        "required_prerequisite_names",
        "local_session_count",
        "support_margin",
        "scale_floor",
        "harmful_relative_regret_threshold",
        "minimum_global_mean_relative_improvement",
        "minimum_global_win_fraction",
        "maximum_global_harmful_fraction",
        "maximum_global_worst_relative_regret",
        "minimum_local_mean_relative_improvement",
        "minimum_local_win_fraction",
        "maximum_local_harmful_fraction",
        "maximum_local_worst_relative_regret",
        "feature_center",
        "feature_scale",
        "maximum_nearest_session_distance",
        "maximum_supported_distance",
        "global_mean_relative_improvement",
        "global_win_fraction",
        "global_harmful_fraction",
        "global_worst_relative_regret",
        "candidate_enabled",
        "source_future_outcomes_used",
        "target_future_outcomes_used",
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


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


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


def _probability(value: Any, *, name: str) -> float:
    return _finite_float(value, name=name, minimum=0.0, maximum=1.0)


def _strictly_below(value: float, threshold: float) -> bool:
    return bool(
        value < threshold and not np.isclose(value, threshold, rtol=1e-12, atol=1e-15)
    )


def _strictly_above(value: float, threshold: float) -> bool:
    return bool(
        value > threshold and not np.isclose(value, threshold, rtol=1e-12, atol=1e-15)
    )


def _positive_integer(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_string_tuple(
    values: Any,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(
        _require_nonempty_string(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not result and not allow_empty:
        raise ValueError(f"{name} must be nonempty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique strings")
    return result


def _contains_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        return value.dtype.kind == "b"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_boolean(item) for item in value)
    return False


def _validated_numeric_vector(value: Any, *, name: str) -> np.ndarray:
    if _contains_boolean(value):
        raise ValueError(f"{name} must contain numbers, not Booleans")
    raw = np.asarray(value)
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name} must contain finite numbers")
    result = readonly_array(raw, dtype=float)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _require_array_match(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    value = _validated_numeric_vector(actual, name=name)
    if value.shape != expected.shape or not np.allclose(
        value,
        expected,
        rtol=1e-13,
        atol=1e-15,
    ):
        raise ValueError(f"{name} does not match the stored source cases")
    return value


def _require_float_match(actual: Any, expected: float, *, name: str) -> float:
    value = _finite_float(actual, name=name)
    if not np.isclose(value, expected, rtol=1e-13, atol=1e-15):
        raise ValueError(f"{name} does not match the stored source cases")
    return value


def _relative_improvement(
    baseline_loss: float,
    candidate_loss: float,
    *,
    metric_direction: str,
) -> float:
    denominator = max(abs(baseline_loss), 1e-12)
    if metric_direction == "lower":
        return (baseline_loss - candidate_loss) / denominator
    if metric_direction == "higher":
        return (candidate_loss - baseline_loss) / denominator
    raise ValueError("metric_direction must be lower or higher")


def _standardized_distances(
    feature: np.ndarray,
    source_features: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    standardized = (source_features - feature[None]) / scale[None]
    return np.sqrt(np.mean(np.square(standardized), axis=1))


@dataclass(frozen=True)
class CounterfactualRegretFeatures:
    """A fixed target-safe feature vector used by one regret protocol."""

    names: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        names = _validated_string_tuple(self.names, name="regret feature names")
        values = _validated_numeric_vector(self.values, name="regret feature values")
        if values.shape != (len(names),):
            raise ValueError("regret feature values must match feature names")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> CounterfactualRegretFeatures:
        """Create a deterministically ordered feature vector from finite scalars."""

        if not isinstance(values, Mapping) or not values:
            raise ValueError("regret features must be a nonempty mapping")
        if any(type(name) is not str or not name for name in values):
            raise ValueError("regret feature names must be nonempty strings")
        names = tuple(sorted(values))
        vector: np.ndarray = np.asarray(
            [
                _finite_float(values[name], name=f"regret feature {name!r}")
                for name in names
            ],
            dtype=float,
        )
        return cls(names=names, values=vector)

    @property
    def feature_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": COUNTERFACTUAL_REGRET_SCHEMA_VERSION,
                "artifact_kind": "Causal4DRegretFeaturesV1",
                "names": list(self.names),
                "values": self.values.tolist(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": COUNTERFACTUAL_REGRET_SCHEMA_VERSION,
            "artifact_kind": "Causal4DRegretFeaturesV1",
            "feature_id": self.feature_id,
            "names": list(self.names),
            "values": self.values.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CounterfactualRegretFeatures:
        _require_exact_fields(payload, _FEATURE_FIELDS, name="regret features")
        if payload["schema_version"] != COUNTERFACTUAL_REGRET_SCHEMA_VERSION:
            raise ValueError("unsupported regret feature schema")
        if payload["artifact_kind"] != "Causal4DRegretFeaturesV1":
            raise ValueError("unexpected regret feature artifact kind")
        expected_id = _require_sha256(payload["feature_id"], name="feature_id")
        features = cls(
            names=tuple(payload["names"]),
            values=np.asarray(payload["values"]),
        )
        if features.feature_id != expected_id:
            raise ValueError("regret feature checksum mismatch")
        return features


@dataclass(frozen=True)
class CounterfactualRegretPrerequisite:
    """One independently content-addressed upstream decision."""

    name: str
    decision_id: str
    accepted: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _require_nonempty_string(self.name, name="prerequisite name"),
        )
        _require_sha256(self.decision_id, name="prerequisite decision_id")
        if type(self.accepted) is not bool:
            raise ValueError("prerequisite accepted must be Boolean")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "decision_id": self.decision_id,
            "accepted": self.accepted,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> CounterfactualRegretPrerequisite:
        _require_exact_fields(
            payload,
            _PREREQUISITE_FIELDS,
            name="regret prerequisite",
        )
        return cls(
            name=payload["name"],
            decision_id=payload["decision_id"],
            accepted=payload["accepted"],
        )


@dataclass(frozen=True)
class CounterfactualRegretSourceCase:
    """One source case with target-safe features and held-out source loss."""

    case_id: str
    session_id: str
    protocol_id: str
    endpoint: Literal[
        "factual_continuation",
        "same_grasp_transfer",
        "new_contact_transfer",
    ]
    baseline_role: str
    candidate_role: str
    metric_id: str
    metric_unit: str
    metric_direction: Literal["lower", "higher"]
    baseline_artifact_id: str
    candidate_artifact_id: str
    features: CounterfactualRegretFeatures
    baseline_loss: float
    candidate_loss: float
    source_future_outcomes_used: bool = True
    target_future_outcomes_used: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "case_id",
            "session_id",
            "protocol_id",
            "baseline_role",
            "candidate_role",
            "metric_id",
            "metric_unit",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_nonempty_string(
                    getattr(self, field_name),
                    name=field_name,
                ),
            )
        if self.endpoint not in COUNTERFACTUAL_REGRET_ENDPOINTS:
            raise ValueError("unsupported counterfactual regret endpoint")
        if self.metric_direction not in METRIC_DIRECTIONS:
            raise ValueError("metric_direction must be lower or higher")
        if self.baseline_role == self.candidate_role:
            raise ValueError("baseline_role and candidate_role must differ")
        _require_sha256(self.baseline_artifact_id, name="baseline_artifact_id")
        _require_sha256(self.candidate_artifact_id, name="candidate_artifact_id")
        if not isinstance(self.features, CounterfactualRegretFeatures):
            raise ValueError("features must be CounterfactualRegretFeatures")
        object.__setattr__(
            self,
            "baseline_loss",
            _finite_float(self.baseline_loss, name="baseline_loss"),
        )
        object.__setattr__(
            self,
            "candidate_loss",
            _finite_float(self.candidate_loss, name="candidate_loss"),
        )
        if (
            self.source_future_outcomes_used is not True
            or self.target_future_outcomes_used is not False
        ):
            raise ValueError("source-case information-boundary flags are invalid")

    @property
    def relative_improvement(self) -> float:
        return _relative_improvement(
            self.baseline_loss,
            self.candidate_loss,
            metric_direction=self.metric_direction,
        )

    @property
    def source_case_artifact_id(self) -> str:
        return _canonical_sha256(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": COUNTERFACTUAL_REGRET_SCHEMA_VERSION,
            "artifact_kind": "Causal4DRegretSourceCaseV1",
            "case_id": self.case_id,
            "session_id": self.session_id,
            "protocol_id": self.protocol_id,
            "endpoint": self.endpoint,
            "baseline_role": self.baseline_role,
            "candidate_role": self.candidate_role,
            "metric_id": self.metric_id,
            "metric_unit": self.metric_unit,
            "metric_direction": self.metric_direction,
            "baseline_artifact_id": self.baseline_artifact_id,
            "candidate_artifact_id": self.candidate_artifact_id,
            "features": self.features.as_dict(),
            "baseline_loss": self.baseline_loss,
            "candidate_loss": self.candidate_loss,
            "source_future_outcomes_used": self.source_future_outcomes_used,
            "target_future_outcomes_used": self.target_future_outcomes_used,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._payload(),
            "source_case_artifact_id": self.source_case_artifact_id,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> CounterfactualRegretSourceCase:
        _require_exact_fields(
            payload,
            _SOURCE_CASE_FIELDS,
            name="counterfactual regret source case",
        )
        if payload["schema_version"] != COUNTERFACTUAL_REGRET_SCHEMA_VERSION:
            raise ValueError("unsupported counterfactual regret source-case schema")
        if payload["artifact_kind"] != "Causal4DRegretSourceCaseV1":
            raise ValueError("unexpected counterfactual regret source-case kind")
        expected_id = _require_sha256(
            payload["source_case_artifact_id"],
            name="source_case_artifact_id",
        )
        features_payload = payload["features"]
        if not isinstance(features_payload, Mapping):
            raise ValueError("source-case features must be a JSON object")
        case = cls(
            case_id=payload["case_id"],
            session_id=payload["session_id"],
            protocol_id=payload["protocol_id"],
            endpoint=payload["endpoint"],
            baseline_role=payload["baseline_role"],
            candidate_role=payload["candidate_role"],
            metric_id=payload["metric_id"],
            metric_unit=payload["metric_unit"],
            metric_direction=payload["metric_direction"],
            baseline_artifact_id=payload["baseline_artifact_id"],
            candidate_artifact_id=payload["candidate_artifact_id"],
            features=CounterfactualRegretFeatures.from_dict(features_payload),
            baseline_loss=payload["baseline_loss"],
            candidate_loss=payload["candidate_loss"],
            source_future_outcomes_used=payload["source_future_outcomes_used"],
            target_future_outcomes_used=payload["target_future_outcomes_used"],
        )
        if case.source_case_artifact_id != expected_id:
            raise ValueError("counterfactual regret source-case checksum mismatch")
        return case


def _derive_certificate(
    source_cases: tuple[CounterfactualRegretSourceCase, ...],
    *,
    support_margin: float,
    scale_floor: float,
    harmful_relative_regret_threshold: float,
    minimum_global_mean_relative_improvement: float,
    minimum_global_win_fraction: float,
    maximum_global_harmful_fraction: float,
    maximum_global_worst_relative_regret: float,
) -> dict[str, Any]:
    feature_matrix = np.stack([case.features.values for case in source_cases], axis=0)
    center = np.median(feature_matrix, axis=0)
    absolute_deviation = np.abs(feature_matrix - center)
    mad_scale = 1.4826 * np.median(absolute_deviation, axis=0)
    half_range = 0.5 * np.ptp(feature_matrix, axis=0)
    magnitude_floor = scale_floor * np.maximum(np.abs(center), 1.0)
    scale = np.maximum.reduce((mad_scale, half_range, magnitude_floor))

    nearest_other_session: list[float] = []
    for index, case in enumerate(source_cases):
        candidate_indices = [
            other_index
            for other_index, other_case in enumerate(source_cases)
            if other_case.session_id != case.session_id
        ]
        if not candidate_indices:
            raise ValueError("source certificate requires independent sessions")
        distances = _standardized_distances(
            feature_matrix[index],
            feature_matrix[candidate_indices],
            scale,
        )
        nearest_other_session.append(float(np.min(distances)))
    maximum_nearest = max(max(nearest_other_session), scale_floor)

    session_gains: dict[str, list[float]] = {}
    for case in source_cases:
        session_gains.setdefault(case.session_id, []).append(case.relative_improvement)
    equal_session_gains: np.ndarray = np.asarray(
        [
            float(np.mean(session_gains[session_id]))
            for session_id in sorted(session_gains)
        ],
        dtype=float,
    )
    mean_gain = float(np.mean(equal_session_gains))
    win_fraction = float(np.mean(equal_session_gains > 0.0))
    harmful_fraction = float(
        np.mean(equal_session_gains < -harmful_relative_regret_threshold)
    )
    worst_regret = float(max(0.0, -float(np.min(equal_session_gains))))
    enabled = not any(
        (
            _strictly_below(
                mean_gain,
                minimum_global_mean_relative_improvement,
            ),
            _strictly_below(win_fraction, minimum_global_win_fraction),
            _strictly_above(
                harmful_fraction,
                maximum_global_harmful_fraction,
            ),
            _strictly_above(
                worst_regret,
                maximum_global_worst_relative_regret,
            ),
        )
    )
    return {
        "feature_center": center,
        "feature_scale": scale,
        "maximum_nearest_session_distance": maximum_nearest,
        "maximum_supported_distance": maximum_nearest * support_margin,
        "global_mean_relative_improvement": mean_gain,
        "global_win_fraction": win_fraction,
        "global_harmful_fraction": harmful_fraction,
        "global_worst_relative_regret": worst_regret,
        "candidate_enabled": enabled,
    }


@dataclass(frozen=True)
class CounterfactualRegretCertificate:
    """Source-only, session-aware selective-regret certificate."""

    protocol_id: str
    endpoint: Literal[
        "factual_continuation",
        "same_grasp_transfer",
        "new_contact_transfer",
    ]
    baseline_role: str
    candidate_role: str
    metric_id: str
    metric_unit: str
    metric_direction: Literal["lower", "higher"]
    feature_names: tuple[str, ...]
    source_cases: tuple[CounterfactualRegretSourceCase, ...]
    required_prerequisite_names: tuple[str, ...]
    local_session_count: int
    support_margin: float
    scale_floor: float
    harmful_relative_regret_threshold: float
    minimum_global_mean_relative_improvement: float
    minimum_global_win_fraction: float
    maximum_global_harmful_fraction: float
    maximum_global_worst_relative_regret: float
    minimum_local_mean_relative_improvement: float
    minimum_local_win_fraction: float
    maximum_local_harmful_fraction: float
    maximum_local_worst_relative_regret: float
    feature_center: np.ndarray
    feature_scale: np.ndarray
    maximum_nearest_session_distance: float
    maximum_supported_distance: float
    global_mean_relative_improvement: float
    global_win_fraction: float
    global_harmful_fraction: float
    global_worst_relative_regret: float
    candidate_enabled: bool
    source_future_outcomes_used: bool = True
    target_future_outcomes_used: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "protocol_id",
            _require_nonempty_string(self.protocol_id, name="protocol_id"),
        )
        if self.endpoint not in COUNTERFACTUAL_REGRET_ENDPOINTS:
            raise ValueError("unsupported counterfactual regret endpoint")
        for field_name in (
            "baseline_role",
            "candidate_role",
            "metric_id",
            "metric_unit",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_nonempty_string(
                    getattr(self, field_name),
                    name=field_name,
                ),
            )
        if self.baseline_role == self.candidate_role:
            raise ValueError("baseline_role and candidate_role must differ")
        if self.metric_direction not in METRIC_DIRECTIONS:
            raise ValueError("metric_direction must be lower or higher")
        feature_names = _validated_string_tuple(
            self.feature_names,
            name="regret feature names",
        )
        prerequisites = tuple(
            sorted(
                _validated_string_tuple(
                    self.required_prerequisite_names,
                    name="required prerequisite names",
                    allow_empty=True,
                )
            )
        )
        cases = tuple(
            sorted(self.source_cases, key=lambda case: (case.session_id, case.case_id))
        )
        if any(not isinstance(case, CounterfactualRegretSourceCase) for case in cases):
            raise ValueError("source_cases must contain regret source cases")
        case_ids = [case.case_id for case in cases]
        artifact_ids = [case.source_case_artifact_id for case in cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("source-case IDs must be unique")
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("source-case artifacts must be unique")
        session_ids = {case.session_id for case in cases}
        local_count = _positive_integer(
            self.local_session_count,
            name="local_session_count",
        )
        if len(session_ids) < max(3, local_count):
            raise ValueError(
                "regret certificate requires at least three independent sessions "
                "and enough sessions for the local neighborhood"
            )
        for case in cases:
            expected = (
                case.protocol_id,
                case.endpoint,
                case.baseline_role,
                case.candidate_role,
                case.metric_id,
                case.metric_unit,
                case.metric_direction,
                case.features.names,
            )
            actual = (
                self.protocol_id,
                self.endpoint,
                self.baseline_role,
                self.candidate_role,
                self.metric_id,
                self.metric_unit,
                self.metric_direction,
                feature_names,
            )
            if expected != actual:
                raise ValueError("source cases do not match the certificate protocol")

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
        harmful_threshold = _finite_float(
            self.harmful_relative_regret_threshold,
            name="harmful_relative_regret_threshold",
            minimum=0.0,
        )
        minimum_global_mean = _finite_float(
            self.minimum_global_mean_relative_improvement,
            name="minimum_global_mean_relative_improvement",
        )
        minimum_global_win = _probability(
            self.minimum_global_win_fraction,
            name="minimum_global_win_fraction",
        )
        maximum_global_harmful = _probability(
            self.maximum_global_harmful_fraction,
            name="maximum_global_harmful_fraction",
        )
        maximum_global_worst = _finite_float(
            self.maximum_global_worst_relative_regret,
            name="maximum_global_worst_relative_regret",
            minimum=0.0,
        )
        minimum_local_mean = _finite_float(
            self.minimum_local_mean_relative_improvement,
            name="minimum_local_mean_relative_improvement",
        )
        minimum_local_win = _probability(
            self.minimum_local_win_fraction,
            name="minimum_local_win_fraction",
        )
        maximum_local_harmful = _probability(
            self.maximum_local_harmful_fraction,
            name="maximum_local_harmful_fraction",
        )
        maximum_local_worst = _finite_float(
            self.maximum_local_worst_relative_regret,
            name="maximum_local_worst_relative_regret",
            minimum=0.0,
        )
        derived = _derive_certificate(
            cases,
            support_margin=support_margin,
            scale_floor=scale_floor,
            harmful_relative_regret_threshold=harmful_threshold,
            minimum_global_mean_relative_improvement=minimum_global_mean,
            minimum_global_win_fraction=minimum_global_win,
            maximum_global_harmful_fraction=maximum_global_harmful,
            maximum_global_worst_relative_regret=maximum_global_worst,
        )
        center = _require_array_match(
            self.feature_center,
            np.asarray(derived["feature_center"], dtype=float),
            name="feature_center",
        )
        scale = _require_array_match(
            self.feature_scale,
            np.asarray(derived["feature_scale"], dtype=float),
            name="feature_scale",
        )
        if center.shape != (len(feature_names),) or scale.shape != center.shape:
            raise ValueError("feature center and scale must match feature names")
        if np.any(scale <= 0.0):
            raise ValueError("feature_scale must be strictly positive")
        maximum_nearest = _require_float_match(
            self.maximum_nearest_session_distance,
            float(derived["maximum_nearest_session_distance"]),
            name="maximum_nearest_session_distance",
        )
        maximum_supported = _require_float_match(
            self.maximum_supported_distance,
            float(derived["maximum_supported_distance"]),
            name="maximum_supported_distance",
        )
        global_mean = _require_float_match(
            self.global_mean_relative_improvement,
            float(derived["global_mean_relative_improvement"]),
            name="global_mean_relative_improvement",
        )
        global_win = _require_float_match(
            self.global_win_fraction,
            float(derived["global_win_fraction"]),
            name="global_win_fraction",
        )
        global_harmful = _require_float_match(
            self.global_harmful_fraction,
            float(derived["global_harmful_fraction"]),
            name="global_harmful_fraction",
        )
        global_worst = _require_float_match(
            self.global_worst_relative_regret,
            float(derived["global_worst_relative_regret"]),
            name="global_worst_relative_regret",
        )
        if type(self.candidate_enabled) is not bool:
            raise ValueError("candidate_enabled must be Boolean")
        if self.candidate_enabled != bool(derived["candidate_enabled"]):
            raise ValueError("candidate_enabled contradicts source regret evidence")
        if (
            self.source_future_outcomes_used is not True
            or self.target_future_outcomes_used is not False
        ):
            raise ValueError("certificate information-boundary flags are invalid")

        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "source_cases", cases)
        object.__setattr__(self, "required_prerequisite_names", prerequisites)
        object.__setattr__(self, "local_session_count", local_count)
        object.__setattr__(self, "support_margin", support_margin)
        object.__setattr__(self, "scale_floor", scale_floor)
        object.__setattr__(
            self,
            "harmful_relative_regret_threshold",
            harmful_threshold,
        )
        object.__setattr__(
            self,
            "minimum_global_mean_relative_improvement",
            minimum_global_mean,
        )
        object.__setattr__(self, "minimum_global_win_fraction", minimum_global_win)
        object.__setattr__(
            self,
            "maximum_global_harmful_fraction",
            maximum_global_harmful,
        )
        object.__setattr__(
            self,
            "maximum_global_worst_relative_regret",
            maximum_global_worst,
        )
        object.__setattr__(
            self,
            "minimum_local_mean_relative_improvement",
            minimum_local_mean,
        )
        object.__setattr__(self, "minimum_local_win_fraction", minimum_local_win)
        object.__setattr__(
            self,
            "maximum_local_harmful_fraction",
            maximum_local_harmful,
        )
        object.__setattr__(
            self,
            "maximum_local_worst_relative_regret",
            maximum_local_worst,
        )
        object.__setattr__(self, "feature_center", center)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "maximum_nearest_session_distance", maximum_nearest)
        object.__setattr__(self, "maximum_supported_distance", maximum_supported)
        object.__setattr__(self, "global_mean_relative_improvement", global_mean)
        object.__setattr__(self, "global_win_fraction", global_win)
        object.__setattr__(self, "global_harmful_fraction", global_harmful)
        object.__setattr__(self, "global_worst_relative_regret", global_worst)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": COUNTERFACTUAL_REGRET_SCHEMA_VERSION,
            "artifact_kind": "Causal4DRegretCertificateV1",
            "protocol_id": self.protocol_id,
            "endpoint": self.endpoint,
            "baseline_role": self.baseline_role,
            "candidate_role": self.candidate_role,
            "metric_id": self.metric_id,
            "metric_unit": self.metric_unit,
            "metric_direction": self.metric_direction,
            "feature_names": list(self.feature_names),
            "source_cases": [case.as_dict() for case in self.source_cases],
            "required_prerequisite_names": list(self.required_prerequisite_names),
            "local_session_count": self.local_session_count,
            "support_margin": self.support_margin,
            "scale_floor": self.scale_floor,
            "harmful_relative_regret_threshold": (
                self.harmful_relative_regret_threshold
            ),
            "minimum_global_mean_relative_improvement": (
                self.minimum_global_mean_relative_improvement
            ),
            "minimum_global_win_fraction": self.minimum_global_win_fraction,
            "maximum_global_harmful_fraction": (self.maximum_global_harmful_fraction),
            "maximum_global_worst_relative_regret": (
                self.maximum_global_worst_relative_regret
            ),
            "minimum_local_mean_relative_improvement": (
                self.minimum_local_mean_relative_improvement
            ),
            "minimum_local_win_fraction": self.minimum_local_win_fraction,
            "maximum_local_harmful_fraction": self.maximum_local_harmful_fraction,
            "maximum_local_worst_relative_regret": (
                self.maximum_local_worst_relative_regret
            ),
            "feature_center": self.feature_center.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "maximum_nearest_session_distance": (self.maximum_nearest_session_distance),
            "maximum_supported_distance": self.maximum_supported_distance,
            "global_mean_relative_improvement": (self.global_mean_relative_improvement),
            "global_win_fraction": self.global_win_fraction,
            "global_harmful_fraction": self.global_harmful_fraction,
            "global_worst_relative_regret": self.global_worst_relative_regret,
            "candidate_enabled": self.candidate_enabled,
            "source_future_outcomes_used": self.source_future_outcomes_used,
            "target_future_outcomes_used": self.target_future_outcomes_used,
        }

    @property
    def certificate_id(self) -> str:
        return _canonical_sha256(self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "certificate_id": self.certificate_id}


def fit_counterfactual_regret_certificate(
    source_cases: Sequence[CounterfactualRegretSourceCase],
    *,
    required_prerequisite_names: Sequence[str] = (),
    local_session_count: int = 3,
    support_margin: float = 1.5,
    scale_floor: float = 1e-6,
    harmful_relative_regret_threshold: float = 0.02,
    minimum_global_mean_relative_improvement: float = 0.0,
    minimum_global_win_fraction: float = 0.5,
    maximum_global_harmful_fraction: float = 1.0 / 3.0,
    maximum_global_worst_relative_regret: float = 0.10,
    minimum_local_mean_relative_improvement: float = 0.0,
    minimum_local_win_fraction: float = 0.5,
    maximum_local_harmful_fraction: float = 1.0 / 3.0,
    maximum_local_worst_relative_regret: float = 0.05,
) -> CounterfactualRegretCertificate:
    """Fit an endpoint-specific certificate from independent source sessions."""

    cases = tuple(
        sorted(source_cases, key=lambda case: (case.session_id, case.case_id))
    )
    if not cases:
        raise ValueError("counterfactual regret certificate requires source cases")
    first = cases[0]
    support_margin_value = _finite_float(
        support_margin,
        name="support_margin",
        minimum=1.0,
    )
    scale_floor_value = _finite_float(
        scale_floor,
        name="scale_floor",
        minimum=float(np.finfo(float).eps),
    )
    harmful_threshold = _finite_float(
        harmful_relative_regret_threshold,
        name="harmful_relative_regret_threshold",
        minimum=0.0,
    )
    minimum_global_mean = _finite_float(
        minimum_global_mean_relative_improvement,
        name="minimum_global_mean_relative_improvement",
    )
    minimum_global_win = _probability(
        minimum_global_win_fraction,
        name="minimum_global_win_fraction",
    )
    maximum_global_harmful = _probability(
        maximum_global_harmful_fraction,
        name="maximum_global_harmful_fraction",
    )
    maximum_global_worst = _finite_float(
        maximum_global_worst_relative_regret,
        name="maximum_global_worst_relative_regret",
        minimum=0.0,
    )
    minimum_local_mean = _finite_float(
        minimum_local_mean_relative_improvement,
        name="minimum_local_mean_relative_improvement",
    )
    minimum_local_win = _probability(
        minimum_local_win_fraction,
        name="minimum_local_win_fraction",
    )
    maximum_local_harmful = _probability(
        maximum_local_harmful_fraction,
        name="maximum_local_harmful_fraction",
    )
    maximum_local_worst = _finite_float(
        maximum_local_worst_relative_regret,
        name="maximum_local_worst_relative_regret",
        minimum=0.0,
    )
    local_count = _positive_integer(
        local_session_count,
        name="local_session_count",
    )
    prerequisites = tuple(
        sorted(
            _validated_string_tuple(
                tuple(required_prerequisite_names),
                name="required prerequisite names",
                allow_empty=True,
            )
        )
    )
    derived = _derive_certificate(
        cases,
        support_margin=support_margin_value,
        scale_floor=scale_floor_value,
        harmful_relative_regret_threshold=harmful_threshold,
        minimum_global_mean_relative_improvement=minimum_global_mean,
        minimum_global_win_fraction=minimum_global_win,
        maximum_global_harmful_fraction=maximum_global_harmful,
        maximum_global_worst_relative_regret=maximum_global_worst,
    )
    return CounterfactualRegretCertificate(
        protocol_id=first.protocol_id,
        endpoint=first.endpoint,
        baseline_role=first.baseline_role,
        candidate_role=first.candidate_role,
        metric_id=first.metric_id,
        metric_unit=first.metric_unit,
        metric_direction=first.metric_direction,
        feature_names=first.features.names,
        source_cases=cases,
        required_prerequisite_names=prerequisites,
        local_session_count=local_count,
        support_margin=support_margin_value,
        scale_floor=scale_floor_value,
        harmful_relative_regret_threshold=harmful_threshold,
        minimum_global_mean_relative_improvement=minimum_global_mean,
        minimum_global_win_fraction=minimum_global_win,
        maximum_global_harmful_fraction=maximum_global_harmful,
        maximum_global_worst_relative_regret=maximum_global_worst,
        minimum_local_mean_relative_improvement=minimum_local_mean,
        minimum_local_win_fraction=minimum_local_win,
        maximum_local_harmful_fraction=maximum_local_harmful,
        maximum_local_worst_relative_regret=maximum_local_worst,
        feature_center=np.asarray(derived["feature_center"], dtype=float),
        feature_scale=np.asarray(derived["feature_scale"], dtype=float),
        maximum_nearest_session_distance=float(
            derived["maximum_nearest_session_distance"]
        ),
        maximum_supported_distance=float(derived["maximum_supported_distance"]),
        global_mean_relative_improvement=float(
            derived["global_mean_relative_improvement"]
        ),
        global_win_fraction=float(derived["global_win_fraction"]),
        global_harmful_fraction=float(derived["global_harmful_fraction"]),
        global_worst_relative_regret=float(derived["global_worst_relative_regret"]),
        candidate_enabled=bool(derived["candidate_enabled"]),
    )


@dataclass(frozen=True)
class CounterfactualRegretTarget:
    """One target query input that contains no future target outcome."""

    case_id: str
    session_id: str
    protocol_id: str
    endpoint: Literal[
        "factual_continuation",
        "same_grasp_transfer",
        "new_contact_transfer",
    ]
    baseline_role: str
    candidate_role: str
    metric_id: str
    metric_unit: str
    metric_direction: Literal["lower", "higher"]
    baseline_artifact_id: str
    candidate_artifact_id: str
    features: CounterfactualRegretFeatures
    prerequisites: tuple[CounterfactualRegretPrerequisite, ...] = ()
    target_future_outcomes_used: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "case_id",
            "session_id",
            "protocol_id",
            "baseline_role",
            "candidate_role",
            "metric_id",
            "metric_unit",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_nonempty_string(
                    getattr(self, field_name),
                    name=field_name,
                ),
            )
        if self.endpoint not in COUNTERFACTUAL_REGRET_ENDPOINTS:
            raise ValueError("unsupported counterfactual regret endpoint")
        if self.metric_direction not in METRIC_DIRECTIONS:
            raise ValueError("metric_direction must be lower or higher")
        if self.baseline_role == self.candidate_role:
            raise ValueError("baseline_role and candidate_role must differ")
        _require_sha256(self.baseline_artifact_id, name="baseline_artifact_id")
        _require_sha256(self.candidate_artifact_id, name="candidate_artifact_id")
        if not isinstance(self.features, CounterfactualRegretFeatures):
            raise ValueError("features must be CounterfactualRegretFeatures")
        prerequisites = tuple(sorted(self.prerequisites, key=lambda value: value.name))
        if any(
            not isinstance(value, CounterfactualRegretPrerequisite)
            for value in prerequisites
        ):
            raise ValueError("prerequisites must contain regret prerequisite records")
        names = [value.name for value in prerequisites]
        if len(names) != len(set(names)):
            raise ValueError("prerequisite names must be unique")
        if self.target_future_outcomes_used is not False:
            raise ValueError("target regret decisions cannot use future outcomes")
        object.__setattr__(self, "prerequisites", prerequisites)

    @property
    def target_input_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": COUNTERFACTUAL_REGRET_SCHEMA_VERSION,
                "artifact_kind": "Causal4DRegretTargetInputV1",
                "case_id": self.case_id,
                "session_id": self.session_id,
                "protocol_id": self.protocol_id,
                "endpoint": self.endpoint,
                "baseline_role": self.baseline_role,
                "candidate_role": self.candidate_role,
                "metric_id": self.metric_id,
                "metric_unit": self.metric_unit,
                "metric_direction": self.metric_direction,
                "baseline_artifact_id": self.baseline_artifact_id,
                "candidate_artifact_id": self.candidate_artifact_id,
                "features": self.features.as_dict(),
                "prerequisites": [value.as_dict() for value in self.prerequisites],
                "target_future_outcomes_used": self.target_future_outcomes_used,
            }
        )


@dataclass(frozen=True)
class CounterfactualRegretDecision:
    """One source-calibrated target decision with local regret diagnostics."""

    certificate_id: str
    target_input_id: str
    case_id: str
    session_id: str
    accepted: bool
    baseline_role: str
    candidate_role: str
    selected_role: str
    reasons: tuple[str, ...]
    nearest_source_distance: float
    neighbor_case_ids: tuple[str, ...]
    neighbor_session_ids: tuple[str, ...]
    neighbor_distances: tuple[float, ...]
    neighbor_relative_improvements: tuple[float, ...]
    harmful_relative_regret_threshold: float
    local_mean_relative_improvement: float
    local_win_fraction: float
    local_harmful_fraction: float
    local_worst_relative_regret: float
    source_global_candidate_enabled: bool
    target_future_observations_read: int = 0
    target_future_outcomes_used: bool = False

    def __post_init__(self) -> None:
        _require_sha256(self.certificate_id, name="certificate_id")
        _require_sha256(self.target_input_id, name="target_input_id")
        object.__setattr__(
            self,
            "case_id",
            _require_nonempty_string(self.case_id, name="decision case_id"),
        )
        object.__setattr__(
            self,
            "session_id",
            _require_nonempty_string(self.session_id, name="decision session_id"),
        )
        baseline_role = _require_nonempty_string(
            self.baseline_role,
            name="baseline_role",
        )
        candidate_role = _require_nonempty_string(
            self.candidate_role,
            name="candidate_role",
        )
        selected_role = _require_nonempty_string(
            self.selected_role,
            name="selected_role",
        )
        if baseline_role == candidate_role:
            raise ValueError("baseline_role and candidate_role must differ")
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be Boolean")
        expected_role = candidate_role if self.accepted else baseline_role
        if selected_role != expected_role:
            raise ValueError("selected_role contradicts the regret decision")
        reasons = tuple(self.reasons)
        if any(type(value) is not str or not value for value in reasons):
            raise ValueError("decision reasons must contain nonempty strings")
        if len(set(reasons)) != len(reasons):
            raise ValueError("decision reasons must be unique")
        if self.accepted and reasons:
            raise ValueError("accepted decisions cannot contain rejection reasons")
        if not self.accepted and not reasons:
            raise ValueError("rejected decisions require at least one reason")
        case_ids = _validated_string_tuple(
            self.neighbor_case_ids,
            name="neighbor_case_ids",
        )
        session_ids = _validated_string_tuple(
            self.neighbor_session_ids,
            name="neighbor_session_ids",
        )
        distances = tuple(
            _finite_float(
                value,
                name=f"neighbor_distances[{index}]",
                minimum=0.0,
            )
            for index, value in enumerate(self.neighbor_distances)
        )
        improvements = tuple(
            _finite_float(
                value,
                name=f"neighbor_relative_improvements[{index}]",
            )
            for index, value in enumerate(self.neighbor_relative_improvements)
        )
        count = len(case_ids)
        if (
            len(session_ids) != count
            or len(distances) != count
            or len(improvements) != count
        ):
            raise ValueError("neighbor diagnostics must have aligned lengths")
        if len(set(session_ids)) != count:
            raise ValueError("neighbor sessions must be independent and unique")
        if tuple(distances) != tuple(sorted(distances)):
            raise ValueError("neighbor distances must use nearest-first order")
        nearest = _finite_float(
            self.nearest_source_distance,
            name="nearest_source_distance",
            minimum=0.0,
        )
        if not np.isclose(nearest, distances[0], rtol=1e-13, atol=1e-15):
            raise ValueError("nearest_source_distance contradicts neighbor distances")
        harmful_threshold = _finite_float(
            self.harmful_relative_regret_threshold,
            name="harmful_relative_regret_threshold",
            minimum=0.0,
        )
        improvement_array: np.ndarray = np.asarray(improvements, dtype=float)
        expected_local_mean = float(np.mean(improvement_array))
        expected_local_win = float(np.mean(improvement_array > 0.0))
        expected_local_harmful = float(np.mean(improvement_array < -harmful_threshold))
        expected_local_worst = float(max(0.0, -float(np.min(improvement_array))))
        local_mean = _require_float_match(
            self.local_mean_relative_improvement,
            expected_local_mean,
            name="local_mean_relative_improvement",
        )
        local_win = _require_float_match(
            self.local_win_fraction,
            expected_local_win,
            name="local_win_fraction",
        )
        local_harmful = _require_float_match(
            self.local_harmful_fraction,
            expected_local_harmful,
            name="local_harmful_fraction",
        )
        local_worst = _require_float_match(
            self.local_worst_relative_regret,
            expected_local_worst,
            name="local_worst_relative_regret",
        )
        if not 0.0 <= local_win <= 1.0 or not 0.0 <= local_harmful <= 1.0:
            raise ValueError("local regret fractions must lie in [0, 1]")
        if local_worst < 0.0:
            raise ValueError("local_worst_relative_regret must be nonnegative")
        if type(self.source_global_candidate_enabled) is not bool:
            raise ValueError("source_global_candidate_enabled must be Boolean")
        if self.target_future_observations_read != 0:
            raise ValueError("counterfactual regret decisions cannot read futures")
        if self.target_future_outcomes_used is not False:
            raise ValueError("counterfactual regret decisions cannot use futures")
        object.__setattr__(self, "baseline_role", baseline_role)
        object.__setattr__(self, "candidate_role", candidate_role)
        object.__setattr__(self, "selected_role", selected_role)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "neighbor_case_ids", case_ids)
        object.__setattr__(self, "neighbor_session_ids", session_ids)
        object.__setattr__(self, "neighbor_distances", distances)
        object.__setattr__(self, "neighbor_relative_improvements", improvements)
        object.__setattr__(self, "nearest_source_distance", nearest)
        object.__setattr__(
            self,
            "harmful_relative_regret_threshold",
            harmful_threshold,
        )
        object.__setattr__(self, "local_mean_relative_improvement", local_mean)
        object.__setattr__(self, "local_win_fraction", local_win)
        object.__setattr__(self, "local_harmful_fraction", local_harmful)
        object.__setattr__(self, "local_worst_relative_regret", local_worst)

    @property
    def decision_id(self) -> str:
        return _canonical_sha256(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": COUNTERFACTUAL_REGRET_SCHEMA_VERSION,
            "artifact_kind": "Causal4DRegretDecisionV1",
            "certificate_id": self.certificate_id,
            "target_input_id": self.target_input_id,
            "case_id": self.case_id,
            "session_id": self.session_id,
            "accepted": self.accepted,
            "baseline_role": self.baseline_role,
            "candidate_role": self.candidate_role,
            "selected_role": self.selected_role,
            "reasons": list(self.reasons),
            "nearest_source_distance": self.nearest_source_distance,
            "neighbor_case_ids": list(self.neighbor_case_ids),
            "neighbor_session_ids": list(self.neighbor_session_ids),
            "neighbor_distances": list(self.neighbor_distances),
            "neighbor_relative_improvements": list(self.neighbor_relative_improvements),
            "harmful_relative_regret_threshold": (
                self.harmful_relative_regret_threshold
            ),
            "local_mean_relative_improvement": (self.local_mean_relative_improvement),
            "local_win_fraction": self.local_win_fraction,
            "local_harmful_fraction": self.local_harmful_fraction,
            "local_worst_relative_regret": self.local_worst_relative_regret,
            "source_global_candidate_enabled": (self.source_global_candidate_enabled),
            "target_future_observations_read": self.target_future_observations_read,
            "target_future_outcomes_used": self.target_future_outcomes_used,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "decision_id": self.decision_id}


def evaluate_counterfactual_regret(
    target: CounterfactualRegretTarget,
    certificate: CounterfactualRegretCertificate,
) -> CounterfactualRegretDecision:
    """Evaluate local source regret without reading a target continuation."""

    protocol = (
        target.protocol_id,
        target.endpoint,
        target.baseline_role,
        target.candidate_role,
        target.metric_id,
        target.metric_unit,
        target.metric_direction,
        target.features.names,
    )
    expected_protocol = (
        certificate.protocol_id,
        certificate.endpoint,
        certificate.baseline_role,
        certificate.candidate_role,
        certificate.metric_id,
        certificate.metric_unit,
        certificate.metric_direction,
        certificate.feature_names,
    )
    if protocol != expected_protocol:
        raise ValueError("target input does not match the regret certificate protocol")
    if target.case_id in {case.case_id for case in certificate.source_cases}:
        raise ValueError("target case must be disjoint from source certificate")
    if target.session_id in {case.session_id for case in certificate.source_cases}:
        raise ValueError("target session must be disjoint from source certificate")
    prerequisite_names = tuple(value.name for value in target.prerequisites)
    if prerequisite_names != certificate.required_prerequisite_names:
        raise ValueError("target prerequisites differ from the frozen protocol")

    source_features = np.stack(
        [case.features.values for case in certificate.source_cases],
        axis=0,
    )
    distances = _standardized_distances(
        target.features.values,
        source_features,
        certificate.feature_scale,
    )
    nearest_by_session: dict[str, tuple[float, str, int]] = {}
    for index, (case, distance) in enumerate(
        zip(certificate.source_cases, distances, strict=True)
    ):
        candidate = (float(distance), case.case_id, index)
        current = nearest_by_session.get(case.session_id)
        if current is None or candidate < current:
            nearest_by_session[case.session_id] = candidate
    ordered = sorted(
        (distance, case_id, session_id, index)
        for session_id, (distance, case_id, index) in nearest_by_session.items()
    )
    neighbors = ordered[: certificate.local_session_count]
    gains: np.ndarray = np.asarray(
        [
            certificate.source_cases[index].relative_improvement
            for _, _, _, index in neighbors
        ],
        dtype=float,
    )
    neighbor_distances: np.ndarray = np.asarray(
        [value[0] for value in neighbors], dtype=float
    )
    nearest_distance = float(np.min(neighbor_distances))
    local_mean = float(np.mean(gains))
    local_win = float(np.mean(gains > 0.0))
    local_harmful = float(
        np.mean(gains < -certificate.harmful_relative_regret_threshold)
    )
    local_worst = float(max(0.0, -float(np.min(gains))))

    reasons: list[str] = []
    for prerequisite in target.prerequisites:
        if not prerequisite.accepted:
            reasons.append(f"prerequisite_rejected:{prerequisite.name}")
    if not certificate.candidate_enabled:
        reasons.append("source_global_regret_policy_failed")
    if _strictly_above(
        nearest_distance,
        certificate.maximum_supported_distance,
    ):
        reasons.append("feature_outside_source_support")
    if _strictly_below(
        local_mean,
        certificate.minimum_local_mean_relative_improvement,
    ):
        reasons.append("local_mean_gain_not_supported")
    if _strictly_below(local_win, certificate.minimum_local_win_fraction):
        reasons.append("local_win_fraction_not_supported")
    if _strictly_above(
        local_harmful,
        certificate.maximum_local_harmful_fraction,
    ):
        reasons.append("local_harmful_fraction_exceeded")
    if _strictly_above(
        local_worst,
        certificate.maximum_local_worst_relative_regret,
    ):
        reasons.append("local_worst_regret_exceeded")
    accepted = not reasons
    return CounterfactualRegretDecision(
        certificate_id=certificate.certificate_id,
        target_input_id=target.target_input_id,
        case_id=target.case_id,
        session_id=target.session_id,
        accepted=accepted,
        baseline_role=certificate.baseline_role,
        candidate_role=certificate.candidate_role,
        selected_role=(
            certificate.candidate_role if accepted else certificate.baseline_role
        ),
        reasons=tuple(reasons),
        nearest_source_distance=nearest_distance,
        neighbor_case_ids=tuple(value[1] for value in neighbors),
        neighbor_session_ids=tuple(value[2] for value in neighbors),
        neighbor_distances=tuple(map(float, neighbor_distances)),
        neighbor_relative_improvements=tuple(map(float, gains)),
        harmful_relative_regret_threshold=(
            certificate.harmful_relative_regret_threshold
        ),
        local_mean_relative_improvement=local_mean,
        local_win_fraction=local_win,
        local_harmful_fraction=local_harmful,
        local_worst_relative_regret=local_worst,
        source_global_candidate_enabled=certificate.candidate_enabled,
    )


@dataclass(frozen=True)
class CounterfactualRegretSelection(Generic[BaselineT, CandidateT]):
    """Candidate selection with byte/object-exact fallback semantics."""

    baseline: BaselineT
    candidate: CandidateT
    deployed: BaselineT | CandidateT
    decision: CounterfactualRegretDecision

    def __post_init__(self) -> None:
        expected = self.candidate if self.decision.accepted else self.baseline
        if self.deployed is not expected:
            raise ValueError("deployed object contradicts the regret decision")


def select_counterfactual_regret_candidate(
    certificate: CounterfactualRegretCertificate,
    target: CounterfactualRegretTarget,
    *,
    baseline: BaselineT,
    candidate: CandidateT,
    baseline_artifact_id: str,
    candidate_artifact_id: str,
) -> CounterfactualRegretSelection[BaselineT, CandidateT]:
    """Deploy the candidate only when the source regret certificate accepts it."""

    if (
        _require_sha256(
            baseline_artifact_id,
            name="baseline_artifact_id",
        )
        != target.baseline_artifact_id
    ):
        raise ValueError("baseline object identity differs from the target contract")
    if (
        _require_sha256(
            candidate_artifact_id,
            name="candidate_artifact_id",
        )
        != target.candidate_artifact_id
    ):
        raise ValueError("candidate object identity differs from the target contract")
    decision = evaluate_counterfactual_regret(target, certificate)
    deployed: BaselineT | CandidateT = candidate if decision.accepted else baseline
    selection = CounterfactualRegretSelection(
        baseline=baseline,
        candidate=candidate,
        deployed=deployed,
        decision=decision,
    )
    if not decision.accepted and selection.deployed is not baseline:
        raise RuntimeError("regret rejection failed to preserve exact baseline")
    return selection


def write_counterfactual_regret_certificate(
    path: str | Path,
    certificate: CounterfactualRegretCertificate,
    *,
    overwrite: bool = False,
) -> None:
    """Publish one finite content-addressed certificate atomically."""

    atomic_write_json(path, certificate.as_dict(), overwrite=overwrite)


def load_counterfactual_regret_certificate(
    path: str | Path,
) -> CounterfactualRegretCertificate:
    """Load and independently reconstruct one source regret certificate."""

    snapshot = read_regular_file(path, name="counterfactual regret certificate")
    payload = load_strict_json_object(
        snapshot.payload,
        name="counterfactual regret certificate",
    )
    _require_exact_fields(
        payload,
        _CERTIFICATE_FIELDS,
        name="counterfactual regret certificate",
    )
    if payload.pop("schema_version") != COUNTERFACTUAL_REGRET_SCHEMA_VERSION:
        raise ValueError("unsupported counterfactual regret certificate schema")
    if payload.pop("artifact_kind") != "Causal4DRegretCertificateV1":
        raise ValueError("unexpected counterfactual regret certificate kind")
    expected_id = _require_sha256(payload.pop("certificate_id"), name="certificate_id")
    source_payloads = payload.get("source_cases")
    if not isinstance(source_payloads, list):
        raise ValueError("source_cases must be a JSON array")
    source_cases: list[CounterfactualRegretSourceCase] = []
    for value in source_payloads:
        if not isinstance(value, Mapping):
            raise ValueError("source_cases must contain JSON objects")
        source_cases.append(CounterfactualRegretSourceCase.from_dict(value))
    certificate = CounterfactualRegretCertificate(
        protocol_id=payload["protocol_id"],
        endpoint=payload["endpoint"],
        baseline_role=payload["baseline_role"],
        candidate_role=payload["candidate_role"],
        metric_id=payload["metric_id"],
        metric_unit=payload["metric_unit"],
        metric_direction=payload["metric_direction"],
        feature_names=tuple(payload["feature_names"]),
        source_cases=tuple(source_cases),
        required_prerequisite_names=tuple(payload["required_prerequisite_names"]),
        local_session_count=payload["local_session_count"],
        support_margin=payload["support_margin"],
        scale_floor=payload["scale_floor"],
        harmful_relative_regret_threshold=(
            payload["harmful_relative_regret_threshold"]
        ),
        minimum_global_mean_relative_improvement=(
            payload["minimum_global_mean_relative_improvement"]
        ),
        minimum_global_win_fraction=payload["minimum_global_win_fraction"],
        maximum_global_harmful_fraction=(payload["maximum_global_harmful_fraction"]),
        maximum_global_worst_relative_regret=(
            payload["maximum_global_worst_relative_regret"]
        ),
        minimum_local_mean_relative_improvement=(
            payload["minimum_local_mean_relative_improvement"]
        ),
        minimum_local_win_fraction=payload["minimum_local_win_fraction"],
        maximum_local_harmful_fraction=payload["maximum_local_harmful_fraction"],
        maximum_local_worst_relative_regret=(
            payload["maximum_local_worst_relative_regret"]
        ),
        feature_center=np.asarray(payload["feature_center"]),
        feature_scale=np.asarray(payload["feature_scale"]),
        maximum_nearest_session_distance=(payload["maximum_nearest_session_distance"]),
        maximum_supported_distance=payload["maximum_supported_distance"],
        global_mean_relative_improvement=(payload["global_mean_relative_improvement"]),
        global_win_fraction=payload["global_win_fraction"],
        global_harmful_fraction=payload["global_harmful_fraction"],
        global_worst_relative_regret=payload["global_worst_relative_regret"],
        candidate_enabled=payload["candidate_enabled"],
        source_future_outcomes_used=payload["source_future_outcomes_used"],
        target_future_outcomes_used=payload["target_future_outcomes_used"],
    )
    if certificate.certificate_id != expected_id:
        raise ValueError("counterfactual regret certificate checksum mismatch")
    return certificate


def load_claim_bearing_counterfactual_regret_certificate(
    path: str | Path,
    *,
    expected_certificate_id: str,
) -> CounterfactualRegretCertificate:
    """Load only the independently frozen claim-bearing certificate identity."""

    expected = _require_sha256(
        expected_certificate_id,
        name="expected_certificate_id",
    )
    certificate = load_counterfactual_regret_certificate(path)
    if certificate.certificate_id != expected:
        raise ValueError("counterfactual regret certificate is not the frozen identity")
    return certificate


def write_counterfactual_regret_decision(
    path: str | Path,
    decision: CounterfactualRegretDecision,
    *,
    overwrite: bool = False,
) -> None:
    """Publish one finite target-safe regret decision atomically."""

    atomic_write_json(path, decision.as_dict(), overwrite=overwrite)


__all__ = [
    "COUNTERFACTUAL_REGRET_ENDPOINTS",
    "COUNTERFACTUAL_REGRET_SCHEMA_VERSION",
    "CounterfactualRegretCertificate",
    "CounterfactualRegretDecision",
    "CounterfactualRegretFeatures",
    "CounterfactualRegretPrerequisite",
    "CounterfactualRegretSelection",
    "CounterfactualRegretSourceCase",
    "CounterfactualRegretTarget",
    "evaluate_counterfactual_regret",
    "fit_counterfactual_regret_certificate",
    "load_claim_bearing_counterfactual_regret_certificate",
    "load_counterfactual_regret_certificate",
    "select_counterfactual_regret_candidate",
    "write_counterfactual_regret_certificate",
    "write_counterfactual_regret_decision",
]
