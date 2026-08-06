"""Source-calibrated, prefix-only reliability gating for hybrid predictions.

The gate is deliberately separate from the frozen controlled benchmark and the
registered physical estimator. Source futures may calibrate whether hybrid
corrections are worth considering, while a target decision reads only the
permitted response prefix and model-produced predictions. Rejection returns the
exact physics-only prediction object.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from causal4d.artifact_io import load_strict_json_object, read_regular_file
from causal4d.atomic_io import atomic_write_json
from causal4d.baselines import PredictiveDistribution, RidgeTrajectoryModel
from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json, validated_json_mapping


HYBRID_RELIABILITY_SCHEMA_VERSION = 1

_CALIBRATION_VECTOR_FIELDS = (
    "source_case_ids",
    "source_case_artifact_ids",
    "source_prefix_input_ids",
    "source_prefix_rmse_relative_improvements",
    "source_prefix_log_score_gains",
    "source_correction_standard_deviation_ratios",
    "source_descriptor_leverages",
    "source_future_relative_improvements",
)
_CALIBRATION_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "calibration_id",
        "source_case_ids",
        "source_case_artifact_ids",
        "source_prefix_input_ids",
        "prefix_frame_count",
        "source_prefix_rmse_relative_improvements",
        "source_prefix_log_score_gains",
        "source_correction_standard_deviation_ratios",
        "source_descriptor_leverages",
        "source_future_relative_improvements",
        "minimum_prefix_rmse_relative_improvement",
        "minimum_prefix_log_score_gain",
        "maximum_correction_standard_deviation_ratio",
        "maximum_descriptor_leverage",
        "minimum_mean_source_future_relative_improvement",
        "minimum_source_future_win_fraction",
        "mean_source_future_relative_improvement",
        "source_future_win_fraction",
        "support_margin",
        "hybrid_enabled",
        "source_futures_used",
        "target_futures_used",
    }
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        plain_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_float(value: Any, *, name: str, minimum: float | None = None) -> float:
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
    return result


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


def ridge_descriptor_leverage(
    model: RidgeTrajectoryModel,
    descriptor: np.ndarray,
) -> float:
    """Return regularized feature leverage without reading response outcomes."""

    values = np.asarray(descriptor, dtype=float)
    if values.shape != model.feature_mean.shape:
        raise ValueError("descriptor shape does not match the residual model")
    if not np.all(np.isfinite(values)):
        raise ValueError("descriptor must be finite")
    standardized = (values - model.feature_mean) / model.feature_scale
    feature = np.concatenate(([1.0], standardized))
    if model.gram_matrix is not None:
        solved = np.linalg.solve(model.gram_matrix, feature)
        leverage = float(feature @ solved)
    else:
        if model.gram_inverse is None:
            raise RuntimeError("ridge model has no Gram representation")
        leverage = float(feature @ model.gram_inverse @ feature)
    if not np.isfinite(leverage):
        raise ValueError("descriptor leverage must be finite")
    return max(leverage, 0.0)


@dataclass(frozen=True)
class HybridReliabilityCase:
    """One source or target case with an explicitly bounded response prefix."""

    case_id: str
    physics: PredictiveDistribution
    hybrid: PredictiveDistribution
    observations: np.ndarray
    descriptor_leverage: float
    prefix_frame_count: int

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id:
            raise ValueError("hybrid reliability case_id must be nonempty")
        if type(self.physics) is not PredictiveDistribution or type(
            self.hybrid
        ) is not PredictiveDistribution:
            raise ValueError("physics and hybrid must be predictive distributions")
        if self.physics.method != "physics_only" or self.hybrid.method != "hybrid":
            raise ValueError(
                "reliability cases require physics_only and hybrid methods"
            )
        if self.physics.mean.shape != self.hybrid.mean.shape:
            raise ValueError("physics and hybrid predictions must have equal shape")
        if self.physics.variance.shape != self.hybrid.variance.shape:
            raise ValueError("physics and hybrid variances must have equal shape")
        observations = readonly_array(self.observations, dtype=float)
        if observations.shape != self.physics.mean.shape:
            raise ValueError("observations must match predictive trajectories")
        if type(self.prefix_frame_count) is not int or not (
            2 <= self.prefix_frame_count < len(observations)
        ):
            raise ValueError(
                "prefix_frame_count must reveal a response and leave a future"
            )
        if not np.all(np.isfinite(observations[1 : self.prefix_frame_count])):
            raise ValueError("the admitted response prefix must be finite")
        leverage = _finite_float(
            self.descriptor_leverage,
            name="descriptor_leverage",
            minimum=0.0,
        )
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "descriptor_leverage", leverage)

    def _identity_payload(
        self,
        *,
        include_future_observations: bool,
        include_case_id: bool,
    ) -> dict[str, Any]:
        observation_values = (
            self.observations
            if include_future_observations
            else self.observations[: self.prefix_frame_count]
        )
        payload = {
            "schema_version": HYBRID_RELIABILITY_SCHEMA_VERSION,
            "prefix_frame_count": self.prefix_frame_count,
            "descriptor_leverage": self.descriptor_leverage,
            "physics_method": self.physics.method,
            "hybrid_method": self.hybrid.method,
            "physics_mean_sha256": array_sha256(self.physics.mean),
            "physics_variance_sha256": array_sha256(self.physics.variance),
            "hybrid_mean_sha256": array_sha256(self.hybrid.mean),
            "hybrid_variance_sha256": array_sha256(self.hybrid.variance),
            "observations_sha256": array_sha256(observation_values),
            "observations_include_future": include_future_observations,
        }
        if include_case_id:
            payload["case_id"] = self.case_id
        return payload

    @property
    def source_case_artifact_id(self) -> str:
        """Identity used when source futures are deliberately admitted."""

        return _canonical_sha256(
            self._identity_payload(
                include_future_observations=True,
                include_case_id=True,
            )
        )

    @property
    def target_prefix_input_id(self) -> str:
        """Target decision identity that excludes every future observation byte."""

        return _canonical_sha256(
            self._identity_payload(
                include_future_observations=False,
                include_case_id=False,
            )
        )


def _gaussian_nll(
    prediction: PredictiveDistribution,
    observations: np.ndarray,
) -> float:
    residual = prediction.mean - observations
    return float(
        np.mean(
            0.5
            * (
                np.log(2.0 * np.pi * prediction.variance)
                + np.square(residual) / prediction.variance
            )
        )
    )


def hybrid_reliability_diagnostics(
    case: HybridReliabilityCase,
) -> dict[str, float | int]:
    """Compute target-safe diagnostics from the response prefix and predictions."""

    prefix = slice(1, case.prefix_frame_count)
    observations = case.observations[prefix]
    physics_mean = case.physics.mean[prefix]
    hybrid_mean = case.hybrid.mean[prefix]
    physics_rmse = float(np.sqrt(np.mean(np.square(physics_mean - observations))))
    hybrid_rmse = float(np.sqrt(np.mean(np.square(hybrid_mean - observations))))
    physics_prefix = PredictiveDistribution(
        method=case.physics.method,
        mean=physics_mean,
        variance=case.physics.variance[prefix],
    )
    hybrid_prefix = PredictiveDistribution(
        method=case.hybrid.method,
        mean=hybrid_mean,
        variance=case.hybrid.variance[prefix],
    )
    physics_nll = _gaussian_nll(physics_prefix, observations)
    hybrid_nll = _gaussian_nll(hybrid_prefix, observations)
    correction = case.hybrid.mean - case.physics.mean
    correction_rms = float(np.sqrt(np.mean(np.square(correction))))
    physics_standard_deviation_rms = float(np.sqrt(np.mean(case.physics.variance)))
    return {
        "prefix_frame_count_including_endpoint": case.prefix_frame_count,
        "response_frames_used": case.prefix_frame_count - 1,
        "future_observation_frames_read": 0,
        "physics_prefix_rmse_m": physics_rmse,
        "hybrid_prefix_rmse_m": hybrid_rmse,
        "prefix_rmse_relative_improvement": 1.0
        - hybrid_rmse / max(physics_rmse, 1e-12),
        "physics_prefix_gaussian_nll": physics_nll,
        "hybrid_prefix_gaussian_nll": hybrid_nll,
        "prefix_gaussian_log_score_gain": physics_nll - hybrid_nll,
        "full_query_correction_rms_m": correction_rms,
        "physics_predictive_standard_deviation_rms_m": (
            physics_standard_deviation_rms
        ),
        "correction_to_physics_standard_deviation_ratio": correction_rms
        / max(physics_standard_deviation_rms, 1e-12),
        "descriptor_leverage": case.descriptor_leverage,
    }


def _future_relative_improvement(case: HybridReliabilityCase) -> float:
    future = slice(case.prefix_frame_count, None)
    observations = case.observations[future]
    if observations.size == 0 or not np.all(np.isfinite(observations)):
        raise ValueError(
            f"source case {case.case_id!r} requires a finite calibration future"
        )
    physics_rmse = float(
        np.sqrt(np.mean(np.square(case.physics.mean[future] - observations)))
    )
    hybrid_rmse = float(
        np.sqrt(np.mean(np.square(case.hybrid.mean[future] - observations)))
    )
    return 1.0 - hybrid_rmse / max(physics_rmse, 1e-12)


@dataclass(frozen=True)
class HybridReliabilityCalibration:
    """Source-only thresholds for prefix-time hybrid acceptance."""

    source_case_ids: tuple[str, ...]
    source_case_artifact_ids: tuple[str, ...]
    source_prefix_input_ids: tuple[str, ...]
    prefix_frame_count: int
    source_prefix_rmse_relative_improvements: tuple[float, ...]
    source_prefix_log_score_gains: tuple[float, ...]
    source_correction_standard_deviation_ratios: tuple[float, ...]
    source_descriptor_leverages: tuple[float, ...]
    source_future_relative_improvements: tuple[float, ...]
    minimum_prefix_rmse_relative_improvement: float
    minimum_prefix_log_score_gain: float
    maximum_correction_standard_deviation_ratio: float
    maximum_descriptor_leverage: float
    minimum_mean_source_future_relative_improvement: float
    minimum_source_future_win_fraction: float
    mean_source_future_relative_improvement: float
    source_future_win_fraction: float
    support_margin: float
    hybrid_enabled: bool
    source_futures_used: bool = True
    target_futures_used: bool = False

    def __post_init__(self) -> None:
        identifiers = tuple(self.source_case_ids)
        artifact_ids = tuple(self.source_case_artifact_ids)
        prefix_input_ids = tuple(self.source_prefix_input_ids)
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("source_case_ids must be nonempty and unique")
        if any(type(value) is not str or not value for value in identifiers):
            raise ValueError("source_case_ids must contain nonempty strings")
        if len(artifact_ids) != len(identifiers) or len(set(artifact_ids)) != len(
            artifact_ids
        ):
            raise ValueError("source case artifact IDs must be aligned and unique")
        for index, value in enumerate(artifact_ids):
            _require_sha256(value, name=f"source_case_artifact_ids[{index}]")
        if (
            len(prefix_input_ids) != len(identifiers)
            or len(set(prefix_input_ids)) != len(prefix_input_ids)
        ):
            raise ValueError("source prefix input IDs must be aligned and unique")
        for index, value in enumerate(prefix_input_ids):
            _require_sha256(value, name=f"source_prefix_input_ids[{index}]")
        if type(self.prefix_frame_count) is not int or self.prefix_frame_count < 2:
            raise ValueError("prefix_frame_count must be an integer of at least two")
        normalized_vectors: dict[str, tuple[float, ...]] = {}
        for name in _CALIBRATION_VECTOR_FIELDS[3:]:
            values = tuple(getattr(self, name))
            if len(values) != len(identifiers):
                raise ValueError("source diagnostics must identify every source case")
            minimum = (
                0.0
                if name
                in {
                    "source_correction_standard_deviation_ratios",
                    "source_descriptor_leverages",
                }
                else None
            )
            normalized_vectors[name] = tuple(
                _finite_float(
                    value,
                    name=f"{name}[{index}]",
                    minimum=minimum,
                )
                for index, value in enumerate(values)
            )

        finite_scalars = (
            "minimum_prefix_rmse_relative_improvement",
            "minimum_prefix_log_score_gain",
            "minimum_mean_source_future_relative_improvement",
            "mean_source_future_relative_improvement",
        )
        for name in finite_scalars:
            _finite_float(getattr(self, name), name=name)
        positive_support = (
            "maximum_correction_standard_deviation_ratio",
            "maximum_descriptor_leverage",
        )
        for name in positive_support:
            value = _finite_float(getattr(self, name), name=name, minimum=0.0)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        support_margin = _finite_float(
            self.support_margin,
            name="support_margin",
            minimum=1.0,
        )
        probabilities = (
            self.minimum_source_future_win_fraction,
            self.source_future_win_fraction,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in probabilities
        ):
            raise ValueError("source future win fractions must lie in [0, 1]")
        if type(self.hybrid_enabled) is not bool:
            raise ValueError("hybrid_enabled must be Boolean")
        expected_enabled = (
            self.mean_source_future_relative_improvement
            >= self.minimum_mean_source_future_relative_improvement
            and self.source_future_win_fraction
            >= self.minimum_source_future_win_fraction
        )
        if self.hybrid_enabled != expected_enabled:
            raise ValueError("hybrid_enabled contradicts source future gates")
        if (
            self.source_futures_used is not True
            or self.target_futures_used is not False
        ):
            raise ValueError("calibration information-boundary flags are invalid")
        object.__setattr__(self, "source_case_ids", identifiers)
        object.__setattr__(self, "source_case_artifact_ids", artifact_ids)
        object.__setattr__(self, "source_prefix_input_ids", prefix_input_ids)
        object.__setattr__(self, "support_margin", support_margin)
        for name, values in normalized_vectors.items():
            object.__setattr__(self, name, values)

    @property
    def calibration_id(self) -> str:
        payload = {
            "schema_version": HYBRID_RELIABILITY_SCHEMA_VERSION,
            "artifact_kind": "Causal4DHybridReliabilityCalibration",
            **asdict(self),
        }
        return _canonical_sha256(payload)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HYBRID_RELIABILITY_SCHEMA_VERSION,
            "artifact_kind": "Causal4DHybridReliabilityCalibration",
            **asdict(self),
            "calibration_id": self.calibration_id,
        }


def fit_hybrid_reliability_calibration(
    source_cases: Sequence[HybridReliabilityCase],
    *,
    minimum_mean_source_future_relative_improvement: float = 0.005,
    minimum_source_future_win_fraction: float = 2.0 / 3.0,
    support_margin: float = 1.5,
    prefix_rmse_margin: float = 0.0,
    prefix_log_score_margin: float = 0.0,
) -> HybridReliabilityCalibration:
    """Fit a conservative gate from disjoint source cases and source futures."""

    cases = tuple(source_cases)
    if len(cases) < 2:
        raise ValueError("hybrid reliability calibration requires two source cases")
    identifiers = tuple(case.case_id for case in cases)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("source reliability case IDs must be unique")
    artifact_ids = tuple(case.source_case_artifact_id for case in cases)
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("source reliability artifacts must be unique")
    prefix_input_ids = tuple(case.target_prefix_input_id for case in cases)
    if len(set(prefix_input_ids)) != len(prefix_input_ids):
        raise ValueError("source reliability prefix inputs must be unique")
    prefix_counts = {case.prefix_frame_count for case in cases}
    if len(prefix_counts) != 1:
        raise ValueError("source reliability cases must share one prefix length")
    minimum_mean = _finite_float(
        minimum_mean_source_future_relative_improvement,
        name="minimum_mean_source_future_relative_improvement",
    )
    minimum_win_fraction = _finite_float(
        minimum_source_future_win_fraction,
        name="minimum_source_future_win_fraction",
        minimum=0.0,
    )
    if minimum_win_fraction > 1.0:
        raise ValueError("minimum_source_future_win_fraction must lie in [0, 1]")
    margin = _finite_float(support_margin, name="support_margin", minimum=1.0)
    rmse_margin = _finite_float(
        prefix_rmse_margin,
        name="prefix_rmse_margin",
        minimum=0.0,
    )
    log_score_margin = _finite_float(
        prefix_log_score_margin,
        name="prefix_log_score_margin",
        minimum=0.0,
    )

    diagnostics = [hybrid_reliability_diagnostics(case) for case in cases]
    prefix_rmse_improvements = tuple(
        float(value["prefix_rmse_relative_improvement"]) for value in diagnostics
    )
    prefix_log_score_gains = tuple(
        float(value["prefix_gaussian_log_score_gain"]) for value in diagnostics
    )
    correction_ratios = tuple(
        float(value["correction_to_physics_standard_deviation_ratio"])
        for value in diagnostics
    )
    leverages = tuple(float(value["descriptor_leverage"]) for value in diagnostics)
    future_improvements = tuple(_future_relative_improvement(case) for case in cases)
    mean_future = float(np.mean(future_improvements))
    future_win_fraction = float(np.mean(np.asarray(future_improvements) > 0.0))
    enabled = (
        mean_future >= minimum_mean
        and future_win_fraction >= minimum_win_fraction
    )
    return HybridReliabilityCalibration(
        source_case_ids=identifiers,
        source_case_artifact_ids=artifact_ids,
        source_prefix_input_ids=prefix_input_ids,
        prefix_frame_count=next(iter(prefix_counts)),
        source_prefix_rmse_relative_improvements=prefix_rmse_improvements,
        source_prefix_log_score_gains=prefix_log_score_gains,
        source_correction_standard_deviation_ratios=correction_ratios,
        source_descriptor_leverages=leverages,
        source_future_relative_improvements=future_improvements,
        minimum_prefix_rmse_relative_improvement=max(
            0.0,
            min(prefix_rmse_improvements) - rmse_margin,
        ),
        minimum_prefix_log_score_gain=max(
            0.0,
            min(prefix_log_score_gains) - log_score_margin,
        ),
        maximum_correction_standard_deviation_ratio=max(
            max(correction_ratios) * margin,
            1e-12,
        ),
        maximum_descriptor_leverage=max(max(leverages) * margin, 1e-12),
        minimum_mean_source_future_relative_improvement=minimum_mean,
        minimum_source_future_win_fraction=minimum_win_fraction,
        mean_source_future_relative_improvement=mean_future,
        source_future_win_fraction=future_win_fraction,
        support_margin=margin,
        hybrid_enabled=enabled,
    )


@dataclass(frozen=True)
class HybridReliabilityDecision:
    """One target decision that is explicitly independent of target futures."""

    calibration_id: str
    case_id: str
    target_prefix_input_id: str
    accepted: bool
    selected_method: Literal["physics_only", "hybrid"]
    reasons: tuple[str, ...]
    diagnostics: Mapping[str, float | int]
    target_future_observations_read: int = 0
    target_future_outcomes_used: bool = False

    def __post_init__(self) -> None:
        _require_sha256(self.calibration_id, name="calibration_id")
        _require_sha256(self.target_prefix_input_id, name="target_prefix_input_id")
        if type(self.case_id) is not str or not self.case_id:
            raise ValueError("decision case_id must be nonempty")
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be Boolean")
        expected_method = "hybrid" if self.accepted else "physics_only"
        if self.selected_method != expected_method:
            raise ValueError("selected_method contradicts accepted")
        reasons = tuple(self.reasons)
        if any(type(value) is not str or not value for value in reasons):
            raise ValueError("decision reasons must contain nonempty strings")
        if len(set(reasons)) != len(reasons):
            raise ValueError("decision reasons must be unique")
        if self.accepted and reasons:
            raise ValueError("accepted decisions cannot contain rejection reasons")
        if not self.accepted and not reasons:
            raise ValueError("rejected decisions require a reason")
        if self.target_future_observations_read != 0:
            raise ValueError("hybrid reliability decisions cannot read target futures")
        if self.target_future_outcomes_used is not False:
            raise ValueError("hybrid reliability decisions cannot use target futures")
        diagnostics = validated_json_mapping(
            self.diagnostics,
            error_message="hybrid reliability diagnostics must be finite JSON",
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            for value in diagnostics.values()
        ):
            raise ValueError("hybrid reliability diagnostics must be finite numbers")
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "diagnostics", diagnostics)

    @property
    def decision_id(self) -> str:
        payload = {
            "schema_version": HYBRID_RELIABILITY_SCHEMA_VERSION,
            "artifact_kind": "Causal4DHybridReliabilityDecision",
            **asdict(self),
        }
        return _canonical_sha256(payload)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HYBRID_RELIABILITY_SCHEMA_VERSION,
            "artifact_kind": "Causal4DHybridReliabilityDecision",
            **asdict(self),
            "decision_id": self.decision_id,
        }


def apply_hybrid_reliability(
    case: HybridReliabilityCase,
    calibration: HybridReliabilityCalibration,
) -> tuple[PredictiveDistribution, HybridReliabilityDecision]:
    """Select hybrid or return the exact physics-only prediction object."""

    if case.case_id in calibration.source_case_ids:
        raise ValueError("target reliability case must be disjoint from calibration")
    if case.target_prefix_input_id in calibration.source_prefix_input_ids:
        raise ValueError("target reliability prefix reuses a calibration source")
    if case.prefix_frame_count != calibration.prefix_frame_count:
        raise ValueError("target prefix length differs from calibration")
    diagnostics = hybrid_reliability_diagnostics(case)
    reasons: list[str] = []
    if not calibration.hybrid_enabled:
        reasons.append("no_source_future_gain")
    if (
        diagnostics["prefix_rmse_relative_improvement"]
        < calibration.minimum_prefix_rmse_relative_improvement
    ):
        reasons.append("prefix_point_score_not_supported")
    if (
        diagnostics["prefix_gaussian_log_score_gain"]
        < calibration.minimum_prefix_log_score_gain
    ):
        reasons.append("prefix_probabilistic_score_not_supported")
    if (
        diagnostics["correction_to_physics_standard_deviation_ratio"]
        > calibration.maximum_correction_standard_deviation_ratio
    ):
        reasons.append("correction_outside_source_scale")
    if diagnostics["descriptor_leverage"] > calibration.maximum_descriptor_leverage:
        reasons.append("descriptor_outside_source_support")
    accepted = not reasons
    selected = case.hybrid if accepted else case.physics
    decision = HybridReliabilityDecision(
        calibration_id=calibration.calibration_id,
        case_id=case.case_id,
        target_prefix_input_id=case.target_prefix_input_id,
        accepted=accepted,
        selected_method="hybrid" if accepted else "physics_only",
        reasons=tuple(reasons),
        diagnostics=diagnostics,
    )
    if not accepted and selected is not case.physics:
        raise RuntimeError("rejection failed to return exact physics-only fallback")
    return selected, decision


def save_hybrid_reliability_calibration(
    path: str | Path,
    calibration: HybridReliabilityCalibration,
    *,
    overwrite: bool = False,
) -> None:
    """Publish one finite, content-addressed calibration atomically."""

    atomic_write_json(path, calibration.as_dict(), overwrite=overwrite)


def load_hybrid_reliability_calibration(
    path: str | Path,
) -> HybridReliabilityCalibration:
    """Load and independently validate one exact-byte calibration artifact."""

    snapshot = read_regular_file(path, name="hybrid reliability calibration")
    payload = load_strict_json_object(
        snapshot.payload,
        name="hybrid reliability calibration",
    )
    _require_exact_fields(
        payload,
        _CALIBRATION_FIELDS,
        name="hybrid reliability calibration",
    )
    if payload.pop("schema_version") != HYBRID_RELIABILITY_SCHEMA_VERSION:
        raise ValueError("unsupported hybrid reliability calibration schema")
    if payload.pop("artifact_kind") != "Causal4DHybridReliabilityCalibration":
        raise ValueError("unexpected hybrid reliability calibration artifact kind")
    expected_id = _require_sha256(
        payload.pop("calibration_id"),
        name="calibration_id",
    )
    for name in _CALIBRATION_VECTOR_FIELDS:
        value = payload.get(name)
        if not isinstance(value, list):
            raise ValueError(f"{name} must be a JSON array")
        payload[name] = tuple(value)
    calibration = HybridReliabilityCalibration(**payload)
    if calibration.calibration_id != expected_id:
        raise ValueError("hybrid reliability calibration checksum mismatch")
    return calibration


def write_hybrid_reliability_decision(
    path: str | Path,
    decision: HybridReliabilityDecision,
    *,
    overwrite: bool = False,
) -> None:
    """Publish one finite target decision atomically."""

    atomic_write_json(path, decision.as_dict(), overwrite=overwrite)


__all__ = [
    "HYBRID_RELIABILITY_SCHEMA_VERSION",
    "HybridReliabilityCalibration",
    "HybridReliabilityCase",
    "HybridReliabilityDecision",
    "apply_hybrid_reliability",
    "fit_hybrid_reliability_calibration",
    "hybrid_reliability_diagnostics",
    "load_hybrid_reliability_calibration",
    "ridge_descriptor_leverage",
    "save_hybrid_reliability_calibration",
    "write_hybrid_reliability_decision",
]
