"""Source-only predictive clock-offset priors for downstream observation models.

The existing actuator-realization diagnostic estimates one timestamp correction
per independent source or dry-run execution. This module aggregates those
execution-level artifacts into a content-addressed predictive prior without
changing the original calibration implementation or frozen evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

OBSERVATION_CLOCK_OFFSET_PRIOR_SCHEMA = "causal4d.observation-clock-offset-prior"
OBSERVATION_CLOCK_OFFSET_PRIOR_VERSION = 1
OBSERVATION_TIME_CORRECTION_CONVENTION = (
    "aligned_observation_time_s = observation_time_s + offset_s"
)

_INFORMATION_BOUNDARY: dict[str, bool] = {
    "source_or_dry_run_only": True,
    "target_outcomes_used": False,
    "hardware_timestamps_authoritative": True,
    "equal_weight_per_execution": True,
}
_CLAIM_BOUNDARY = (
    "This predictive timing prior does not identify contact slip, material "
    "relaxation, controller-frame physics, or downstream physical-query benefit."
)
_PRIOR_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "clock_domain",
        "reference_clock_domain",
        "time_scale",
        "offset_convention",
        "source_revision",
        "source_artifact_ids",
        "execution_ids",
        "source_offsets_s",
        "source_group_count",
        "mean_offset_s",
        "sample_standard_deviation_s",
        "grid_quantization_standard_deviation_s",
        "minimum_predictive_standard_deviation_s",
        "predictive_standard_deviation_s",
        "information_boundary",
        "claim_boundary",
    }
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _nonempty_string(value: object, *, name: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{name} must be nonempty")
    result = str(value)
    _require(result == result.strip(), f"{name} has surrounding whitespace")
    return result


def _sha256(value: object, *, name: str) -> str:
    digest = _nonempty_string(value, name=name)
    _require(
        len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return digest


def _revision(value: object, *, name: str) -> str:
    revision = _nonempty_string(value, name=name)
    _require(
        len(revision) in {40, 64}
        and all(character in "0123456789abcdef" for character in revision),
        f"{name} must be an exact lowercase Git commit",
    )
    return revision


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be finite")
    raw = array.item()
    if isinstance(raw, bool) or not isinstance(
        raw,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be finite")
    result = float(raw)
    _require(np.isfinite(result), f"{name} must be finite")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    _require(result >= 1, f"{name} must be a positive integer")
    return result


def _sequence(value: object, *, name: str) -> Sequence[object]:
    _require(
        not isinstance(value, (str, bytes)) and isinstance(value, Sequence),
        f"{name} must be a sequence",
    )
    return value


def _ordered_unique_strings(value: object, *, name: str) -> tuple[str, ...]:
    values = _sequence(value, name=name)
    result = tuple(_nonempty_string(item, name=f"{name} entry") for item in values)
    _require(bool(result), f"{name} must not be empty")
    _require(len(set(result)) == len(result), f"{name} must be unique")
    return result


def _ordered_sha256s(value: object, *, name: str) -> tuple[str, ...]:
    values = _sequence(value, name=name)
    result = tuple(_sha256(item, name=f"{name} entry") for item in values)
    _require(bool(result), f"{name} must not be empty")
    _require(len(set(result)) == len(result), f"{name} must be unique")
    return result


def _ordered_finite_floats(value: object, *, name: str) -> tuple[float, ...]:
    values = _sequence(value, name=name)
    result = tuple(_finite_float(item, name=f"{name} entry") for item in values)
    _require(bool(result), f"{name} must not be empty")
    return result


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("artifact contains non-JSON or non-finite values") from error
    return encoded.encode("utf-8")


def _content_id(value: Mapping[str, Any]) -> str:
    canonical = dict(value)
    canonical.pop("artifact_id", None)
    return hashlib.sha256(_canonical_json_bytes(canonical)).hexdigest()


def _predictive_summary(
    offsets_s: Sequence[float],
    *,
    grid_standard_deviation_s: float,
    minimum_standard_deviation_s: float,
) -> tuple[float, float, float]:
    values = np.asarray(offsets_s, dtype=np.float64)
    _require(len(values) >= 3, "at least three source executions are required")
    mean = float(np.mean(values))
    sample_standard_deviation = float(np.std(values, ddof=1))
    predictive_variance = (
        1.0 + 1.0 / len(values)
    ) * sample_standard_deviation**2 + grid_standard_deviation_s**2
    predictive_standard_deviation = max(
        minimum_standard_deviation_s,
        math.sqrt(max(0.0, predictive_variance)),
    )
    return mean, sample_standard_deviation, predictive_standard_deviation


@dataclass(frozen=True, slots=True)
class ObservationClockOffsetPriorV1:
    """Equal-execution predictive prior for one observation clock domain."""

    clock_domain: str
    reference_clock_domain: str
    time_scale: str
    source_revision: str
    source_artifact_ids: tuple[str, ...]
    execution_ids: tuple[str, ...]
    source_offsets_s: tuple[float, ...]
    mean_offset_s: float
    sample_standard_deviation_s: float
    grid_quantization_standard_deviation_s: float
    minimum_predictive_standard_deviation_s: float
    predictive_standard_deviation_s: float
    source_group_count: int
    offset_convention: str = OBSERVATION_TIME_CORRECTION_CONVENTION
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        clock_domain = _nonempty_string(self.clock_domain, name="clock_domain")
        reference_clock_domain = _nonempty_string(
            self.reference_clock_domain,
            name="reference_clock_domain",
        )
        _require(
            clock_domain != reference_clock_domain,
            "clock and reference clock domains must differ",
        )
        time_scale = _nonempty_string(self.time_scale, name="time_scale")
        source_revision = _revision(self.source_revision, name="source_revision")
        source_artifact_ids = _ordered_sha256s(
            self.source_artifact_ids,
            name="source_artifact_ids",
        )
        execution_ids = _ordered_unique_strings(
            self.execution_ids,
            name="execution_ids",
        )
        source_offsets = _ordered_finite_floats(
            self.source_offsets_s,
            name="source_offsets_s",
        )
        _require(
            len(source_artifact_ids) == len(execution_ids) == len(source_offsets),
            "source timing evidence counts differ",
        )
        _require(
            execution_ids == tuple(sorted(execution_ids)),
            "execution IDs must use deterministic sorted order",
        )
        source_group_count = _positive_integer(
            self.source_group_count,
            name="source_group_count",
        )
        _require(
            source_group_count == len(execution_ids) and source_group_count >= 3,
            "source_group_count must equal at least three executions",
        )
        _require(
            self.offset_convention == OBSERVATION_TIME_CORRECTION_CONVENTION,
            "observation time-correction convention changed",
        )
        grid_standard_deviation = _finite_float(
            self.grid_quantization_standard_deviation_s,
            name="grid_quantization_standard_deviation_s",
        )
        minimum_standard_deviation = _finite_float(
            self.minimum_predictive_standard_deviation_s,
            name="minimum_predictive_standard_deviation_s",
        )
        _require(
            grid_standard_deviation > 0.0,
            "grid quantization standard deviation must be positive",
        )
        _require(
            minimum_standard_deviation > 0.0,
            "minimum predictive standard deviation must be positive",
        )
        expected = _predictive_summary(
            source_offsets,
            grid_standard_deviation_s=grid_standard_deviation,
            minimum_standard_deviation_s=minimum_standard_deviation,
        )
        supplied = (
            _finite_float(self.mean_offset_s, name="mean_offset_s"),
            _finite_float(
                self.sample_standard_deviation_s,
                name="sample_standard_deviation_s",
            ),
            _finite_float(
                self.predictive_standard_deviation_s,
                name="predictive_standard_deviation_s",
            ),
        )
        _require(
            all(
                np.isclose(actual, wanted, rtol=1e-13, atol=1e-15)
                for actual, wanted in zip(supplied, expected, strict=True)
            ),
            "clock-offset prior summary does not match source offsets",
        )
        _require(
            supplied[2] >= minimum_standard_deviation,
            "predictive standard deviation is below its declared floor",
        )

        object.__setattr__(self, "clock_domain", clock_domain)
        object.__setattr__(
            self,
            "reference_clock_domain",
            reference_clock_domain,
        )
        object.__setattr__(self, "time_scale", time_scale)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "source_artifact_ids", source_artifact_ids)
        object.__setattr__(self, "execution_ids", execution_ids)
        object.__setattr__(self, "source_offsets_s", source_offsets)
        object.__setattr__(self, "source_group_count", source_group_count)
        object.__setattr__(self, "mean_offset_s", supplied[0])
        object.__setattr__(self, "sample_standard_deviation_s", supplied[1])
        object.__setattr__(
            self,
            "grid_quantization_standard_deviation_s",
            grid_standard_deviation,
        )
        object.__setattr__(
            self,
            "minimum_predictive_standard_deviation_s",
            minimum_standard_deviation,
        )
        object.__setattr__(self, "predictive_standard_deviation_s", supplied[2])

        expected_id = _content_id(self.identity_record())
        if self.artifact_id is not None:
            supplied_id = _sha256(self.artifact_id, name="artifact_id")
            _require(
                supplied_id == expected_id,
                "observation clock-offset prior artifact ID mismatch",
            )
        object.__setattr__(self, "artifact_id", expected_id)

    def identity_record(self) -> dict[str, Any]:
        """Return the canonical payload without its derived content ID."""

        return {
            "schema": OBSERVATION_CLOCK_OFFSET_PRIOR_SCHEMA,
            "schema_version": OBSERVATION_CLOCK_OFFSET_PRIOR_VERSION,
            "clock_domain": self.clock_domain,
            "reference_clock_domain": self.reference_clock_domain,
            "time_scale": self.time_scale,
            "offset_convention": self.offset_convention,
            "source_revision": self.source_revision,
            "source_artifact_ids": list(self.source_artifact_ids),
            "execution_ids": list(self.execution_ids),
            "source_offsets_s": list(self.source_offsets_s),
            "source_group_count": self.source_group_count,
            "mean_offset_s": self.mean_offset_s,
            "sample_standard_deviation_s": self.sample_standard_deviation_s,
            "grid_quantization_standard_deviation_s": (
                self.grid_quantization_standard_deviation_s
            ),
            "minimum_predictive_standard_deviation_s": (
                self.minimum_predictive_standard_deviation_s
            ),
            "predictive_standard_deviation_s": (self.predictive_standard_deviation_s),
            "information_boundary": dict(_INFORMATION_BOUNDARY),
            "claim_boundary": _CLAIM_BOUNDARY,
        }

    def to_record(self) -> dict[str, Any]:
        """Return the complete portable JSON record."""

        return {**self.identity_record(), "artifact_id": self.artifact_id}

    def bayesian_phystwin_prior_payload(self) -> dict[str, Any]:
        """Return fields required by BayesianPhysTwin's timing prior."""

        return {
            "clock_domain": self.clock_domain,
            "mean_offset_s": self.mean_offset_s,
            "standard_deviation_s": self.predictive_standard_deviation_s,
            "source_artifact_id": self.artifact_id,
            "offset_convention": self.offset_convention,
        }

    @classmethod
    def from_record(
        cls,
        value: Mapping[str, Any],
    ) -> ObservationClockOffsetPriorV1:
        """Load and fully revalidate one closed-schema JSON value."""

        _require(
            set(value) == _PRIOR_FIELDS,
            "observation clock-offset prior fields changed",
        )
        _require(
            value.get("schema") == OBSERVATION_CLOCK_OFFSET_PRIOR_SCHEMA,
            "unexpected observation clock-offset prior schema",
        )
        _require(
            value.get("schema_version") == OBSERVATION_CLOCK_OFFSET_PRIOR_VERSION,
            "unsupported observation clock-offset prior version",
        )
        _require(
            value.get("information_boundary") == _INFORMATION_BOUNDARY,
            "observation clock-offset information boundary changed",
        )
        _require(
            value.get("claim_boundary") == _CLAIM_BOUNDARY,
            "observation clock-offset claim boundary changed",
        )
        return cls(
            clock_domain=_nonempty_string(
                value.get("clock_domain"),
                name="clock_domain",
            ),
            reference_clock_domain=_nonempty_string(
                value.get("reference_clock_domain"),
                name="reference_clock_domain",
            ),
            time_scale=_nonempty_string(value.get("time_scale"), name="time_scale"),
            offset_convention=_nonempty_string(
                value.get("offset_convention"),
                name="offset_convention",
            ),
            source_revision=_revision(
                value.get("source_revision"),
                name="source_revision",
            ),
            source_artifact_ids=_ordered_sha256s(
                value.get("source_artifact_ids"),
                name="source_artifact_ids",
            ),
            execution_ids=_ordered_unique_strings(
                value.get("execution_ids"),
                name="execution_ids",
            ),
            source_offsets_s=_ordered_finite_floats(
                value.get("source_offsets_s"),
                name="source_offsets_s",
            ),
            source_group_count=_positive_integer(
                value.get("source_group_count"),
                name="source_group_count",
            ),
            mean_offset_s=_finite_float(
                value.get("mean_offset_s"),
                name="mean_offset_s",
            ),
            sample_standard_deviation_s=_finite_float(
                value.get("sample_standard_deviation_s"),
                name="sample_standard_deviation_s",
            ),
            grid_quantization_standard_deviation_s=_finite_float(
                value.get("grid_quantization_standard_deviation_s"),
                name="grid_quantization_standard_deviation_s",
            ),
            minimum_predictive_standard_deviation_s=_finite_float(
                value.get("minimum_predictive_standard_deviation_s"),
                name="minimum_predictive_standard_deviation_s",
            ),
            predictive_standard_deviation_s=_finite_float(
                value.get("predictive_standard_deviation_s"),
                name="predictive_standard_deviation_s",
            ),
            artifact_id=_sha256(value.get("artifact_id"), name="artifact_id"),
        )


def _validated_calibration(
    value: Mapping[str, Any],
) -> tuple[str, str, float, float]:
    _require(
        value.get("artifact_kind") == "ActuatorRealizationCalibration",
        "source artifact is not an actuator-realization calibration",
    )
    _require(
        value.get("schema_version") == 1,
        "unsupported actuator-realization calibration schema",
    )
    artifact_id = _sha256(value.get("artifact_id"), name="source artifact ID")
    _require(
        artifact_id == _content_id(value),
        "actuator-realization calibration artifact ID mismatch",
    )
    execution_id = _nonempty_string(value.get("execution_id"), name="execution_id")
    boundary = value.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("source_or_dry_run_only") is True
        and boundary.get("target_outcomes_used") is False
        and boundary.get("hardware_timestamps_authoritative") is True,
        "actuator timing artifact violates the source-only information boundary",
    )
    alignment = value.get("timestamp_alignment")
    _require(
        isinstance(alignment, Mapping),
        "actuator timing artifact lacks timestamp alignment",
    )
    _require(
        alignment.get("convention")
        == "aligned_measurement_time_s = measurement_time_s + offset_s",
        "actuator timing convention changed",
    )
    offset = _finite_float(alignment.get("best_offset_s"), name="best_offset_s")
    grid = _ordered_finite_floats(
        alignment.get("offset_grid_s"),
        name="offset_grid_s",
    )
    _require(len(grid) >= 2, "offset grid must contain at least two points")
    differences = np.diff(np.asarray(grid, dtype=np.float64))
    _require(np.all(differences > 0.0), "offset grid must be strictly increasing")
    _require(
        grid[0] <= offset <= grid[-1],
        "best offset lies outside the declared grid",
    )
    return execution_id, artifact_id, offset, float(np.max(differences))


def fit_observation_clock_offset_prior(
    calibrations: Sequence[Mapping[str, Any]],
    *,
    clock_domain: str,
    reference_clock_domain: str,
    time_scale: str,
    source_revision: str,
    minimum_predictive_standard_deviation_s: float = 5e-4,
) -> ObservationClockOffsetPriorV1:
    """Aggregate independent source executions into a predictive timing prior."""

    _require(
        not isinstance(calibrations, (str, bytes))
        and isinstance(calibrations, Sequence),
        "calibrations must be a sequence",
    )
    validated = [_validated_calibration(value) for value in calibrations]
    _require(
        len(validated) >= 3,
        "at least three source executions are required",
    )
    validated.sort(key=lambda row: row[0])
    execution_ids = tuple(row[0] for row in validated)
    artifact_ids = tuple(row[1] for row in validated)
    offsets = tuple(row[2] for row in validated)
    _require(
        len(set(execution_ids)) == len(execution_ids),
        "source execution IDs must be unique",
    )
    _require(
        len(set(artifact_ids)) == len(artifact_ids),
        "source calibration artifact IDs must be unique",
    )
    minimum_standard_deviation = _finite_float(
        minimum_predictive_standard_deviation_s,
        name="minimum_predictive_standard_deviation_s",
    )
    _require(
        minimum_standard_deviation > 0.0,
        "minimum predictive standard deviation must be positive",
    )
    maximum_grid_step = max(row[3] for row in validated)
    grid_standard_deviation = maximum_grid_step / math.sqrt(12.0)
    mean, sample_standard_deviation, predictive_standard_deviation = (
        _predictive_summary(
            offsets,
            grid_standard_deviation_s=grid_standard_deviation,
            minimum_standard_deviation_s=minimum_standard_deviation,
        )
    )
    return ObservationClockOffsetPriorV1(
        clock_domain=clock_domain,
        reference_clock_domain=reference_clock_domain,
        time_scale=time_scale,
        source_revision=source_revision,
        source_artifact_ids=artifact_ids,
        execution_ids=execution_ids,
        source_offsets_s=offsets,
        source_group_count=len(offsets),
        mean_offset_s=mean,
        sample_standard_deviation_s=sample_standard_deviation,
        grid_quantization_standard_deviation_s=grid_standard_deviation,
        minimum_predictive_standard_deviation_s=minimum_standard_deviation,
        predictive_standard_deviation_s=predictive_standard_deviation,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_observation_clock_offset_prior(
    path: str | Path,
) -> ObservationClockOffsetPriorV1:
    """Load an exact JSON snapshot and revalidate all derived values."""

    artifact_path = Path(path)
    if artifact_path.is_symlink():
        raise ValueError("observation clock-offset prior must not be a symlink")
    try:
        payload = artifact_path.read_bytes()
    except OSError as error:
        raise ValueError("observation clock-offset prior is unreadable") from error
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("observation clock-offset prior is invalid JSON") from error
    _require(
        isinstance(value, Mapping),
        "observation clock-offset prior must be a JSON object",
    )
    return ObservationClockOffsetPriorV1.from_record(value)


def write_observation_clock_offset_prior(
    prior: ObservationClockOffsetPriorV1,
    path: str | Path,
) -> None:
    """Publish one complete prior without replacing different content."""

    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.is_symlink():
        raise ValueError("observation clock-offset prior target must not be a symlink")
    if artifact_path.exists():
        existing = load_observation_clock_offset_prior(artifact_path)
        _require(
            existing.artifact_id == prior.artifact_id,
            "observation clock-offset prior path contains different content",
        )
        return
    payload = (
        json.dumps(
            prior.to_record(),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{artifact_path.name}.",
        suffix=".tmp",
        dir=artifact_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, artifact_path)
        except FileExistsError:
            if artifact_path.is_symlink():
                raise ValueError(
                    "observation clock-offset prior target must not be a symlink"
                ) from None
            existing = load_observation_clock_offset_prior(artifact_path)
            if existing.artifact_id != prior.artifact_id:
                raise ValueError(
                    "observation clock-offset prior publication raced with "
                    "different content"
                ) from None
    finally:
        temporary_path.unlink(missing_ok=True)


__all__ = [
    "OBSERVATION_CLOCK_OFFSET_PRIOR_SCHEMA",
    "OBSERVATION_CLOCK_OFFSET_PRIOR_VERSION",
    "OBSERVATION_TIME_CORRECTION_CONVENTION",
    "ObservationClockOffsetPriorV1",
    "fit_observation_clock_offset_prior",
    "load_observation_clock_offset_prior",
    "write_observation_clock_offset_prior",
]
