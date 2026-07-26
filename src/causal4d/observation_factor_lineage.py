"""Validate and bind exact Prob4D observation-factor bundle lineage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import TwinBelief

OBSERVATION_FACTOR_SCHEMA = "prob4d.observation-factor-bundle"
OBSERVATION_FACTOR_SCHEMA_VERSION = 3
GAUGE_PARAMETERIZATION = "log-scale-rotvec-translation-v1"
_REQUIRED_FACTOR_ARRAYS = {
    "point_ids",
    "points_local_m",
    "valid_mask",
    "local_covariance_m2",
    "association_probability",
    "prior_reliability",
}


def file_sha256(path: str | Path) -> str:
    """Hash one artifact file without loading it fully into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def compute_observation_factor_bundle_id(
    manifest_sha256: str,
    payload_sha256: str,
) -> str:
    """Content-address the exact manifest and payload byte pair."""

    _validate_sha256(manifest_sha256, name="manifest_sha256")
    _validate_sha256(payload_sha256, name="payload_sha256")
    digest = hashlib.sha256()
    digest.update(f"{OBSERVATION_FACTOR_SCHEMA}\0".encode("utf-8"))
    digest.update(str(OBSERVATION_FACTOR_SCHEMA_VERSION).encode("ascii"))
    digest.update(b"\0")
    digest.update(manifest_sha256.encode("ascii"))
    digest.update(b"\0")
    digest.update(payload_sha256.encode("ascii"))
    return digest.hexdigest()


def _nonempty_string(value: Any, *, name: str) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def _bounded_probability(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _composite_weight(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError("factor composite_weight must lie in (0, 1]")
    return result


def _probability_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(
        (result < 0.0) | (result > 1.0)
    ):
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _require_psd(
    values: np.ndarray,
    *,
    name: str,
    tolerance: float = 1e-12,
) -> None:
    symmetric = 0.5 * (values + np.swapaxes(values, -1, -2))
    if not np.allclose(values, symmetric, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if np.any(eigenvalues < -tolerance):
        raise ValueError(f"{name} must be positive semidefinite")


def _safe_payload_path(manifest: Path, relative: Any) -> Path:
    relative_path = Path(str(relative))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("factor-bundle payload path must stay below the manifest")
    root = manifest.parent.resolve()
    payload = (root / relative_path).resolve()
    if payload != root and root not in payload.parents:
        raise ValueError("factor-bundle payload path escapes the manifest directory")
    if not payload.is_file():
        raise ValueError("factor-bundle payload file is missing")
    return payload


@dataclass(frozen=True)
class ObservationFactorLineage:
    """Immutable summary of a validated Prob4D schema-v3 factor bundle."""

    artifact_id: str
    manifest_sha256: str
    payload_sha256: str
    case_id: str
    stream_id: str
    sequence_id: str
    causal_frame_stop: int
    minimum_frame_id: int
    maximum_frame_id: int
    factor_count: int
    observation_count: int
    active_observation_count: int
    gauge_count: int
    correlation_group_count: int
    source_repository: str
    source_revision: str

    def __post_init__(self) -> None:
        _validate_sha256(self.artifact_id, name="artifact_id")
        _validate_sha256(self.manifest_sha256, name="manifest_sha256")
        _validate_sha256(self.payload_sha256, name="payload_sha256")
        for name, value in (
            ("case_id", self.case_id),
            ("stream_id", self.stream_id),
            ("sequence_id", self.sequence_id),
            ("source_repository", self.source_repository),
            ("source_revision", self.source_revision),
        ):
            if not value:
                raise ValueError(f"{name} must be nonempty")
        if self.causal_frame_stop < 1:
            raise ValueError("factor-bundle causal frame stop must be positive")
        if not 0 <= self.minimum_frame_id <= self.maximum_frame_id:
            raise ValueError("factor-bundle frame range is invalid")
        if self.maximum_frame_id >= self.causal_frame_stop:
            raise ValueError("factor-bundle lineage crosses its causal stop")
        if min(
            self.factor_count,
            self.observation_count,
            self.active_observation_count,
            self.gauge_count,
            self.correlation_group_count,
        ) < 1:
            raise ValueError("factor-bundle lineage counts must be positive")
        if self.active_observation_count > self.observation_count:
            raise ValueError("active observation count exceeds total observations")

    def metadata(self) -> dict[str, Any]:
        return {
            "source_observation_factor_bundle_id": self.artifact_id,
            "source_observation_factor_schema": OBSERVATION_FACTOR_SCHEMA,
            "source_observation_factor_schema_version": (
                OBSERVATION_FACTOR_SCHEMA_VERSION
            ),
            "source_observation_factor_case_id": self.case_id,
            "source_observation_factor_stream_id": self.stream_id,
            "source_observation_factor_sequence_id": self.sequence_id,
            "source_observation_factor_causal_frame_stop": (
                self.causal_frame_stop
            ),
            "source_observation_factor_repository": self.source_repository,
            "source_observation_factor_revision": self.source_revision,
            "source_observation_factor_manifest_sha256": (
                self.manifest_sha256
            ),
            "source_observation_factor_payload_sha256": self.payload_sha256,
        }


def load_observation_factor_lineage(
    manifest_path: str | Path,
) -> ObservationFactorLineage:
    """Validate a Prob4D factor bundle without importing its producer."""

    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise ValueError("factor-bundle manifest file is missing")
    manifest_sha = file_sha256(manifest)
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("factor-bundle manifest is not valid UTF-8 JSON") from error
    if record.get("schema") != OBSERVATION_FACTOR_SCHEMA:
        raise ValueError("unsupported observation-factor schema")
    if int(record.get("schema_version", -1)) != (
        OBSERVATION_FACTOR_SCHEMA_VERSION
    ):
        raise ValueError("unsupported observation-factor schema version")
    if record.get("gauge_parameterization") != GAUGE_PARAMETERIZATION:
        raise ValueError("unsupported observation-factor gauge parameterization")
    if record.get("causal_frame_stop_convention") != "exclusive":
        raise ValueError("factor-bundle causal frame stop must be exclusive")

    sequence_id = _nonempty_string(
        record.get("sequence_id", ""),
        name="sequence_id",
    )
    case_id = _nonempty_string(record.get("case_id", ""), name="case_id")
    stream_id = _nonempty_string(record.get("stream_id", ""), name="stream_id")
    source_repository = _nonempty_string(
        record.get("source_repository", ""),
        name="source_repository",
    )
    source_revision = _nonempty_string(
        record.get("source_revision", ""),
        name="source_revision",
    )
    causal_stop = int(record.get("causal_frame_stop", -1))
    if causal_stop < 1:
        raise ValueError("factor-bundle causal frame stop must be positive")

    payload_record = record.get("payload")
    if not isinstance(payload_record, Mapping):
        raise ValueError("factor-bundle payload descriptor is missing")
    if payload_record.get("allow_pickle") is not False:
        raise ValueError("factor-bundle payload must disable pickle")
    declared_payload_sha = str(payload_record.get("sha256", ""))
    _validate_sha256(declared_payload_sha, name="payload sha256")
    payload = _safe_payload_path(manifest, payload_record.get("path", ""))
    actual_payload_sha = file_sha256(payload)
    if actual_payload_sha != declared_payload_sha:
        raise ValueError("factor-bundle payload checksum mismatch")

    gauge_records = record.get("gauges")
    factor_records = record.get("factors")
    if not isinstance(gauge_records, list) or not gauge_records:
        raise ValueError("factor bundle must contain gauge records")
    if not isinstance(factor_records, list) or not factor_records:
        raise ValueError("factor bundle must contain factor records")

    expected_array_names: list[str] = []
    gauge_ids: set[str] = set()
    factor_ids: set[str] = set()
    group_parameters: dict[str, tuple[float, float]] = {}
    frame_ids: list[int] = []
    total_observations = 0
    active_observations = 0

    with np.load(payload, allow_pickle=False) as arrays:
        for position, gauge_record in enumerate(gauge_records):
            if not isinstance(gauge_record, Mapping):
                raise ValueError("factor-bundle gauge record must be a mapping")
            gauge_id = _nonempty_string(
                gauge_record.get("gauge_id", ""),
                name=f"gauge {position} ID",
            )
            if gauge_id in gauge_ids:
                raise ValueError("factor-bundle gauge IDs must be unique")
            gauge_ids.add(gauge_id)
            mean_key = _nonempty_string(
                gauge_record.get("mean_key", ""),
                name=f"gauge {position} mean key",
            )
            covariance_key = _nonempty_string(
                gauge_record.get("covariance_key", ""),
                name=f"gauge {position} covariance key",
            )
            expected_array_names.extend((mean_key, covariance_key))
            if mean_key not in arrays or covariance_key not in arrays:
                raise ValueError("factor-bundle gauge payload arrays are missing")
            mean = np.asarray(arrays[mean_key], dtype=np.float64)
            covariance = np.asarray(
                arrays[covariance_key],
                dtype=np.float64,
            )
            if mean.shape != (7,) or not np.all(np.isfinite(mean)):
                raise ValueError("factor-bundle gauge mean must have shape (7,)")
            if covariance.shape != (7, 7) or not np.all(
                np.isfinite(covariance)
            ):
                raise ValueError(
                    "factor-bundle gauge covariance must have shape (7, 7)"
                )
            _require_psd(covariance, name="factor-bundle gauge covariance")

        for position, factor_record in enumerate(factor_records):
            if not isinstance(factor_record, Mapping):
                raise ValueError("factor-bundle factor record must be a mapping")
            identifiers = {
                name: _nonempty_string(
                    factor_record.get(name, ""),
                    name=f"factor {position} {name}",
                )
                for name in (
                    "factor_id",
                    "view_id",
                    "window_id",
                    "gauge_id",
                    "correlation_group_id",
                )
            }
            factor_id = identifiers["factor_id"]
            if factor_id in factor_ids:
                raise ValueError("factor-bundle factor IDs must be unique")
            factor_ids.add(factor_id)
            if identifiers["gauge_id"] not in gauge_ids:
                raise ValueError("factor references an unavailable gauge")
            frame_index = int(factor_record.get("frame_index", -1))
            factor_stop = int(factor_record.get("causal_frame_stop", -1))
            if frame_index < 0 or frame_index >= causal_stop:
                raise ValueError("factor frame crosses the bundle causal stop")
            if factor_stop != causal_stop:
                raise ValueError("factor and bundle causal frame stops differ")
            frame_ids.append(frame_index)

            nominal_probability = _bounded_probability(
                factor_record.get("prior_nominal_probability"),
                name="factor prior_nominal_probability",
            )
            composite_weight = _composite_weight(
                factor_record.get("composite_weight")
            )
            group_id = identifiers["correlation_group_id"]
            parameters = (nominal_probability, composite_weight)
            previous = group_parameters.setdefault(group_id, parameters)
            if previous != parameters:
                raise ValueError(
                    "one correlation group has inconsistent factor metadata"
                )

            key_record = factor_record.get("arrays")
            if not isinstance(key_record, Mapping):
                raise ValueError("factor payload array mapping is missing")
            missing_keys = _REQUIRED_FACTOR_ARRAYS - key_record.keys()
            extra_keys = key_record.keys() - _REQUIRED_FACTOR_ARRAYS
            if missing_keys or extra_keys:
                raise ValueError(
                    "factor payload array contract changed; "
                    f"missing={sorted(missing_keys)}, "
                    f"extra={sorted(extra_keys)}"
                )
            factor_keys = {
                name: _nonempty_string(
                    key_record[name],
                    name=f"factor {position} {name} key",
                )
                for name in _REQUIRED_FACTOR_ARRAYS
            }
            expected_array_names.extend(factor_keys.values())
            ray_key_value = factor_record.get("ray_directions_local_key")
            ray_key = None
            if ray_key_value is not None:
                ray_key = _nonempty_string(
                    ray_key_value,
                    name=f"factor {position} ray key",
                )
                expected_array_names.append(ray_key)
            if any(key not in arrays for key in factor_keys.values()):
                raise ValueError("factor payload arrays are missing")
            if ray_key is not None and ray_key not in arrays:
                raise ValueError("factor ray payload array is missing")

            point_ids = np.asarray(arrays[factor_keys["point_ids"]])
            points = np.asarray(
                arrays[factor_keys["points_local_m"]],
                dtype=np.float64,
            )
            valid = np.asarray(arrays[factor_keys["valid_mask"]])
            covariance = np.asarray(
                arrays[factor_keys["local_covariance_m2"]],
                dtype=np.float64,
            )
            association = _probability_vector(
                arrays[factor_keys["association_probability"]],
                name="factor association probability",
            )
            reliability = _probability_vector(
                arrays[factor_keys["prior_reliability"]],
                name="factor prior reliability",
            )
            if point_ids.ndim != 1 or len(point_ids) == 0:
                raise ValueError("factor point IDs must be a nonempty vector")
            if not np.issubdtype(point_ids.dtype, np.integer):
                raise ValueError("factor point IDs must use an integer dtype")
            point_ids = np.asarray(point_ids, dtype=np.int64)
            if np.any(point_ids < 0) or len(np.unique(point_ids)) != len(
                point_ids
            ):
                raise ValueError(
                    "factor point IDs must be nonnegative and unique"
                )
            count = len(point_ids)
            if points.shape != (count, 3):
                raise ValueError("factor points must have shape (N, 3)")
            if valid.dtype != np.dtype(bool) or valid.shape != (count,):
                raise ValueError("factor valid mask must be a Boolean vector")
            if covariance.shape != (count, 3, 3):
                raise ValueError(
                    "factor local covariance must have shape (N, 3, 3)"
                )
            if association.shape != (count,) or reliability.shape != (count,):
                raise ValueError(
                    "factor probability vectors must identify every point"
                )
            active = valid & (association > 0.0) & (reliability > 0.0)
            if not np.all(np.isfinite(points[active])):
                raise ValueError("active factor points must be finite")
            if not np.all(np.isfinite(covariance[active])):
                raise ValueError("active factor covariance must be finite")
            if np.any(active):
                _require_psd(
                    covariance[active],
                    name="active factor covariance",
                )
            if ray_key is not None:
                rays = np.asarray(arrays[ray_key], dtype=np.float64)
                if rays.shape != (count, 3):
                    raise ValueError("factor rays must have shape (N, 3)")
                if not np.all(np.isfinite(rays[active])):
                    raise ValueError("active factor rays must be finite")
                if np.any(
                    active
                    & (
                        np.linalg.norm(rays, axis=1)
                        <= np.finfo(np.float64).eps
                    )
                ):
                    raise ValueError("active factor rays must be nonzero")
            total_observations += count
            active_observations += int(np.sum(active))

        if len(expected_array_names) != len(set(expected_array_names)):
            raise ValueError("factor-bundle payload array keys are reused")
        expected_arrays = set(expected_array_names)
        actual_arrays = set(arrays.files)
        missing_arrays = expected_arrays - actual_arrays
        extra_arrays = actual_arrays - expected_arrays
        if missing_arrays or extra_arrays:
            raise ValueError(
                "factor-bundle payload array set changed; "
                f"missing={sorted(missing_arrays)}, "
                f"extra={sorted(extra_arrays)}"
            )

    if active_observations < 1:
        raise ValueError("factor bundle has no active observation rows")
    artifact_id = compute_observation_factor_bundle_id(
        manifest_sha,
        actual_payload_sha,
    )
    return ObservationFactorLineage(
        artifact_id=artifact_id,
        manifest_sha256=manifest_sha,
        payload_sha256=actual_payload_sha,
        case_id=case_id,
        stream_id=stream_id,
        sequence_id=sequence_id,
        causal_frame_stop=causal_stop,
        minimum_frame_id=min(frame_ids),
        maximum_frame_id=max(frame_ids),
        factor_count=len(factor_records),
        observation_count=total_observations,
        active_observation_count=active_observations,
        gauge_count=len(gauge_records),
        correlation_group_count=len(group_parameters),
        source_repository=source_repository,
        source_revision=source_revision,
    )


def validate_twin_belief_observation_factor_lineage(
    twin_belief: TwinBelief,
    lineage: ObservationFactorLineage,
    *,
    require_bound: bool = True,
) -> dict[str, Any]:
    """Check case, O-minus containment, and exact factor-bundle binding."""

    if twin_belief.context.case_id != lineage.case_id:
        raise ValueError("factor bundle and TwinBelief identify different cases")
    if lineage.minimum_frame_id < twin_belief.context.o_minus.frame_start:
        raise ValueError("factor bundle begins before the TwinBelief O- boundary")
    if lineage.causal_frame_stop > twin_belief.context.o_minus.frame_stop:
        raise ValueError("factor bundle extends beyond the TwinBelief O- boundary")

    expected_metadata = lineage.metadata()
    metadata = twin_belief.metadata
    bound_id = metadata.get("source_observation_factor_bundle_id")
    if bound_id is not None and bound_id != lineage.artifact_id:
        raise ValueError("TwinBelief is bound to a different factor bundle")
    if require_bound and bound_id is None:
        raise ValueError("TwinBelief has no source factor-bundle binding")
    if bound_id is not None:
        for key, expected in expected_metadata.items():
            actual = metadata.get(key)
            if actual != expected:
                raise ValueError(
                    f"TwinBelief factor-bundle metadata mismatch for {key}"
                )
    return {
        "status": "valid",
        "twin_belief_id": twin_belief.artifact_id,
        "observation_factor_bundle_id": lineage.artifact_id,
        "case_id": lineage.case_id,
        "stream_id": lineage.stream_id,
        "sequence_id": lineage.sequence_id,
        "lineage_bound": bound_id == lineage.artifact_id,
        "observation_frame_range": [
            lineage.minimum_frame_id,
            lineage.maximum_frame_id,
        ],
        "observation_causal_frame_stop": lineage.causal_frame_stop,
        "twin_o_minus_frame_range": [
            twin_belief.context.o_minus.frame_start,
            twin_belief.context.o_minus.frame_stop,
        ],
        "factor_count": lineage.factor_count,
        "observation_count": lineage.observation_count,
        "active_observation_count": lineage.active_observation_count,
        "gauge_count": lineage.gauge_count,
        "correlation_group_count": lineage.correlation_group_count,
        "manifest_sha256": lineage.manifest_sha256,
        "payload_sha256": lineage.payload_sha256,
    }


def bind_twin_belief_observation_factor_lineage(
    twin_belief: TwinBelief,
    lineage: ObservationFactorLineage,
) -> TwinBelief:
    """Bind the exact factor bundle actually consumed by the estimator."""

    validate_twin_belief_observation_factor_lineage(
        twin_belief,
        lineage,
        require_bound=False,
    )
    metadata = dict(twin_belief.metadata)
    existing = metadata.get("source_observation_factor_bundle_id")
    if existing is not None and existing != lineage.artifact_id:
        raise ValueError("TwinBelief already has incompatible factor lineage")
    metadata.update(lineage.metadata())
    return replace(twin_belief, metadata=metadata)


__all__ = [
    "GAUGE_PARAMETERIZATION",
    "OBSERVATION_FACTOR_SCHEMA",
    "OBSERVATION_FACTOR_SCHEMA_VERSION",
    "ObservationFactorLineage",
    "bind_twin_belief_observation_factor_lineage",
    "compute_observation_factor_bundle_id",
    "file_sha256",
    "load_observation_factor_lineage",
    "validate_twin_belief_observation_factor_lineage",
]
