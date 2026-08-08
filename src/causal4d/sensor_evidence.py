"""Typed independent actuator and contact-wrench evidence artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import numpy as np

from causal4d.artifact_io import (
    ArtifactValidationError,
    load_npz_bytes,
    load_strict_json_object,
    read_regular_file,
)
from causal4d.atomic_io import atomic_write_binary
from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json, validated_json_mapping


INDEPENDENT_SENSOR_SCHEMA_VERSION = 1

_ACTUATOR_ARCHIVE_ARRAYS = frozenset(
    {
        "descriptor_json",
        "sample_times_s",
        "positions_m",
        "variance_m2",
        "valid_mask",
    }
)
_CONTACT_WRENCH_ARCHIVE_ARRAYS = frozenset(
    {
        "descriptor_json",
        "sample_times_s",
        "wrench",
        "variance",
        "valid_mask",
    }
)


def _readonly(values: np.ndarray, *, dtype: type | None = float) -> np.ndarray:
    return readonly_array(values, dtype=dtype)


def _validate_identity(
    *,
    protocol_id: str,
    case_id: str,
    observed_action_id: str,
    stream_id: str,
    clock_id: str,
    provenance: str,
) -> None:
    identities = {
        "protocol_id": protocol_id,
        "case_id": case_id,
        "observed_action_id": observed_action_id,
        "stream_id": stream_id,
        "clock_id": clock_id,
        "provenance": provenance,
    }
    missing = [name for name, value in identities.items() if not value]
    if missing:
        raise ValueError(f"sensor evidence identities must be nonempty: {missing}")


def _identity_payload(
    evidence: ActuatorEvidence | ContactWrenchEvidence,
) -> dict[str, str]:
    return {
        "protocol_id": evidence.protocol_id,
        "case_id": evidence.case_id,
        "observed_action_id": evidence.observed_action_id,
        "stream_id": evidence.stream_id,
        "clock_id": evidence.clock_id,
        "provenance": evidence.provenance,
    }


def _broadcast_variance(
    values: np.ndarray,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    supplied = np.asarray(values, dtype=float)
    try:
        result = np.broadcast_to(supplied, shape).copy()
    except ValueError as error:
        raise ValueError(f"{name} must broadcast to {shape}") from error
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return readonly_array(result)


def _broadcast_mask(
    values: np.ndarray | None,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    if values is None:
        result = np.ones(shape, dtype=bool)
    else:
        supplied = np.asarray(values, dtype=bool)
        try:
            result = np.broadcast_to(supplied, shape).copy()
        except ValueError as error:
            raise ValueError(f"{name} must broadcast to {shape}") from error
    return readonly_array(result)


def _validate_times(values: np.ndarray, sample_count: int) -> np.ndarray:
    times = _readonly(values)
    if times.shape != (sample_count,) or not np.all(np.isfinite(times)):
        raise ValueError("sample_times_s must contain one finite value per sample")
    if sample_count > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("sample_times_s must be strictly increasing")
    return times


def _artifact_id(
    *,
    artifact_kind: str,
    scalar_payload: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> str:
    digest = hashlib.sha256()
    digest.update(artifact_kind.encode("utf-8"))
    digest.update(
        json.dumps(
            plain_json(scalar_payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    for name, values in sorted(arrays.items()):
        digest.update(name.encode("utf-8"))
        digest.update(array_sha256(values).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ActuatorEvidence:
    """Measured end-effector positions independent of object observations."""

    protocol_id: str
    case_id: str
    observed_action_id: str
    stream_id: str
    clock_id: str
    provenance: str
    sample_times_s: np.ndarray
    positions_m: np.ndarray
    variance_m2: np.ndarray
    evidence_frame_stop: int
    valid_mask: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positions = _readonly(self.positions_m)
        if positions.ndim != 3 or positions.shape[2] != 3:
            raise ValueError("positions_m must have shape (sample, controller, 3)")
        times = _validate_times(self.sample_times_s, positions.shape[0])
        variance = _broadcast_variance(
            self.variance_m2,
            positions.shape,
            "variance_m2",
        )
        mask = _broadcast_mask(self.valid_mask, positions.shape, "valid_mask")
        _validate_identity(
            protocol_id=self.protocol_id,
            case_id=self.case_id,
            observed_action_id=self.observed_action_id,
            stream_id=self.stream_id,
            clock_id=self.clock_id,
            provenance=self.provenance,
        )
        if self.evidence_frame_stop < 1:
            raise ValueError("evidence_frame_stop must be positive")
        if not np.all(np.isfinite(positions[mask])):
            raise ValueError("valid actuator positions must be finite")
        object.__setattr__(self, "sample_times_s", times)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "variance_m2", variance)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="metadata must contain finite JSON values",
            ),
        )

    @property
    def artifact_id(self) -> str:
        return _artifact_id(
            artifact_kind="ActuatorEvidence",
            scalar_payload={
                **_identity_payload(self),
                "evidence_frame_stop": self.evidence_frame_stop,
                "metadata": plain_json(self.metadata),
                "independent_of_object_observations": True,
            },
            arrays={
                "sample_times_s": self.sample_times_s,
                "positions_m": self.positions_m,
                "variance_m2": self.variance_m2,
                "valid_mask": self.valid_mask,
            },
        )


@dataclass(frozen=True)
class ContactWrenchEvidence:
    """Measured contact force or wrench evidence on a trusted clock."""

    protocol_id: str
    case_id: str
    observed_action_id: str
    stream_id: str
    clock_id: str
    provenance: str
    sample_times_s: np.ndarray
    wrench: np.ndarray
    variance: np.ndarray
    quantity_names: tuple[str, ...]
    evidence_frame_stop: int
    valid_mask: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        wrench = _readonly(self.wrench)
        if wrench.ndim != 2 or wrench.shape[1] == 0:
            raise ValueError("wrench must have shape (sample, quantity)")
        if len(self.quantity_names) != wrench.shape[1] or len(
            set(self.quantity_names)
        ) != len(self.quantity_names):
            raise ValueError("quantity_names must uniquely identify every column")
        if any(not name for name in self.quantity_names):
            raise ValueError("quantity_names must be nonempty")
        times = _validate_times(self.sample_times_s, wrench.shape[0])
        variance = _broadcast_variance(self.variance, wrench.shape, "variance")
        mask = _broadcast_mask(self.valid_mask, wrench.shape, "valid_mask")
        _validate_identity(
            protocol_id=self.protocol_id,
            case_id=self.case_id,
            observed_action_id=self.observed_action_id,
            stream_id=self.stream_id,
            clock_id=self.clock_id,
            provenance=self.provenance,
        )
        if self.evidence_frame_stop < 1:
            raise ValueError("evidence_frame_stop must be positive")
        if not np.all(np.isfinite(wrench[mask])):
            raise ValueError("valid wrench values must be finite")
        object.__setattr__(self, "sample_times_s", times)
        object.__setattr__(self, "wrench", wrench)
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="metadata must contain finite JSON values",
            ),
        )

    @property
    def artifact_id(self) -> str:
        return _artifact_id(
            artifact_kind="ContactWrenchEvidence",
            scalar_payload={
                **_identity_payload(self),
                "quantity_names": list(self.quantity_names),
                "evidence_frame_stop": self.evidence_frame_stop,
                "metadata": plain_json(self.metadata),
                "independent_of_object_observations": True,
            },
            arrays={
                "sample_times_s": self.sample_times_s,
                "wrench": self.wrench,
                "variance": self.variance,
                "valid_mask": self.valid_mask,
            },
        )


def _sensor_archive_payload(
    evidence: ActuatorEvidence | ContactWrenchEvidence,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if isinstance(evidence, ActuatorEvidence):
        descriptor = {
            "schema_version": INDEPENDENT_SENSOR_SCHEMA_VERSION,
            "artifact_kind": "ActuatorEvidence",
            **_identity_payload(evidence),
            "evidence_frame_stop": evidence.evidence_frame_stop,
            "metadata": plain_json(evidence.metadata),
            "artifact_id": evidence.artifact_id,
        }
        arrays = {
            "sample_times_s": evidence.sample_times_s,
            "positions_m": evidence.positions_m,
            "variance_m2": evidence.variance_m2,
            "valid_mask": evidence.valid_mask,
        }
    elif isinstance(evidence, ContactWrenchEvidence):
        descriptor = {
            "schema_version": INDEPENDENT_SENSOR_SCHEMA_VERSION,
            "artifact_kind": "ContactWrenchEvidence",
            **_identity_payload(evidence),
            "quantity_names": list(evidence.quantity_names),
            "evidence_frame_stop": evidence.evidence_frame_stop,
            "metadata": plain_json(evidence.metadata),
            "artifact_id": evidence.artifact_id,
        }
        arrays = {
            "sample_times_s": evidence.sample_times_s,
            "wrench": evidence.wrench,
            "variance": evidence.variance,
            "valid_mask": evidence.valid_mask,
        }
    else:
        raise TypeError("unsupported independent-sensor evidence type")
    return descriptor, arrays


def save_independent_sensor_evidence(
    path: str | Path,
    evidence: ActuatorEvidence | ContactWrenchEvidence,
    *,
    overwrite: bool = False,
) -> None:
    """Publish one checksummed NPZ atomically and exactly once by default."""

    if type(overwrite) is not bool:
        raise TypeError("overwrite must be an exact boolean")
    descriptor, arrays = _sensor_archive_payload(evidence)
    encoded = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    def write_archive(handle: BinaryIO) -> None:
        np.savez_compressed(
            handle,
            descriptor_json=np.frombuffer(encoded, dtype=np.uint8),
            **arrays,
        )

    def validate_archive(candidate: Path) -> None:
        restored = load_independent_sensor_evidence(candidate)
        if type(restored) is not type(evidence):
            raise ArtifactValidationError(
                "published independent-sensor evidence kind changed"
            )
        if restored.artifact_id != evidence.artifact_id:
            raise ArtifactValidationError(
                "published independent-sensor evidence identity changed"
            )

    atomic_write_binary(
        path,
        write_archive,
        overwrite=overwrite,
        validate=validate_archive,
    )


def _descriptor_identity(descriptor: Mapping[str, Any]) -> dict[str, str]:
    identity: dict[str, str] = {}
    for field_name in (
        "protocol_id",
        "case_id",
        "observed_action_id",
        "stream_id",
        "clock_id",
        "provenance",
    ):
        value = descriptor.get(field_name)
        if type(value) is not str or not value:
            raise ArtifactValidationError(
                f"independent-sensor {field_name} must be a nonempty string"
            )
        identity[field_name] = value
    return identity


def _load_sensor_archive_arrays(
    payload: bytes,
) -> tuple[dict[str, np.ndarray], str]:
    failures: list[ArtifactValidationError] = []
    for kind, expected_arrays in (
        ("ActuatorEvidence", _ACTUATOR_ARCHIVE_ARRAYS),
        ("ContactWrenchEvidence", _CONTACT_WRENCH_ARCHIVE_ARRAYS),
    ):
        try:
            return (
                load_npz_bytes(
                    payload,
                    name="independent-sensor evidence",
                    expected_arrays=expected_arrays,
                ),
                kind,
            )
        except ArtifactValidationError as error:
            failures.append(error)
    raise ArtifactValidationError(
        "independent-sensor evidence does not match a supported closed array inventory"
    ) from failures[-1]


def load_independent_sensor_evidence(
    path: str | Path,
) -> ActuatorEvidence | ContactWrenchEvidence:
    """Load exact ordinary-file bytes and verify the closed NPZ contract."""

    snapshot = read_regular_file(path, name="independent-sensor evidence")
    arrays, inventory_kind = _load_sensor_archive_arrays(snapshot.payload)
    descriptor_array = arrays["descriptor_json"]
    if descriptor_array.dtype != np.dtype(np.uint8) or descriptor_array.ndim != 1:
        raise ArtifactValidationError(
            "independent-sensor descriptor_json must be a uint8 vector"
        )
    descriptor = load_strict_json_object(
        descriptor_array.tobytes(),
        name="independent-sensor descriptor",
    )
    schema_version = descriptor.get("schema_version")
    if (
        type(schema_version) is not int
        or schema_version != INDEPENDENT_SENSOR_SCHEMA_VERSION
    ):
        raise ValueError("unsupported independent-sensor schema version")
    kind = descriptor.get("artifact_kind")
    if kind != inventory_kind:
        raise ArtifactValidationError(
            "independent-sensor descriptor kind disagrees with its array inventory"
        )
    identity = _descriptor_identity(descriptor)
    evidence_frame_stop = descriptor.get("evidence_frame_stop")
    if type(evidence_frame_stop) is not int:
        raise ArtifactValidationError(
            "independent-sensor evidence_frame_stop must be an integer"
        )
    if kind == "ActuatorEvidence":
        evidence: ActuatorEvidence | ContactWrenchEvidence = ActuatorEvidence(
            **identity,
            sample_times_s=arrays["sample_times_s"],
            positions_m=arrays["positions_m"],
            variance_m2=arrays["variance_m2"],
            evidence_frame_stop=evidence_frame_stop,
            valid_mask=arrays["valid_mask"],
            metadata=descriptor.get("metadata", {}),
        )
    else:
        quantity_names = descriptor.get("quantity_names")
        if (
            type(quantity_names) is not list
            or not quantity_names
            or any(type(name) is not str or not name for name in quantity_names)
        ):
            raise ArtifactValidationError(
                "independent-sensor quantity_names must be nonempty strings"
            )
        evidence = ContactWrenchEvidence(
            **identity,
            sample_times_s=arrays["sample_times_s"],
            wrench=arrays["wrench"],
            variance=arrays["variance"],
            quantity_names=tuple(quantity_names),
            evidence_frame_stop=evidence_frame_stop,
            valid_mask=arrays["valid_mask"],
            metadata=descriptor.get("metadata", {}),
        )
    if evidence.artifact_id != descriptor.get("artifact_id"):
        raise ValueError("independent-sensor evidence checksum mismatch")
    return evidence
