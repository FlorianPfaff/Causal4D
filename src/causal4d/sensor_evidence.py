"""Typed independent actuator and contact-wrench evidence artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from causal4d.immutable_json import validated_json_mapping

from causal4d.contracts import array_sha256


INDEPENDENT_SENSOR_SCHEMA_VERSION = 1


def _readonly(values: np.ndarray, *, dtype: type | None = float) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


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
    result.setflags(write=False)
    return result


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
    result.setflags(write=False)
    return result


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
            dict(scalar_payload),
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
                "metadata": self.metadata,
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
                "metadata": self.metadata,
                "independent_of_object_observations": True,
            },
            arrays={
                "sample_times_s": self.sample_times_s,
                "wrench": self.wrench,
                "variance": self.variance,
                "valid_mask": self.valid_mask,
            },
        )


def save_independent_sensor_evidence(
    path: str | Path,
    evidence: ActuatorEvidence | ContactWrenchEvidence,
) -> None:
    """Serialize one evidence artifact as a non-pickled checksummed NPZ."""

    if isinstance(evidence, ActuatorEvidence):
        descriptor = {
            "schema_version": INDEPENDENT_SENSOR_SCHEMA_VERSION,
            "artifact_kind": "ActuatorEvidence",
            **_identity_payload(evidence),
            "evidence_frame_stop": evidence.evidence_frame_stop,
            "metadata": evidence.metadata,
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
            "metadata": evidence.metadata,
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
    encoded = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            descriptor_json=np.frombuffer(encoded, dtype=np.uint8),
            **arrays,
        )


def _descriptor_identity(descriptor: Mapping[str, Any]) -> dict[str, str]:
    return {
        "protocol_id": str(descriptor["protocol_id"]),
        "case_id": str(descriptor["case_id"]),
        "observed_action_id": str(descriptor["observed_action_id"]),
        "stream_id": str(descriptor["stream_id"]),
        "clock_id": str(descriptor["clock_id"]),
        "provenance": str(descriptor["provenance"]),
    }


def load_independent_sensor_evidence(
    path: str | Path,
) -> ActuatorEvidence | ContactWrenchEvidence:
    """Load and verify a serialized independent-sensor evidence artifact."""

    with np.load(Path(path), allow_pickle=False) as archive:
        if "descriptor_json" not in archive.files:
            raise ValueError("sensor evidence archive is missing descriptor_json")
        descriptor = json.loads(
            np.asarray(archive["descriptor_json"], dtype=np.uint8)
            .tobytes()
            .decode("utf-8")
        )
        if descriptor.get("schema_version") != INDEPENDENT_SENSOR_SCHEMA_VERSION:
            raise ValueError("unsupported independent-sensor schema version")
        kind = descriptor.get("artifact_kind")
        identity = _descriptor_identity(descriptor)
        if kind == "ActuatorEvidence":
            required = {
                "sample_times_s",
                "positions_m",
                "variance_m2",
                "valid_mask",
            }
            if not required.issubset(archive.files):
                raise ValueError("actuator evidence archive is incomplete")
            evidence: ActuatorEvidence | ContactWrenchEvidence = ActuatorEvidence(
                **identity,
                sample_times_s=archive["sample_times_s"],
                positions_m=archive["positions_m"],
                variance_m2=archive["variance_m2"],
                evidence_frame_stop=int(descriptor["evidence_frame_stop"]),
                valid_mask=archive["valid_mask"],
                metadata=descriptor.get("metadata", {}),
            )
        elif kind == "ContactWrenchEvidence":
            required = {
                "sample_times_s",
                "wrench",
                "variance",
                "valid_mask",
            }
            if not required.issubset(archive.files):
                raise ValueError("contact-wrench evidence archive is incomplete")
            evidence = ContactWrenchEvidence(
                **identity,
                sample_times_s=archive["sample_times_s"],
                wrench=archive["wrench"],
                variance=archive["variance"],
                quantity_names=tuple(map(str, descriptor["quantity_names"])),
                evidence_frame_stop=int(descriptor["evidence_frame_stop"]),
                valid_mask=archive["valid_mask"],
                metadata=descriptor.get("metadata", {}),
            )
        else:
            raise ValueError(f"unsupported sensor evidence kind: {kind!r}")
    if evidence.artifact_id != descriptor.get("artifact_id"):
        raise ValueError("independent-sensor evidence checksum mismatch")
    return evidence
