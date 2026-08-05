"""Portable, content-addressed physical evaluation targets."""

from __future__ import annotations

import hashlib
import io
import json
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from causal4d.atomic_io import atomic_write_binary
from causal4d.contracts import (
    CausalContext,
    ObservationWindow,
    PhysicalPosterior,
    array_sha256,
)
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.stage_provenance import EvaluationTarget

PHYSICAL_TARGET_SCHEMA = "causal4d.physical_evaluation_target"
PHYSICAL_TARGET_SCHEMA_VERSION = 1
PHYSICAL_TARGET_OBSERVATION_SEMANTICS = (
    "causal4d.phystwin_backend.object_points_float32"
)
PHYSICAL_TARGET_VALIDITY_SEMANTICS = (
    "bayesian_phystwin.causal4d_provider_v1.target_validity"
)

_ARCHIVE_FIELDS = frozenset(
    {
        "descriptor_json",
        "object_points",
        "validity_mask",
    }
)
_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "context",
        "source",
        "alignment",
        "metadata",
        "payload",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "format",
        "observation_semantics",
        "sha256",
        "validity_semantics",
    }
)
_ALIGNMENT_FIELDS = frozenset(
    {
        "anchor_frame",
        "frame_stop",
        "stream_id",
    }
)
_PAYLOAD_FIELDS = frozenset(
    {
        "object_points_sha256",
        "validity_mask_sha256",
    }
)
_LOWER_HEX = frozenset("0123456789abcdef")


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _require_exact_fields(
    value: Any,
    *,
    name: str,
    required: frozenset[str],
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, name=name)
    actual = set(mapping)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if missing or unexpected:
        raise ValueError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return mapping


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _validate_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _parse_descriptor(value: np.ndarray) -> Mapping[str, Any]:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError("descriptor_json must be one scalar string")
    scalar = array.item()
    if isinstance(scalar, bytes):
        try:
            scalar = scalar.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("descriptor_json is not valid UTF-8") from error
    try:
        parsed = json.loads(
            scalar,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("descriptor_json is not valid canonical JSON") from error
    return _require_mapping(parsed, name="physical target descriptor")


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"physical target path contains a symlink: {current}")


@dataclass(frozen=True)
class PhysicalTargetBundle:
    """Held-out physical observations aligned to one Causal4D trajectory."""

    context: CausalContext
    object_points: np.ndarray
    validity_mask: np.ndarray
    source_final_data_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.context) is not CausalContext:
            raise ValueError("context must be a CausalContext")
        source_sha256 = _validate_sha256(
            self.source_final_data_sha256,
            name="source_final_data_sha256",
        )

        raw_points = np.asarray(self.object_points)
        if raw_points.dtype != np.dtype(np.float32):
            raise ValueError("object_points must use the canonical float32 dtype")
        points = np.ascontiguousarray(raw_points).copy()
        if points.ndim != 3 or points.shape[1] < 1 or points.shape[2] != 3:
            raise ValueError("object_points must have shape (T, N>=1, 3)")

        raw_valid = np.asarray(self.validity_mask)
        if raw_valid.dtype != np.dtype(bool):
            raise ValueError("validity_mask must use the boolean dtype")
        valid = np.ascontiguousarray(raw_valid).copy()
        if valid.shape != points.shape[:2]:
            raise ValueError("validity_mask must have shape (T, N)")

        anchor_frame = self.anchor_frame
        expected_frames = self.context.o_plus.frame_stop - anchor_frame
        if expected_frames < 2 or len(points) != expected_frames:
            raise ValueError(
                "object_points must cover the pre-intervention endpoint through O+"
            )
        o_plus_offset = self.context.o_plus.frame_start - anchor_frame
        if not 0 <= o_plus_offset < len(points):
            raise ValueError("O+ does not lie inside the aligned physical target")
        if array_sha256(points[o_plus_offset:]) != (
            self.context.o_plus.content_sha256
        ):
            raise ValueError("physical target does not match the declared O+ digest")
        if np.any(valid & ~np.all(np.isfinite(points), axis=2)):
            raise ValueError("valid physical target points must be finite")
        if not np.any(valid[o_plus_offset:]):
            raise ValueError("physical target has no valid O+ point frames")

        points.setflags(write=False)
        valid.setflags(write=False)
        metadata = validated_json_mapping(
            self.metadata,
            error_message="physical target metadata must be finite JSON data",
        )
        object.__setattr__(self, "object_points", points)
        object.__setattr__(self, "validity_mask", valid)
        object.__setattr__(self, "source_final_data_sha256", source_sha256)
        object.__setattr__(self, "metadata", metadata)

    @property
    def anchor_frame(self) -> int:
        """Global frame represented by local trajectory frame zero."""

        return self.context.o_minus.frame_stop - 1

    @property
    def frame_stop(self) -> int:
        return self.context.o_plus.frame_stop

    @property
    def frame_count(self) -> int:
        return len(self.object_points)

    @property
    def point_count(self) -> int:
        return int(self.object_points.shape[1])

    def _descriptor_without_id(self) -> dict[str, Any]:
        return {
            "schema": PHYSICAL_TARGET_SCHEMA,
            "schema_version": PHYSICAL_TARGET_SCHEMA_VERSION,
            "context": self.context.as_dict(),
            "source": {
                "format": "trusted_python_pickle",
                "observation_semantics": PHYSICAL_TARGET_OBSERVATION_SEMANTICS,
                "sha256": self.source_final_data_sha256,
                "validity_semantics": PHYSICAL_TARGET_VALIDITY_SEMANTICS,
            },
            "alignment": {
                "anchor_frame": self.anchor_frame,
                "frame_stop": self.frame_stop,
                "stream_id": self.context.o_plus.stream_id,
            },
            "metadata": plain_json(self.metadata),
            "payload": {
                "object_points_sha256": array_sha256(self.object_points),
                "validity_mask_sha256": array_sha256(self.validity_mask),
            },
        }

    @property
    def artifact_id(self) -> str:
        encoded = _canonical_json(self._descriptor_without_id()).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def descriptor(self) -> dict[str, Any]:
        descriptor = self._descriptor_without_id()
        descriptor["artifact_id"] = self.artifact_id
        return descriptor

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "protocol_id": self.context.protocol_id,
            "case_id": self.context.case_id,
            "stream_id": self.context.o_plus.stream_id,
            "anchor_frame": self.anchor_frame,
            "frame_stop": self.frame_stop,
            "frame_count": self.frame_count,
            "point_count": self.point_count,
            "valid_point_frames": int(np.sum(self.validity_mask)),
            "source_final_data_sha256": self.source_final_data_sha256,
        }

    def evaluation_target(self, *, start_frame: int) -> EvaluationTarget:
        """Build a content identity for the selected held-out suffix."""

        local_start = _require_integer(
            start_frame,
            name="start_frame",
        )
        if not 0 <= local_start < self.frame_count:
            raise ValueError("start_frame must lie inside the physical target")
        global_start = self.anchor_frame + local_start
        if not (
            self.context.o_plus.frame_start
            <= global_start
            < self.context.o_plus.frame_stop
        ):
            raise ValueError("evaluation suffix must be a nonempty part of O+")
        target = ObservationWindow(
            case_id=self.context.case_id,
            stream_id=self.context.o_plus.stream_id,
            frame_start=global_start,
            frame_stop=self.context.o_plus.frame_stop,
            content_sha256=array_sha256(self.object_points[local_start:]),
        )
        return EvaluationTarget(
            protocol_id=self.context.protocol_id,
            target=target,
        )

    def aligned_for_posterior(
        self,
        posterior: PhysicalPosterior,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate context and return the target restricted to posterior nodes."""

        if type(posterior) is not PhysicalPosterior:
            raise ValueError("posterior must be a PhysicalPosterior")
        if posterior.context != self.context:
            raise ValueError("physical target context does not match the posterior")
        trajectory = posterior.readout_trajectories_m
        if trajectory.shape[1] != self.frame_count:
            raise ValueError("physical target frame count does not match the posterior")
        node_count = int(trajectory.shape[2])
        if self.point_count < node_count:
            raise ValueError("physical target does not cover all posterior nodes")
        return (
            self.object_points[:, :node_count],
            self.validity_mask[:, :node_count],
        )


def build_physical_target(
    context: CausalContext,
    object_points: np.ndarray,
    validity_mask: np.ndarray,
    *,
    source_final_data_sha256: str,
    metadata: Mapping[str, Any] | None = None,
) -> PhysicalTargetBundle:
    """Build an aligned target from an exact legacy acquisition payload."""

    raw_points = np.asarray(object_points)
    if raw_points.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("source object_points must use float32 or float64")
    points = np.asarray(raw_points, dtype=np.float32)
    valid = np.asarray(validity_mask)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("source object_points must have shape (T, N, 3)")
    if valid.shape != points.shape[:2]:
        raise ValueError("source validity_mask must have shape (T, N)")
    anchor = context.o_minus.frame_stop - 1
    if anchor < 0 or context.o_plus.frame_stop > len(points):
        raise ValueError("source object_points do not cover the Causal4D context")
    if array_sha256(
        points[context.o_minus.frame_start : context.o_minus.frame_stop]
    ) != context.o_minus.content_sha256:
        raise ValueError("source object_points do not match the declared O- digest")
    if array_sha256(
        points[context.o_plus.frame_start : context.o_plus.frame_stop]
    ) != context.o_plus.content_sha256:
        raise ValueError("source object_points do not match the declared O+ digest")
    return PhysicalTargetBundle(
        context=context,
        object_points=points[anchor : context.o_plus.frame_stop],
        validity_mask=valid[anchor : context.o_plus.frame_stop],
        source_final_data_sha256=source_final_data_sha256,
        metadata={} if metadata is None else metadata,
    )


def _save_archive(handle: BinaryIO, bundle: PhysicalTargetBundle) -> None:
    np.savez_compressed(
        handle,
        descriptor_json=np.asarray(_canonical_json(bundle.descriptor())),
        object_points=bundle.object_points,
        validity_mask=bundle.validity_mask,
    )


def save_physical_target(
    path: str | Path,
    bundle: PhysicalTargetBundle,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish and independently reload one target artifact."""

    target = Path(path)
    _reject_symlink_components(target)
    expected_id = bundle.artifact_id

    def validate(temporary: Path) -> None:
        loaded = load_physical_target(temporary)
        if loaded.artifact_id != expected_id:
            raise ValueError("physical target changed during serialization")

    atomic_write_binary(
        target,
        lambda handle: _save_archive(handle, bundle),
        overwrite=overwrite,
        validate=validate,
    )


def load_physical_target(path: str | Path) -> PhysicalTargetBundle:
    """Load a strict non-pickled physical target from one immutable byte snapshot."""

    supplied = Path(path)
    _reject_symlink_components(supplied)
    try:
        metadata = supplied.stat()
    except FileNotFoundError:
        raise FileNotFoundError(f"physical target does not exist: {supplied}") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"physical target must be an ordinary file: {supplied}")
    payload = supplied.read_bytes()

    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            actual_fields = set(archive.files)
            if actual_fields != _ARCHIVE_FIELDS:
                raise ValueError(
                    "physical target archive fields do not match schema; "
                    f"missing={sorted(_ARCHIVE_FIELDS - actual_fields)}, "
                    f"unexpected={sorted(actual_fields - _ARCHIVE_FIELDS)}"
                )
            descriptor = _require_exact_fields(
                _parse_descriptor(archive["descriptor_json"]),
                name="physical target descriptor",
                required=_DESCRIPTOR_FIELDS,
            )
            points = np.asarray(archive["object_points"])
            valid = np.asarray(archive["validity_mask"])
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("physical target"):
            raise
        raise ValueError("physical target is not a valid non-pickled NPZ") from error

    if type(descriptor["schema"]) is not str or (
        descriptor["schema"] != PHYSICAL_TARGET_SCHEMA
    ):
        raise ValueError("unexpected physical target schema")
    if type(descriptor["schema_version"]) is not int or (
        descriptor["schema_version"] != PHYSICAL_TARGET_SCHEMA_VERSION
    ):
        raise ValueError("unsupported physical target schema version")
    _validate_sha256(
        descriptor["artifact_id"],
        name="physical target artifact_id",
    )
    source = _require_exact_fields(
        descriptor["source"],
        name="physical target source",
        required=_SOURCE_FIELDS,
    )
    if type(source["format"]) is not str or (
        source["format"] != "trusted_python_pickle"
    ):
        raise ValueError("unsupported physical target source format")
    if type(source["observation_semantics"]) is not str or (
        source["observation_semantics"] != PHYSICAL_TARGET_OBSERVATION_SEMANTICS
    ):
        raise ValueError("unsupported physical target observation semantics")
    if type(source["validity_semantics"]) is not str or (
        source["validity_semantics"] != PHYSICAL_TARGET_VALIDITY_SEMANTICS
    ):
        raise ValueError("unsupported physical target validity semantics")
    alignment = _require_exact_fields(
        descriptor["alignment"],
        name="physical target alignment",
        required=_ALIGNMENT_FIELDS,
    )
    if type(alignment["anchor_frame"]) is not int or (
        alignment["anchor_frame"] < 0
    ):
        raise ValueError("physical target anchor_frame must be a nonnegative integer")
    if type(alignment["frame_stop"]) is not int or alignment["frame_stop"] < 1:
        raise ValueError("physical target frame_stop must be a positive integer")
    _require_nonempty_string(alignment["stream_id"], name="alignment.stream_id")
    payload_descriptor = _require_exact_fields(
        descriptor["payload"],
        name="physical target payload",
        required=_PAYLOAD_FIELDS,
    )
    for name, value in payload_descriptor.items():
        _validate_sha256(value, name=f"physical target payload.{name}")
    context = CausalContext.from_dict(
        _require_mapping(descriptor["context"], name="physical target context")
    )
    bundle = PhysicalTargetBundle(
        context=context,
        object_points=points,
        validity_mask=valid,
        source_final_data_sha256=_validate_sha256(
            source["sha256"],
            name="physical target source.sha256",
        ),
        metadata=_require_mapping(
            descriptor["metadata"],
            name="physical target metadata",
        ),
    )
    if alignment != bundle.descriptor()["alignment"]:
        raise ValueError("physical target alignment does not match its arrays")
    if payload_descriptor != bundle.descriptor()["payload"]:
        raise ValueError("physical target payload hashes do not match its arrays")
    if descriptor["artifact_id"] != bundle.artifact_id:
        raise ValueError("physical target artifact_id does not match its content")
    if dict(descriptor) != bundle.descriptor():
        raise ValueError("physical target descriptor is not canonical")
    return bundle


__all__ = [
    "PHYSICAL_TARGET_SCHEMA",
    "PHYSICAL_TARGET_SCHEMA_VERSION",
    "PHYSICAL_TARGET_OBSERVATION_SEMANTICS",
    "PHYSICAL_TARGET_VALIDITY_SEMANTICS",
    "PhysicalTargetBundle",
    "build_physical_target",
    "load_physical_target",
    "save_physical_target",
]
