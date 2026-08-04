"""Atomic, content-addressed persistence for finite rollout banks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from causal4d.atomic_io import atomic_write_binary
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.rollout_bank import JointRolloutBank

ROLLOUT_BANK_ARCHIVE_SCHEMA_VERSION = 2

_LEGACY_MEMBERS = frozenset(
    {
        "hypothesis_ids",
        "hypothesis_metadata_json",
        "hypothesis_prior_weights",
        "parameter_particles",
        "parameter_weights",
        "trajectories",
        "variance_floor_m2",
        "confidence_level",
        "manifest_json",
    }
)
_VERSION_2_MEMBERS = _LEGACY_MEMBERS | {
    "archive_schema_version",
    "rollout_bank_id",
}


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_json_mapping(value: str, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            value,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain strict finite JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    normalized = validated_json_mapping(
        payload,
        error_message=f"{name} must contain finite JSON data",
    )
    return plain_json(normalized)


def _text_value(value: Any, *, name: str) -> str:
    item = value.item() if isinstance(value, np.generic) else value
    if isinstance(item, bytes):
        try:
            return item.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{name} must contain UTF-8 text") from error
    if not isinstance(item, str):
        raise ValueError(f"{name} must contain text")
    return item


def _scalar_text(value: np.ndarray, *, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a scalar text member")
    return _text_value(array.item(), name=name)


def _scalar_integer(value: np.ndarray, *, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be a scalar integer member")
    return int(array.item())


def _scalar_float(value: np.ndarray, *, name: str) -> float:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{name} must be a scalar numeric member")
    return float(array.item())


def _archive_payload(
    bank: JointRolloutBank,
    manifest: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    normalized_manifest = plain_json(
        validated_json_mapping(
            manifest,
            error_message="rollout bank manifest must be finite JSON data",
        )
    )
    return {
        "hypothesis_ids": np.asarray(bank.hypothesis_ids),
        "hypothesis_metadata_json": np.asarray(
            [
                json.dumps(
                    plain_json(value),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                for value in bank.hypothesis_metadata
            ]
        ),
        "hypothesis_prior_weights": bank.hypothesis_prior_weights,
        "parameter_particles": bank.parameter_particles,
        "parameter_weights": bank.parameter_weights,
        "trajectories": bank.trajectories,
        "variance_floor_m2": np.asarray(bank.variance_floor_m2),
        "confidence_level": np.asarray(bank.confidence_level),
        "manifest_json": np.asarray(
            json.dumps(
                normalized_manifest,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
        "archive_schema_version": np.asarray(
            ROLLOUT_BANK_ARCHIVE_SCHEMA_VERSION,
            dtype=np.int64,
        ),
        "rollout_bank_id": np.asarray(bank.artifact_id),
    }


def save_rollout_bank(
    path: str | Path,
    bank: JointRolloutBank,
    manifest: Mapping[str, Any],
    *,
    overwrite: bool = True,
) -> None:
    """Atomically publish one validated version-2 rollout-bank archive."""

    target = Path(path)
    payload = _archive_payload(bank, manifest)
    expected_manifest = _load_json_mapping(
        str(payload["manifest_json"]),
        name="rollout bank manifest",
    )

    def write_archive(handle: BinaryIO) -> None:
        np.savez_compressed(handle, **payload)

    def validate_archive(temporary: Path) -> None:
        restored_bank, restored_manifest = load_rollout_bank(temporary)
        if restored_bank.artifact_id != bank.artifact_id:
            raise ValueError("written rollout bank changed its content identity")
        if restored_manifest != expected_manifest:
            raise ValueError("written rollout bank changed its manifest")

    atomic_write_binary(
        target,
        write_archive,
        overwrite=overwrite,
        validate=validate_archive,
    )


def load_rollout_bank(path: str | Path) -> tuple[JointRolloutBank, dict[str, Any]]:
    """Load and revalidate a legacy or version-2 rollout-bank archive."""

    with np.load(path, allow_pickle=False) as archive:
        members = frozenset(archive.files)
        if members == _LEGACY_MEMBERS:
            schema_version = 1
        elif members == _VERSION_2_MEMBERS:
            schema_version = _scalar_integer(
                archive["archive_schema_version"],
                name="archive_schema_version",
            )
            if schema_version != ROLLOUT_BANK_ARCHIVE_SCHEMA_VERSION:
                raise ValueError("unsupported rollout-bank archive schema version")
        else:
            missing = sorted(_VERSION_2_MEMBERS - members)
            extra = sorted(members - _VERSION_2_MEMBERS)
            raise ValueError(
                "rollout-bank archive members changed; "
                f"missing={missing}, extra={extra}"
            )

        hypothesis_ids_array = np.asarray(archive["hypothesis_ids"])
        metadata_array = np.asarray(archive["hypothesis_metadata_json"])
        if hypothesis_ids_array.ndim != 1 or hypothesis_ids_array.dtype.kind not in {
            "U",
            "S",
        }:
            raise ValueError("hypothesis_ids must be a text vector")
        if metadata_array.shape != hypothesis_ids_array.shape or (
            metadata_array.dtype.kind not in {"U", "S"}
        ):
            raise ValueError(
                "hypothesis_metadata_json must be a text vector matching hypothesis_ids"
            )
        metadata = tuple(
            _load_json_mapping(
                _text_value(value, name="hypothesis metadata"),
                name="hypothesis metadata",
            )
            for value in metadata_array
        )
        hypothesis_ids = tuple(
            _text_value(value, name="hypothesis ID")
            for value in hypothesis_ids_array
        )
        bank = JointRolloutBank(
            hypothesis_ids=hypothesis_ids,
            hypothesis_metadata=metadata,
            hypothesis_prior_weights=np.asarray(
                archive["hypothesis_prior_weights"], dtype=float
            ),
            parameter_particles=np.asarray(archive["parameter_particles"], dtype=float),
            parameter_weights=np.asarray(archive["parameter_weights"], dtype=float),
            trajectories=np.asarray(archive["trajectories"], dtype=np.float32),
            variance_floor_m2=_scalar_float(
                archive["variance_floor_m2"],
                name="variance_floor_m2",
            ),
            confidence_level=_scalar_float(
                archive["confidence_level"],
                name="confidence_level",
            ),
        )
        manifest = _load_json_mapping(
            _scalar_text(archive["manifest_json"], name="manifest_json"),
            name="rollout bank manifest",
        )
        if schema_version == ROLLOUT_BANK_ARCHIVE_SCHEMA_VERSION:
            stored_id = _scalar_text(
                archive["rollout_bank_id"],
                name="rollout_bank_id",
            )
            if stored_id != bank.artifact_id:
                raise ValueError("rollout bank ID does not match its payload")
    return bank, manifest


__all__ = [
    "ROLLOUT_BANK_ARCHIVE_SCHEMA_VERSION",
    "load_rollout_bank",
    "save_rollout_bank",
]
