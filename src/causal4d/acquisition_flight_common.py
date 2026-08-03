"""Shared safety and content-addressing utilities for acquisition operations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

JOURNAL_SCHEMA_VERSION = 1
JOURNAL_EVENT_KIND = "Causal4DAcquisitionJournalEvent"
JOURNAL_SEAL_KIND = "Causal4DAcquisitionJournalSeal"
HEALTH_SNAPSHOT_KIND = "Causal4DAcquisitionHealthSnapshot"
DOCTOR_REPORT_KIND = "Causal4DAcquisitionDoctorReport"

_FORBIDDEN_OUTCOME_KEYS = frozenset(
    {
        "target_outcomes",
        "target_metrics",
        "held_out_metrics",
        "prediction_error",
        "coordinate_rmse_m",
        "track_error_m",
        "chamfer_distance_m",
        "negative_log_likelihood",
        "nll",
        "coverage",
        "oracle_error",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any], *, omitted: str) -> str:
    payload = dict(value)
    payload.pop(omitted, None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parse_utc(value: Any, *, name: str) -> datetime:
    _require(isinstance(value, str) and bool(value), f"{name} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} is not ISO 8601") from error
    _require(parsed.tzinfo is not None, f"{name} must include a timezone")
    _require(
        parsed.utcoffset() == timezone.utc.utcoffset(parsed),
        f"{name} must be UTC",
    )
    return parsed


def _assert_json_value(value: Any, *, name: str) -> None:
    try:
        _canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite JSON") from error


def _walk_mapping_keys(value: Any) -> Sequence[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_walk_mapping_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            keys.extend(_walk_mapping_keys(child))
    return keys


def _reject_target_outcomes(value: Any) -> None:
    forbidden = sorted(
        key
        for key in _walk_mapping_keys(value)
        if key.lower() in _FORBIDDEN_OUTCOME_KEYS
    )
    _require(
        not forbidden,
        "acquisition operations must not contain target-outcome fields: "
        + ", ".join(forbidden),
    )


def _assert_no_symlink_components(path: Path, *, name: str) -> None:
    candidate = path.absolute()
    for component in (candidate, *candidate.parents):
        if component.is_symlink():
            raise ValueError(f"{name} contains a symlink component: {component}")


def _assert_ordinary_file_or_missing(path: Path, *, name: str) -> None:
    _assert_no_symlink_components(path, name=name)
    if path.exists():
        _require(path.is_file(), f"{name} must be an ordinary file")


def journal_seal_path(journal_path: str | Path) -> Path:
    journal = Path(journal_path)
    return journal.with_name(f"{journal.name}.seal.json")
