"""Strict result-bundle identity and embedded-manifest verification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence


RESULT_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class ArtifactRecord:
    """Verified immutable identity for one result-bundle payload."""

    name: str
    path: Path
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class VerifiedResultBundle:
    """A result bundle whose embedded manifest matches its payload bytes."""

    manifest_path: Path
    directory: Path
    schema_version: int
    benchmark: str
    artifacts: tuple[ArtifactRecord, ...]

    @property
    def artifacts_by_name(self) -> Mapping[str, ArtifactRecord]:
        return MappingProxyType(
            {artifact.name: artifact for artifact in self.artifacts}
        )


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of an ordinary file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _require_finite_json_numbers(value: Any, *, path: str) -> None:
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"non-finite JSON number is forbidden at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_json_numbers(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json_numbers(item, path=f"{path}[{index}]")


def load_strict_json_bytes(payload: bytes, *, source: str) -> Any:
    """Decode strict UTF-8 JSON, rejecting duplicates and non-finite numbers."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"JSON is not valid UTF-8: {source}") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
        _require_finite_json_numbers(value, path="$")
        return value
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON in {source}: {error}") from error


def load_strict_json(path: Path) -> Any:
    """Read strict JSON from an ordinary, non-symlink file."""

    _require_ordinary_file(path, label="JSON file")
    return load_strict_json_bytes(path.read_bytes(), source=str(path))


def _safe_artifact_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("result manifest artifact names must be strings")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise ValueError(f"unsafe artifact name: {value!r}")
    return path.name


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_ordinary_file(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")


def verify_result_manifest(
    manifest_path: Path,
    bundle_directory: Path | None = None,
) -> VerifiedResultBundle:
    """Verify a schema-v1 result manifest and its exact payload inventory."""

    _require_ordinary_file(manifest_path, label="result manifest")
    manifest_path = manifest_path.resolve()
    raw_directory = bundle_directory or manifest_path.parent
    if raw_directory.is_symlink():
        raise ValueError(
            f"result bundle directory must not be a symlink: {raw_directory}"
        )
    directory = raw_directory.resolve(strict=False)
    if not directory.is_dir():
        raise FileNotFoundError(f"result bundle directory is missing: {directory}")
    if manifest_path.parent != directory:
        raise ValueError(
            "the result manifest must be embedded directly in the bundle directory"
        )

    manifest = load_strict_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("result manifest must be a JSON object")
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("result manifest schema_version must equal integer 1")
    benchmark = manifest.get("benchmark")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("result manifest benchmark must be nonempty")
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, dict) or not raw_artifacts:
        raise ValueError("result manifest artifacts must be a nonempty object")

    records: list[ArtifactRecord] = []
    expected_names: set[str] = set()
    for raw_name, raw_record in raw_artifacts.items():
        name = _safe_artifact_name(raw_name)
        if name in expected_names:
            raise ValueError(f"duplicate normalized artifact name: {name!r}")
        expected_names.add(name)
        if not isinstance(raw_record, dict):
            raise ValueError(f"artifact record for {name!r} must be an object")
        expected_hash = raw_record.get("sha256")
        expected_bytes = raw_record.get("bytes")
        if not _valid_sha256(expected_hash):
            raise ValueError(f"artifact {name!r} has an invalid SHA-256 digest")
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise ValueError(f"artifact {name!r} has an invalid byte count")
        path = directory / name
        _require_ordinary_file(path, label=f"bundle artifact {name!r}")
        actual_bytes = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"artifact {name!r} byte count changed: "
                f"{actual_bytes} != {expected_bytes}"
            )
        if actual_hash != expected_hash:
            raise ValueError(
                f"artifact {name!r} checksum changed: {actual_hash} != {expected_hash}"
            )
        records.append(
            ArtifactRecord(
                name=name,
                path=path,
                byte_count=actual_bytes,
                sha256=actual_hash,
            )
        )

    actual_names: set[str] = set()
    for path in directory.iterdir():
        if path.name == manifest_path.name:
            continue
        if path.is_symlink():
            raise ValueError(f"result bundle contains a symlink: {path}")
        if not path.is_file():
            raise ValueError(
                f"result bundle contains an undeclared non-file entry: {path}"
            )
        actual_names.add(path.name)
    unexpected = sorted(actual_names - expected_names)
    missing = sorted(expected_names - actual_names)
    if unexpected or missing:
        raise ValueError(
            "bundle file set differs from the manifest: "
            f"missing={missing}, unexpected={unexpected}"
        )

    return VerifiedResultBundle(
        manifest_path=manifest_path,
        directory=directory,
        schema_version=schema_version,
        benchmark=benchmark,
        artifacts=tuple(sorted(records, key=lambda record: record.name)),
    )
