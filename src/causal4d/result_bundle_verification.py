"""Fail-closed verification for flat Causal4D result bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_nonfinite_json(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_nonfinite_json,
    )
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path.name}")
    return value


def _safe_artifact_name(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or path.name in {"", ".", ".."}
    ):
        raise ValueError(f"unsafe artifact name: {value!r}")
    return path.name


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"result-bundle path contains a symlink: {current}")


def verify_embedded_result_bundle(
    bundle_directory: str | Path,
) -> dict[str, Any]:
    """Verify exact inventory, byte counts, and SHA-256 identities."""

    supplied = Path(bundle_directory)
    _reject_symlink_components(supplied)
    if not supplied.is_dir():
        raise FileNotFoundError(
            f"result bundle directory does not exist: {supplied}"
        )
    bundle = supplied.resolve()
    manifest_path = bundle / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("bundle manifest must not be a symlink")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "bundle must contain an ordinary manifest.json file"
        )

    manifest = _read_json_object(manifest_path)
    expected_manifest_keys = {"schema_version", "benchmark", "artifacts"}
    if set(manifest) != expected_manifest_keys:
        raise ValueError(
            "result manifest keys differ from the exact schema: "
            f"{sorted(manifest)}"
        )
    if manifest["schema_version"] != 1:
        raise ValueError("result manifest schema_version must equal 1")
    benchmark = manifest["benchmark"]
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("result manifest benchmark must be nonempty")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(
            "result manifest artifacts must be a nonempty object"
        )

    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_record in artifacts.items():
        if not isinstance(raw_name, str):
            raise ValueError("result manifest artifact names must be strings")
        name = _safe_artifact_name(raw_name)
        if name == manifest_path.name:
            raise ValueError("manifest.json cannot declare itself as an artifact")
        if not isinstance(raw_record, dict) or set(raw_record) != {
            "bytes",
            "sha256",
        }:
            raise ValueError(
                f"artifact record for {name!r} has an invalid schema"
            )
        expected_hash = raw_record["sha256"]
        expected_bytes = raw_record["bytes"]
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_hash
            )
        ):
            raise ValueError(
                f"artifact {name!r} has an invalid SHA-256 digest"
            )
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
        ):
            raise ValueError(f"artifact {name!r} has an invalid byte count")
        path = bundle / name
        if path.is_symlink():
            raise ValueError(f"bundle artifact must not be a symlink: {name}")
        if not path.is_file():
            raise FileNotFoundError(
                f"bundle artifact must be an ordinary file: {name}"
            )
        actual_bytes = path.stat().st_size
        actual_hash = _sha256(path)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"artifact {name!r} byte count changed: "
                f"{actual_bytes} != {expected_bytes}"
            )
        if actual_hash != expected_hash:
            raise ValueError(
                f"artifact {name!r} checksum changed: "
                f"{actual_hash} != {expected_hash}"
            )
        normalized[name] = {
            "bytes": expected_bytes,
            "sha256": expected_hash,
        }

    expected_names = set(normalized) | {manifest_path.name}
    actual_names: set[str] = set()
    for entry in bundle.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(
                "result bundle entries must be ordinary files: "
                f"{entry.name}"
            )
        actual_names.add(entry.name)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            "bundle file set differs from the manifest: "
            f"missing={missing}, unexpected={unexpected}"
        )

    return {
        "schema_version": 1,
        "benchmark": benchmark,
        "bundle_name": bundle.name,
        "manifest_sha256": _sha256(manifest_path),
        "artifact_count": len(normalized),
        "artifacts": {
            name: normalized[name] for name in sorted(normalized)
        },
    }