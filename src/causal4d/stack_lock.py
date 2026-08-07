"""Content-addressed wheel locks for the Prob4D -> BPT -> Causal4D stack."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from causal4d.atomic_io import atomic_write_json
from causal4d.immutable_json import plain_json

STACK_LOCK_SCHEMA_NAME = "causal4d.stack-lock"
STACK_LOCK_SCHEMA_VERSION = 1
STACK_VERIFICATION_SCHEMA_NAME = "causal4d.stack-verification"
STACK_VERIFICATION_SCHEMA_VERSION = 1
STACK_LOCK_GENERATOR = "causal4d stack create"
STACK_PIPELINE = ("prob4d", "bayesian-phystwin", "causal4d")

SOURCE_REPOSITORIES = {
    "prob4d": "IPS-Stuttgart/Prob4D",
    "bayesian-phystwin": "IPS-Stuttgart/BayesianPhysTwin",
    "causal4d": "IPS-Stuttgart/Causal4D",
}
REQUIRED_MODULES = {
    "prob4d": (
        "prob4d.provider_v2",
        "prob4d.provider_v2_loading",
    ),
    "bayesian-phystwin": (
        "bayesian_phystwin.causal4d_provider_v2",
        "bayesian_phystwin.causal4d_belief_provider_v2",
    ),
    "causal4d": (
        "causal4d.claim_bearing_observation_lineage",
        "causal4d.belief_provider_v2_contract",
    ),
}

_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class WheelIdentity:
    """Identity extracted from one wheel without importing or installing it."""

    name: str
    version: str
    filename: str
    sha256: str
    size_bytes: int
    path: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_text(text: str, *, label: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains non-finite value {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error


def _canonical_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _payload_id(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload_bytes(payload)).hexdigest()


def _canonical_version(value: str, *, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be nonempty")
    try:
        normalized = str(Version(value))
    except InvalidVersion as error:
        raise ValueError(f"{label} is invalid: {value!r}") from error
    _require(normalized == value, f"{label} must use canonical form {normalized!r}")
    return normalized


def _normalize_revisions(revisions: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_name, raw_revision in revisions.items():
        name = canonicalize_name(raw_name)
        _require(name not in normalized, f"duplicate source revision for {name}")
        revision = str(raw_revision).lower()
        _require(
            _HEX_40.fullmatch(revision) is not None,
            f"source revision for {name} must be a 40-character hexadecimal SHA",
        )
        normalized[name] = revision
    expected = set(STACK_PIPELINE)
    observed = set(normalized)
    _require(
        observed == expected,
        "source revisions must cover exactly "
        f"{sorted(expected)}; missing={sorted(expected - observed)}, "
        f"unexpected={sorted(observed - expected)}",
    )
    return normalized


def inspect_wheel(path: str | Path) -> WheelIdentity:
    """Return the content and metadata identity of one wheel file."""

    wheel_path = Path(path).resolve()
    _require(wheel_path.is_file(), f"wheel does not exist: {wheel_path}")
    _require(wheel_path.suffix == ".whl", f"wheel path must end in .whl: {wheel_path}")
    try:
        with ZipFile(wheel_path) as archive:
            metadata_members = sorted(
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            )
            _require(
                len(metadata_members) == 1,
                f"wheel must contain exactly one .dist-info/METADATA: {wheel_path}",
            )
            metadata_text = archive.read(metadata_members[0]).decode("utf-8")
    except (BadZipFile, UnicodeDecodeError, KeyError) as error:
        raise ValueError(f"wheel metadata cannot be read: {wheel_path}") from error

    message = Parser().parsestr(metadata_text)
    raw_name = message.get("Name")
    raw_version = message.get("Version")
    _require(raw_name is not None, f"wheel metadata has no Name: {wheel_path}")
    _require(raw_version is not None, f"wheel metadata has no Version: {wheel_path}")
    name = canonicalize_name(raw_name)
    version = _canonical_version(raw_version, label=f"wheel version for {name}")
    return WheelIdentity(
        name=name,
        version=version,
        filename=wheel_path.name,
        sha256=_sha256_file(wheel_path),
        size_bytes=wheel_path.stat().st_size,
        path=wheel_path,
    )


def build_stack_lock(
    wheel_paths: Sequence[str | Path],
    *,
    source_revisions: Mapping[str, str],
) -> dict[str, Any]:
    """Build a deterministic lock from exactly the three compatibility wheels."""

    revisions = _normalize_revisions(source_revisions)
    identities: dict[str, WheelIdentity] = {}
    for path in wheel_paths:
        identity = inspect_wheel(path)
        _require(
            identity.name not in identities,
            f"duplicate wheel for distribution {identity.name}",
        )
        identities[identity.name] = identity
    expected = set(STACK_PIPELINE)
    observed = set(identities)
    _require(
        observed == expected,
        "wheel set must contain exactly "
        f"{sorted(expected)}; missing={sorted(expected - observed)}, "
        f"unexpected={sorted(observed - expected)}",
    )

    distributions = []
    for name in STACK_PIPELINE:
        identity = identities[name]
        distributions.append(
            {
                "name": name,
                "version": identity.version,
                "wheel": {
                    "filename": identity.filename,
                    "sha256": identity.sha256,
                    "size_bytes": identity.size_bytes,
                },
                "source": {
                    "repository": SOURCE_REPOSITORIES[name],
                    "revision": revisions[name],
                },
                "required_modules": list(REQUIRED_MODULES[name]),
            }
        )
    payload: dict[str, Any] = {
        "schema_name": STACK_LOCK_SCHEMA_NAME,
        "schema_version": STACK_LOCK_SCHEMA_VERSION,
        "generated_by": STACK_LOCK_GENERATOR,
        "compatibility": {
            "pipeline": list(STACK_PIPELINE),
            "claim_bearing_provider_v2_required": True,
            "wheel_identity_required": True,
        },
        "distributions": distributions,
    }
    return {**payload, "lock_id": _payload_id(payload)}


def _validate_distribution(entry: Any, *, expected_name: str) -> None:
    _require(isinstance(entry, dict), "stack distribution entries must be objects")
    _require(
        set(entry) == {"name", "version", "wheel", "source", "required_modules"},
        f"stack entry for {expected_name} has unexpected fields",
    )
    _require(entry["name"] == expected_name, "stack distribution order is invalid")
    _canonical_version(entry["version"], label=f"locked version for {expected_name}")

    wheel = entry["wheel"]
    _require(
        isinstance(wheel, dict),
        f"wheel record for {expected_name} must be an object",
    )
    _require(
        set(wheel) == {"filename", "sha256", "size_bytes"},
        f"wheel record for {expected_name} has unexpected fields",
    )
    filename = wheel["filename"]
    _require(
        isinstance(filename, str)
        and bool(filename)
        and Path(filename).name == filename
        and filename.endswith(".whl"),
        f"wheel filename for {expected_name} is invalid",
    )
    _require(
        isinstance(wheel["sha256"], str)
        and _HEX_64.fullmatch(wheel["sha256"]) is not None,
        f"wheel SHA-256 for {expected_name} is invalid",
    )
    _require(
        isinstance(wheel["size_bytes"], int)
        and not isinstance(wheel["size_bytes"], bool)
        and wheel["size_bytes"] > 0,
        f"wheel size for {expected_name} must be a positive integer",
    )

    source = entry["source"]
    _require(
        isinstance(source, dict),
        f"source record for {expected_name} must be an object",
    )
    _require(
        set(source) == {"repository", "revision"},
        f"source record for {expected_name} has unexpected fields",
    )
    _require(
        source["repository"] == SOURCE_REPOSITORIES[expected_name],
        f"source repository for {expected_name} is not canonical",
    )
    _require(
        isinstance(source["revision"], str)
        and _HEX_40.fullmatch(source["revision"]) is not None,
        f"source revision for {expected_name} is invalid",
    )
    _require(
        entry["required_modules"] == list(REQUIRED_MODULES[expected_name]),
        f"required modules for {expected_name} are invalid",
    )


def validate_stack_lock(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a stack lock and return ordinary finite JSON containers."""

    try:
        normalized = json.loads(
            json.dumps(plain_json(value), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("stack lock must be finite JSON data") from error
    _require(isinstance(normalized, dict), "stack lock must be a JSON object")
    _require(
        set(normalized)
        == {
            "schema_name",
            "schema_version",
            "generated_by",
            "compatibility",
            "distributions",
            "lock_id",
        },
        "stack lock has missing or unexpected top-level fields",
    )
    _require(
        normalized["schema_name"] == STACK_LOCK_SCHEMA_NAME,
        "unsupported stack-lock schema name",
    )
    _require(
        normalized["schema_version"] == STACK_LOCK_SCHEMA_VERSION,
        "unsupported stack-lock schema version",
    )
    _require(
        normalized["generated_by"] == STACK_LOCK_GENERATOR,
        "unsupported stack-lock generator",
    )

    compatibility = normalized["compatibility"]
    _require(isinstance(compatibility, dict), "stack compatibility must be an object")
    _require(
        compatibility
        == {
            "pipeline": list(STACK_PIPELINE),
            "claim_bearing_provider_v2_required": True,
            "wheel_identity_required": True,
        },
        "stack compatibility declaration is invalid",
    )

    distributions = normalized["distributions"]
    _require(isinstance(distributions, list), "stack distributions must be an array")
    _require(
        len(distributions) == len(STACK_PIPELINE),
        "stack lock must contain exactly three distributions",
    )
    for expected_name, entry in zip(STACK_PIPELINE, distributions, strict=True):
        _validate_distribution(entry, expected_name=expected_name)

    lock_id = normalized["lock_id"]
    _require(
        isinstance(lock_id, str) and _HEX_64.fullmatch(lock_id) is not None,
        "stack lock_id is invalid",
    )
    payload_without_id = {
        key: item for key, item in normalized.items() if key != "lock_id"
    }
    _require(
        lock_id == _payload_id(payload_without_id),
        "stack lock_id does not match its payload",
    )
    return normalized


def load_stack_lock(path: str | Path) -> dict[str, Any]:
    """Load a duplicate-key-safe stack lock from JSON."""

    lock_path = Path(path)
    try:
        text = lock_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"stack lock cannot be read: {lock_path}") from error
    value = _load_json_text(text, label="stack lock")
    _require(isinstance(value, dict), "stack lock must be a JSON object")
    return validate_stack_lock(value)


def write_stack_lock(
    path: str | Path,
    lock: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Validate and atomically publish a stack lock."""

    atomic_write_json(path, validate_stack_lock(lock), overwrite=overwrite)


def verify_stack_lock(
    lock: Mapping[str, Any],
    *,
    wheel_paths: Sequence[str | Path] = (),
    require_wheels: bool = True,
) -> dict[str, Any]:
    """Verify lock integrity and, when supplied, exact wheel identities."""

    validated = validate_stack_lock(lock)
    expected = {entry["name"]: entry for entry in validated["distributions"]}
    if not wheel_paths:
        errors: list[str] = []
        if require_wheels:
            errors.append("exact verification requires all three locked wheels")
        return {
            "schema_name": STACK_VERIFICATION_SCHEMA_NAME,
            "schema_version": STACK_VERIFICATION_SCHEMA_VERSION,
            "lock_id": validated["lock_id"],
            "valid": not errors,
            "requirements": {"wheels": require_wheels},
            "wheel_set": {
                "provided": False,
                "complete": False,
                "verified": False,
                "entries": [],
            },
            "errors": errors,
        }

    observed: dict[str, WheelIdentity] = {}
    errors: list[str] = []
    for path in wheel_paths:
        try:
            identity = inspect_wheel(path)
        except ValueError as error:
            errors.append(str(error))
            continue
        if identity.name in observed:
            errors.append(f"duplicate wheel for distribution {identity.name}")
            continue
        observed[identity.name] = identity

    entries = []
    for name in STACK_PIPELINE:
        locked = expected[name]
        identity = observed.get(name)
        if identity is None:
            entries.append(
                {
                    "name": name,
                    "path": None,
                    "expected_version": locked["version"],
                    "observed_version": None,
                    "expected_sha256": locked["wheel"]["sha256"],
                    "observed_sha256": None,
                    "valid": False,
                }
            )
            continue
        valid = (
            identity.version == locked["version"]
            and identity.sha256 == locked["wheel"]["sha256"]
            and identity.size_bytes == locked["wheel"]["size_bytes"]
        )
        if not valid:
            errors.append(f"wheel identity mismatch for {name}")
        entries.append(
            {
                "name": name,
                "path": str(identity.path),
                "expected_version": locked["version"],
                "observed_version": identity.version,
                "expected_sha256": locked["wheel"]["sha256"],
                "observed_sha256": identity.sha256,
                "valid": valid,
            }
        )

    expected_names = set(expected)
    observed_names = set(observed)
    complete = expected_names == observed_names
    if wheel_paths and not complete:
        errors.append(
            "provided wheels do not match locked distributions: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"unexpected={sorted(observed_names - expected_names)}"
        )
    wheels_verified = complete and not errors and all(item["valid"] for item in entries)
    if require_wheels and not wheels_verified:
        errors.append("exact verification requires all three locked wheels")
    valid = not errors and (wheels_verified or not require_wheels)
    return {
        "schema_name": STACK_VERIFICATION_SCHEMA_NAME,
        "schema_version": STACK_VERIFICATION_SCHEMA_VERSION,
        "lock_id": validated["lock_id"],
        "valid": valid,
        "requirements": {"wheels": require_wheels},
        "wheel_set": {
            "provided": bool(wheel_paths),
            "complete": complete,
            "verified": wheels_verified,
            "entries": entries,
        },
        "errors": errors,
    }


__all__ = [
    "REQUIRED_MODULES",
    "SOURCE_REPOSITORIES",
    "STACK_LOCK_SCHEMA_NAME",
    "STACK_LOCK_SCHEMA_VERSION",
    "STACK_PIPELINE",
    "WheelIdentity",
    "build_stack_lock",
    "inspect_wheel",
    "load_stack_lock",
    "validate_stack_lock",
    "verify_stack_lock",
    "write_stack_lock",
]
