"""Validation of runtime-bound result-bundle reproduction sidecars."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from reproduction_manifest_runtime import (
    COMPARISON_CONTRACT_VERSION,
    REPRODUCTION_MANIFEST_KIND,
    REPRODUCTION_MANIFEST_SCHEMA_VERSION,
    _DISTRIBUTIONS,
    _RUNTIME_ENVIRONMENT_KEYS,
)
from result_bundle_identity import (
    VerifiedResultBundle,
    _require_ordinary_file,
    _valid_sha256,
    load_strict_json,
    sha256_file,
)


_COMMIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")

def _require_exact_keys(
    record: dict[str, Any],
    expected_keys: set[str],
    *,
    label: str,
) -> None:
    actual_keys = set(record)
    if actual_keys != expected_keys:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )


def _require_nonempty_optional_string(value: Any, *, label: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{label} must be null or a nonempty string")


def _validate_runtime_identity(runtime: Any) -> None:
    if not isinstance(runtime, dict):
        raise ValueError("reproduction manifest runtime must be an object")
    _require_exact_keys(
        runtime,
        {"python", "platform", "distributions", "numerical_backends", "environment"},
        label="reproduction manifest runtime",
    )

    python_record = runtime["python"]
    if not isinstance(python_record, dict):
        raise ValueError("runtime python identity must be an object")
    _require_exact_keys(
        python_record,
        {"implementation", "version", "compiler", "build", "executable"},
        label="runtime python identity",
    )
    for key in ("implementation", "version", "compiler", "executable"):
        if not isinstance(python_record[key], str):
            raise ValueError(f"runtime python {key} must be a string")
    build = python_record["build"]
    if not (
        isinstance(build, list)
        and len(build) == 2
        and all(isinstance(item, str) for item in build)
    ):
        raise ValueError("runtime python build must contain two strings")

    platform_record = runtime["platform"]
    if not isinstance(platform_record, dict):
        raise ValueError("runtime platform identity must be an object")
    _require_exact_keys(
        platform_record,
        {"system", "release", "version", "machine", "processor", "libc"},
        label="runtime platform identity",
    )
    for key in ("system", "release", "version", "machine", "processor"):
        if not isinstance(platform_record[key], str):
            raise ValueError(f"runtime platform {key} must be a string")
    libc = platform_record["libc"]
    if not (
        isinstance(libc, list)
        and len(libc) == 2
        and all(isinstance(item, str) for item in libc)
    ):
        raise ValueError("runtime platform libc must contain two strings")

    distributions = runtime["distributions"]
    if not isinstance(distributions, dict):
        raise ValueError("runtime distributions must be an object")
    _require_exact_keys(
        distributions,
        set(_DISTRIBUTIONS),
        label="runtime distributions",
    )
    for distribution, value in distributions.items():
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"runtime distribution {distribution!r} must be null or a string"
            )

    backends = runtime["numerical_backends"]
    if not isinstance(backends, dict):
        raise ValueError("runtime numerical_backends must be an object")
    _require_exact_keys(
        backends,
        {"numpy", "scipy"},
        label="runtime numerical_backends",
    )
    for backend_name, backend in backends.items():
        if not isinstance(backend, dict):
            raise ValueError(f"runtime backend {backend_name!r} must be an object")
        available = backend.get("available")
        if type(available) is not bool:
            raise ValueError(
                f"runtime backend {backend_name!r} available must be a boolean"
            )
        if not available:
            _require_exact_keys(
                backend,
                {"available"},
                label=f"runtime backend {backend_name!r}",
            )
            continue
        _require_exact_keys(
            backend,
            {"available", "format", "value"},
            label=f"runtime backend {backend_name!r}",
        )
        format_name = backend["format"]
        if format_name == "mapping":
            if not isinstance(backend["value"], dict):
                raise ValueError(
                    f"runtime backend {backend_name!r} mapping value must be an object"
                )
        elif format_name == "text":
            if not isinstance(backend["value"], str):
                raise ValueError(
                    f"runtime backend {backend_name!r} text value must be a string"
                )
        else:
            raise ValueError(
                f"runtime backend {backend_name!r} format is unsupported"
            )

    environment = runtime["environment"]
    if not isinstance(environment, dict):
        raise ValueError("runtime environment must be an object")
    _require_exact_keys(
        environment,
        set(_RUNTIME_ENVIRONMENT_KEYS),
        label="runtime environment",
    )
    for key, value in environment.items():
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"runtime environment variable {key!r} must be null or a string"
            )


def _validate_source_identity(source: Any) -> None:
    if not isinstance(source, dict):
        raise ValueError("reproduction manifest source must be an object")
    _require_exact_keys(
        source,
        {"repository", "commit_sha", "workflow_run_id", "runner_name"},
        label="reproduction manifest source",
    )
    for key in ("repository", "workflow_run_id", "runner_name"):
        _require_nonempty_optional_string(source[key], label=f"source {key}")
    commit_sha = source["commit_sha"]
    _require_nonempty_optional_string(commit_sha, label="source commit_sha")
    if commit_sha is not None and not _COMMIT_SHA.fullmatch(commit_sha):
        raise ValueError(
            "source commit_sha must be a lowercase 40- or 64-character hex digest"
        )


def _require_record_matches_file(
    record: Any,
    *,
    path: Path,
    label: str,
    expected_keys: set[str],
) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{label} record must be an object")
    _require_exact_keys(record, expected_keys, label=f"{label} record")
    expected_bytes = record.get("bytes")
    expected_hash = record.get("sha256")
    if type(expected_bytes) is not int or expected_bytes < 0:
        raise ValueError(f"{label} has an invalid byte count")
    if not _valid_sha256(expected_hash):
        raise ValueError(f"{label} has an invalid SHA-256 digest")
    _require_ordinary_file(path, label=label)
    actual_bytes = path.stat().st_size
    actual_hash = sha256_file(path)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{label} byte count changed: {actual_bytes} != {expected_bytes}"
        )
    if actual_hash != expected_hash:
        raise ValueError(f"{label} checksum changed: {actual_hash} != {expected_hash}")


def validate_reproduction_manifest(
    path: Path,
    bundle: VerifiedResultBundle,
) -> dict[str, Any]:
    """Validate a reproduction sidecar against a verified result bundle."""

    document = load_strict_json(path)
    if not isinstance(document, dict):
        raise ValueError("reproduction manifest must be a JSON object")
    _require_exact_keys(
        document,
        {
            "schema_version",
            "artifact_kind",
            "comparison_contract_version",
            "bundle",
            "runtime",
            "source",
            "claim_boundary",
        },
        label="reproduction manifest",
    )
    schema_version = document["schema_version"]
    if (
        type(schema_version) is not int
        or schema_version != REPRODUCTION_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("reproduction manifest schema_version must equal integer 1")
    comparison_contract_version = document["comparison_contract_version"]
    if (
        type(comparison_contract_version) is not int
        or comparison_contract_version != COMPARISON_CONTRACT_VERSION
    ):
        raise ValueError("reproduction manifest comparison contract is unsupported")
    if document["artifact_kind"] != REPRODUCTION_MANIFEST_KIND:
        raise ValueError("reproduction manifest artifact_kind is invalid")
    claim_boundary = document["claim_boundary"]
    if not isinstance(claim_boundary, str) or not claim_boundary:
        raise ValueError("reproduction manifest claim_boundary must be nonempty")

    bundle_record = document["bundle"]
    if not isinstance(bundle_record, dict):
        raise ValueError("reproduction manifest bundle must be an object")
    _require_exact_keys(
        bundle_record,
        {
            "benchmark",
            "result_manifest_schema_version",
            "result_manifest",
            "payloads",
        },
        label="reproduction manifest bundle",
    )
    if bundle_record["benchmark"] != bundle.benchmark:
        raise ValueError("reproduction manifest benchmark does not match the bundle")
    result_schema_version = bundle_record["result_manifest_schema_version"]
    if (
        type(result_schema_version) is not int
        or result_schema_version != bundle.schema_version
    ):
        raise ValueError(
            "reproduction manifest result schema does not match the bundle"
        )
    result_manifest_record = bundle_record["result_manifest"]
    if not isinstance(result_manifest_record, dict):
        raise ValueError("reproduction manifest result_manifest must be an object")
    if result_manifest_record.get("name") != bundle.manifest_path.name:
        raise ValueError("reproduction manifest result manifest name is invalid")
    _require_record_matches_file(
        result_manifest_record,
        path=bundle.manifest_path,
        label="result manifest",
        expected_keys={"name", "bytes", "sha256"},
    )

    raw_payloads = bundle_record["payloads"]
    if not isinstance(raw_payloads, dict):
        raise ValueError("reproduction manifest payloads must be an object")
    expected_names = {artifact.name for artifact in bundle.artifacts}
    if set(raw_payloads) != expected_names:
        raise ValueError(
            "reproduction manifest payload inventory differs from the bundle: "
            f"expected={sorted(expected_names)}, actual={sorted(raw_payloads)}"
        )
    for artifact in bundle.artifacts:
        _require_record_matches_file(
            raw_payloads[artifact.name],
            path=artifact.path,
            label=f"payload {artifact.name!r}",
            expected_keys={"bytes", "sha256"},
        )

    runtime = document["runtime"]
    source = document["source"]
    _validate_runtime_identity(runtime)
    _validate_source_identity(source)
    return {
        "valid": True,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "runtime": runtime,
        "source": source,
    }
