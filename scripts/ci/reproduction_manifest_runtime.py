"""Runtime capture and construction of result-bundle reproduction sidecars."""

from __future__ import annotations

from contextlib import redirect_stdout
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

from result_bundle_identity import VerifiedResultBundle, sha256_file


REPRODUCTION_MANIFEST_SCHEMA_VERSION = 1
COMPARISON_CONTRACT_VERSION = 2
REPRODUCTION_MANIFEST_KIND = "Causal4DResultBundleReproductionManifest"
_RUNTIME_ENVIRONMENT_KEYS = (
    "BLIS_NUM_THREADS",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONHASHSEED",
    "VECLIB_MAXIMUM_THREADS",
)
_DISTRIBUTIONS = ("causal4d", "numpy", "scipy")

def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    return str(value)


def _distribution_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in _DISTRIBUTIONS:
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = None
    return versions


def _module_configuration(module_name: str) -> dict[str, Any]:
    module = import_module(module_name)
    configuration_module = getattr(module, "__config__", None)
    if configuration_module is None:
        return {"available": False}
    raw_configuration = getattr(configuration_module, "CONFIG", None)
    if isinstance(raw_configuration, Mapping):
        return {
            "available": True,
            "format": "mapping",
            "value": _normalize_json_value(raw_configuration),
        }
    show = getattr(configuration_module, "show", None)
    if callable(show):
        stream = io.StringIO()
        with redirect_stdout(stream):
            show()
        return {
            "available": True,
            "format": "text",
            "value": stream.getvalue().strip(),
        }
    return {"available": False}


def capture_runtime_identity() -> dict[str, Any]:
    """Capture stable runtime identity fields relevant to numerical reproduction."""

    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "compiler": platform.python_compiler(),
            "build": list(platform.python_build()),
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "libc": list(platform.libc_ver()),
        },
        "distributions": _distribution_versions(),
        "numerical_backends": {
            "numpy": _module_configuration("numpy"),
            "scipy": _module_configuration("scipy"),
        },
        "environment": {
            key: os.environ.get(key) for key in _RUNTIME_ENVIRONMENT_KEYS
        },
    }


def build_reproduction_manifest(
    bundle: VerifiedResultBundle,
    *,
    repository: str | None = None,
    commit_sha: str | None = None,
    workflow_run_id: str | None = None,
    runner_name: str | None = None,
) -> dict[str, Any]:
    """Build a runtime sidecar without modifying the result bundle itself."""

    manifest_record = {
        "name": bundle.manifest_path.name,
        "bytes": bundle.manifest_path.stat().st_size,
        "sha256": sha256_file(bundle.manifest_path),
    }
    payloads = {
        artifact.name: {
            "bytes": artifact.byte_count,
            "sha256": artifact.sha256,
        }
        for artifact in bundle.artifacts
    }
    return {
        "schema_version": REPRODUCTION_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": REPRODUCTION_MANIFEST_KIND,
        "comparison_contract_version": COMPARISON_CONTRACT_VERSION,
        "bundle": {
            "benchmark": bundle.benchmark,
            "result_manifest_schema_version": bundle.schema_version,
            "result_manifest": manifest_record,
            "payloads": payloads,
        },
        "runtime": capture_runtime_identity(),
        "source": {
            "repository": repository,
            "commit_sha": commit_sha,
            "workflow_run_id": workflow_run_id,
            "runner_name": runner_name,
        },
        "claim_boundary": (
            "This sidecar records the runtime and exact bytes used for an "
            "independent numerical reproduction. It does not replace, rewrite, "
            "or rehash the frozen result bundle."
        ),
    }


def write_reproduction_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write a canonical UTF-8 reproduction manifest."""

    if path.is_symlink():
        raise ValueError("reproduction manifest must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

