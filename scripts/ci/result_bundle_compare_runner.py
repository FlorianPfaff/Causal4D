"""Orchestration and reporting for strict result-bundle comparison."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from reproduction_manifest_runtime import COMPARISON_CONTRACT_VERSION
from reproduction_manifest_validation import validate_reproduction_manifest
from result_bundle_compare_payloads import _compare_artifact
from result_bundle_compare_values import Comparison, _DIRECTION_ANGLE_FIELD
from result_bundle_identity import (
    RESULT_MANIFEST_NAME,
    VerifiedResultBundle,
    sha256_file,
    verify_result_manifest,
)


def _load_bundle(
    directory: Path,
    *,
    role: str,
    comparison: Comparison,
) -> VerifiedResultBundle | None:
    try:
        return verify_result_manifest(
            directory / RESULT_MANIFEST_NAME,
            directory,
        )
    except (FileNotFoundError, ValueError) as error:
        comparison.add_mismatch(f"{role} result bundle is invalid: {error}")
        return None


def _manifest_identity(bundle: VerifiedResultBundle | None) -> dict[str, Any] | None:
    if bundle is None:
        return None
    return {
        "path": str(bundle.manifest_path),
        "bytes": bundle.manifest_path.stat().st_size,
        "sha256": sha256_file(bundle.manifest_path),
        "schema_version": bundle.schema_version,
        "benchmark": bundle.benchmark,
        "artifact_names": [artifact.name for artifact in bundle.artifacts],
    }


def _validate_tolerance(value: float, *, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _comparison_contract(
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    direction_angle_tolerance_deg: float,
) -> dict[str, Any]:
    return {
        "version": COMPARISON_CONTRACT_VERSION,
        "result_manifest": (
            "Each bundle manifest must independently match its exact payload "
            "inventory, byte counts, and SHA-256 digests. Benchmark, schema, and "
            "artifact names are exact across bundles."
        ),
        "exact_semantics": [
            "JSON value types, object keys, list lengths, and list order",
            "JSON integers and CSV integer lexemes",
            "strings, booleans, nulls, categories, and CSV row order",
            "success-gate names, comparisons, thresholds, decisions, and "
            "overall decision",
            "internal success-gate truth consistency",
        ],
        "floating_semantics": {
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
            "direction_angle_tolerance_deg": direction_angle_tolerance_deg,
            "direction_angle_field": _DIRECTION_ANGLE_FIELD,
        },
        "forbidden": [
            "duplicate JSON object keys",
            "non-finite JSON numbers or CSV numeric tokens",
            "symlinks in result bundles",
            "using tolerance to change or rescue a scientific gate",
        ],
    }


def compare_result_bundles(
    expected_dir: Path,
    actual_dir: Path,
    *,
    relative_tolerance: float = 2e-12,
    absolute_tolerance: float = 2e-15,
    direction_angle_tolerance_deg: float = 2e-6,
    expected_reproduction_manifest: Path | None = None,
    actual_reproduction_manifest: Path | None = None,
    require_actual_reproduction_manifest: bool = False,
) -> dict[str, Any]:
    """Return a byte-identity and strict semantic comparison report."""

    _validate_tolerance(relative_tolerance, name="relative tolerance")
    _validate_tolerance(absolute_tolerance, name="absolute tolerance")
    _validate_tolerance(
        direction_angle_tolerance_deg,
        name="direction-angle tolerance",
    )
    expected_dir = expected_dir.absolute()
    actual_dir = actual_dir.absolute()
    comparison = Comparison()
    expected_bundle = _load_bundle(
        expected_dir,
        role="expected",
        comparison=comparison,
    )
    actual_bundle = _load_bundle(
        actual_dir,
        role="actual",
        comparison=comparison,
    )

    file_records: dict[str, Any] = {}
    payload_inventory_matches = False
    if expected_bundle is not None and actual_bundle is not None:
        if expected_bundle.schema_version != actual_bundle.schema_version:
            comparison.add_mismatch(
                "result manifest schema differs "
                f"({expected_bundle.schema_version} versus "
                f"{actual_bundle.schema_version})"
            )
        if expected_bundle.benchmark != actual_bundle.benchmark:
            comparison.add_mismatch(
                "result manifest benchmark differs "
                f"({expected_bundle.benchmark!r} versus "
                f"{actual_bundle.benchmark!r})"
            )
        expected_artifacts = expected_bundle.artifacts_by_name
        actual_artifacts = actual_bundle.artifacts_by_name
        expected_names = set(expected_artifacts)
        actual_names = set(actual_artifacts)
        payload_inventory_matches = expected_names == actual_names
        if not payload_inventory_matches:
            comparison.add_mismatch(
                "result payload inventory differs; "
                f"missing={sorted(expected_names - actual_names)}, "
                f"extra={sorted(actual_names - expected_names)}"
            )
        for name in sorted(expected_names & actual_names):
            file_records[name] = _compare_artifact(
                expected_artifacts[name],
                actual_artifacts[name],
                comparison=comparison,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                direction_angle_tolerance_deg=direction_angle_tolerance_deg,
            )
        for name in sorted(expected_names - actual_names):
            file_records[name] = {"present": False}

    reproduction_records: dict[str, Any] = {"expected": None, "actual": None}
    if expected_reproduction_manifest is not None:
        if expected_bundle is None:
            comparison.add_mismatch(
                "expected reproduction manifest cannot be validated because the "
                "expected result bundle is invalid"
            )
        else:
            try:
                reproduction_records["expected"] = validate_reproduction_manifest(
                    expected_reproduction_manifest,
                    expected_bundle,
                )
            except (FileNotFoundError, ValueError) as error:
                comparison.add_mismatch(
                    f"expected reproduction manifest is invalid: {error}"
                )
    if actual_reproduction_manifest is not None:
        if actual_bundle is None:
            comparison.add_mismatch(
                "actual reproduction manifest cannot be validated because the "
                "actual result bundle is invalid"
            )
        else:
            try:
                reproduction_records["actual"] = validate_reproduction_manifest(
                    actual_reproduction_manifest,
                    actual_bundle,
                )
            except (FileNotFoundError, ValueError) as error:
                comparison.add_mismatch(
                    f"actual reproduction manifest is invalid: {error}"
                )
    elif require_actual_reproduction_manifest:
        comparison.add_mismatch("an actual reproduction manifest is required")

    expected_manifest_identity = _manifest_identity(expected_bundle)
    actual_manifest_identity = _manifest_identity(actual_bundle)
    result_manifests_byte_identical = bool(
        expected_manifest_identity
        and actual_manifest_identity
        and expected_manifest_identity["sha256"] == actual_manifest_identity["sha256"]
    )
    all_payload_bytes_match = bool(
        payload_inventory_matches
        and file_records
        and all(
            record.get("byte_identical") is True for record in file_records.values()
        )
    )
    report = {
        "schema_version": 2,
        "artifact_kind": "Causal4DResultBundleComparison",
        "expected_dir": str(expected_dir),
        "actual_dir": str(actual_dir),
        "comparison_contract": _comparison_contract(
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            direction_angle_tolerance_deg=direction_angle_tolerance_deg,
        ),
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "direction_angle_tolerance_deg": direction_angle_tolerance_deg,
        "result_manifests_byte_identical": result_manifests_byte_identical,
        "all_payload_bytes_match": all_payload_bytes_match,
        "semantic_match": not comparison.mismatches,
        "numeric_comparisons": comparison.numeric_comparisons,
        "exact_numeric_comparisons": comparison.exact_numeric_comparisons,
        "maximum_absolute_difference": comparison.maximum_absolute_difference,
        "maximum_relative_difference": comparison.maximum_relative_difference,
        "maximum_direction_angle_difference_deg": (
            comparison.maximum_direction_angle_difference_deg
        ),
        "mismatch_count": len(comparison.mismatches),
        "mismatches": comparison.mismatches[:100],
        "result_manifests": {
            "expected": expected_manifest_identity,
            "actual": actual_manifest_identity,
        },
        "reproduction_manifests": reproduction_records,
        "files": file_records,
        "claim_boundary": (
            "Frozen artifact identity is exact byte and digest equality under its "
            "recorded environment. Independent reproduction is a separate, "
            "field-aware semantic contract. Float tolerances never apply to "
            "schema, identity, ordering, integer fields, thresholds, or gate "
            "decisions, and byte mismatches remain explicit diagnostics."
        ),
    }
    return report
