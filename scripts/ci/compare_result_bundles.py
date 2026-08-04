"""Compare result bundles bytewise and with field-aware numeric semantics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


_JSON_FILES = ("protocol.json", "success_gates.json", "summary.json")
_CSV_FILES = ("contact_recovery.csv", "fold_calibration.csv", "interventions.csv")
_FLOAT_TEXT = re.compile(
    r"^[+-]?(?:(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)$"
)
_DIRECTION_ANGLE_SUFFIX = "direction_error_deg"
_GATE_KEYS = frozenset({"name", "value", "comparison", "threshold", "passed"})
_EXACT_NUMERIC_FIELDS = frozenset(
    {
        "bytes",
        "case_count",
        "contact_hypothesis_count",
        "delay_map",
        "delay_truth",
        "fit_frame_stride",
        "forecast_start_frame",
        "frame_count",
        "held_out_topology_count",
        "object_count",
        "parameter_grid_count",
        "parameter_particle_count",
        "repeat_id",
        "schema_version",
        "seed",
        "source_condition_count",
        "training_repeats",
    }
)
_TOLERANCE_POLICY_ID = "causal4d-field-aware-cross-platform-v1"


def _path_field_name(path: str) -> str:
    if ":field=" in path:
        return path.rsplit(":field=", 1)[1]
    tail = path.rsplit(".", 1)[-1]
    return tail.split("[", 1)[0]


def _numeric_policy(path: str) -> str:
    if path.endswith(_DIRECTION_ANGLE_SUFFIX):
        return "near_zero_direction_angle"
    field_name = _path_field_name(path)
    if (
        field_name in _EXACT_NUMERIC_FIELDS
        or ".seeds[" in path
        or ".contact_nodes[" in path
    ):
        return "exact"
    return "floating"


@dataclass
class Comparison:
    """Accumulate structural differences and field-aware floating-point drift."""

    mismatches: list[str] = field(default_factory=list)
    numeric_comparisons: int = 0
    maximum_absolute_difference: float = 0.0
    maximum_relative_difference: float = 0.0
    maximum_direction_angle_difference_deg: float = 0.0
    gate_checks: int = 0
    numeric_policy_counts: dict[str, int] = field(default_factory=dict)

    def compare_number(
        self,
        expected: float,
        actual: float,
        *,
        path: str,
        relative_tolerance: float,
        absolute_tolerance: float,
        direction_angle_tolerance_deg: float,
    ) -> None:
        self.numeric_comparisons += 1
        policy = _numeric_policy(path)
        self.numeric_policy_counts[policy] = (
            self.numeric_policy_counts.get(policy, 0) + 1
        )
        if math.isnan(expected) or math.isnan(actual):
            if not (math.isnan(expected) and math.isnan(actual)):
                self.mismatches.append(
                    f"{path}: expected {expected!r}, received {actual!r}"
                )
            return
        if math.isinf(expected) or math.isinf(actual):
            if expected != actual:
                self.mismatches.append(
                    f"{path}: expected {expected!r}, received {actual!r}"
                )
            return
        absolute = abs(actual - expected)
        denominator = max(abs(expected), abs(actual), absolute_tolerance, 1e-300)
        relative = absolute / denominator
        self.maximum_absolute_difference = max(
            self.maximum_absolute_difference, absolute
        )
        self.maximum_relative_difference = max(
            self.maximum_relative_difference, relative
        )
        if policy == "exact":
            if expected != actual:
                self.mismatches.append(
                    f"{path}: exact numeric field changed from "
                    f"{expected!r} to {actual!r}"
                )
            return
        effective_absolute_tolerance = absolute_tolerance
        if policy == "near_zero_direction_angle":
            self.maximum_direction_angle_difference_deg = max(
                self.maximum_direction_angle_difference_deg,
                absolute,
            )
            effective_absolute_tolerance = max(
                effective_absolute_tolerance,
                direction_angle_tolerance_deg,
            )
        if not math.isclose(
            expected,
            actual,
            rel_tol=relative_tolerance,
            abs_tol=effective_absolute_tolerance,
        ):
            self.mismatches.append(
                f"{path}: expected {expected!r}, received {actual!r}; "
                f"absolute difference {absolute:.17g}"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_gate_record(value: Any) -> bool:
    return isinstance(value, dict) and _GATE_KEYS.issubset(value)


def _gate_decision(value: float, threshold: float, comparison: str) -> bool:
    if comparison == ">=":
        return value >= threshold
    if comparison == "<=":
        return value <= threshold
    raise ValueError(f"unsupported gate comparison: {comparison!r}")


def _compare_gate_record(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    path: str,
    comparison: Comparison,
    relative_tolerance: float,
    absolute_tolerance: float,
    direction_angle_tolerance_deg: float,
) -> None:
    comparison.gate_checks += 1
    expected_keys = set(expected)
    actual_keys = set(actual)
    if expected_keys != actual_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        comparison.mismatches.append(
            f"{path}: gate keys differ; missing={missing!r}, extra={extra!r}"
        )

    for key in ("name", "comparison", "threshold", "passed"):
        if key not in expected or key not in actual:
            continue
        expected_value = expected[key]
        actual_value = actual[key]
        if key == "threshold" and _is_number(expected_value) and _is_number(
            actual_value
        ):
            if float(expected_value) != float(actual_value):
                comparison.mismatches.append(
                    f"{path}.{key}: registered gate threshold changed from "
                    f"{expected_value!r} to {actual_value!r}"
                )
        elif (
            type(expected_value) is not type(actual_value)
            or expected_value != actual_value
        ):
            comparison.mismatches.append(
                f"{path}.{key}: expected {expected_value!r}, "
                f"received {actual_value!r}"
            )

    if "value" in expected and "value" in actual:
        if _is_number(expected["value"]) and _is_number(actual["value"]):
            comparison.compare_number(
                float(expected["value"]),
                float(actual["value"]),
                path=f"{path}.value",
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                direction_angle_tolerance_deg=direction_angle_tolerance_deg,
            )
        else:
            comparison.mismatches.append(
                f"{path}.value: gate values must both be numeric"
            )

    for label, record in (("expected", expected), ("actual", actual)):
        try:
            derived = _gate_decision(
                float(record["value"]),
                float(record["threshold"]),
                str(record["comparison"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            comparison.mismatches.append(
                f"{path}: {label} gate cannot be evaluated: {error}"
            )
            continue
        if not isinstance(record["passed"], bool):
            comparison.mismatches.append(
                f"{path}.passed: {label} gate decision is not boolean"
            )
        elif record["passed"] != derived:
            comparison.mismatches.append(
                f"{path}: {label} gate is internally inconsistent; "
                f"reported passed={record['passed']!r}, derived={derived!r}"
            )

    try:
        expected_decision = _gate_decision(
            float(expected["value"]),
            float(expected["threshold"]),
            str(expected["comparison"]),
        )
        actual_decision = _gate_decision(
            float(actual["value"]),
            float(actual["threshold"]),
            str(actual["comparison"]),
        )
    except (KeyError, TypeError, ValueError):
        expected_decision = actual_decision = None
    if (
        expected_decision is not None
        and actual_decision is not None
        and expected_decision != actual_decision
    ):
        comparison.mismatches.append(
            f"{path}: gate decision changed from "
            f"{expected_decision!r} to {actual_decision!r}; "
            "numeric tolerance cannot revise or rescue a gate"
        )

    for key in sorted((expected_keys & actual_keys) - _GATE_KEYS):
        _compare_json_value(
            expected[key],
            actual[key],
            path=f"{path}.{key}",
            comparison=comparison,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            direction_angle_tolerance_deg=direction_angle_tolerance_deg,
        )


def _compare_json_value(
    expected: Any,
    actual: Any,
    *,
    path: str,
    comparison: Comparison,
    relative_tolerance: float,
    absolute_tolerance: float,
    direction_angle_tolerance_deg: float,
) -> None:
    if _is_gate_record(expected) and _is_gate_record(actual):
        _compare_gate_record(
            expected,
            actual,
            path=path,
            comparison=comparison,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            direction_angle_tolerance_deg=direction_angle_tolerance_deg,
        )
        return
    if _is_number(expected) and _is_number(actual):
        comparison.compare_number(
            float(expected),
            float(actual),
            path=path,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            direction_angle_tolerance_deg=direction_angle_tolerance_deg,
        )
        return
    if type(expected) is not type(actual):
        comparison.mismatches.append(
            f"{path}: type differs ({type(expected).__name__} versus "
            f"{type(actual).__name__})"
        )
        return
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            comparison.mismatches.append(
                f"{path}: object keys differ; missing={missing!r}, extra={extra!r}"
            )
        for key in sorted(expected_keys & actual_keys):
            _compare_json_value(
                expected[key],
                actual[key],
                path=f"{path}.{key}",
                comparison=comparison,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                direction_angle_tolerance_deg=direction_angle_tolerance_deg,
            )
        return
    if isinstance(expected, list):
        if len(expected) != len(actual):
            comparison.mismatches.append(
                f"{path}: list length differs ({len(expected)} versus "
                f"{len(actual)})"
            )
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=False)
        ):
            _compare_json_value(
                expected_item,
                actual_item,
                path=f"{path}[{index}]",
                comparison=comparison,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                direction_angle_tolerance_deg=direction_angle_tolerance_deg,
            )
        return
    if expected != actual:
        comparison.mismatches.append(
            f"{path}: expected {expected!r}, received {actual!r}"
        )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _numeric_text(value: str) -> bool:
    return bool(_FLOAT_TEXT.fullmatch(value.strip()))


def _structured_json_text(value: str) -> bool:
    stripped = value.strip()
    return len(stripped) >= 2 and stripped[0] in "[{" and stripped[-1] in "]}"


def _compare_csv(
    expected_path: Path,
    actual_path: Path,
    *,
    comparison: Comparison,
    relative_tolerance: float,
    absolute_tolerance: float,
    direction_angle_tolerance_deg: float,
) -> None:
    expected_fields, expected_rows = _read_csv(expected_path)
    actual_fields, actual_rows = _read_csv(actual_path)
    label = expected_path.name
    if expected_fields != actual_fields:
        comparison.mismatches.append(
            f"{label}: header differs ({expected_fields!r} versus "
            f"{actual_fields!r})"
        )
        return
    if len(expected_rows) != len(actual_rows):
        comparison.mismatches.append(
            f"{label}: row count differs ({len(expected_rows)} versus "
            f"{len(actual_rows)})"
        )
    for row_index, (expected_row, actual_row) in enumerate(
        zip(expected_rows, actual_rows, strict=False), start=2
    ):
        for field_name in expected_fields:
            expected = expected_row[field_name]
            actual = actual_row[field_name]
            if expected == actual:
                continue
            path = f"{label}:row={row_index}:field={field_name}"
            if _numeric_text(expected) and _numeric_text(actual):
                comparison.compare_number(
                    float(expected),
                    float(actual),
                    path=path,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                    direction_angle_tolerance_deg=direction_angle_tolerance_deg,
                )
            elif _structured_json_text(expected) and _structured_json_text(actual):
                try:
                    expected_value = json.loads(expected)
                    actual_value = json.loads(actual)
                except json.JSONDecodeError:
                    comparison.mismatches.append(
                        f"{path}: expected {expected!r}, received {actual!r}"
                    )
                else:
                    _compare_json_value(
                        expected_value,
                        actual_value,
                        path=path,
                        comparison=comparison,
                        relative_tolerance=relative_tolerance,
                        absolute_tolerance=absolute_tolerance,
                        direction_angle_tolerance_deg=direction_angle_tolerance_deg,
                    )
            else:
                comparison.mismatches.append(
                    f"{path}: expected {expected!r}, received {actual!r}"
                )


def _compare_file(
    expected_path: Path,
    actual_path: Path,
    *,
    comparison: Comparison,
    relative_tolerance: float,
    absolute_tolerance: float,
    direction_angle_tolerance_deg: float,
) -> dict[str, Any]:
    if not expected_path.is_file():
        comparison.mismatches.append(f"missing expected file: {expected_path}")
        return {"present": False}
    if not actual_path.is_file():
        comparison.mismatches.append(f"missing actual file: {actual_path}")
        return {"present": False}
    expected_bytes = expected_path.read_bytes()
    actual_bytes = actual_path.read_bytes()
    record = {
        "present": True,
        "byte_identical": expected_bytes == actual_bytes,
        "expected_bytes": len(expected_bytes),
        "actual_bytes": len(actual_bytes),
        "expected_sha256": _sha256(expected_path),
        "actual_sha256": _sha256(actual_path),
    }
    if expected_path.suffix == ".json":
        expected = json.loads(expected_bytes)
        actual = json.loads(actual_bytes)
        _compare_json_value(
            expected,
            actual,
            path=expected_path.name,
            comparison=comparison,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            direction_angle_tolerance_deg=direction_angle_tolerance_deg,
        )
    elif expected_path.suffix == ".csv":
        _compare_csv(
            expected_path,
            actual_path,
            comparison=comparison,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            direction_angle_tolerance_deg=direction_angle_tolerance_deg,
        )
    else:
        raise ValueError(f"unsupported comparison file: {expected_path}")
    return record


def _runtime_environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a regenerated deterministic result bundle with an archived "
            "bundle, preserving byte mismatch as a diagnostic while failing on "
            "structural, gate, categorical, or substantive numeric drift."
        )
    )
    parser.add_argument("expected_dir")
    parser.add_argument("actual_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--relative-tolerance", type=float, default=2e-12)
    parser.add_argument("--absolute-tolerance", type=float, default=2e-15)
    parser.add_argument(
        "--direction-angle-tolerance-deg",
        type=float,
        default=2e-6,
        help=(
            "Absolute tolerance for direction angles near zero, where arccos "
            "amplifies machine-level cosine differences."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tolerances = (
        args.relative_tolerance,
        args.absolute_tolerance,
        args.direction_angle_tolerance_deg,
    )
    if any(value < 0.0 or not math.isfinite(value) for value in tolerances):
        raise ValueError("comparison tolerances must be finite and nonnegative")
    expected_dir = Path(args.expected_dir).resolve()
    actual_dir = Path(args.actual_dir).resolve()
    comparison = Comparison()
    file_records: dict[str, Any] = {}
    for file_name in (*_JSON_FILES, *_CSV_FILES):
        file_records[file_name] = _compare_file(
            expected_dir / file_name,
            actual_dir / file_name,
            comparison=comparison,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance=args.absolute_tolerance,
            direction_angle_tolerance_deg=args.direction_angle_tolerance_deg,
        )
    report = {
        "schema_version": 2,
        "artifact_kind": "Causal4DResultBundleComparison",
        "tolerance_policy_id": _TOLERANCE_POLICY_ID,
        "expected_dir": str(expected_dir),
        "actual_dir": str(actual_dir),
        "comparison_environment": _runtime_environment(),
        "relative_tolerance": args.relative_tolerance,
        "absolute_tolerance": args.absolute_tolerance,
        "direction_angle_tolerance_deg": args.direction_angle_tolerance_deg,
        "all_payload_bytes_match": all(
            record.get("byte_identical") is True for record in file_records.values()
        ),
        "semantic_match": not comparison.mismatches,
        "numeric_comparisons": comparison.numeric_comparisons,
        "numeric_policy_counts": dict(sorted(comparison.numeric_policy_counts.items())),
        "gate_checks": comparison.gate_checks,
        "maximum_absolute_difference": comparison.maximum_absolute_difference,
        "maximum_relative_difference": comparison.maximum_relative_difference,
        "maximum_direction_angle_difference_deg": (
            comparison.maximum_direction_angle_difference_deg
        ),
        "mismatch_count": len(comparison.mismatches),
        "mismatches": comparison.mismatches[:100],
        "files": file_records,
        "claim_boundary": (
            "Frozen artifact identity requires the archived bytes and SHA-256 "
            "values. Independent numerical reproduction requires exact structure, "
            "categorical values, row order, registered gate definitions and "
            "decisions, plus field-aware numerical agreement. Tolerances can "
            "never change, revise, or rescue a scientific gate."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["semantic_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
