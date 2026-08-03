"""Compare result bundles bytewise and with strict numeric semantics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


_JSON_FILES = ("protocol.json", "success_gates.json", "summary.json")
_CSV_FILES = ("contact_recovery.csv", "fold_calibration.csv", "interventions.csv")
_FLOAT_TEXT = re.compile(
    r"^[+-]?(?:(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)$"
)


@dataclass
class Comparison:
    """Accumulate strict semantic differences and floating-point drift."""

    mismatches: list[str] = field(default_factory=list)
    numeric_comparisons: int = 0
    maximum_absolute_difference: float = 0.0
    maximum_relative_difference: float = 0.0

    def compare_number(
        self,
        expected: float,
        actual: float,
        *,
        path: str,
        relative_tolerance: float,
        absolute_tolerance: float,
    ) -> None:
        self.numeric_comparisons += 1
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
        denominator = max(abs(expected), abs(actual), absolute_tolerance)
        relative = absolute / denominator
        self.maximum_absolute_difference = max(
            self.maximum_absolute_difference, absolute
        )
        self.maximum_relative_difference = max(
            self.maximum_relative_difference, relative
        )
        if not math.isclose(
            expected,
            actual,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
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


def _compare_json_value(
    expected: Any,
    actual: Any,
    *,
    path: str,
    comparison: Comparison,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    if _is_number(expected) and _is_number(actual):
        comparison.compare_number(
            float(expected),
            float(actual),
            path=path,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
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
            )
        return
    if isinstance(expected, list):
        if len(expected) != len(actual):
            comparison.mismatches.append(
                f"{path}: list length differs ({len(expected)} versus {len(actual)})"
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


def _compare_csv(
    expected_path: Path,
    actual_path: Path,
    *,
    comparison: Comparison,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> None:
    expected_fields, expected_rows = _read_csv(expected_path)
    actual_fields, actual_rows = _read_csv(actual_path)
    label = expected_path.name
    if expected_fields != actual_fields:
        comparison.mismatches.append(
            f"{label}: header differs ({expected_fields!r} versus {actual_fields!r})"
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
        )
    elif expected_path.suffix == ".csv":
        _compare_csv(
            expected_path,
            actual_path,
            comparison=comparison,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
    else:
        raise ValueError(f"unsupported comparison file: {expected_path}")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a regenerated deterministic result bundle with an archived "
            "bundle, preserving byte mismatch as a diagnostic while failing on "
            "semantic drift."
        )
    )
    parser.add_argument("expected_dir")
    parser.add_argument("actual_dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--relative-tolerance", type=float, default=1e-12)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.relative_tolerance < 0.0 or args.absolute_tolerance < 0.0:
        raise ValueError("comparison tolerances must be nonnegative")
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
        )
    report = {
        "schema_version": 1,
        "artifact_kind": "Causal4DResultBundleComparison",
        "expected_dir": str(expected_dir),
        "actual_dir": str(actual_dir),
        "relative_tolerance": args.relative_tolerance,
        "absolute_tolerance": args.absolute_tolerance,
        "all_payload_bytes_match": all(
            record.get("byte_identical") is True for record in file_records.values()
        ),
        "semantic_match": not comparison.mismatches,
        "numeric_comparisons": comparison.numeric_comparisons,
        "maximum_absolute_difference": comparison.maximum_absolute_difference,
        "maximum_relative_difference": comparison.maximum_relative_difference,
        "mismatch_count": len(comparison.mismatches),
        "mismatches": comparison.mismatches[:100],
        "files": file_records,
        "claim_boundary": (
            "Byte equality is reported separately from strict numeric and "
            "structural equivalence; tolerance is never used to revise a gate."
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
