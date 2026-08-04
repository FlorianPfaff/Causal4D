"""Strict CSV and payload comparison for verified result bundles."""

from __future__ import annotations

import csv
import io
import re
from typing import Any

from result_bundle_compare_values import (
    Comparison,
    _compare_json_value,
    _validate_success_gates_document,
)
from result_bundle_identity import ArtifactRecord, load_strict_json_bytes


_FLOAT_TEXT = re.compile(
    r"^[+-]?(?:(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)$"
)
_INTEGER_TEXT = re.compile(r"^[+-]?\d+$")
_NONFINITE_TEXT = {
    "nan",
    "+nan",
    "-nan",
    "inf",
    "+inf",
    "-inf",
    "infinity",
    "+infinity",
    "-infinity",
}


def _read_csv_bytes(
    payload: bytes,
    *,
    source: str,
) -> tuple[list[str], list[list[str]]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"CSV is not valid UTF-8: {source}") from error
    reader = csv.reader(io.StringIO(text, newline=""))
    rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {source}")
    header = rows[0]
    if not header or any(not field for field in header):
        raise ValueError(f"CSV header contains an empty field: {source}")
    if len(set(header)) != len(header):
        raise ValueError(f"CSV header contains duplicate fields: {source}")
    data_rows = rows[1:]
    for line_number, row in enumerate(data_rows, start=2):
        if len(row) != len(header):
            raise ValueError(
                f"CSV row {line_number} has {len(row)} fields; "
                f"expected {len(header)}: {source}"
            )
    return header, data_rows


def _numeric_text(value: str) -> bool:
    return bool(_FLOAT_TEXT.fullmatch(value.strip()))


def _integer_text(value: str) -> bool:
    return bool(_INTEGER_TEXT.fullmatch(value.strip()))


def _nonfinite_text(value: str) -> bool:
    return value.strip().lower() in _NONFINITE_TEXT


def _structured_json_text(value: str) -> bool:
    stripped = value.strip()
    return len(stripped) >= 2 and stripped[0] in "[{" and stripped[-1] in "]}"


def _compare_csv_bytes(
    expected_payload: bytes,
    actual_payload: bytes,
    *,
    label: str,
    comparison: Comparison,
    relative_tolerance: float,
    absolute_tolerance: float,
    direction_angle_tolerance_deg: float,
) -> None:
    try:
        expected_fields, expected_rows = _read_csv_bytes(
            expected_payload,
            source=f"expected {label}",
        )
    except ValueError as error:
        comparison.add_mismatch(str(error))
        return
    try:
        actual_fields, actual_rows = _read_csv_bytes(
            actual_payload,
            source=f"actual {label}",
        )
    except ValueError as error:
        comparison.add_mismatch(str(error))
        return
    if expected_fields != actual_fields:
        comparison.add_mismatch(
            f"{label}: header differs ({expected_fields!r} versus {actual_fields!r})"
        )
        return
    if len(expected_rows) != len(actual_rows):
        comparison.add_mismatch(
            f"{label}: row count differs "
            f"({len(expected_rows)} versus {len(actual_rows)})"
        )
    for row_index, (expected_row, actual_row) in enumerate(
        zip(expected_rows, actual_rows, strict=False),
        start=2,
    ):
        for field_index, field_name in enumerate(expected_fields):
            expected = expected_row[field_index]
            actual = actual_row[field_index]
            path = f"{label}:row={row_index}:field={field_name}"
            if _nonfinite_text(expected) or _nonfinite_text(actual):
                comparison.add_mismatch(f"{path}: non-finite numeric text is forbidden")
            elif _structured_json_text(expected) and _structured_json_text(actual):
                try:
                    expected_value = load_strict_json_bytes(
                        expected.encode("utf-8"),
                        source=f"expected {path}",
                    )
                    actual_value = load_strict_json_bytes(
                        actual.encode("utf-8"),
                        source=f"actual {path}",
                    )
                except ValueError as error:
                    comparison.add_mismatch(str(error))
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
            elif _integer_text(expected) and _integer_text(actual):
                comparison.exact_numeric_comparisons += 1
                if expected != actual:
                    comparison.add_mismatch(
                        f"{path}: exact integer text differs "
                        f"({expected!r} versus {actual!r})"
                    )
            elif _numeric_text(expected) and _numeric_text(actual):
                comparison.compare_float(
                    float(expected),
                    float(actual),
                    path=path,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                    direction_angle_tolerance_deg=direction_angle_tolerance_deg,
                )
            elif expected != actual:
                comparison.add_mismatch(
                    f"{path}: expected {expected!r}, received {actual!r}"
                )


def _compare_json_bytes(
    expected_payload: bytes,
    actual_payload: bytes,
    *,
    label: str,
    comparison: Comparison,
    relative_tolerance: float,
    absolute_tolerance: float,
    direction_angle_tolerance_deg: float,
) -> None:
    try:
        expected = load_strict_json_bytes(
            expected_payload,
            source=f"expected {label}",
        )
    except ValueError as error:
        comparison.add_mismatch(str(error))
        return
    try:
        actual = load_strict_json_bytes(
            actual_payload,
            source=f"actual {label}",
        )
    except ValueError as error:
        comparison.add_mismatch(str(error))
        return
    if label == "success_gates.json":
        _validate_success_gates_document(
            expected,
            label="expected success_gates.json",
            comparison=comparison,
        )
        _validate_success_gates_document(
            actual,
            label="actual success_gates.json",
            comparison=comparison,
        )
    _compare_json_value(
        expected,
        actual,
        path=label,
        comparison=comparison,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        direction_angle_tolerance_deg=direction_angle_tolerance_deg,
    )


def _compare_artifact(
    expected: ArtifactRecord,
    actual: ArtifactRecord,
    *,
    comparison: Comparison,
    relative_tolerance: float,
    absolute_tolerance: float,
    direction_angle_tolerance_deg: float,
) -> dict[str, Any]:
    expected_payload = expected.path.read_bytes()
    actual_payload = actual.path.read_bytes()
    record = {
        "present": True,
        "byte_identical": expected_payload == actual_payload,
        "expected_bytes": expected.byte_count,
        "actual_bytes": actual.byte_count,
        "expected_sha256": expected.sha256,
        "actual_sha256": actual.sha256,
    }
    suffix = expected.path.suffix.lower()
    if suffix == ".json":
        _compare_json_bytes(
            expected_payload,
            actual_payload,
            label=expected.name,
            comparison=comparison,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            direction_angle_tolerance_deg=direction_angle_tolerance_deg,
        )
    elif suffix == ".csv":
        _compare_csv_bytes(
            expected_payload,
            actual_payload,
            label=expected.name,
            comparison=comparison,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            direction_angle_tolerance_deg=direction_angle_tolerance_deg,
        )
    elif expected_payload != actual_payload:
        comparison.add_mismatch(
            f"{expected.name}: unsupported payload type changed bytewise"
        )
    return record
