"""Strict scalar, JSON, and success-gate comparison semantics."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any


_SUCCESS_GATE_THRESHOLD_PATH = re.compile(
    r"^success_gates\.json\.gates\[\d+\]\.threshold$"
)
_DIRECTION_ANGLE_FIELD = "direction_error_deg"


@dataclass
class Comparison:
    """Accumulate exact and tolerance-governed semantic differences."""

    mismatches: list[str] = field(default_factory=list)
    numeric_comparisons: int = 0
    exact_numeric_comparisons: int = 0
    maximum_absolute_difference: float = 0.0
    maximum_relative_difference: float = 0.0
    maximum_direction_angle_difference_deg: float = 0.0

    def add_mismatch(self, message: str) -> None:
        self.mismatches.append(message)

    def compare_exact_number(
        self,
        expected: int | float,
        actual: int | float,
        *,
        path: str,
    ) -> None:
        self.exact_numeric_comparisons += 1
        if expected != actual:
            self.add_mismatch(
                f"{path}: exact numeric value differs ({expected!r} versus {actual!r})"
            )

    def compare_float(
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
        if not math.isfinite(expected) or not math.isfinite(actual):
            self.add_mismatch(f"{path}: non-finite floating-point values are forbidden")
            return
        absolute = abs(actual - expected)
        denominator = max(abs(expected), abs(actual), absolute_tolerance)
        relative = absolute / denominator if denominator else 0.0
        self.maximum_absolute_difference = max(
            self.maximum_absolute_difference,
            absolute,
        )
        self.maximum_relative_difference = max(
            self.maximum_relative_difference,
            relative,
        )
        effective_absolute_tolerance = absolute_tolerance
        if _is_direction_angle_path(path):
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
            self.add_mismatch(
                f"{path}: expected {expected!r}, received {actual!r}; "
                f"absolute difference {absolute:.17g}"
            )


def _is_direction_angle_path(path: str) -> bool:
    return path.endswith(f".{_DIRECTION_ANGLE_FIELD}") or path.endswith(
        f"field={_DIRECTION_ANGLE_FIELD}"
    )


def _is_number(value: Any) -> bool:
    return type(value) in {int, float}


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
    if type(expected) is not type(actual):
        comparison.add_mismatch(
            f"{path}: type differs "
            f"({type(expected).__name__} versus {type(actual).__name__})"
        )
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if expected != actual:
            comparison.add_mismatch(
                f"{path}: expected {expected!r}, received {actual!r}"
            )
        return
    if type(expected) is int:
        comparison.compare_exact_number(expected, actual, path=path)
        return
    if type(expected) is float:
        if _SUCCESS_GATE_THRESHOLD_PATH.fullmatch(path):
            comparison.compare_exact_number(expected, actual, path=path)
        else:
            comparison.compare_float(
                expected,
                actual,
                path=path,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                direction_angle_tolerance_deg=direction_angle_tolerance_deg,
            )
        return
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            comparison.add_mismatch(
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
            comparison.add_mismatch(
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
                direction_angle_tolerance_deg=direction_angle_tolerance_deg,
            )
        return
    comparison.add_mismatch(
        f"{path}: unsupported JSON value type {type(expected).__name__}"
    )


def _gate_truth(comparison_operator: str, value: float, threshold: float) -> bool:
    if comparison_operator == ">=":
        return value >= threshold
    if comparison_operator == "<=":
        return value <= threshold
    if comparison_operator == ">":
        return value > threshold
    if comparison_operator == "<":
        return value < threshold
    if comparison_operator == "==":
        return value == threshold
    raise ValueError(f"unsupported gate comparison: {comparison_operator!r}")


def _validate_success_gates_document(
    document: Any,
    *,
    label: str,
    comparison: Comparison,
) -> None:
    if not isinstance(document, dict):
        comparison.add_mismatch(f"{label}: success-gate document must be an object")
        return
    gates = document.get("gates")
    overall_passed = document.get("overall_passed")
    if not isinstance(gates, list) or not gates:
        comparison.add_mismatch(f"{label}.gates: must be a nonempty list")
        return
    if type(overall_passed) is not bool:
        comparison.add_mismatch(f"{label}.overall_passed: must be a boolean")

    names: set[str] = set()
    gate_passes: list[bool] = []
    for index, gate in enumerate(gates):
        path = f"{label}.gates[{index}]"
        if not isinstance(gate, dict):
            comparison.add_mismatch(f"{path}: gate must be an object")
            continue
        name = gate.get("name")
        operator = gate.get("comparison")
        passed = gate.get("passed")
        threshold = gate.get("threshold")
        value = gate.get("value")
        if not isinstance(name, str) or not name:
            comparison.add_mismatch(f"{path}.name: must be a nonempty string")
        elif name in names:
            comparison.add_mismatch(f"{path}.name: duplicate gate name {name!r}")
        else:
            names.add(name)
        if not isinstance(operator, str):
            comparison.add_mismatch(f"{path}.comparison: must be a string")
            continue
        if type(passed) is not bool:
            comparison.add_mismatch(f"{path}.passed: must be a boolean")
            continue
        gate_passes.append(passed)
        if not _is_number(threshold) or not _is_number(value):
            comparison.add_mismatch(
                f"{path}: threshold and value must be integer or floating numbers"
            )
            continue
        try:
            threshold_float = float(threshold)
            value_float = float(value)
        except OverflowError:
            comparison.add_mismatch(f"{path}: gate numbers are out of range")
            continue
        if not math.isfinite(threshold_float) or not math.isfinite(value_float):
            comparison.add_mismatch(f"{path}: gate numbers must be finite")
            continue
        try:
            computed = _gate_truth(operator, value_float, threshold_float)
        except ValueError as error:
            comparison.add_mismatch(f"{path}.comparison: {error}")
            continue
        if computed is not passed:
            comparison.add_mismatch(
                f"{path}.passed: {passed!r} is inconsistent with "
                f"{value!r} {operator} {threshold!r}"
            )
    if type(overall_passed) is bool and overall_passed is not all(gate_passes):
        comparison.add_mismatch(
            f"{label}.overall_passed: {overall_passed!r} is inconsistent with "
            "the individual gate decisions"
        )
