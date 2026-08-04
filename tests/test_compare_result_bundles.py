from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "compare_result_bundles.py"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_bundle(
    root: Path,
    *,
    angle_deg: float = 0.0,
    trajectory_rmse_m: float = 0.010,
    gate_value: float = 0.51,
    gate_passed: bool = True,
    intervention_order: tuple[int, ...] = (1, 2),
    schema_version: int = 1,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    gate = {
        "name": "quality",
        "value": gate_value,
        "comparison": ">=",
        "threshold": 0.5,
        "passed": gate_passed,
    }
    success_gates = {
        "overall_passed": gate_passed,
        "gates": [gate],
        "derived": {"score": gate_value},
    }
    _write_json(
        root / "protocol.json",
        {"schema_version": schema_version, "name": "fixture"},
    )
    _write_json(root / "success_gates.json", success_gates)
    _write_json(
        root / "summary.json",
        {
            "schema_version": schema_version,
            "seeds": [1, 2],
            "aggregate": {"trajectory_rmse_m": trajectory_rmse_m},
            "success_gates": success_gates,
        },
    )
    _write_csv(
        root / "contact_recovery.csv",
        ["seed", "node_correct", "node_confidence"],
        [{"seed": 1, "node_correct": True, "node_confidence": 0.9}],
    )
    _write_csv(
        root / "fold_calibration.csv",
        ["seed", "likelihood_scale_m"],
        [{"seed": 1, "likelihood_scale_m": 0.0025}],
    )
    _write_csv(
        root / "interventions.csv",
        ["seed", "method", "trajectory_rmse_m", "direction_error_deg"],
        [
            {
                "seed": seed,
                "method": "latent_contact",
                "trajectory_rmse_m": trajectory_rmse_m + seed * 0.001,
                "direction_error_deg": angle_deg,
            }
            for seed in intervention_order
        ],
    )


def _compare(
    tmp_path: Path,
    expected: Path,
    actual: Path,
    *extra_arguments: str,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    output = tmp_path / "comparison.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(expected),
            str(actual),
            "--output",
            str(output),
            *extra_arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed, json.loads(output.read_text(encoding="utf-8"))


def test_near_zero_angle_drift_is_semantic_but_not_byte_identical(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _write_bundle(expected)
    _write_bundle(actual, angle_deg=1e-6)

    completed, report = _compare(tmp_path, expected, actual)

    assert completed.returncode == 0
    assert report["semantic_match"] is True
    assert report["all_payload_bytes_match"] is False
    assert report["maximum_direction_angle_difference_deg"] == 1e-6
    assert report["tolerance_policy_id"] == (
        "causal4d-field-aware-cross-platform-v1"
    )
    assert report["comparison_environment"]["python_version"]


def test_substantive_numeric_change_and_row_reordering_are_rejected(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected"
    changed = tmp_path / "changed"
    reordered = tmp_path / "reordered"
    _write_bundle(expected)
    _write_bundle(changed, trajectory_rmse_m=0.1)
    _write_bundle(reordered, intervention_order=(2, 1))

    changed_process, changed_report = _compare(
        tmp_path,
        expected,
        changed,
    )
    reordered_process, reordered_report = _compare(
        tmp_path,
        expected,
        reordered,
    )

    assert changed_process.returncode == 2
    assert changed_report["semantic_match"] is False
    assert any(
        "trajectory_rmse_m" in mismatch
        for mismatch in changed_report["mismatches"]
    )
    assert reordered_process.returncode == 2
    assert reordered_report["semantic_match"] is False
    assert any(
        "field=seed" in mismatch for mismatch in reordered_report["mismatches"]
    )


def test_gate_crossing_cannot_be_rescued_by_loose_numeric_tolerance(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _write_bundle(expected, gate_value=0.51, gate_passed=True)
    _write_bundle(actual, gate_value=0.49, gate_passed=False)

    completed, report = _compare(
        tmp_path,
        expected,
        actual,
        "--relative-tolerance",
        "1",
        "--absolute-tolerance",
        "1",
    )

    assert completed.returncode == 2
    assert report["semantic_match"] is False
    assert report["gate_checks"] == 2
    assert any(
        "numeric tolerance cannot revise or rescue a gate" in mismatch
        for mismatch in report["mismatches"]
    )


def test_internally_inconsistent_gate_is_rejected(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _write_bundle(expected, gate_value=0.51, gate_passed=True)
    _write_bundle(actual, gate_value=0.49, gate_passed=True)

    completed, report = _compare(
        tmp_path,
        expected,
        actual,
        "--relative-tolerance",
        "1",
        "--absolute-tolerance",
        "1",
    )

    assert completed.returncode == 2
    assert any(
        "actual gate is internally inconsistent" in mismatch
        for mismatch in report["mismatches"]
    )


def test_schema_and_registered_thresholds_are_exact(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected"
    schema_changed = tmp_path / "schema-changed"
    threshold_changed = tmp_path / "threshold-changed"
    _write_bundle(expected)
    _write_bundle(schema_changed, schema_version=2)
    _write_bundle(threshold_changed)

    success_path = threshold_changed / "success_gates.json"
    success = json.loads(success_path.read_text(encoding="utf-8"))
    success["gates"][0]["threshold"] = 0.49
    _write_json(success_path, success)

    schema_process, schema_report = _compare(
        tmp_path,
        expected,
        schema_changed,
        "--relative-tolerance",
        "1",
        "--absolute-tolerance",
        "1",
    )
    threshold_process, threshold_report = _compare(
        tmp_path,
        expected,
        threshold_changed,
        "--relative-tolerance",
        "1",
        "--absolute-tolerance",
        "1",
    )

    assert schema_process.returncode == 2
    assert any(
        "exact numeric field changed" in mismatch
        for mismatch in schema_report["mismatches"]
    )
    assert threshold_process.returncode == 2
    assert any(
        "registered gate threshold changed" in mismatch
        for mismatch in threshold_report["mismatches"]
    )
