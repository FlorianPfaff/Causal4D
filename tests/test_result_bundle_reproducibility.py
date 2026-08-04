from __future__ import annotations

import csv
from pathlib import Path

from result_bundle_test_support import (
    _copy_bundle,
    _read_json,
    _refresh_result_manifest,
    _run_comparison,
    _write_bundle,
    _write_json,
)


def test_identical_bundles_are_byte_and_semantic_matches(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _write_bundle(expected)
    _copy_bundle(expected, actual)

    process, report = _run_comparison(expected, actual, tmp_path)

    assert process.returncode == 0
    assert report["semantic_match"] is True
    assert report["all_payload_bytes_match"] is True
    assert report["result_manifests_byte_identical"] is True
    assert report["comparison_contract"]["version"] == 2


def test_tiny_float_drift_is_semantic_but_not_byte_identical(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _write_bundle(expected)
    _copy_bundle(expected, actual)
    summary = _read_json(actual / "summary.json")
    summary["metric"] = 1.0 + 1e-13
    _write_json(actual / "summary.json", summary)
    _refresh_result_manifest(actual)

    process, report = _run_comparison(expected, actual, tmp_path)

    assert process.returncode == 0
    assert report["semantic_match"] is True
    assert report["all_payload_bytes_match"] is False
    assert report["maximum_absolute_difference"] > 0.0


def test_json_integer_type_is_exact_even_with_large_tolerance(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _write_bundle(expected)
    _copy_bundle(expected, actual)
    protocol = _read_json(actual / "protocol.json")
    protocol["seed"] = 1.0
    _write_json(actual / "protocol.json", protocol)
    _refresh_result_manifest(actual)

    process, report = _run_comparison(
        expected,
        actual,
        tmp_path,
        "--relative-tolerance",
        "10",
        "--absolute-tolerance",
        "10",
    )

    assert process.returncode == 2
    assert report["semantic_match"] is False
    assert any(
        "protocol.json.seed: type differs" in item for item in report["mismatches"]
    )


def test_csv_integer_lexemes_and_row_order_are_exact(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _write_bundle(expected)
    _copy_bundle(expected, actual)
    rows = list(csv.reader((actual / "contact_recovery.csv").open(encoding="utf-8")))
    rows[1], rows[2] = rows[2], rows[1]
    with (actual / "contact_recovery.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        csv.writer(handle).writerows(rows)
    _refresh_result_manifest(actual)

    process, report = _run_comparison(expected, actual, tmp_path)

    assert process.returncode == 2
    assert report["semantic_match"] is False
    assert any("field=seed" in item for item in report["mismatches"])


def test_gate_threshold_is_exact_even_inside_float_tolerance(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _write_bundle(expected)
    _copy_bundle(expected, actual)
    gates = _read_json(actual / "success_gates.json")
    gates["gates"][0]["threshold"] = 0.5 + 1e-13
    _write_json(actual / "success_gates.json", gates)
    _refresh_result_manifest(actual)

    process, report = _run_comparison(expected, actual, tmp_path)

    assert process.returncode == 2
    assert report["semantic_match"] is False
    assert any("gates[0].threshold" in item for item in report["mismatches"])


def test_float_tolerance_cannot_rescue_inconsistent_gate(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _write_bundle(expected)
    _copy_bundle(expected, actual)
    gates = _read_json(actual / "success_gates.json")
    gates["gates"][0]["value"] = 0.4999999999999
    gates["derived"]["metric"] = 0.4999999999999
    _write_json(actual / "success_gates.json", gates)
    _refresh_result_manifest(actual)

    process, report = _run_comparison(expected, actual, tmp_path)

    assert process.returncode == 2
    assert report["semantic_match"] is False
    assert any("is inconsistent with" in item for item in report["mismatches"])


def test_near_zero_angle_has_only_the_declared_absolute_tolerance(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected"
    within = tmp_path / "within"
    beyond = tmp_path / "beyond"
    _write_bundle(expected)
    _copy_bundle(expected, within)
    _copy_bundle(expected, beyond)

    within_rows = list(
        csv.reader((within / "interventions.csv").open(encoding="utf-8"))
    )
    within_rows[1][2] = "1.2e-6"
    with (within / "interventions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        csv.writer(handle).writerows(within_rows)
    _refresh_result_manifest(within)

    beyond_rows = list(
        csv.reader((beyond / "interventions.csv").open(encoding="utf-8"))
    )
    beyond_rows[1][2] = "2.1e-6"
    with (beyond / "interventions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        csv.writer(handle).writerows(beyond_rows)
    _refresh_result_manifest(beyond)

    within_process, within_report = _run_comparison(expected, within, tmp_path)
    beyond_process, beyond_report = _run_comparison(expected, beyond, tmp_path)

    assert within_process.returncode == 0
    assert within_report["semantic_match"] is True
    assert within_report["maximum_direction_angle_difference_deg"] == 1.2e-6
    assert beyond_process.returncode == 2
    assert beyond_report["semantic_match"] is False


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _write_bundle(expected)
    _copy_bundle(expected, actual)
    (actual / "summary.json").write_text(
        '{"schema_version": 1, "metric": 1.0, "metric": 1.0}\n',
        encoding="utf-8",
    )
    _refresh_result_manifest(actual)

    process, report = _run_comparison(expected, actual, tmp_path)

    assert process.returncode == 2
    assert report["semantic_match"] is False
    assert any("duplicate JSON object key" in item for item in report["mismatches"])


def test_overflowing_json_number_is_rejected(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _write_bundle(expected)
    _copy_bundle(expected, actual)
    (actual / "summary.json").write_text(
        '{"schema_version": 1, "metric": 1e999, "category": "controlled"}\n',
        encoding="utf-8",
    )
    _refresh_result_manifest(actual)

    process, report = _run_comparison(expected, actual, tmp_path)

    assert process.returncode == 2
    assert report["semantic_match"] is False
    assert any("non-finite JSON number" in item for item in report["mismatches"])


def test_overflowing_csv_number_is_rejected(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _write_bundle(expected)
    _copy_bundle(expected, actual)
    rows = list(csv.reader((actual / "interventions.csv").open(encoding="utf-8")))
    rows[1][3] = "1e999"
    with (actual / "interventions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        csv.writer(handle).writerows(rows)
    _refresh_result_manifest(actual)

    process, report = _run_comparison(expected, actual, tmp_path)

    assert process.returncode == 2
    assert report["semantic_match"] is False
    assert any(
        "non-finite floating-point values are forbidden" in item
        for item in report["mismatches"]
    )


def test_out_of_range_gate_number_is_rejected_without_crashing(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _write_bundle(expected)
    _copy_bundle(expected, actual)
    gates = _read_json(actual / "success_gates.json")
    gates["gates"][0]["value"] = 10**400
    gates["derived"]["metric"] = 10**400
    _write_json(actual / "success_gates.json", gates)
    _refresh_result_manifest(actual)

    process, report = _run_comparison(expected, actual, tmp_path)

    assert process.returncode == 2
    assert report["semantic_match"] is False
    assert any("gate numbers are out of range" in item for item in report["mismatches"])


def test_whitelisted_contact_diagnostics_are_additive_only(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected-additive"
    actual = tmp_path / "actual-additive"
    _write_bundle(expected)
    _copy_bundle(expected, actual)

    expected_summary = _read_json(expected / "summary.json")
    expected_summary["aggregate"] = {
        "contact_recovery": [
            {
                "case_count": 2,
                "setting": "online_adaptation",
                "world_condition": "shifted_contact",
            }
        ]
    }
    _write_json(expected / "summary.json", expected_summary)
    _refresh_result_manifest(expected)

    actual_summary = _read_json(actual / "summary.json")
    actual_summary["aggregate"] = {
        "contact_recovery": [
            {
                "case_count": 2,
                "setting": "online_adaptation",
                "world_condition": "shifted_contact",
                "mean_node_map_set_size": 1.25,
                "node_map_set_coverage": 0.95,
            }
        ]
    }
    _write_json(actual / "summary.json", actual_summary)

    rows = list(csv.reader((actual / "contact_recovery.csv").open(encoding="utf-8")))
    rows[0].extend(["node_map_set", "node_map_set_size"])
    rows[1].extend(["2", "1"])
    rows[2].extend(["3;4", "2"])
    with (actual / "contact_recovery.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        csv.writer(handle).writerows(rows)
    _refresh_result_manifest(actual)

    process, report = _run_comparison(expected, actual, tmp_path)

    assert process.returncode == 0
    assert report["semantic_match"] is True
    assert report["all_payload_bytes_match"] is False
    assert report["additive_diagnostic_fields"]["contact_recovery.csv"] == [
        "node_map_set",
        "node_map_set_size",
    ]
    assert any(
        path.startswith("summary.json.aggregate.contact_recovery[")
        for path in report["additive_diagnostic_fields"]
    )


def test_unregistered_additive_field_remains_a_semantic_mismatch(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected-extra"
    actual = tmp_path / "actual-extra"
    _write_bundle(expected)
    _copy_bundle(expected, actual)

    rows = list(csv.reader((actual / "contact_recovery.csv").open(encoding="utf-8")))
    rows[0].append("invented_diagnostic")
    rows[1].append("1")
    rows[2].append("1")
    with (actual / "contact_recovery.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        csv.writer(handle).writerows(rows)
    _refresh_result_manifest(actual)

    process, report = _run_comparison(expected, actual, tmp_path)

    assert process.returncode == 2
    assert report["semantic_match"] is False
    assert any("invented_diagnostic" in mismatch for mismatch in report["mismatches"])
