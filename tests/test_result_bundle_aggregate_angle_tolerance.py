from __future__ import annotations

import math
from pathlib import Path

from result_bundle_test_support import (
    _copy_bundle,
    _read_json,
    _refresh_result_manifest,
    _run_comparison,
    _write_bundle,
    _write_json,
)


def _set_mean_direction_error(bundle: Path, value: float) -> None:
    summary = _read_json(bundle / "summary.json")
    summary["aggregate"] = {
        "interventions": [
            {
                "method": "latent_contact",
                "mean_direction_error_deg": value,
            }
        ]
    }
    _write_json(bundle / "summary.json", summary)
    _refresh_result_manifest(bundle)


def test_aggregate_direction_angle_uses_declared_tolerance(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    within = tmp_path / "within"
    beyond = tmp_path / "beyond"
    expected_value = 0.15694540479337696
    observed_workstation2_delta = 1.2074182697257333e-6
    _write_bundle(expected)
    _set_mean_direction_error(expected, expected_value)
    _copy_bundle(expected, within)
    _copy_bundle(expected, beyond)

    _set_mean_direction_error(within, expected_value + observed_workstation2_delta)
    _set_mean_direction_error(beyond, expected_value + 2.1e-6)

    within_process, within_report = _run_comparison(expected, within, tmp_path)
    beyond_process, beyond_report = _run_comparison(expected, beyond, tmp_path)

    assert within_process.returncode == 0
    assert within_report["semantic_match"] is True
    assert math.isclose(
        within_report["maximum_direction_angle_difference_deg"],
        observed_workstation2_delta,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert beyond_process.returncode == 2
    assert beyond_report["semantic_match"] is False
    assert any(
        "mean_direction_error_deg" in mismatch
        for mismatch in beyond_report["mismatches"]
    )
