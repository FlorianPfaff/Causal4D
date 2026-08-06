"""Preacquisition interval diagnostics for session-clustered real effects.

The registered percentile interval remains unchanged.  This module publishes
source-verified Student-t and bootstrap-t sensitivity intervals that are
explicitly ineligible to alter the primary decision unless a separate,
content-addressed preacquisition amendment promotes one of them.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t_distribution

from causal4d.atomic_io import atomic_write_json
from causal4d.real_analysis_reporting import (
    BOOTSTRAP_CONFIDENCE_LEVEL,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    build_real_analysis_effect_report,
    load_real_analysis_effect_table,
)


REAL_ANALYSIS_INTERVAL_DIAGNOSTICS_SCHEMA_VERSION = 1
OPERATING_CHARACTERISTIC_TARGET_SHA = (
    "fa6a64b2442474321e453e9e8fdccd591e0a282d"
)
BOOTSTRAP_OPERATING_CHARACTERISTIC_RUN_ID = 31091137654
BOOTSTRAP_OPERATING_CHARACTERISTIC_AUDIT_ID = (
    "7dbea2a9b99cbc98acd03fa28af9583f0e95d4d0772e58853af4f05d0584267a"
)
INTERVAL_COMPARISON_RUN_ID = 31091652355
INTERVAL_COMPARISON_AUDIT_ID = (
    "5a13c416d7efd522f5123f98afacaacd218838583d78256d463eeb5e1d478576"
)

FloatArray = NDArray[np.float64]


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _validated_values(values: Sequence[float]) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("session effects must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("session effects must be finite")
    return array


def _not_estimable_interval(
    *,
    method: str,
    sample_count: int,
    confidence_level: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "estimable": False,
        "method": method,
        "confidence_level": confidence_level,
        "sample_count": sample_count,
        "point_estimate": None,
        "lower": None,
        "upper": None,
        "reason": reason,
        "finite_sample_coverage_guaranteed": False,
        "may_change_primary_decision": False,
    }


def student_t_sensitivity_interval(
    values: Sequence[float],
    *,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Return a transparent small-sample mean interval under t assumptions."""

    array = _validated_values(values)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if len(array) < 2:
        return _not_estimable_interval(
            method="student_t_mean_sensitivity",
            sample_count=len(array),
            confidence_level=confidence_level,
            reason="at least two included sessions are required",
        )

    point = float(np.mean(array))
    sample_sd = float(np.std(array, ddof=1))
    standard_error = sample_sd / math.sqrt(len(array))
    tail = 0.5 * (1.0 - confidence_level)
    critical_value = float(
        student_t_distribution.ppf(1.0 - tail, df=len(array) - 1)
    )
    half_width = critical_value * standard_error
    return {
        "estimable": True,
        "method": "student_t_mean_sensitivity",
        "confidence_level": confidence_level,
        "sample_count": len(array),
        "degrees_of_freedom": len(array) - 1,
        "point_estimate": point,
        "sample_standard_deviation": sample_sd,
        "standard_error": standard_error,
        "critical_value": critical_value,
        "lower": point - half_width,
        "upper": point + half_width,
        "degenerate_sample": sample_sd == 0.0,
        "coverage_assumptions": [
            "independent session effects",
            "approximately normal sampling distribution of the session mean",
        ],
        "finite_sample_coverage_guaranteed": False,
        "may_change_primary_decision": False,
    }


def bootstrap_t_sensitivity_interval(
    values: Sequence[float],
    *,
    confidence_level: float = BOOTSTRAP_CONFIDENCE_LEVEL,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return a deterministic studentized bootstrap interval for the mean."""

    array = _validated_values(values)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if len(array) < 2:
        return _not_estimable_interval(
            method="bootstrap_t_mean_sensitivity",
            sample_count=len(array),
            confidence_level=confidence_level,
            reason="at least two included sessions are required",
        )

    sample_count = len(array)
    point = float(np.mean(array))
    sample_sd = float(np.std(array, ddof=1))
    sample_standard_error = sample_sd / math.sqrt(sample_count)
    if sample_standard_error == 0.0:
        return {
            "estimable": True,
            "method": "bootstrap_t_mean_sensitivity",
            "confidence_level": confidence_level,
            "sample_count": sample_count,
            "point_estimate": point,
            "lower": point,
            "upper": point,
            "replicates": replicates,
            "seed": seed,
            "finite_studentized_replicate_count": replicates,
            "finite_studentized_replicate_fraction": 1.0,
            "degenerate_sample": True,
            "finite_sample_coverage_guaranteed": False,
            "may_change_primary_decision": False,
        }

    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        sample_count,
        size=(replicates, sample_count),
    )
    resamples = array[indices]
    resample_means = np.mean(resamples, axis=1)
    resample_sd = np.std(resamples, axis=1, ddof=1)
    resample_standard_error = resample_sd / math.sqrt(sample_count)
    differences = resample_means - point
    studentized = np.empty(replicates, dtype=np.float64)
    finite = resample_standard_error > 0.0
    studentized[finite] = differences[finite] / resample_standard_error[finite]
    zero_error = ~finite
    studentized[zero_error] = np.where(
        differences[zero_error] > 0.0,
        np.inf,
        np.where(differences[zero_error] < 0.0, -np.inf, 0.0),
    )

    tail = 0.5 * (1.0 - confidence_level)
    lower_pivot, upper_pivot = np.quantile(
        studentized,
        [tail, 1.0 - tail],
    )
    lower = point - float(upper_pivot) * sample_standard_error
    upper = point - float(lower_pivot) * sample_standard_error
    if not np.isfinite(lower) or not np.isfinite(upper):
        result = _not_estimable_interval(
            method="bootstrap_t_mean_sensitivity",
            sample_count=sample_count,
            confidence_level=confidence_level,
            reason="too many degenerate bootstrap resamples for finite pivots",
        )
        result.update(
            replicates=replicates,
            seed=seed,
            finite_studentized_replicate_count=int(np.sum(finite)),
            finite_studentized_replicate_fraction=float(np.mean(finite)),
        )
        return result

    return {
        "estimable": True,
        "method": "bootstrap_t_mean_sensitivity",
        "confidence_level": confidence_level,
        "sample_count": sample_count,
        "point_estimate": point,
        "sample_standard_deviation": sample_sd,
        "standard_error": sample_standard_error,
        "lower_pivot_quantile": float(lower_pivot),
        "upper_pivot_quantile": float(upper_pivot),
        "lower": lower,
        "upper": upper,
        "replicates": replicates,
        "seed": seed,
        "finite_studentized_replicate_count": int(np.sum(finite)),
        "finite_studentized_replicate_fraction": float(np.mean(finite)),
        "degenerate_sample": False,
        "finite_sample_coverage_guaranteed": False,
        "may_change_primary_decision": False,
    }


def _included_session_effects(
    records: Sequence[Mapping[str, Any]],
    *,
    lower_is_better: bool,
) -> tuple[float, ...]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if not bool(record["included"]):
            continue
        baseline = float(record["baseline_value"])
        candidate = float(record["candidate_value"])
        improvement = (
            baseline - candidate if lower_is_better else candidate - baseline
        )
        grouped[str(record["session_id"])].append(improvement)
    return tuple(float(np.mean(values)) for values in grouped.values())


def build_real_analysis_interval_diagnostics(
    effect_table_path: str | Path,
    protocol_path: str | Path,
    *,
    method_freeze_path: str | Path,
    analysis_manifest_path: str | Path,
) -> dict[str, Any]:
    """Build source-verified companion intervals without changing the report."""

    primary_report = build_real_analysis_effect_report(
        effect_table_path,
        protocol_path,
        method_freeze_path=method_freeze_path,
        analysis_manifest_path=analysis_manifest_path,
    )
    table, _ = load_real_analysis_effect_table(effect_table_path)
    values = _included_session_effects(
        table.records,
        lower_is_better=table.lower_is_better,
    )
    primary = primary_report["primary_session_clustered_effect"]
    primary_summary = primary["equal_session_weighted_improvement"]
    expected_point = None if primary_summary is None else float(primary_summary["mean"])
    actual_point = None if not values else float(np.mean(values))
    if expected_point is None:
        if actual_point is not None:
            raise ValueError("companion interval point estimate differs from primary report")
    elif actual_point is None or not np.isclose(expected_point, actual_point):
        raise ValueError("companion interval point estimate differs from primary report")

    payload: dict[str, Any] = {
        "schema_version": REAL_ANALYSIS_INTERVAL_DIAGNOSTICS_SCHEMA_VERSION,
        "artifact_kind": "Causal4DRealAnalysisIntervalDiagnostics",
        "protocol_id": primary_report["protocol_id"],
        "protocol_design_sha256": primary_report["protocol_design_sha256"],
        "preacquisition_amendment_sha256": primary_report[
            "preacquisition_amendment_sha256"
        ],
        "method_freeze_sha256": primary_report["method_freeze_sha256"],
        "analysis_manifest_sha256": primary_report["analysis_manifest_sha256"],
        "endpoint": primary_report["endpoint"],
        "metric_id": primary_report["metric_id"],
        "metric_unit": primary_report["metric_unit"],
        "source_primary_report_id": primary_report["report_id"],
        "source_effect_table": primary_report["source_effect_table"],
        "source_protocol": primary_report["source_protocol"],
        "source_verification": primary_report["source_verification"],
        "included_session_count": len(values),
        "point_estimate": actual_point,
        "primary_percentile_interval": {
            "interval": primary["confidence_interval"],
            "frozen_primary_output_unchanged": True,
            "finite_sample_coverage_guaranteed": False,
            "coverage_evidence": {
                "workflow_run_id": BOOTSTRAP_OPERATING_CHARACTERISTIC_RUN_ID,
                "audit_id": BOOTSTRAP_OPERATING_CHARACTERISTIC_AUDIT_ID,
                "implementation_target_sha": OPERATING_CHARACTERISTIC_TARGET_SHA,
                "tested_session_counts": [12, 18],
                "uses_target_outcomes": False,
            },
        },
        "sensitivity_intervals": {
            "student_t": student_t_sensitivity_interval(values),
            "bootstrap_t": bootstrap_t_sensitivity_interval(values),
            "may_change_primary_decision": False,
            "promotion_requires_explicit_preacquisition_amendment": True,
        },
        "interval_method_comparison_evidence": {
            "workflow_run_id": INTERVAL_COMPARISON_RUN_ID,
            "audit_id": INTERVAL_COMPARISON_AUDIT_ID,
            "implementation_target_sha": OPERATING_CHARACTERISTIC_TARGET_SHA,
            "bootstrap_t_best_mean_absolute_coverage_error": 0.019,
            "student_t_best_maximum_favorable_type_i_error": 0.02666666666666667,
            "uses_target_outcomes": False,
            "automatically_selected_method": False,
        },
        "interpretation": {
            "percentile_interval_is_reproduced_not_recalibrated": True,
            "bootstrap_t_is_general_calibration_sensitivity": True,
            "student_t_is_transparent_symmetric_sensitivity": True,
            "sensitivity_intervals_cannot_rescue_failed_primary_endpoint": True,
            "target_informed_selection": False,
        },
        "claim_boundary": {
            **primary_report["claim_boundary"],
            "companion_intervals_are_non_decision_making": True,
            "physical_target_outcomes_used_to_choose_interval": False,
        },
    }
    payload["diagnostic_id"] = _canonical_sha256(payload)
    return payload


def write_real_analysis_interval_diagnostics(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish a finite companion interval artifact."""

    atomic_write_json(path, payload, overwrite=overwrite)


__all__ = [
    "BOOTSTRAP_OPERATING_CHARACTERISTIC_AUDIT_ID",
    "BOOTSTRAP_OPERATING_CHARACTERISTIC_RUN_ID",
    "INTERVAL_COMPARISON_AUDIT_ID",
    "INTERVAL_COMPARISON_RUN_ID",
    "REAL_ANALYSIS_INTERVAL_DIAGNOSTICS_SCHEMA_VERSION",
    "bootstrap_t_sensitivity_interval",
    "build_real_analysis_interval_diagnostics",
    "student_t_sensitivity_interval",
    "write_real_analysis_interval_diagnostics",
]
