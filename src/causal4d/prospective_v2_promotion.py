"""Two-phase promotion experiment for prospective Causal4D V2 candidates.

The freeze artifact locks the candidate ladder, independent evaluation units,
stack identity, source evidence, and all decision thresholds before target
outcomes are opened. The result evaluates the complete Cartesian product once,
at the registered independent-unit level, and retains the exact baseline when
no predeclared candidate passes every endpoint gate.

This module is prospective infrastructure. It does not modify the frozen
physical-acquisition estimator or reinterpret an existing target result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d._prospective_v2_promotion_candidate_result import (
    ProspectiveV2CandidateResultV1,
)
from causal4d._prospective_v2_promotion_common import (
    PROSPECTIVE_V2_CANDIDATE_KINDS,
    PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
)
from causal4d._prospective_v2_promotion_freeze import (
    ProspectiveV2PromotionFreezeV1,
)
from causal4d._prospective_v2_promotion_inputs import (
    ProspectiveV2CandidateV1,
    ProspectiveV2EvaluationUnitV1,
    ProspectiveV2PromotionPolicyV1,
)
from causal4d._prospective_v2_promotion_metrics import (
    ProspectiveV2EndpointMetricsV1,
    ProspectiveV2UnitMetricsV1,
)
from causal4d._prospective_v2_promotion_result import (
    ProspectiveV2PromotionResultV1,
)
from causal4d.atomic_io import atomic_write_json
from causal4d.decision_trace import DECISION_TRACE_ENDPOINTS


def _mean(values: Sequence[float]) -> float:
    result = float(np.mean(np.asarray(values, dtype=float)))
    if not np.isfinite(result):
        raise ValueError("aggregate metric mean must be finite")
    return result


def _evaluate_endpoint(
    *,
    candidate: ProspectiveV2CandidateV1,
    endpoint: str,
    units: Sequence[ProspectiveV2EvaluationUnitV1],
    baseline_metrics: Sequence[ProspectiveV2UnitMetricsV1],
    candidate_metrics: Sequence[ProspectiveV2UnitMetricsV1],
    policy: ProspectiveV2PromotionPolicyV1,
) -> ProspectiveV2EndpointMetricsV1:
    if len(units) != len(baseline_metrics) or len(units) != len(candidate_metrics):
        raise ValueError("endpoint metric alignment is incomplete")
    log_score_gain = _mean(
        tuple(
            candidate_metric.log_score - baseline_metric.log_score
            for baseline_metric, candidate_metric in zip(
                baseline_metrics,
                candidate_metrics,
                strict=True,
            )
        )
    )
    brier_change = _mean(
        tuple(
            candidate_metric.brier_score - baseline_metric.brier_score
            for baseline_metric, candidate_metric in zip(
                baseline_metrics,
                candidate_metrics,
                strict=True,
            )
        )
    )
    trajectory_regret = _mean(
        tuple(
            candidate_metric.trajectory_error_m - baseline_metric.trajectory_error_m
            for baseline_metric, candidate_metric in zip(
                baseline_metrics,
                candidate_metrics,
                strict=True,
            )
        )
    )
    coverage_error = _mean(
        tuple(metric.coverage_error for metric in candidate_metrics)
    )
    interval_width = _mean(
        tuple(metric.interval_width_m for metric in candidate_metrics)
    )
    baseline_interval_width = _mean(
        tuple(metric.interval_width_m for metric in baseline_metrics)
    )
    interval_width_ratio = interval_width / max(
        baseline_interval_width,
        policy.interval_width_floor_m,
    )
    accepted_count = sum(metric.candidate_accepted for metric in candidate_metrics)
    harmful_count = sum(metric.harmful_update for metric in candidate_metrics)
    unit_count = len(candidate_metrics)
    accepted_rate = accepted_count / unit_count
    harmful_accepted_rate = harmful_count / accepted_count if accepted_count else 0.0
    fallback_rate = sum(metric.fallback_used for metric in candidate_metrics) / (
        unit_count
    )

    reasons: list[str] = []
    if log_score_gain < policy.minimum_mean_log_score_gain:
        reasons.append("mean_log_score_gain_below_limit")
    if brier_change > policy.maximum_mean_brier_change:
        reasons.append("mean_brier_change_exceeds_limit")
    if trajectory_regret > policy.maximum_mean_trajectory_regret_m:
        reasons.append("mean_trajectory_regret_exceeds_limit")
    if coverage_error > policy.maximum_mean_coverage_error:
        reasons.append("mean_coverage_error_exceeds_limit")
    if interval_width_ratio > policy.maximum_mean_interval_width_ratio:
        reasons.append("mean_interval_width_ratio_exceeds_limit")
    if accepted_rate < policy.minimum_accepted_update_rate:
        reasons.append("accepted_update_rate_below_limit")
    if harmful_accepted_rate > policy.maximum_harmful_accepted_update_rate:
        reasons.append("harmful_accepted_update_rate_exceeds_limit")
    if fallback_rate > policy.maximum_fallback_rate:
        reasons.append("fallback_rate_exceeds_limit")
    return ProspectiveV2EndpointMetricsV1(
        candidate_id=candidate.candidate_id,
        endpoint=endpoint,
        unit_count=unit_count,
        mean_log_score_gain=log_score_gain,
        mean_brier_change=brier_change,
        mean_trajectory_regret_m=trajectory_regret,
        mean_coverage_error=coverage_error,
        mean_interval_width_m=interval_width,
        mean_interval_width_ratio=interval_width_ratio,
        accepted_update_rate=accepted_rate,
        harmful_accepted_update_rate=harmful_accepted_rate,
        fallback_rate=fallback_rate,
        accepted=not reasons,
        reasons=tuple(reasons),
    )


def evaluate_prospective_v2_promotion_v1(
    freeze: ProspectiveV2PromotionFreezeV1,
    metrics: Sequence[ProspectiveV2UnitMetricsV1],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ProspectiveV2PromotionResultV1:
    """Evaluate the frozen ladder once and retain exact baseline fallback."""

    if type(freeze) is not ProspectiveV2PromotionFreezeV1:
        raise ValueError("freeze must be a ProspectiveV2PromotionFreezeV1")
    metric_tuple = tuple(metrics)
    if not metric_tuple or any(
        type(value) is not ProspectiveV2UnitMetricsV1 for value in metric_tuple
    ):
        raise ValueError("metrics must contain ProspectiveV2UnitMetricsV1 values")
    keys = tuple((metric.unit_id, metric.candidate_id) for metric in metric_tuple)
    if len(set(keys)) != len(keys):
        raise ValueError("unit/candidate metric pairs must be unique")
    expected_keys = {
        (unit.unit_id, candidate.candidate_id)
        for unit in freeze.evaluation_units
        for candidate in freeze.candidates
    }
    actual_keys = set(keys)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "evaluation metrics must cover the frozen unit/candidate product; "
            f"missing={missing}, unexpected={unexpected}"
        )
    unit_index = {unit.unit_id: unit for unit in freeze.evaluation_units}
    opening_ids = {metric.evaluation_opening_id for metric in metric_tuple}
    if len(opening_ids) != 1:
        raise ValueError("all evaluation metrics must come from one target opening")
    for metric in metric_tuple:
        unit = unit_index[metric.unit_id]
        if metric.endpoint != unit.endpoint:
            raise ValueError("unit metric endpoint disagrees with the frozen unit")
        if not metric.target_outcomes_used:
            raise ValueError("evaluation result metrics must use target outcomes")

    metric_index = {
        (metric.unit_id, metric.candidate_id): metric for metric in metric_tuple
    }
    baseline = freeze.candidates[0]
    for unit in freeze.evaluation_units:
        baseline_metric = metric_index[(unit.unit_id, baseline.candidate_id)]
        if (
            not baseline_metric.candidate_accepted
            or baseline_metric.fallback_used
            or baseline_metric.harmful_update
        ):
            raise ValueError(
                "registered baseline metrics must be accepted, nonfallback, "
                "and nonharmful"
            )

    candidate_results: list[ProspectiveV2CandidateResultV1] = []
    for candidate in freeze.candidates[1:]:
        endpoint_metrics: list[ProspectiveV2EndpointMetricsV1] = []
        for endpoint in DECISION_TRACE_ENDPOINTS:
            units = tuple(
                unit for unit in freeze.evaluation_units if unit.endpoint == endpoint
            )
            baseline_metrics = tuple(
                metric_index[(unit.unit_id, baseline.candidate_id)] for unit in units
            )
            candidate_metrics = tuple(
                metric_index[(unit.unit_id, candidate.candidate_id)] for unit in units
            )
            endpoint_metrics.append(
                _evaluate_endpoint(
                    candidate=candidate,
                    endpoint=endpoint,
                    units=units,
                    baseline_metrics=baseline_metrics,
                    candidate_metrics=candidate_metrics,
                    policy=freeze.policy,
                )
            )
        reasons = tuple(
            f"{metric.endpoint}:{reason}"
            for metric in endpoint_metrics
            for reason in metric.reasons
        )
        candidate_results.append(
            ProspectiveV2CandidateResultV1(
                candidate_id=candidate.candidate_id,
                candidate_kind=candidate.candidate_kind,
                candidate_artifact_id=candidate.artifact_id,
                endpoint_metrics=tuple(endpoint_metrics),
                accepted=not reasons,
                reasons=reasons,
            )
        )

    accepted_ids = {
        result.candidate_id for result in candidate_results if result.accepted
    }
    selected = baseline
    for candidate in freeze.candidates[1:]:
        if candidate.candidate_id in accepted_ids:
            selected = candidate
    ordered_metric_ids = tuple(
        metric_index[(unit.unit_id, candidate.candidate_id)].metric_id
        for unit in freeze.evaluation_units
        for candidate in freeze.candidates
    )
    return ProspectiveV2PromotionResultV1(
        freeze_id=freeze.freeze_id,
        evaluation_opening_id=next(iter(opening_ids)),
        baseline_candidate_id=baseline.candidate_id,
        baseline_artifact_id=baseline.artifact_id,
        selected_candidate_id=selected.candidate_id,
        selected_candidate_kind=selected.candidate_kind,
        selected_artifact_id=selected.artifact_id,
        candidate_results=tuple(candidate_results),
        evaluation_metric_ids=ordered_metric_ids,
        exact_artifact_fallback_verified=True,
        one_target_opening_verified=True,
        target_outcomes_used=True,
        metadata={} if metadata is None else metadata,
    )


def write_prospective_v2_promotion_freeze(
    path: str | Path,
    freeze: ProspectiveV2PromotionFreezeV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish a validated target-free promotion freeze."""

    if type(freeze) is not ProspectiveV2PromotionFreezeV1:
        raise ValueError("freeze must be a ProspectiveV2PromotionFreezeV1")
    atomic_write_json(path, freeze.as_dict(), overwrite=overwrite)


def write_prospective_v2_promotion_result(
    path: str | Path,
    result: ProspectiveV2PromotionResultV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish a validated one-opening promotion result."""

    if type(result) is not ProspectiveV2PromotionResultV1:
        raise ValueError("result must be a ProspectiveV2PromotionResultV1")
    atomic_write_json(path, result.as_dict(), overwrite=overwrite)


__all__ = [
    "PROSPECTIVE_V2_CANDIDATE_KINDS",
    "PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION",
    "ProspectiveV2CandidateResultV1",
    "ProspectiveV2CandidateV1",
    "ProspectiveV2EndpointMetricsV1",
    "ProspectiveV2EvaluationUnitV1",
    "ProspectiveV2PromotionFreezeV1",
    "ProspectiveV2PromotionPolicyV1",
    "ProspectiveV2PromotionResultV1",
    "ProspectiveV2UnitMetricsV1",
    "evaluate_prospective_v2_promotion_v1",
    "write_prospective_v2_promotion_freeze",
    "write_prospective_v2_promotion_result",
]
