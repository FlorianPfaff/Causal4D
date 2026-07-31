"""Equal-execution failure attribution for real Causal4D oracle audits."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np

from causal4d.atomic_io import atomic_write_json, atomic_write_text

AUDIT_EXPERIMENT = "real_oracle_gap_and_variance_audit"
AGGREGATE_NAME = "real_failure_attribution_v1"
METRICS = ("track_error_m", "coordinate_rmse_m")
PREDICTORS = (
    "nominal_phystwin",
    "bayesian_phystwin_mixture_nominal_z",
    "current_causal4d_posterior",
)
GAP_NAMES = ("inference_gap", "proposal_gap", "model_gap")


def _reject_nonfinite_json(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON evidence file: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON evidence must be an object: {source}")
    return payload


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _finite_number(
    value: Any,
    *,
    name: str,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("summary statistics require at least one value")
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("summary values must be finite")
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _optional_stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"defined_count": 0, "statistics": None}
    return {"defined_count": len(values), "statistics": _stats(values)}


def _validate_predictors(summary: Mapping[str, Any], *, case: str) -> None:
    predictors = _mapping(summary.get("predictors"), name=f"{case}.predictors")
    for predictor_name in PREDICTORS:
        predictor = _mapping(
            predictors.get(predictor_name),
            name=f"{case}.predictors.{predictor_name}",
        )
        if predictor.get("label_use_for_prediction") is not False:
            raise ValueError(
                f"{case}.{predictor_name} must not use holdout labels for prediction"
            )
        for metric in METRICS:
            _finite_number(
                predictor.get(metric),
                name=f"{case}.{predictor_name}.{metric}",
                nonnegative=True,
            )


def _validate_information_boundary(summary: Mapping[str, Any], *, case: str) -> None:
    boundary = _mapping(
        summary.get("information_boundary"),
        name=f"{case}.information_boundary",
    )
    required_flags = {
        "holdout_labels_used_for_prediction": False,
        "holdout_labels_used_for_evaluation": True,
        "holdout_labels_used_for_oracle_selection": True,
        "oracle_outputs_deployable": False,
    }
    for key, expected in required_flags.items():
        if boundary.get(key) is not expected:
            raise ValueError(f"{case}.information_boundary.{key} must be {expected}")
    protocol = _mapping(boundary.get("protocol"), name=f"{case}.protocol")
    if protocol.get("label_use") != "diagnostic_only":
        raise ValueError(f"{case} oracle labels must remain diagnostic-only")
    if protocol.get("selection_metric") not in METRICS:
        raise ValueError(f"{case} uses an unsupported oracle selection metric")
    start_frame = protocol.get("start_frame")
    stop_frame = protocol.get("stop_frame")
    if (
        isinstance(start_frame, bool)
        or not isinstance(start_frame, int)
        or isinstance(stop_frame, bool)
        or not isinstance(stop_frame, int)
        or start_frame < 1
        or stop_frame <= start_frame
    ):
        raise ValueError(f"{case} has an invalid holdout interval")
    prefix = _sequence(
        boundary.get("o_plus_prefix_frame_interval"),
        name=f"{case}.o_plus_prefix_frame_interval",
    )
    if (
        len(prefix) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in prefix
        )
        or prefix[0] < 0
        or prefix[1] <= prefix[0]
        or prefix[1] != start_frame
    ):
        raise ValueError(f"{case} has an invalid O+ prefix interval")


def _validate_oracles(summary: Mapping[str, Any], *, case: str) -> None:
    boundary = _mapping(
        summary["information_boundary"],
        name=f"{case}.information_boundary",
    )
    protocol = _mapping(boundary["protocol"], name=f"{case}.protocol")
    caps = []
    for name in ("current_bank_oracle", "expanded_bank_oracle"):
        oracle = _mapping(summary.get(name), name=f"{case}.{name}")
        if oracle.get("deployable") is not False:
            raise ValueError(f"{case}.{name} must be marked non-deployable")
        if oracle.get("label_use") != "diagnostic_only":
            raise ValueError(f"{case}.{name} must be diagnostic-only")
        if oracle.get("selection_metric") != protocol.get("selection_metric"):
            raise ValueError(f"{case}.{name} selection metric disagrees with protocol")
        caps.append(
            _finite_number(
                oracle.get("discrepancy_cap_m"),
                name=f"{case}.{name}.discrepancy_cap_m",
                nonnegative=True,
            )
        )
    if not math.isclose(caps[0], caps[1], rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"{case} oracle discrepancy caps disagree")
    nesting = _mapping(summary.get("bank_nesting"), name=f"{case}.bank_nesting")
    if nesting.get("verified") is not True:
        raise ValueError(f"{case} expanded rollout bank was not verified as nested")


def _validate_gap_arithmetic(summary: Mapping[str, Any], *, case: str) -> None:
    predictors = _mapping(summary["predictors"], name=f"{case}.predictors")
    current_predictor = _mapping(
        predictors["current_causal4d_posterior"],
        name=f"{case}.current_causal4d_posterior",
    )
    gaps = _mapping(summary.get("gaps"), name=f"{case}.gaps")
    for metric in METRICS:
        values = _mapping(gaps.get(metric), name=f"{case}.gaps.{metric}")
        posterior = _finite_number(
            values.get("current_causal4d_posterior"),
            name=f"{case}.gaps.{metric}.current_causal4d_posterior",
            nonnegative=True,
        )
        predictor_value = _finite_number(
            current_predictor.get(metric),
            name=f"{case}.predictors.current_causal4d_posterior.{metric}",
            nonnegative=True,
        )
        current_oracle = _finite_number(
            values.get("current_bank_oracle"),
            name=f"{case}.gaps.{metric}.current_bank_oracle",
            nonnegative=True,
        )
        expanded_oracle = _finite_number(
            values.get("expanded_bank_oracle"),
            name=f"{case}.gaps.{metric}.expanded_bank_oracle",
            nonnegative=True,
        )
        ceiling = _finite_number(
            values.get("oracle_discrepancy_ceiling"),
            name=f"{case}.gaps.{metric}.oracle_discrepancy_ceiling",
            nonnegative=True,
        )
        expected = {
            "inference_gap": posterior - current_oracle,
            "proposal_gap": current_oracle - expanded_oracle,
            "model_gap": expanded_oracle - ceiling,
        }
        if not math.isclose(posterior, predictor_value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{case} predictor and gap report disagree for {metric}")
        for gap_name, expected_value in expected.items():
            actual = _finite_number(
                values.get(gap_name),
                name=f"{case}.gaps.{metric}.{gap_name}",
            )
            if not math.isclose(actual, expected_value, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(
                    f"{case}.{metric}.{gap_name} does not close arithmetically"
                )
        total = _finite_number(
            values.get("total_diagnostic_headroom"),
            name=f"{case}.gaps.{metric}.total_diagnostic_headroom",
        )
        if not math.isclose(
            total,
            posterior - ceiling,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{case}.{metric} total headroom does not close")
        if values.get("dominant_gap") not in GAP_NAMES:
            raise ValueError(f"{case}.{metric} has an invalid dominant gap")


def _validate_variance(summary: Mapping[str, Any], *, case: str) -> None:
    variance = _mapping(
        summary.get("variance_decomposition"),
        name=f"{case}.variance_decomposition",
    )
    if variance.get("method") != "weighted_shapley_variance_of_conditional_means":
        raise ValueError(f"{case} uses an unsupported variance decomposition")
    all_holdout = _mapping(
        variance.get("all_holdout"),
        name=f"{case}.variance_decomposition.all_holdout",
    )
    _finite_number(
        all_holdout.get("total_predictive_variance_m2"),
        name=f"{case}.total_predictive_variance_m2",
        nonnegative=True,
    )
    ratio = all_holdout.get("residual_mse_to_predictive_variance_ratio")
    if ratio is not None:
        _finite_number(
            ratio,
            name=f"{case}.residual_mse_to_predictive_variance_ratio",
            nonnegative=True,
        )
    closure = _mapping(all_holdout.get("closure"), name=f"{case}.variance.closure")
    if not closure:
        raise ValueError(f"{case} variance closure diagnostics must be nonempty")
    for key, value in closure.items():
        _finite_number(value, name=f"{case}.variance.closure.{key}", nonnegative=True)
    contributions = _mapping(
        all_holdout.get("contributions"),
        name=f"{case}.variance.contributions",
    )
    if not contributions:
        raise ValueError(f"{case} variance contributions must be nonempty")
    for name, raw_entry in contributions.items():
        entry = _mapping(raw_entry, name=f"{case}.variance.contributions.{name}")
        _finite_number(
            entry.get("fraction_of_total_predictive_variance"),
            name=f"{case}.variance.contributions.{name}.fraction",
        )


def _validate_audit(summary: Mapping[str, Any], *, source: Path) -> dict[str, Any]:
    if summary.get("schema_version") != 1:
        raise ValueError(f"unsupported oracle-audit schema in {source}")
    if summary.get("experiment") != AUDIT_EXPERIMENT:
        raise ValueError(f"input is not a real oracle-gap audit: {source}")
    case = summary.get("case")
    if not isinstance(case, str) or not case:
        raise ValueError(f"audit case must be a nonempty string: {source}")
    context = _mapping(summary.get("causal_context"), name=f"{case}.causal_context")
    protocol_id = context.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError(f"{case} causal context lacks a protocol_id")
    for window_name in ("o_minus", "o_plus", "u_obs", "u_cf"):
        window = _mapping(context.get(window_name), name=f"{case}.{window_name}")
        if window.get("case_id") != case:
            raise ValueError(f"{case}.{window_name} identifies another case")
    _validate_information_boundary(summary, case=case)
    _validate_predictors(summary, case=case)
    _validate_oracles(summary, case=case)
    _validate_gap_arithmetic(summary, case=case)
    _validate_variance(summary, case=case)
    likelihood = _mapping(
        summary.get("abduction_likelihood"),
        name=f"{case}.abduction_likelihood",
    )
    for key in (
        "observation_scale_m",
        "likelihood_power",
        "dynamic_likelihood_weight",
        "degrees_of_freedom",
    ):
        _finite_number(
            likelihood.get(key),
            name=f"{case}.abduction_likelihood.{key}",
            nonnegative=True,
        )
    return {
        "case": case,
        "protocol_id": protocol_id,
        "selection_metric": summary["information_boundary"]["protocol"][
            "selection_metric"
        ],
        "prefix_interval": list(
            summary["information_boundary"]["o_plus_prefix_frame_interval"]
        ),
        "abduction_likelihood": dict(likelihood),
        "discrepancy_cap_m": float(summary["current_bank_oracle"]["discrepancy_cap_m"]),
        "gap_definitions": dict(summary["gaps"]["definitions"]),
        "summary": summary,
        "source": source,
        "sha256": _sha256(source),
    }


def _paired_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    candidate: str,
    baseline: str,
) -> dict[str, Any]:
    deltas = [
        float(row["predictors"][candidate][metric])
        - float(row["predictors"][baseline][metric])
        for row in rows
    ]
    percent_changes = []
    for row in rows:
        baseline_value = float(row["predictors"][baseline][metric])
        candidate_value = float(row["predictors"][candidate][metric])
        if baseline_value > np.finfo(float).eps:
            percent_changes.append(100.0 * (candidate_value / baseline_value - 1.0))
    tolerance = 1e-12
    better = sum(value < -tolerance for value in deltas)
    tied = sum(abs(value) <= tolerance for value in deltas)
    worse = len(deltas) - better - tied
    return {
        "candidate": candidate,
        "baseline": baseline,
        "paired_delta_m": _stats(deltas),
        "paired_percent_change": _optional_stats(percent_changes),
        "candidate_better_case_count": better,
        "candidate_tied_case_count": tied,
        "candidate_worse_case_count": worse,
        "candidate_better_case_fraction": float(better / len(deltas)),
        "worst_regression_m": float(max(0.0, max(deltas))),
        "largest_improvement_m": float(max(0.0, -min(deltas))),
    }


def _dominant_gap_summary(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> dict[str, Any]:
    counts = {name: 0 for name in GAP_NAMES}
    for row in rows:
        counts[str(row["gaps"][metric]["dominant_gap"])] += 1
    maximum = max(counts.values())
    winners = [name for name in GAP_NAMES if counts[name] == maximum]
    dominant = winners[0] if len(winners) == 1 else "mixed"
    interpretations = {
        "inference_gap": "abduction_or_posterior_weighting_headroom",
        "proposal_gap": "intervention_support_headroom",
        "model_gap": "physical_model_or_discrepancy_headroom",
        "mixed": "heterogeneous_failure_boundary",
    }
    return {
        "case_counts": counts,
        "case_fractions": {
            name: float(count / len(rows)) for name, count in counts.items()
        },
        "most_common": dominant,
        "interpretation": interpretations[dominant],
    }


def _per_case_row(audit: Mapping[str, Any], sha256: str) -> dict[str, Any]:
    predictors = audit["predictors"]
    gaps = audit["gaps"]
    variance = audit["variance_decomposition"]["all_holdout"]
    closure = variance["closure"]
    row: dict[str, Any] = {
        "case": audit["case"],
        "protocol_id": audit["causal_context"]["protocol_id"],
        "audit_sha256": sha256,
        "selection_metric": audit["information_boundary"]["protocol"][
            "selection_metric"
        ],
        "bpt_effective_component_count": predictors[
            "bayesian_phystwin_mixture_nominal_z"
        ].get("effective_component_count"),
        "causal4d_effective_component_count": predictors[
            "current_causal4d_posterior"
        ].get("effective_component_count"),
        "residual_mse_to_predictive_variance_ratio": variance.get(
            "residual_mse_to_predictive_variance_ratio"
        ),
        "maximum_variance_closure_error_m2": float(max(closure.values())),
    }
    for predictor in PREDICTORS:
        for metric in METRICS:
            row[f"{predictor}_{metric}"] = float(predictors[predictor][metric])
    for metric in METRICS:
        row[f"causal4d_minus_bpt_{metric}"] = float(
            predictors["current_causal4d_posterior"][metric]
            - predictors["bayesian_phystwin_mixture_nominal_z"][metric]
        )
        row[f"bpt_minus_nominal_{metric}"] = float(
            predictors["bayesian_phystwin_mixture_nominal_z"][metric]
            - predictors["nominal_phystwin"][metric]
        )
        for gap_name in GAP_NAMES:
            row[f"{metric}_{gap_name}"] = float(gaps[metric][gap_name])
        row[f"{metric}_total_diagnostic_headroom"] = float(
            gaps[metric]["total_diagnostic_headroom"]
        )
        row[f"{metric}_dominant_gap"] = gaps[metric]["dominant_gap"]
    return row


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise ValueError("CSV output requires at least one included case")
    preferred = ["case", "protocol_id", "audit_sha256", "selection_metric"]
    fieldnames = preferred + sorted(
        set().union(*(row.keys() for row in rows)) - set(preferred)
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def aggregate_real_failure_attribution(
    audit_paths: Sequence[str | Path],
    output_json: str | Path,
    *,
    output_csv: str | Path | None = None,
    expected_case_ids: Sequence[str] | None = None,
    expected_protocol_id: str | None = None,
    excluded_cases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Aggregate registered oracle audits with exact case accounting.

    Holdout-selected oracle values remain diagnostic-only. Supplying
    ``expected_case_ids`` makes the accounting fail closed: every expected case
    must have exactly one audit or one explicit exclusion reason.
    """

    if not audit_paths:
        raise ValueError("failure attribution requires at least one oracle audit")
    validated = [
        _validate_audit(_load_json(path), source=Path(path)) for path in audit_paths
    ]
    cases = [str(item["case"]) for item in validated]
    if len(cases) != len(set(cases)):
        raise ValueError("oracle-audit cases must be unique")

    protocol_ids = {str(item["protocol_id"]) for item in validated}
    if len(protocol_ids) != 1:
        raise ValueError("oracle audits must use one protocol_id")
    protocol_id = next(iter(protocol_ids))
    if expected_protocol_id is not None and protocol_id != expected_protocol_id:
        raise ValueError("oracle audits do not match the expected protocol_id")

    invariant_fields = (
        "selection_metric",
        "prefix_interval",
        "abduction_likelihood",
        "discrepancy_cap_m",
        "gap_definitions",
    )
    reference = validated[0]
    for item in validated[1:]:
        for field in invariant_fields:
            if item[field] != reference[field]:
                raise ValueError(
                    f"oracle-audit invariant differs across cases: {field}"
                )

    exclusions = dict(excluded_cases or {})
    for case, reason in exclusions.items():
        if not isinstance(case, str) or not case:
            raise ValueError("excluded case IDs must be nonempty strings")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"excluded case {case} lacks a reason")
    overlap = set(cases) & set(exclusions)
    if overlap:
        raise ValueError(
            f"cases cannot be both audited and excluded: {sorted(overlap)}"
        )

    expected: list[str] | None = None
    complete_accounting = False
    if expected_case_ids is not None:
        expected = list(expected_case_ids)
        if not expected or any(
            not isinstance(case, str) or not case for case in expected
        ):
            raise ValueError("expected case IDs must be nonempty strings")
        if len(expected) != len(set(expected)):
            raise ValueError("expected case IDs must be unique")
        extras = (set(cases) | set(exclusions)) - set(expected)
        missing = set(expected) - set(cases) - set(exclusions)
        if extras:
            raise ValueError(f"unregistered cases were supplied: {sorted(extras)}")
        if missing:
            raise ValueError(f"expected cases are unaccounted: {sorted(missing)}")
        complete_accounting = True
        order = {case: index for index, case in enumerate(expected)}
        validated.sort(key=lambda item: order[str(item["case"])])
    else:
        validated.sort(key=lambda item: str(item["case"]))

    summaries = [item["summary"] for item in validated]
    rows = [_per_case_row(item["summary"], str(item["sha256"])) for item in validated]
    predictor_summary = {
        predictor: {
            metric: _stats(
                [
                    float(summary["predictors"][predictor][metric])
                    for summary in summaries
                ]
            )
            for metric in METRICS
        }
        for predictor in PREDICTORS
    }
    comparisons = {
        metric: {
            "bayesian_phystwin_vs_nominal_phystwin": _paired_summary(
                summaries,
                metric=metric,
                candidate="bayesian_phystwin_mixture_nominal_z",
                baseline="nominal_phystwin",
            ),
            "causal4d_vs_bayesian_phystwin_nominal_z": _paired_summary(
                summaries,
                metric=metric,
                candidate="current_causal4d_posterior",
                baseline="bayesian_phystwin_mixture_nominal_z",
            ),
            "causal4d_vs_nominal_phystwin": _paired_summary(
                summaries,
                metric=metric,
                candidate="current_causal4d_posterior",
                baseline="nominal_phystwin",
            ),
        }
        for metric in METRICS
    }
    gap_attribution: dict[str, Any] = {}
    for metric in METRICS:
        fraction_values: dict[str, list[float]] = {name: [] for name in GAP_NAMES}
        for summary in summaries:
            fractions = summary["gaps"][metric].get(
                "fraction_of_total_diagnostic_headroom", {}
            )
            for gap_name in GAP_NAMES:
                value = fractions.get(gap_name) if isinstance(fractions, dict) else None
                if value is not None:
                    fraction_values[gap_name].append(
                        _finite_number(
                            value,
                            name=f"{summary['case']}.{metric}.{gap_name}.fraction",
                        )
                    )
        gap_attribution[metric] = {
            "gap_m": {
                gap_name: _stats(
                    [float(summary["gaps"][metric][gap_name]) for summary in summaries]
                )
                for gap_name in GAP_NAMES
            },
            "fraction_of_total_diagnostic_headroom": {
                gap_name: _optional_stats(values)
                for gap_name, values in fraction_values.items()
            },
            "dominant_gap": _dominant_gap_summary(summaries, metric),
        }

    contribution_names = tuple(
        summaries[0]["variance_decomposition"]["all_holdout"]["contributions"]
    )
    for summary in summaries[1:]:
        names = tuple(summary["variance_decomposition"]["all_holdout"]["contributions"])
        if names != contribution_names:
            raise ValueError("variance contribution inventory differs across cases")
    variance_ratios = [
        summary["variance_decomposition"]["all_holdout"].get(
            "residual_mse_to_predictive_variance_ratio"
        )
        for summary in summaries
    ]
    defined_variance_ratios = [
        float(value) for value in variance_ratios if value is not None
    ]
    variance_summary = {
        "contribution_fraction_of_total_predictive_variance": {
            name: _stats(
                [
                    float(
                        summary["variance_decomposition"]["all_holdout"][
                            "contributions"
                        ][name]["fraction_of_total_predictive_variance"]
                    )
                    for summary in summaries
                ]
            )
            for name in contribution_names
        },
        "residual_mse_to_predictive_variance_ratio": _optional_stats(
            defined_variance_ratios
        ),
        "maximum_closure_error_m2": {
            key: float(
                max(
                    summary["variance_decomposition"]["all_holdout"]["closure"][key]
                    for summary in summaries
                )
            )
            for key in summaries[0]["variance_decomposition"]["all_holdout"]["closure"]
        },
    }

    input_records = [
        {
            "case": item["case"],
            "path": str(item["source"].resolve()),
            "sha256": item["sha256"],
        }
        for item in validated
    ]
    excluded_output = [
        {"case": case, "reason": exclusions[case]}
        for case in (expected if expected is not None else sorted(exclusions))
        if case in exclusions
    ]
    result: dict[str, Any] = {
        "schema_version": 1,
        "aggregate": AGGREGATE_NAME,
        "status": "diagnostic_not_confirmatory",
        "analysis_unit": "equal_execution",
        "protocol_id": protocol_id,
        "case_accounting": {
            "expected_case_count": len(expected) if expected is not None else None,
            "included_case_count": len(cases),
            "excluded_case_count": len(exclusions),
            "complete": complete_accounting,
            "expected_case_ids": expected,
            "included_case_ids": [str(item["case"]) for item in validated],
            "excluded_cases": excluded_output,
        },
        "frozen_invariants": {field: reference[field] for field in invariant_fields},
        "predictors": predictor_summary,
        "paired_comparisons": comparisons,
        "diagnostic_gap_attribution": gap_attribution,
        "variance_decomposition": variance_summary,
        "per_case": rows,
        "claim_boundary": {
            "holdout_labels_used_for_oracle_selection": True,
            "oracle_outputs_deployable": False,
            "may_select_or_tune_the_confirmatory_method": False,
            "may_replace_registered_exclusions": False,
            "confirmatory_conclusion_requires_registered_analysis": True,
        },
        "artifacts": {"input_audits": input_records},
    }
    fingerprint_payload = dict(result)
    fingerprint_payload["artifacts"] = {
        "input_audits": [
            {"case": record["case"], "sha256": record["sha256"]}
            for record in input_records
        ]
    }
    result["evidence_fingerprint"] = _canonical_sha256(fingerprint_payload)
    atomic_write_json(output_json, result)
    if output_csv is not None:
        atomic_write_text(output_csv, _csv_text(rows))
    return result


__all__ = [
    "AGGREGATE_NAME",
    "AUDIT_EXPERIMENT",
    "GAP_NAMES",
    "METRICS",
    "PREDICTORS",
    "aggregate_real_failure_attribution",
]
