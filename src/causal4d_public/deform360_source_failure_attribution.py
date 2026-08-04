"""Source-only attribution of the failed Deform360 replication backend gate."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.atomic_io import atomic_write_json

from .deform360_replication import validate_deform360_replication_protocol
from .deform360_replication_backend import (
    validate_source_backend_decision_artifact,
)
from .deform360_replication_fit import (
    validate_pooled_source_warp_fit,
    validate_source_warp_candidate_grid,
)


SOURCE_FAILURE_ATTRIBUTION_SCHEMA_VERSION = 1
SOURCE_FAILURE_ATTRIBUTION_KIND = "Deform360SourceFailureAttribution"
_ALLOWED_CLASSIFICATIONS = frozenset(
    {
        "observation_geometry_failure",
        "per_episode_physical_feasibility_failure",
        "strain_constraint_failure",
        "episode_level_backend_competence_failure",
        "episode_level_backend_and_shared_feasibility_failure",
        "shared_physical_feasibility_failure",
        "shared_parameter_transfer_failure",
        "source_backend_competence_passed",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _relative_improvement(score: float, persistence: float) -> float:
    _require(persistence > 0.0, "persistence score must be positive")
    return (persistence - score) / persistence


def classify_source_failure(
    *,
    complete_source_episode_set: bool,
    every_episode_has_quality_candidate: bool,
    quality_oracle_gate_passed: bool,
    unconstrained_oracle_gate_passed: bool,
    common_quality_candidate_count: int,
    common_quality_gate_passed: bool,
) -> str:
    """Classify the first source-only boundary that prevents admission."""

    _require(common_quality_candidate_count >= 0, "common candidate count is negative")
    if not complete_source_episode_set:
        return "observation_geometry_failure"
    if not every_episode_has_quality_candidate:
        return "per_episode_physical_feasibility_failure"
    if not quality_oracle_gate_passed:
        if unconstrained_oracle_gate_passed:
            return "strain_constraint_failure"
        if common_quality_candidate_count == 0:
            return "episode_level_backend_and_shared_feasibility_failure"
        return "episode_level_backend_competence_failure"
    if common_quality_candidate_count == 0:
        return "shared_physical_feasibility_failure"
    if not common_quality_gate_passed:
        return "shared_parameter_transfer_failure"
    return "source_backend_competence_passed"


def _score_gate(
    scores: Sequence[float | None],
    persistence: Sequence[float],
    *,
    minimum_relative_improvement: float,
    minimum_win_fraction: float,
    complete: bool,
) -> dict[str, Any]:
    _require(len(scores) == len(persistence), "gate score count differs")
    valid = [score is not None for score in scores]
    finite_scores = [float(score) for score in scores if score is not None]
    finite_persistence = [
        float(baseline)
        for baseline, keep in zip(persistence, valid, strict=True)
        if keep
    ]
    if not finite_scores:
        return {
            "complete": False,
            "evaluated_episode_count": 0,
            "mean_chamfer_m": None,
            "persistence_mean_chamfer_m": None,
            "relative_improvement_vs_persistence": None,
            "win_fraction": None,
            "minimum_relative_improvement": minimum_relative_improvement,
            "minimum_win_fraction": minimum_win_fraction,
            "passed": False,
        }
    mean_score = float(np.mean(finite_scores))
    mean_persistence = float(np.mean(finite_persistence))
    improvement = _relative_improvement(mean_score, mean_persistence)
    wins = [
        score < baseline
        for score, baseline in zip(finite_scores, finite_persistence, strict=True)
    ]
    win_fraction = float(np.mean(wins))
    gate_complete = bool(complete and all(valid))
    return {
        "complete": gate_complete,
        "evaluated_episode_count": len(finite_scores),
        "mean_chamfer_m": mean_score,
        "persistence_mean_chamfer_m": mean_persistence,
        "relative_improvement_vs_persistence": improvement,
        "win_fraction": win_fraction,
        "minimum_relative_improvement": minimum_relative_improvement,
        "minimum_win_fraction": minimum_win_fraction,
        "passed": bool(
            gate_complete
            and improvement >= minimum_relative_improvement
            and win_fraction >= minimum_win_fraction
        ),
    }


def _best_candidate(
    scores: np.ndarray,
    *,
    required_support: int | None = None,
) -> int | None:
    _require(scores.ndim == 2, "candidate score matrix must be two-dimensional")
    support = np.sum(np.isfinite(scores), axis=1)
    required = scores.shape[1] if required_support is None else required_support
    candidates = np.flatnonzero(support >= required)
    if not len(candidates):
        return None
    return int(
        min(
            candidates,
            key=lambda index: (
                float(np.mean(scores[index, np.isfinite(scores[index])])),
                int(index),
            ),
        )
    )


def _oracle_record(
    scores: np.ndarray,
    p99_strain: np.ndarray,
    persistence: float,
    parameters: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    indices = np.flatnonzero(np.isfinite(scores))
    if not len(indices):
        return None
    index = int(
        min(indices, key=lambda candidate: (float(scores[candidate]), int(candidate)))
    )
    score = float(scores[index])
    strain = float(p99_strain[index]) if np.isfinite(p99_strain[index]) else None
    return {
        "candidate_index": index,
        "parameters": dict(parameters[index]),
        "mean_chamfer_m": score,
        "p99_relative_edge_strain": strain,
        "relative_improvement_vs_persistence": _relative_improvement(
            score, persistence
        ),
        "win_vs_persistence": score < persistence,
    }


def _candidate_record(
    candidate_index: int | None,
    scores: np.ndarray,
    persistence: np.ndarray,
    parameters: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if candidate_index is None:
        return None
    values = scores[candidate_index]
    finite = np.isfinite(values)
    score_list = [float(value) if keep else None for value, keep in zip(values, finite)]
    gate = _score_gate(
        score_list,
        persistence.tolist(),
        minimum_relative_improvement=0.0,
        minimum_win_fraction=0.0,
        complete=bool(np.all(finite)),
    )
    return {
        "candidate_index": candidate_index,
        "parameters": dict(parameters[candidate_index]),
        "episode_support_count": int(np.sum(finite)),
        "episode_scores_m": score_list,
        "mean_chamfer_m_over_supported_episodes": gate["mean_chamfer_m"],
        "win_fraction_over_supported_episodes": gate["win_fraction"],
    }


def _transfer_summary(
    episode_records: Sequence[Mapping[str, Any]],
    quality_scores: np.ndarray,
    persistence: np.ndarray,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(episode_records):
        oracle = source.get("quality_oracle")
        if not isinstance(oracle, Mapping):
            continue
        candidate_index = int(oracle["candidate_index"])
        for target_index, target in enumerate(episode_records):
            if source_index == target_index:
                continue
            score = quality_scores[candidate_index, target_index]
            target_oracle = target.get("quality_oracle")
            target_oracle_score = (
                float(target_oracle["mean_chamfer_m"])
                if isinstance(target_oracle, Mapping)
                else None
            )
            valid = bool(np.isfinite(score))
            row: dict[str, Any] = {
                "fit_episode_id": source["episode_id"],
                "evaluation_episode_id": target["episode_id"],
                "candidate_index": candidate_index,
                "same_bimanual_condition": (source["bimanual"] is target["bimanual"]),
                "quality_valid": valid,
                "candidate_chamfer_m": float(score) if valid else None,
                "persistence_chamfer_m": float(persistence[target_index]),
                "win_vs_persistence": bool(valid and score < persistence[target_index]),
                "target_oracle_chamfer_m": target_oracle_score,
                "normalized_regret_vs_target_oracle": None,
            }
            if valid and target_oracle_score is not None:
                row["normalized_regret_vs_target_oracle"] = (
                    float(score) - target_oracle_score
                ) / float(persistence[target_index])
            rows.append(row)
    total = len(rows)
    valid_rows = [row for row in rows if row["quality_valid"]]
    regrets = [
        float(row["normalized_regret_vs_target_oracle"])
        for row in valid_rows
        if row["normalized_regret_vs_target_oracle"] is not None
    ]

    def group(condition: bool) -> dict[str, Any]:
        selected = [
            row for row in rows if bool(row["same_bimanual_condition"]) is condition
        ]
        valid_selected = [row for row in selected if row["quality_valid"]]
        return {
            "pair_count": len(selected),
            "quality_valid_pair_count": len(valid_selected),
            "quality_coverage": (
                len(valid_selected) / len(selected) if selected else None
            ),
            "win_fraction_over_all_pairs": (
                float(np.mean([row["win_vs_persistence"] for row in selected]))
                if selected
                else None
            ),
            "conditional_win_fraction": (
                float(np.mean([row["win_vs_persistence"] for row in valid_selected]))
                if valid_selected
                else None
            ),
        }

    return {
        "pair_count": total,
        "quality_valid_pair_count": len(valid_rows),
        "quality_coverage": len(valid_rows) / total if total else None,
        "win_fraction_over_all_pairs": (
            float(np.mean([row["win_vs_persistence"] for row in rows]))
            if rows
            else None
        ),
        "conditional_win_fraction": (
            float(np.mean([row["win_vs_persistence"] for row in valid_rows]))
            if valid_rows
            else None
        ),
        "mean_normalized_regret_vs_target_oracle": (
            float(np.mean(regrets)) if regrets else None
        ),
        "same_bimanual_condition": group(True),
        "cross_bimanual_condition": group(False),
        "rows": rows,
    }


def _analyze_object(
    cohort_record: Mapping[str, Any],
    source_grids: Sequence[Mapping[str, Any]],
    decision_record: Mapping[str, Any],
    pooled_fit: Mapping[str, Any] | None,
    *,
    minimum_relative_improvement: float,
    minimum_win_fraction: float,
) -> dict[str, Any]:
    object_id = str(cohort_record["object_id"])
    stratum = str(cohort_record["stratum"])
    expected_episode_ids = [
        f"{object_id}/episode_{int(index):04d}"
        for index in cohort_record["source_episode_ids"]
    ]
    by_episode: dict[str, Mapping[str, Any]] = {}
    for grid in source_grids:
        validate_source_warp_candidate_grid(grid)
        episode_id = str(grid["episode_id"])
        _require(
            episode_id.startswith(f"{object_id}/episode_"),
            "source grid belongs to another object",
        )
        _require(grid["stratum"] == stratum, "source grid stratum changed")
        _require(episode_id not in by_episode, "source grid episode repeated")
        by_episode[episode_id] = grid
    _require(
        set(by_episode).issubset(expected_episode_ids),
        "source grid includes an unregistered episode",
    )
    available_ids = [
        episode_id for episode_id in expected_episode_ids if episode_id in by_episode
    ]
    missing_ids = [
        episode_id
        for episode_id in expected_episode_ids
        if episode_id not in by_episode
    ]
    complete = not missing_ids
    if not available_ids:
        classification = classify_source_failure(
            complete_source_episode_set=False,
            every_episode_has_quality_candidate=False,
            quality_oracle_gate_passed=False,
            unconstrained_oracle_gate_passed=False,
            common_quality_candidate_count=0,
            common_quality_gate_passed=False,
        )
        return {
            "object_id": object_id,
            "stratum": stratum,
            "expected_source_episode_ids": expected_episode_ids,
            "available_source_episode_ids": [],
            "missing_source_episode_ids": missing_ids,
            "recorded_source_outcome": dict(decision_record),
            "classification": classification,
            "recommended_next_test": (
                "repair source geometry availability before changing dynamics"
            ),
        }

    grids = [by_episode[episode_id] for episode_id in available_ids]
    candidate_count = int(grids[0]["candidate_count"])
    _require(candidate_count == 200, "candidate grid size changed")
    quality_scores = np.full((candidate_count, len(grids)), np.inf, dtype=np.float64)
    unconstrained_scores = np.full_like(quality_scores, np.inf)
    p99_strain = np.full_like(quality_scores, np.inf)
    persistence = np.empty(len(grids), dtype=np.float64)
    parameters: list[Mapping[str, Any]] = [dict() for _ in range(candidate_count)]
    quality_limit: float | None = None
    episode_records: list[dict[str, Any]] = []

    for episode_index, grid in enumerate(grids):
        configured_limit = float(grid["config"]["maximum_p99_relative_edge_strain"])
        if quality_limit is None:
            quality_limit = configured_limit
        else:
            _require(configured_limit == quality_limit, "strain limit changed")
        baseline = _finite_float(grid["persistence"]["mean_m"])
        _require(baseline is not None and baseline > 0.0, "invalid persistence score")
        persistence[episode_index] = baseline
        finite_prediction_count = 0
        quality_valid_count = 0
        for row in grid["candidate_scores"]:
            candidate_index = int(row["candidate_index"])
            candidate_parameters = dict(row["parameters"])
            if episode_index == 0:
                parameters[candidate_index] = candidate_parameters
            else:
                _require(
                    parameters[candidate_index] == candidate_parameters,
                    "candidate parameters changed across episodes",
                )
            score = _finite_float(row.get("mean_chamfer_m"))
            strain = _finite_float(row.get("p99_relative_edge_strain"))
            if row.get("finite") is True and score is not None:
                finite_prediction_count += 1
                unconstrained_scores[candidate_index, episode_index] = score
            if strain is not None:
                p99_strain[candidate_index, episode_index] = strain
            if (
                row.get("finite") is True
                and score is not None
                and strain is not None
                and strain <= configured_limit
            ):
                quality_valid_count += 1
                quality_scores[candidate_index, episode_index] = score
        episode_number = int(available_ids[episode_index].rsplit("_", maxsplit=1)[1])
        metadata = cohort_record["episodes"][str(episode_number)]
        unconstrained_oracle = _oracle_record(
            unconstrained_scores[:, episode_index],
            p99_strain[:, episode_index],
            baseline,
            parameters,
        )
        quality_oracle = _oracle_record(
            quality_scores[:, episode_index],
            p99_strain[:, episode_index],
            baseline,
            parameters,
        )
        episode_records.append(
            {
                "episode_id": available_ids[episode_index],
                "episode_index": episode_number,
                "action": metadata["action"],
                "bimanual": metadata["bimanual"] == "yes",
                "nonprehensile": metadata["nonprehensile"] == "yes",
                "persistence_chamfer_m": baseline,
                "finite_prediction_candidate_count": finite_prediction_count,
                "quality_valid_candidate_count": quality_valid_count,
                "strain_rejected_candidate_count": (
                    finite_prediction_count - quality_valid_count
                ),
                "numerical_failure_candidate_count": (
                    candidate_count - finite_prediction_count
                ),
                "unconstrained_oracle": unconstrained_oracle,
                "quality_oracle": quality_oracle,
            }
        )

    unconstrained_oracle_scores = [
        (
            float(record["unconstrained_oracle"]["mean_chamfer_m"])
            if record["unconstrained_oracle"] is not None
            else None
        )
        for record in episode_records
    ]
    quality_oracle_scores = [
        (
            float(record["quality_oracle"]["mean_chamfer_m"])
            if record["quality_oracle"] is not None
            else None
        )
        for record in episode_records
    ]
    unconstrained_oracle_gate = _score_gate(
        unconstrained_oracle_scores,
        persistence.tolist(),
        minimum_relative_improvement=minimum_relative_improvement,
        minimum_win_fraction=minimum_win_fraction,
        complete=complete,
    )
    quality_oracle_gate = _score_gate(
        quality_oracle_scores,
        persistence.tolist(),
        minimum_relative_improvement=minimum_relative_improvement,
        minimum_win_fraction=minimum_win_fraction,
        complete=complete,
    )

    quality_support = np.sum(np.isfinite(quality_scores), axis=1)
    unconstrained_support = np.sum(np.isfinite(unconstrained_scores), axis=1)
    common_quality = np.flatnonzero(quality_support == len(grids))
    common_unconstrained = np.flatnonzero(unconstrained_support == len(grids))
    common_quality_index = _best_candidate(quality_scores)
    common_unconstrained_index = _best_candidate(unconstrained_scores)
    maximum_quality_support = int(np.max(quality_support))
    maximum_support_index = _best_candidate(
        quality_scores, required_support=maximum_quality_support
    )

    common_quality_scores = (
        [float(value) for value in quality_scores[common_quality_index]]
        if common_quality_index is not None
        else [None] * len(grids)
    )
    common_unconstrained_scores = (
        [float(value) for value in unconstrained_scores[common_unconstrained_index]]
        if common_unconstrained_index is not None
        else [None] * len(grids)
    )
    common_quality_gate = _score_gate(
        common_quality_scores,
        persistence.tolist(),
        minimum_relative_improvement=minimum_relative_improvement,
        minimum_win_fraction=minimum_win_fraction,
        complete=complete,
    )
    common_unconstrained_gate = _score_gate(
        common_unconstrained_scores,
        persistence.tolist(),
        minimum_relative_improvement=minimum_relative_improvement,
        minimum_win_fraction=minimum_win_fraction,
        complete=complete,
    )

    if pooled_fit is not None:
        validate_pooled_source_warp_fit(pooled_fit)
        _require(pooled_fit["object_id"] == object_id, "pooled fit object changed")
        _require(
            pooled_fit["source_episode_ids"] == expected_episode_ids,
            "pooled fit source episodes changed",
        )
        pooled_index = int(pooled_fit["selection"]["pooled_candidate_index"])
        _require(
            pooled_index == common_quality_index,
            "archived pooled candidate differs from recomputed common optimum",
        )
        _require(
            np.isclose(
                float(pooled_fit["pooled_source_mean_chamfer_m"]),
                float(common_quality_gate["mean_chamfer_m"]),
                rtol=0.0,
                atol=1e-15,
            ),
            "archived pooled score differs from recomputation",
        )
        pooled_fit_record: dict[str, Any] | None = {
            "result_sha256": pooled_fit["result_sha256"],
            "candidate_index": pooled_index,
            "archived_competence_passed": pooled_fit["source_backend_competence"][
                "passed"
            ],
        }
    else:
        pooled_fit_record = None

    every_episode_has_quality_candidate = all(
        record["quality_oracle"] is not None for record in episode_records
    )
    classification = classify_source_failure(
        complete_source_episode_set=complete,
        every_episode_has_quality_candidate=every_episode_has_quality_candidate,
        quality_oracle_gate_passed=bool(quality_oracle_gate["passed"]),
        unconstrained_oracle_gate_passed=bool(unconstrained_oracle_gate["passed"]),
        common_quality_candidate_count=len(common_quality),
        common_quality_gate_passed=bool(common_quality_gate["passed"]),
    )
    recommendation = {
        "observation_geometry_failure": (
            "repair source geometry availability before changing dynamics"
        ),
        "per_episode_physical_feasibility_failure": (
            "revise the graph representation or strain model on source episodes"
        ),
        "strain_constraint_failure": (
            "improve representation and material constraints without relaxing the gate"
        ),
        "episode_level_backend_competence_failure": (
            "improve source dynamics, contact realization, or support registration"
        ),
        "episode_level_backend_and_shared_feasibility_failure": (
            "address episode competence and cross-episode physical feasibility jointly"
        ),
        "shared_physical_feasibility_failure": (
            "test stratum-specific representations or hierarchical physical parameters"
        ),
        "shared_parameter_transfer_failure": (
            "test hierarchical or action-conditioned shared physics on a new source lock"
        ),
        "source_backend_competence_passed": (
            "retain the method and proceed only under a separately authorized protocol"
        ),
    }[classification]

    oracle_indices = [
        int(record["quality_oracle"]["candidate_index"])
        for record in episode_records
        if record["quality_oracle"] is not None
    ]
    oracle_counts = Counter(oracle_indices)
    support_histogram = {
        str(support): int(np.sum(quality_support == support))
        for support in range(len(grids) + 1)
        if np.any(quality_support == support)
    }
    return {
        "object_id": object_id,
        "stratum": stratum,
        "expected_source_episode_ids": expected_episode_ids,
        "available_source_episode_ids": available_ids,
        "missing_source_episode_ids": missing_ids,
        "source_episode_set_complete": complete,
        "candidate_count": candidate_count,
        "maximum_p99_relative_edge_strain": quality_limit,
        "episodes": episode_records,
        "unconstrained_episode_oracle_gate": unconstrained_oracle_gate,
        "quality_episode_oracle_gate": quality_oracle_gate,
        "candidate_support": {
            "quality_support_histogram": support_histogram,
            "common_quality_candidate_count": len(common_quality),
            "common_unconstrained_candidate_count": len(common_unconstrained),
            "maximum_quality_support_episode_count": maximum_quality_support,
            "maximum_quality_support_fraction": maximum_quality_support / len(grids),
            "best_maximum_support_candidate": _candidate_record(
                maximum_support_index,
                quality_scores,
                persistence,
                parameters,
            ),
        },
        "common_quality_candidate": _candidate_record(
            common_quality_index,
            quality_scores,
            persistence,
            parameters,
        ),
        "common_quality_candidate_gate": common_quality_gate,
        "common_unconstrained_candidate": _candidate_record(
            common_unconstrained_index,
            unconstrained_scores,
            persistence,
            parameters,
        ),
        "common_unconstrained_candidate_gate": common_unconstrained_gate,
        "episode_oracle_parameter_instability": {
            "distinct_quality_oracle_candidate_count": len(oracle_counts),
            "quality_oracle_candidate_counts": {
                str(index): count for index, count in sorted(oracle_counts.items())
            },
        },
        "cross_episode_oracle_transfer": _transfer_summary(
            episode_records, quality_scores, persistence
        ),
        "archived_pooled_fit": pooled_fit_record,
        "recorded_source_outcome": dict(decision_record),
        "classification": classification,
        "recommended_next_test": recommendation,
    }


def build_source_failure_attribution(
    protocol: Mapping[str, Any],
    source_backend_decision: Mapping[str, Any],
    source_grids_by_object: Mapping[str, Sequence[Mapping[str, Any]]],
    pooled_fits_by_object: Mapping[str, Mapping[str, Any]],
    *,
    input_verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic source-only decomposition without target access."""

    validate_deform360_replication_protocol(protocol)
    validate_source_backend_decision_artifact(source_backend_decision)
    _require(
        source_backend_decision["protocol_config_sha256"] == protocol["config_sha256"],
        "source decision belongs to another protocol",
    )
    boundary = source_backend_decision["information_boundary"]
    _require(
        boundary["target_prefix_read"] is False
        and boundary["target_future_geometry_read"] is False
        and boundary["target_future_tactile_read"] is False,
        "source decision crossed the target boundary",
    )
    gate = protocol["config"]["gates"]["source_backend_competence"]
    minimum_relative_improvement = float(
        gate["minimum_pooled_chamfer_improvement_vs_persistence"]
    )
    minimum_win_fraction = float(gate["minimum_leave_one_source_win_fraction"])
    decision_by_object = {
        str(record["object_id"]): record
        for record in source_backend_decision["object_results"]
    }
    objects = []
    for cohort_record in protocol["config"]["cohort"]:
        object_id = str(cohort_record["object_id"])
        _require(object_id in decision_by_object, "source decision object is missing")
        objects.append(
            _analyze_object(
                cohort_record,
                source_grids_by_object.get(object_id, ()),
                decision_by_object[object_id],
                pooled_fits_by_object.get(object_id),
                minimum_relative_improvement=minimum_relative_improvement,
                minimum_win_fraction=minimum_win_fraction,
            )
        )
    classifications = Counter(record["classification"] for record in objects)
    stratum_summary = []
    for stratum in ("filament", "sheet", "volumetric"):
        selected = [record for record in objects if record["stratum"] == stratum]
        stratum_summary.append(
            {
                "stratum": stratum,
                "object_count": len(selected),
                "classification_counts": dict(
                    sorted(
                        Counter(record["classification"] for record in selected).items()
                    )
                ),
                "quality_episode_oracle_gate_pass_count": sum(
                    bool(record.get("quality_episode_oracle_gate", {}).get("passed"))
                    for record in selected
                ),
                "common_quality_candidate_gate_pass_count": sum(
                    bool(record.get("common_quality_candidate_gate", {}).get("passed"))
                    for record in selected
                ),
            }
        )
    episode_records = [
        episode for record in objects for episode in record.get("episodes", [])
    ]
    quality_oracle_wins = [
        bool(episode["quality_oracle"]["win_vs_persistence"])
        for episode in episode_records
        if episode["quality_oracle"] is not None
    ]
    payload: dict[str, Any] = {
        "schema_version": SOURCE_FAILURE_ATTRIBUTION_SCHEMA_VERSION,
        "artifact_kind": SOURCE_FAILURE_ATTRIBUTION_KIND,
        "protocol_config_sha256": protocol["config_sha256"],
        "source_backend_decision_result_sha256": source_backend_decision[
            "result_sha256"
        ],
        "registered_thresholds": {
            "minimum_relative_improvement": minimum_relative_improvement,
            "minimum_win_fraction": minimum_win_fraction,
            "maximum_p99_relative_edge_strain": 0.5,
        },
        "objects": objects,
        "strata": stratum_summary,
        "cohort_summary": {
            "object_count": len(objects),
            "classification_counts": dict(sorted(classifications.items())),
            "complete_source_geometry_object_count": sum(
                not record["missing_source_episode_ids"] for record in objects
            ),
            "quality_episode_oracle_gate_pass_object_count": sum(
                bool(record.get("quality_episode_oracle_gate", {}).get("passed"))
                for record in objects
            ),
            "common_quality_candidate_gate_pass_object_count": sum(
                bool(record.get("common_quality_candidate_gate", {}).get("passed"))
                for record in objects
            ),
            "available_source_episode_count": len(episode_records),
            "quality_oracle_available_episode_count": len(quality_oracle_wins),
            "quality_oracle_win_fraction": (
                float(np.mean(quality_oracle_wins)) if quality_oracle_wins else None
            ),
        },
        "decision": {
            "diagnostic_only": True,
            "registered_method_changed": False,
            "target_prefix_access_permitted": False,
            "target_future_access_permitted": False,
            "interpretation": (
                "The artifact localizes source-stage failure boundaries. It does not "
                "rescue the failed gate, authorize target access, or select a new method."
            ),
        },
        "information_boundary": {
            "source_candidate_scores_read": True,
            "source_geometry_failure_records_read": True,
            "calibration_outcomes_read": False,
            "target_prefix_read": False,
            "target_future_geometry_read": False,
            "target_future_tactile_read": False,
        },
        "input_verification": dict(input_verification or {}),
    }
    payload["result_sha256"] = _artifact_sha256(payload)
    return payload


def validate_source_failure_attribution(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate attribution identity, object partition, and target boundary."""

    _require(
        payload.get("schema_version") == SOURCE_FAILURE_ATTRIBUTION_SCHEMA_VERSION,
        "source-failure attribution schema changed",
    )
    _require(
        payload.get("artifact_kind") == SOURCE_FAILURE_ATTRIBUTION_KIND,
        "source-failure attribution kind changed",
    )
    _require(
        payload.get("result_sha256") == _artifact_sha256(payload),
        "source-failure attribution checksum mismatch",
    )
    objects = payload.get("objects")
    _require(isinstance(objects, list) and len(objects) == 6, "object cohort changed")
    identifiers = [record.get("object_id") for record in objects]
    _require(len(set(identifiers)) == len(identifiers), "object identity repeated")
    for record in objects:
        _require(
            record.get("classification") in _ALLOWED_CLASSIFICATIONS,
            "unknown source-failure classification",
        )
    boundary = payload.get("information_boundary", {})
    decision = payload.get("decision", {})
    _require(
        boundary.get("calibration_outcomes_read") is False
        and boundary.get("target_prefix_read") is False
        and boundary.get("target_future_geometry_read") is False
        and boundary.get("target_future_tactile_read") is False,
        "source-failure attribution crossed the target boundary",
    )
    _require(
        decision.get("diagnostic_only") is True
        and decision.get("registered_method_changed") is False
        and decision.get("target_prefix_access_permitted") is False
        and decision.get("target_future_access_permitted") is False,
        "source-failure attribution changed the registered decision",
    )
    return {
        "passed": True,
        "object_count": len(objects),
        "classification_counts": payload["cohort_summary"]["classification_counts"],
        "result_sha256": payload["result_sha256"],
    }


def _verify_milestone_manifest(milestone_root: Path) -> dict[str, Any]:
    manifest_path = milestone_root / "artifact-manifest.json"
    _require(manifest_path.is_file(), "milestone artifact manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries")
    _require(isinstance(entries, list) and entries, "artifact manifest is empty")
    repository_root = milestone_root.parent.parent
    for entry in entries:
        source = (repository_root / str(entry["source_path"])).resolve()
        try:
            source.relative_to(repository_root.resolve())
        except ValueError as error:
            raise ValueError("artifact manifest path escapes the repository") from error
        _require(source.is_file(), f"artifact manifest file is missing: {source}")
        _require(
            source.stat().st_size == int(entry["bytes"]), "artifact byte count changed"
        )
        _require(_sha256_file(source) == entry["sha256"], "artifact checksum changed")
    return {
        "artifact_manifest_sha256": _sha256_file(manifest_path),
        "artifact_manifest_bytes": manifest_path.stat().st_size,
        "verified_entry_count": len(entries),
        "all_entries_verified": True,
    }


def analyze_source_failure_milestone(
    protocol_path: str | Path,
    milestone_root: str | Path,
) -> dict[str, Any]:
    """Load and verify the frozen source artifacts, then build the attribution."""

    protocol_file = Path(protocol_path).resolve()
    root = Path(milestone_root).resolve()
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    validate_deform360_replication_protocol(protocol)
    verification = _verify_milestone_manifest(root)
    decision: Mapping[str, Any] | None = None
    grids: dict[str, list[Mapping[str, Any]]] = {}
    pooled: dict[str, Mapping[str, Any]] = {}
    for path in sorted((root / "artifacts").rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        kind = payload.get("artifact_kind") if isinstance(payload, Mapping) else None
        if kind == "Deform360ReplicationSourceBackendDecision":
            _require(decision is None, "source-backend decision is repeated")
            decision = payload
        elif kind == "Deform360ReplicationSourceWarpCandidateGrid":
            object_id = str(payload["episode_id"]).split("/episode_", maxsplit=1)[0]
            grids.setdefault(object_id, []).append(payload)
        elif kind == "Deform360ReplicationPooledSourceWarpFit":
            object_id = str(payload["object_id"])
            _require(object_id not in pooled, "pooled fit is repeated")
            pooled[object_id] = payload
    _require(decision is not None, "source-backend decision is missing")
    verification.update(
        {
            "protocol_file_sha256": _sha256_file(protocol_file),
            "protocol_file_bytes": protocol_file.stat().st_size,
            "source_grid_count": sum(len(values) for values in grids.values()),
            "pooled_fit_count": len(pooled),
        }
    )
    return build_source_failure_attribution(
        protocol,
        decision,
        grids,
        pooled,
        input_verification=verification,
    )


def write_source_failure_attribution(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Validate and atomically publish one attribution artifact."""

    validate_source_failure_attribution(payload)
    output = Path(path)
    atomic_write_json(output, dict(payload), overwrite=True)
    reopened = json.loads(output.read_text(encoding="utf-8"))
    validate_source_failure_attribution(reopened)
    return output


__all__ = [
    "SOURCE_FAILURE_ATTRIBUTION_KIND",
    "SOURCE_FAILURE_ATTRIBUTION_SCHEMA_VERSION",
    "analyze_source_failure_milestone",
    "build_source_failure_attribution",
    "classify_source_failure",
    "validate_source_failure_attribution",
    "write_source_failure_attribution",
]
