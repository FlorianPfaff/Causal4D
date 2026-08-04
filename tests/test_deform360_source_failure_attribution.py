from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from causal4d_public.deform360_source_failure_attribution import (
    analyze_source_failure_milestone,
    classify_source_failure,
    validate_source_failure_attribution,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "configs/causal4d_public/deform360_replication_v1.json"
MILESTONE = REPOSITORY_ROOT / "milestones/deform360-replication-source-backend-v1"
SUMMARY = (
    REPOSITORY_ROOT
    / "milestones/deform360-source-failure-attribution-v1/summary.json"
)
DEPENDENCY_LOCK = (
    REPOSITORY_ROOT
    / "requirements/diagnostics/deform360-source-failure-py312.txt"
)
EXPECTED_RESULT_SHA256 = (
    "8da1a6112a7afb959a6bf81c3f870a8135d8414b599db98d67656ba40e98c9eb"
)
EXPECTED_DEPENDENCY_LOCK_SHA256 = (
    "e8258eeae724e775651af0449d3e0708dae2fd4d04d2316f109792aad2862895"
)
EXPECTED_CLASSIFICATION_COUNTS = {
    "episode_level_backend_competence_failure": 4,
    "observation_geometry_failure": 1,
    "per_episode_physical_feasibility_failure": 1,
}


@pytest.mark.parametrize(
    (
        "complete",
        "every_episode_has_quality_candidate",
        "quality_oracle_passed",
        "unconstrained_oracle_passed",
        "common_quality_candidate_count",
        "common_quality_passed",
        "expected",
    ),
    (
        (False, False, False, False, 0, False, "observation_geometry_failure"),
        (
            True,
            False,
            False,
            False,
            0,
            False,
            "per_episode_physical_feasibility_failure",
        ),
        (True, True, False, True, 4, False, "strain_constraint_failure"),
        (
            True,
            True,
            False,
            False,
            3,
            False,
            "episode_level_backend_competence_failure",
        ),
        (
            True,
            True,
            False,
            False,
            0,
            False,
            "episode_level_backend_and_shared_feasibility_failure",
        ),
        (
            True,
            True,
            True,
            True,
            0,
            False,
            "shared_physical_feasibility_failure",
        ),
        (
            True,
            True,
            True,
            True,
            2,
            False,
            "shared_parameter_transfer_failure",
        ),
        (
            True,
            True,
            True,
            True,
            2,
            True,
            "source_backend_competence_passed",
        ),
    ),
)
def test_source_failure_classification_is_ordered_and_fail_closed(
    complete: bool,
    every_episode_has_quality_candidate: bool,
    quality_oracle_passed: bool,
    unconstrained_oracle_passed: bool,
    common_quality_candidate_count: int,
    common_quality_passed: bool,
    expected: str,
) -> None:
    assert (
        classify_source_failure(
            complete_source_episode_set=complete,
            every_episode_has_quality_candidate=(
                every_episode_has_quality_candidate
            ),
            quality_oracle_gate_passed=quality_oracle_passed,
            unconstrained_oracle_gate_passed=unconstrained_oracle_passed,
            common_quality_candidate_count=common_quality_candidate_count,
            common_quality_gate_passed=common_quality_passed,
        )
        == expected
    )


def test_frozen_source_attribution_is_locked_and_target_closed() -> None:
    first = analyze_source_failure_milestone(PROTOCOL, MILESTONE)
    second = analyze_source_failure_milestone(PROTOCOL, MILESTONE)

    assert first == second
    validation = validate_source_failure_attribution(first)
    assert validation["object_count"] == 6
    assert validation["result_sha256"] == EXPECTED_RESULT_SHA256
    assert validation["classification_counts"] == EXPECTED_CLASSIFICATION_COUNTS
    assert first["result_sha256"] == EXPECTED_RESULT_SHA256
    assert first["input_verification"]["source_grid_count"] == 30
    assert first["input_verification"]["pooled_fit_count"] == 4
    assert first["input_verification"]["all_entries_verified"] is True

    cohort = first["cohort_summary"]
    assert cohort["object_count"] == 6
    assert cohort["classification_counts"] == EXPECTED_CLASSIFICATION_COUNTS
    assert cohort["available_source_episode_count"] == 30
    assert cohort["complete_source_geometry_object_count"] == 5
    assert cohort["quality_episode_oracle_gate_pass_object_count"] == 0
    assert cohort["common_quality_candidate_gate_pass_object_count"] == 0
    assert cohort["quality_oracle_available_episode_count"] == 28
    assert cohort["quality_oracle_win_fraction"] == pytest.approx(6 / 28)

    assert first["decision"]["diagnostic_only"] is True
    assert first["decision"]["registered_method_changed"] is False
    assert first["decision"]["target_prefix_access_permitted"] is False
    assert first["decision"]["target_future_access_permitted"] is False
    assert first["information_boundary"]["calibration_outcomes_read"] is False
    assert first["information_boundary"]["target_prefix_read"] is False
    assert first["information_boundary"]["target_future_geometry_read"] is False
    assert first["information_boundary"]["target_future_tactile_read"] is False

    by_object = {record["object_id"]: record for record in first["objects"]}
    assert by_object["002-rope-silk"]["classification"] == (
        "episode_level_backend_competence_failure"
    )
    assert by_object["081-stripe-rope"]["classification"] == (
        "episode_level_backend_competence_failure"
    )
    assert by_object["085-scarf-cloth"]["classification"] == (
        "episode_level_backend_competence_failure"
    )
    assert by_object["083-blanket-cloth"]["classification"] == (
        "per_episode_physical_feasibility_failure"
    )
    assert by_object["092-squirrel"]["classification"] == (
        "observation_geometry_failure"
    )
    assert by_object["170-spider"]["classification"] == (
        "episode_level_backend_competence_failure"
    )
    assert by_object["083-blanket-cloth"]["recorded_source_outcome"][
        "failed_stage"
    ] == "source-pooling"
    assert by_object["083-blanket-cloth"]["common_quality_candidate"] is None


def test_compact_milestone_summary_matches_the_full_attribution() -> None:
    full = analyze_source_failure_milestone(PROTOCOL, MILESTONE)
    compact = json.loads(SUMMARY.read_text(encoding="utf-8"))
    dependency_lock_bytes = DEPENDENCY_LOCK.read_bytes()

    assert compact["full_attribution_result_sha256"] == EXPECTED_RESULT_SHA256
    assert compact["environment"]["python"] == "3.12"
    assert compact["environment"]["dependency_lock_path"] == (
        "requirements/diagnostics/deform360-source-failure-py312.txt"
    )
    assert compact["environment"]["dependency_lock_sha256"] == (
        EXPECTED_DEPENDENCY_LOCK_SHA256
    )
    assert compact["environment"]["dependency_lock_bytes"] == len(
        dependency_lock_bytes
    )
    assert hashlib.sha256(dependency_lock_bytes).hexdigest() == (
        EXPECTED_DEPENDENCY_LOCK_SHA256
    )
    assert compact["cohort"]["classification_counts"] == (
        EXPECTED_CLASSIFICATION_COUNTS
    )
    assert compact["cohort"]["quality_episode_oracle_gate_pass_object_count"] == 0
    assert compact["cohort"]["common_quality_candidate_gate_pass_object_count"] == 0
    assert compact["cohort"]["quality_oracle_available_episode_count"] == 28
    assert compact["cohort"]["quality_oracle_win_count"] == 6
    assert compact["cohort"]["quality_oracle_win_fraction"] == pytest.approx(6 / 28)
    assert compact["decision"]["registered_method_changed"] is False
    assert compact["decision"]["target_prefix_access_permitted"] is False
    assert compact["decision"]["target_future_access_permitted"] is False
    assert compact["information_boundary"]["calibration_outcomes_read"] is False
    assert compact["information_boundary"]["target_prefix_read"] is False
    assert compact["information_boundary"]["target_future_geometry_read"] is False
    assert compact["information_boundary"]["target_future_tactile_read"] is False

    full_by_object = {record["object_id"]: record for record in full["objects"]}
    compact_by_object = {
        record["object_id"]: record for record in compact["objects"]
    }
    assert set(compact_by_object) == set(full_by_object)
    for object_id, record in compact_by_object.items():
        assert record["classification"] == full_by_object[object_id]["classification"]


def test_source_failure_attribution_rejects_mutated_content() -> None:
    payload = analyze_source_failure_milestone(PROTOCOL, MILESTONE)
    tampered = deepcopy(payload)
    tampered["objects"][0]["recommended_next_test"] = "open the target"

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_source_failure_attribution(tampered)
