from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT / "milestones" / "deform360-prefix-kinematics-v1" / "summary.json"
)


def test_completed_prefix_kinematics_milestone_is_negative_and_target_closed() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert payload["artifact_kind"] == "Deform360SourcePrefixKinematicsSummary"
    assert payload["schema_version"] == 1
    assert payload["status"] == "source-only-diagnostic-complete-negative"
    assert payload["workflow"] == {
        "repository": "IPS-Stuttgart/Causal4D",
        "run_id": 30972643551,
        "head_sha": "77caf44dbd749e37b34dbecf47cba03799d4289f",
        "artifact_id": 8917112270,
        "artifact_name": (
            "deform360-prefix-kinematics-evidence-"
            "77caf44dbd749e37b34dbecf47cba03799d4289f"
        ),
        "artifact_archive_sha256": (
            "77955b80f4ef5ff1b9d796d3d816a9dde1342f471120d4d9724d1d82454ba9f4"
        ),
    }
    assert payload["artifacts"]["result_sha256"] == (
        "3f1eaa75800cd7bb24d3be82da112ec5c6ab93d2873508cb147a1e2de3a323b3"
    )
    assert payload["cohort"]["episode_count"] == 30
    assert payload["cohort"]["baseline_reproduction_episode_count"] == 30
    assert payload["cohort"]["baseline_reproduction_passed"] is True
    assert payload["runtime"]["recorded_numpy"] == "2.5.1"
    assert payload["runtime"]["conditional_reproduction_numpy"] == "1.26.4"
    assert payload["runtime"]["zero_baseline_reproduction_required"] is True

    zero = payload["policies"]["zero_v1"]
    global_translation = payload["policies"]["global_contact_translation_v1"]
    graph_harmonic = payload["policies"]["graph_harmonic_contact_v1"]
    assert zero["mean_chamfer_mm"] == pytest.approx(58.040161213749776)
    assert global_translation["relative_improvement_vs_zero"] == pytest.approx(
        -0.004592926475892943
    )
    assert global_translation["win_fraction_vs_zero"] == pytest.approx(0.4)
    assert graph_harmonic["relative_improvement_vs_zero"] == pytest.approx(
        -0.001754550461647353
    )
    assert graph_harmonic["win_fraction_vs_zero"] == pytest.approx(0.3)
    assert graph_harmonic["quality_valid_episode_count"] == (
        zero["quality_valid_episode_count"]
    )

    decision = payload["decision"]
    assert decision["passed"] is False
    assert decision["initial_velocity_is_dominant_general_repair"] is False
    assert decision["registered_method_changed"] is False
    assert decision["target_prefix_access_permitted"] is False
    assert decision["target_future_access_permitted"] is False
