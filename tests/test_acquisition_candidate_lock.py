from __future__ import annotations

import hashlib
import json
from pathlib import Path

from causal4d.real_experiment_freeze import (
    ACQUISITION_CANDIDATE_PATH,
    BPT_PIN_PATH,
    REQUIRED_LOCKED_PATHS,
)


ROOT = Path(__file__).resolve().parents[1]


def _canonical_sha256(values: dict[str, object]) -> str:
    payload = dict(values)
    payload.pop("candidate_sha256")
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_acquisition_candidate_is_content_addressed_and_freeze_bound() -> None:
    candidate = json.loads((ROOT / ACQUISITION_CANDIDATE_PATH).read_text())

    assert candidate["candidate_sha256"] == _canonical_sha256(candidate)
    assert ACQUISITION_CANDIDATE_PATH in REQUIRED_LOCKED_PATHS
    assert candidate["information_boundary"] == {
        "allowed_post_intervention_prefix_frames": 6,
        "confirmation_outcomes_used": False,
        "source_or_target_outcomes_used_for_selection": False,
        "target_outcomes_may_select_method_or_hyperparameters": False,
    }
    assert candidate["observation_path"]["prob4d"]["used"] is False
    assert candidate["semantic_path"]["molmomotion_beta"] == 0


def test_acquisition_candidate_and_provider_ci_use_one_bpt_revision() -> None:
    candidate = json.loads((ROOT / ACQUISITION_CANDIDATE_PATH).read_text())
    bpt_pin = (ROOT / BPT_PIN_PATH).read_text(encoding="utf-8").strip()

    assert candidate["physical_model"]["bayesian_phystwin_commit_sha"] == bpt_pin
    for relative in (".github/workflows/ci.yml", ".github/workflows/merge-gate.yml"):
        workflow = (ROOT / relative).read_text(encoding="utf-8")
        assert f"ref: {bpt_pin}" in workflow
