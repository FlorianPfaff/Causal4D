from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from causal4d.contact_concentration_diagnostic import (
    ConcentrationPolicy,
    _aggregate,
    _policy_comparison,
    scale_probability_weights,
    write_contact_concentration_diagnostic,
)


def test_logit_scaling_preserves_support_and_softens_or_sharpens() -> None:
    weights = np.asarray([0.9, 0.1, 0.0])

    softened = scale_probability_weights(weights, 0.5)
    unchanged = scale_probability_weights(weights, 1.0)
    sharpened = scale_probability_weights(weights, 2.0)

    assert softened[2] == 0.0
    assert sharpened[2] == 0.0
    assert softened[0] < weights[0]
    assert sharpened[0] > weights[0]
    np.testing.assert_array_equal(unchanged, weights)
    assert np.sum(softened) == pytest.approx(1.0)
    assert np.sum(sharpened) == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [0.0, -1.0, np.inf, np.nan])
def test_logit_scaling_rejects_invalid_scales(scale: float) -> None:
    with pytest.raises(ValueError, match="logit_scale"):
        scale_probability_weights(np.asarray([0.5, 0.5]), scale)


@pytest.mark.parametrize(
    "weights, message",
    [
        (np.asarray([]), "nonempty"),
        (np.asarray([0.5, np.nan]), "finite"),
        (np.asarray([1.1, -0.1]), "nonnegative"),
        (np.asarray([0.4, 0.4]), "sum to one"),
    ],
)
def test_logit_scaling_rejects_invalid_weights(
    weights: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        scale_probability_weights(weights, 1.0)


def test_concentration_policy_requires_unique_positive_candidates() -> None:
    policy = ConcentrationPolicy("expanded", (0.5, 1.0, 2.0))
    assert policy.as_dict() == {
        "name": "expanded",
        "logit_scales": [0.5, 1.0, 2.0],
    }

    with pytest.raises(ValueError, match="unique"):
        ConcentrationPolicy("duplicate", (1.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        ConcentrationPolicy("invalid", (0.0, 1.0))


def _row(
    policy: str,
    *,
    correct: bool,
    confidence: float,
    brier: float,
    rmse: float,
) -> dict[str, object]:
    return {
        "policy": policy,
        "world_condition": "shifted_contact",
        "node_correct": correct,
        "node_confidence": confidence,
        "node_truth_probability": 0.7 if correct else 0.2,
        "node_brier": brier,
        "node_credible_covered": correct,
        "trajectory_rmse_m": rmse,
    }


def test_aggregate_and_policy_comparison_keep_metric_signs_explicit() -> None:
    registered_rows = [
        _row(
            "registered_candidates",
            correct=True,
            confidence=0.95,
            brier=0.12,
            rmse=0.0012,
        ),
        _row(
            "registered_candidates",
            correct=False,
            confidence=0.90,
            brier=0.44,
            rmse=0.0014,
        ),
    ]
    expanded_rows = [
        _row(
            "expanded_with_softening",
            correct=True,
            confidence=0.80,
            brier=0.10,
            rmse=0.0012,
        ),
        _row(
            "expanded_with_softening",
            correct=False,
            confidence=0.65,
            brier=0.30,
            rmse=0.0014,
        ),
    ]
    registered = {
        "policy": "registered_candidates",
        "world_condition": "shifted_contact",
        **_aggregate(registered_rows),
    }
    expanded = {
        "policy": "expanded_with_softening",
        "world_condition": "shifted_contact",
        **_aggregate(expanded_rows),
    }

    comparison = _policy_comparison(
        [registered, expanded],
        registered_policy="registered_candidates",
        expanded_policy="expanded_with_softening",
    )

    assert comparison[0]["expanded_minus_registered_mean_brier"] < 0.0
    assert comparison[0]["expanded_minus_registered_calibration_error"] < 0.0
    assert comparison[0]["expanded_minus_registered_trajectory_rmse_m"] == (
        pytest.approx(0.0)
    )


def test_writer_records_exact_payload_identities(tmp_path: Path) -> None:
    result = {
        "schema_version": 1,
        "artifact_kind": "Causal4DContactConcentrationDiagnostic",
        "seeds": [200],
        "policies": [],
        "aggregate": [],
        "by_topology": [],
        "comparison": [],
        "claim_boundary": "exploratory",
        "rows": [
            {
                "seed": 200,
                "policy": "registered_candidates",
                "trajectory_rmse_m": 0.001,
            }
        ],
        "selection_rows": [
            {
                "seed": 200,
                "policy": "registered_candidates",
                "selected_logit_scale": 1.0,
            }
        ],
    }

    paths = write_contact_concentration_diagnostic(result, tmp_path)
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))

    for name, record in manifest["artifacts"].items():
        payload = tmp_path / name
        assert record["bytes"] == payload.stat().st_size
        assert record["sha256"] == hashlib.sha256(payload.read_bytes()).hexdigest()
