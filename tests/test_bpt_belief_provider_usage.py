from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("bayesian_phystwin")

import causal4d.bpt_belief as belief_module
from bayesian_phystwin.causal4d_belief_provider_v1 import (
    FixedBayesianAnchorConfigV1,
    RobustEndpointPosteriorV1,
)
from causal4d.bpt_belief import (
    BPTBeliefExportConfig,
    build_twin_belief_from_replays,
)
from causal4d.contracts import build_causal_context


def test_belief_export_validates_and_uses_fixed_anchor_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    received: dict[str, Any] = {}

    def require_provider():
        events.append("validate-provider")
        return object()

    def infer_endpoint(
        residual_m: np.ndarray,
        valid: np.ndarray,
        *,
        end_frame: int,
        config: FixedBayesianAnchorConfigV1,
    ) -> RobustEndpointPosteriorV1:
        events.append("infer-endpoint")
        received["residual"] = residual_m.copy()
        received["valid"] = valid.copy()
        received["end_frame"] = end_frame
        received["config"] = config
        return RobustEndpointPosteriorV1(
            mean_m=np.asarray([[0.002, 0.0, 0.0]]),
            variance_m2=np.asarray([4e-6]),
            final_nominal_probability=np.asarray([0.8]),
            update_count=np.asarray([2], dtype=np.int64),
        )

    monkeypatch.setattr(
        belief_module,
        "require_bayesian_phystwin_belief_provider",
        require_provider,
    )
    monkeypatch.setattr(
        belief_module,
        "infer_fixed_bayesian_anchor_endpoint",
        infer_endpoint,
    )
    monkeypatch.setattr(
        belief_module,
        "build_lift_map",
        lambda *args, **kwargs: (
            np.asarray([[0]], dtype=np.int64),
            np.asarray([[1.0]]),
        ),
    )

    frame_count = 4
    intervention_frame = 3
    observed = np.zeros((frame_count, 1, 3), dtype=float)
    replay = np.zeros((1, frame_count, 2, 3), dtype=float)
    replay[:, :, 1] = replay[:, :, 0]
    velocity = np.zeros_like(replay)
    valid = np.ones((frame_count, 1), dtype=bool)
    actions = np.zeros((frame_count, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="belief-provider-v1",
        case_id="synthetic",
        observations=observed,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=intervention_frame,
    )
    settings = BPTBeliefExportConfig(interpolation_neighbors=1)

    belief = build_twin_belief_from_replays(
        context=context,
        replay_positions_m=replay,
        replay_velocities_mps=velocity,
        observed_positions_m=observed,
        observed_valid=valid,
        theta=np.asarray([[0.0]]),
        theta_names=("parameter",),
        weights=np.asarray([1.0]),
        config=settings,
    )

    assert events == ["validate-provider", "infer-endpoint"]
    assert received["end_frame"] == intervention_frame
    assert received["residual"].shape == (intervention_frame, 1, 3)
    assert received["valid"].shape == (intervention_frame, 1)
    assert received["config"] == settings.fixed_anchor_config()
    assert belief.metadata["particle_update_counts"] == [2]
    assert belief.metadata["particle_mean_final_inlier_probability"] == [0.8]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"process_std_m": float("nan")},
        {"observation_std_m": float("inf")},
        {"interpolation_neighbors": True},
        {"maximum_discrepancy_m": float("nan")},
    ],
)
def test_belief_export_config_rejects_nonfinite_or_boolean_settings(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        BPTBeliefExportConfig(**kwargs)
