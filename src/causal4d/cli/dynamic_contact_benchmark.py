"""Controlled delayed-contact benchmark for dynamic Causal4D interventions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.dynamic_contact import (
    ContactRegime,
    ContactTransitionConfig,
    DynamicContactInferenceConfig,
    DynamicContactPathBank,
    enumerate_contact_paths,
    first_activation_frame,
    infer_dynamic_contact_posterior,
)


def _simulate_path(
    regime_path: np.ndarray,
    command_activation: np.ndarray,
    *,
    sticking_step_m: float = 0.006,
    slipping_step_m: float = 0.003,
) -> np.ndarray:
    frame_count = len(regime_path)
    trajectory = np.zeros((frame_count, 1, 3), dtype=float)
    for frame in range(1, frame_count):
        regime = int(regime_path[frame])
        if regime == ContactRegime.STICKING:
            step = sticking_step_m * command_activation[frame]
        elif regime == ContactRegime.SLIPPING:
            step = slipping_step_m * command_activation[frame]
        else:
            step = 0.0
        trajectory[frame] = trajectory[frame - 1]
        trajectory[frame, 0, 0] += step
    return trajectory


def delayed_contact_case(
    *,
    seed: int = 0,
    frame_count: int = 24,
    prefix_frame_count: int = 6,
) -> dict[str, Any]:
    """Run one action-known contact-onset case without reading future observations."""

    if not 2 <= prefix_frame_count < frame_count - 2:
        raise ValueError("prefix_frame_count must leave a meaningful future")
    rng = np.random.default_rng(seed)
    activation = np.zeros(frame_count, dtype=float)
    activation[prefix_frame_count:] = 1.0
    transition = ContactTransitionConfig(
        activation_floor=0.0,
        activation_gain=0.96,
        slip_floor=0.0,
        slip_change_gain=0.0,
        release_floor=0.0,
        release_gain=0.0,
        slip_recovery_probability=0.0,
        reattachment_gain=0.0,
        maximum_paths=32,
    )
    prior = enumerate_contact_paths(activation, config=transition)
    trajectories = np.stack(
        [_simulate_path(path, activation) for path in prior.regime_paths],
        axis=0,
    )
    bank = DynamicContactPathBank(
        path_ids=prior.path_ids,
        regime_paths=prior.regime_paths,
        trajectories_m=trajectories,
        prior_weights=prior.weights,
        base_variance_m2=(0.0005**2),
    )

    truth_path = np.full(frame_count, ContactRegime.INACTIVE, dtype=np.int8)
    truth_path[prefix_frame_count:] = ContactRegime.STICKING
    truth = _simulate_path(truth_path, activation)
    observations = truth + rng.normal(scale=0.0005, size=truth.shape)
    inference = DynamicContactInferenceConfig(
        observation_scale_m=0.001,
        degrees_of_freedom=4.0,
        likelihood_power=1.0,
        dynamic_likelihood_weight=0.25,
        switch_variance_m2=0.00075**2,
        command_change_variance_m2=0.00025**2,
        ood_variance_m2=0.0,
        confidence_level=0.90,
    )
    posterior = infer_dynamic_contact_posterior(
        bank,
        observations,
        prefix_frame_count=prefix_frame_count,
        command_activation=activation,
        config=inference,
    )

    static_inactive = _simulate_path(
        np.full(frame_count, ContactRegime.INACTIVE, dtype=np.int8),
        activation,
    )
    future = slice(prefix_frame_count, frame_count)

    def rmse(prediction: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(prediction[future] - truth[future]))))

    static_rmse = rmse(static_inactive)
    dynamic_rmse = rmse(posterior.mean_m)
    oracle_rmse = rmse(truth)
    active_frames = [first_activation_frame(path) for path in prior.regime_paths]
    onset_values = np.asarray(
        [frame_count if value is None else value for value in active_frames],
        dtype=float,
    )
    expected_onset = float(np.dot(posterior.weights, onset_values))
    truth_covered = (truth >= posterior.interval_lower_m) & (
        truth <= posterior.interval_upper_m
    )
    future_coverage = float(np.mean(truth_covered[future]))
    improvement = 1.0 - dynamic_rmse / static_rmse
    return {
        "protocol": "delayed_contact_onset_v1",
        "seed": seed,
        "frame_count": frame_count,
        "prefix_frame_count": prefix_frame_count,
        "future_observations_read": posterior.metadata["future_observations_read"],
        "path_count": len(prior.weights),
        "retained_prior_mass": prior.retained_prior_mass,
        "transition_config": asdict(transition),
        "inference_config": asdict(inference),
        "static_prefix_persistence_rmse_m": static_rmse,
        "dynamic_contact_rmse_m": dynamic_rmse,
        "oracle_rmse_m": oracle_rmse,
        "relative_rmse_improvement": improvement,
        "true_contact_onset_frame": prefix_frame_count,
        "posterior_expected_contact_onset_frame": expected_onset,
        "contact_onset_absolute_error_frames": abs(
            expected_onset - prefix_frame_count
        ),
        "future_coverage": future_coverage,
        "map_path_id": posterior.map_path_id,
        "maximum_switch_probability": float(np.max(posterior.switch_probability)),
        "gates": {
            "dynamic_beats_static_by_50_percent": improvement >= 0.50,
            "onset_error_at_most_one_frame": (
                abs(expected_onset - prefix_frame_count) <= 1.0
            ),
            "prefix_only_inference": (
                posterior.metadata["future_observations_read"] == 0
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=24)
    parser.add_argument("--prefix-frame-count", type=int, default=6)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--require-gates", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = delayed_contact_case(
        seed=args.seed,
        frame_count=args.frame_count,
        prefix_frame_count=args.prefix_frame_count,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is None:
        print(payload, end="")
    else:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    if args.require_gates and not all(result["gates"].values()):
        raise SystemExit("dynamic-contact benchmark failed one or more gates")


if __name__ == "__main__":
    main()
