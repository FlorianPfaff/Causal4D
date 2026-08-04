#!/usr/bin/env python3
"""Run the topology-conditioned contact-prefix covariance diagnostic."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import math
from pathlib import Path

from causal4d.benchmark import CounterfactualBenchmarkConfig
from causal4d.contact_inference import LatentContactConfig
from causal4d.contact_topology_covariance_diagnostic import (
    TopologyCovarianceDiagnosticConfig,
    run_contact_topology_covariance_diagnostic,
    write_contact_topology_covariance_diagnostic,
)


def _parse_seeds(value: str) -> tuple[int, ...]:
    text = value.strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("seed range must use START:STOP")
        try:
            start, stop = map(int, parts)
        except ValueError as error:
            raise argparse.ArgumentTypeError("seed bounds must be integers") from error
        if start < 0 or stop <= start:
            raise argparse.ArgumentTypeError(
                "seed range must satisfy 0 <= START < STOP"
            )
        return tuple(range(start, stop))
    try:
        seeds = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be integers") from error
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be nonnegative and unique")
    return seeds


def _parse_unit_grid(value: str, *, allow_zero: bool) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("grid values must be numeric") from error
    lower_invalid = (
        any(item < 0.0 for item in values)
        if allow_zero
        else any(item <= 0.0 for item in values)
    )
    if (
        not values
        or lower_invalid
        or any(not math.isfinite(item) or item > 1.0 for item in values)
        or len(set(values)) != len(values)
    ):
        interval = "[0, 1]" if allow_zero else "(0, 1]"
        raise argparse.ArgumentTypeError(
            f"grid values must be unique finite numbers in {interval}"
        )
    return values


def _parse_shared_grid(value: str) -> tuple[float, ...]:
    return _parse_unit_grid(value, allow_zero=True)


def _parse_identity_grid(value: str) -> tuple[float, ...]:
    values = _parse_unit_grid(value, allow_zero=False)
    if 1.0 not in values:
        raise argparse.ArgumentTypeError(
            "identity shrinkages must contain 1.0 as the no-op candidate"
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--development-seeds",
        type=_parse_seeds,
        default=_parse_seeds("300:320"),
    )
    parser.add_argument(
        "--evaluation-seeds",
        type=_parse_seeds,
        default=_parse_seeds("400:420"),
    )
    parser.add_argument("--frames", type=int, default=56)
    parser.add_argument("--training-repeats", type=int, default=2)
    parser.add_argument("--parameter-grid-count", type=int, default=5)
    parser.add_argument("--contact-parameter-particles", type=int, default=12)
    parser.add_argument("--observation-fraction", type=float, default=0.20)
    parser.add_argument("--observation-noise-mm", type=float, default=1.5)
    parser.add_argument(
        "--shared-correlation-weights",
        type=_parse_shared_grid,
        default=_parse_shared_grid("0,0.25,0.50,0.75,1.00"),
    )
    parser.add_argument(
        "--identity-shrinkages",
        type=_parse_identity_grid,
        default=_parse_identity_grid("0.10,0.25,0.50,0.75,1.00"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if set(args.development_seeds) & set(args.evaluation_seeds):
        raise SystemExit("development and evaluation seeds must be disjoint")
    benchmark = CounterfactualBenchmarkConfig(
        frame_count=args.frames,
        training_repeats=args.training_repeats,
        parameter_grid_count=args.parameter_grid_count,
    )
    contact = LatentContactConfig(
        observation_fraction=args.observation_fraction,
        observation_noise_std_m=args.observation_noise_mm / 1000.0,
        parameter_particle_count=args.contact_parameter_particles,
    )
    diagnostic = TopologyCovarianceDiagnosticConfig(
        shared_correlation_weights=args.shared_correlation_weights,
        identity_shrinkages=args.identity_shrinkages,
    )
    result = run_contact_topology_covariance_diagnostic(
        args.development_seeds,
        args.evaluation_seeds,
        benchmark_config=benchmark,
        contact_config=contact,
        diagnostic_config=diagnostic,
    )
    paths = write_contact_topology_covariance_diagnostic(
        result,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "development_seeds": result["development_seeds"],
                "evaluation_seeds": result["evaluation_seeds"],
                "selected_global_candidate": result["selected_global_candidate"],
                "selected_topology_candidates": result["selected_topology_candidates"],
                "aggregate": result["aggregate"],
                "comparison": result["comparison"],
                "decision": result["decision"],
                "topology_hypothesis_supported": result[
                    "topology_hypothesis_supported"
                ],
                "artifacts": paths,
                "claim_boundary": result["claim_boundary"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
