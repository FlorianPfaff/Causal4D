#!/usr/bin/env python3
"""Run the fresh-panel source-only contact-correlation diagnostic."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import math
from pathlib import Path

from causal4d.benchmark import CounterfactualBenchmarkConfig
from causal4d.contact_correlation_diagnostic import (
    CorrelationDiagnosticConfig,
    run_contact_correlation_diagnostic,
    write_contact_correlation_diagnostic,
)
from causal4d.contact_inference import LatentContactConfig


def _parse_seeds(value: str) -> tuple[int, ...]:
    text = value.strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("seed range must use START:STOP")
        start, stop = map(int, parts)
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


def _parse_int_grid(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("integer grid values are required") from error
    if (
        not values
        or any(item < 2 for item in values)
        or len(set(values)) != len(values)
    ):
        raise argparse.ArgumentTypeError(
            "frame block sizes must be unique integers of at least two"
        )
    return values


def _parse_unit_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("grid values must be numeric") from error
    if (
        not values
        or any(not math.isfinite(item) or not 0.0 < item <= 1.0 for item in values)
        or len(set(values)) != len(values)
    ):
        raise argparse.ArgumentTypeError(
            "grid values must be unique finite numbers in (0, 1]"
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=_parse_seeds("300:320"),
    )
    parser.add_argument("--frames", type=int, default=56)
    parser.add_argument("--training-repeats", type=int, default=2)
    parser.add_argument("--parameter-grid-count", type=int, default=5)
    parser.add_argument("--contact-parameter-particles", type=int, default=12)
    parser.add_argument("--observation-fraction", type=float, default=0.20)
    parser.add_argument("--observation-noise-mm", type=float, default=1.5)
    parser.add_argument(
        "--frame-block-sizes",
        type=_parse_int_grid,
        default=_parse_int_grid("2,3,4"),
    )
    parser.add_argument(
        "--whitening-shrinkages",
        type=_parse_unit_grid,
        default=_parse_unit_grid("0.10,0.25,0.50,0.75"),
    )
    parser.add_argument(
        "--generalized-bayes-rates",
        type=_parse_unit_grid,
        default=_parse_unit_grid("0.25,0.50,0.75,1.00"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    diagnostic = CorrelationDiagnosticConfig(
        frame_block_sizes=args.frame_block_sizes,
        whitening_shrinkages=args.whitening_shrinkages,
        generalized_bayes_rates=args.generalized_bayes_rates,
    )
    result = run_contact_correlation_diagnostic(
        args.seeds,
        benchmark_config=benchmark,
        contact_config=contact,
        diagnostic_config=diagnostic,
    )
    paths = write_contact_correlation_diagnostic(result, args.output_dir)
    print(
        json.dumps(
            {
                "seeds": result["seeds"],
                "aggregate": result["aggregate"],
                "comparison": result["comparison"],
                "decision": result["decision"],
                "any_promotion_candidate": result["any_promotion_candidate"],
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
