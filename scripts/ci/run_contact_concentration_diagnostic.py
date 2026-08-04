#!/usr/bin/env python3
"""Run the fresh-seed source-only contact concentration diagnostic."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from causal4d.benchmark import CounterfactualBenchmarkConfig
from causal4d.contact_concentration_diagnostic import (
    run_contact_concentration_diagnostic,
    write_contact_concentration_diagnostic,
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
            raise argparse.ArgumentTypeError("seed range must satisfy 0 <= START < STOP")
        return tuple(range(start, stop))
    try:
        seeds = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be integers") from error
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be nonnegative and unique")
    return seeds


def _parse_scales(value: str) -> tuple[float, ...]:
    try:
        scales = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError("logit scales must be numeric") from error
    if not scales:
        raise argparse.ArgumentTypeError("at least one softening scale is required")
    return scales


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default="200:220")
    parser.add_argument("--frames", type=int, default=56)
    parser.add_argument("--training-repeats", type=int, default=2)
    parser.add_argument("--parameter-grid-count", type=int, default=5)
    parser.add_argument("--contact-parameter-particles", type=int, default=12)
    parser.add_argument("--observation-fraction", type=float, default=0.20)
    parser.add_argument("--observation-noise-mm", type=float, default=1.5)
    parser.add_argument(
        "--softening-logit-scales",
        type=_parse_scales,
        default="0.25,0.50,0.75",
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
    result = run_contact_concentration_diagnostic(
        args.seeds,
        benchmark_config=benchmark,
        contact_config=contact,
        softening_logit_scales=args.softening_logit_scales,
    )
    paths = write_contact_concentration_diagnostic(result, args.output_dir)
    print(
        json.dumps(
            {
                "seeds": result["seeds"],
                "policies": result["policies"],
                "aggregate": result["aggregate"],
                "comparison": result["comparison"],
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
