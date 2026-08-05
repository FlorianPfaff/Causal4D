"""Evaluate a PhysicalPosterior against an identity-bound held-out target."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence


def _load_runtime_dependencies() -> None:
    """Load integrations only after argparse handles ``--help``."""

    global PhysicalPosterior
    global build_physical_counterfactual_evaluation_record
    global evaluate_beta_zero_physical_posterior
    global load_contract
    global load_held_out_physical_target
    global save_physical_counterfactual_evaluation_record

    from causal4d.contracts import PhysicalPosterior, load_contract
    from causal4d.held_out_target import load_held_out_physical_target
    from causal4d.physical_evaluation_record import (
        build_physical_counterfactual_evaluation_record,
        save_physical_counterfactual_evaluation_record,
    )
    from causal4d.physical_validation import evaluate_beta_zero_physical_posterior


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a discrepancy-aware physical posterior at beta=0 against "
            "a non-pickled, content-addressed held-out target artifact."
        )
    )
    parser.add_argument("physical_posterior_npz")
    parser.add_argument("held_out_target_npz")
    parser.add_argument("output_json")
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--confidence-level", type=float, default=0.90)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="fail rather than replacing an existing evaluation artifact",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_runtime_dependencies()
    artifact = load_contract(args.physical_posterior_npz)
    if not isinstance(artifact, PhysicalPosterior):
        raise TypeError("physical_posterior_npz must contain a PhysicalPosterior")
    target = load_held_out_physical_target(args.held_out_target_npz)
    target.require_compatible_physical_posterior(artifact)

    metrics = evaluate_beta_zero_physical_posterior(
        artifact,
        target.positions_m,
        mask=target.validity_mask,
        start_frame=args.start_frame,
        confidence_level=args.confidence_level,
    )
    result = build_physical_counterfactual_evaluation_record(
        artifact,
        target,
        metrics,
        start_frame=args.start_frame,
        confidence_level=args.confidence_level,
    )
    save_physical_counterfactual_evaluation_record(
        args.output_json,
        result,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
