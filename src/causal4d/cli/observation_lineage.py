"""Validate or explicitly bind ObservationBeliefV1 lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from causal4d.contracts import TwinBelief, load_contract, save_contract
from causal4d.observation_lineage import (
    bind_twin_belief_observation_lineage,
    load_observation_lineage,
    validate_twin_belief_observation_lineage,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("observation_belief", type=Path)
    validate.add_argument("twin_belief", type=Path)
    validate.add_argument("--require-bound", action="store_true")

    bind = subparsers.add_parser("bind")
    bind.add_argument("observation_belief", type=Path)
    bind.add_argument("twin_belief", type=Path)
    bind.add_argument("output_twin_belief", type=Path)
    bind.add_argument(
        "--confirm-observation-was-consumed",
        action="store_true",
        help=(
            "required acknowledgement that the estimator actually consumed "
            "this exact observation artifact"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lineage = load_observation_lineage(args.observation_belief)
    artifact = load_contract(args.twin_belief)
    if not isinstance(artifact, TwinBelief):
        raise ValueError("twin_belief must contain a Causal4D TwinBelief")

    if args.command == "validate":
        result = validate_twin_belief_observation_lineage(
            artifact,
            lineage,
            require_bound=args.require_bound,
        )
    else:
        if not args.confirm_observation_was_consumed:
            raise ValueError(
                "binding requires --confirm-observation-was-consumed; "
                "validation alone does not establish estimator provenance"
            )
        bound = bind_twin_belief_observation_lineage(artifact, lineage)
        save_contract(args.output_twin_belief, bound)
        result = validate_twin_belief_observation_lineage(
            bound,
            lineage,
            require_bound=True,
        )
        result["output"] = str(args.output_twin_belief.resolve())
        result["source_twin_belief_id"] = artifact.artifact_id
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
