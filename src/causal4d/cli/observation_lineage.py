"""Validate or explicitly bind portable observation lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from causal4d.contracts import TwinBelief, load_contract, save_contract
from causal4d.observation_factor_lineage import (
    bind_twin_belief_observation_factor_lineage,
    load_observation_factor_lineage,
    validate_twin_belief_observation_factor_lineage,
)
from causal4d.observation_lineage import (
    bind_twin_belief_observation_lineage,
    load_observation_lineage,
    validate_twin_belief_observation_lineage,
)


def _add_twin_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("twin_belief", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="validate a marginalized ObservationBeliefV1 artifact",
    )
    validate.add_argument("observation_belief", type=Path)
    _add_twin_argument(validate)
    validate.add_argument("--require-bound", action="store_true")

    bind = subparsers.add_parser(
        "bind",
        help="bind a consumed ObservationBeliefV1 artifact",
    )
    bind.add_argument("observation_belief", type=Path)
    _add_twin_argument(bind)
    bind.add_argument("output_twin_belief", type=Path)
    bind.add_argument(
        "--confirm-observation-was-consumed",
        action="store_true",
        help=(
            "required acknowledgement that the estimator actually consumed "
            "this exact observation artifact"
        ),
    )

    validate_factor = subparsers.add_parser(
        "validate-factor-bundle",
        help="validate an exact Prob4D schema-v3 or schema-v4 factor bundle",
    )
    validate_factor.add_argument("factor_bundle_manifest", type=Path)
    _add_twin_argument(validate_factor)
    validate_factor.add_argument("--require-bound", action="store_true")

    bind_factor = subparsers.add_parser(
        "bind-factor-bundle",
        help="bind the exact Prob4D schema-v3 or schema-v4 factor bundle consumed",
    )
    bind_factor.add_argument("factor_bundle_manifest", type=Path)
    _add_twin_argument(bind_factor)
    bind_factor.add_argument("output_twin_belief", type=Path)
    bind_factor.add_argument(
        "--confirm-factor-bundle-was-consumed",
        action="store_true",
        help=(
            "required acknowledgement that the estimator consumed the exact "
            "manifest and payload pair"
        ),
    )
    return parser


def _load_twin_belief(path: Path) -> TwinBelief:
    artifact = load_contract(path)
    if not isinstance(artifact, TwinBelief):
        raise ValueError("twin_belief must contain a Causal4D TwinBelief")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = _load_twin_belief(args.twin_belief)

    if args.command in {"validate", "bind"}:
        lineage = load_observation_lineage(args.observation_belief)
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
    else:
        lineage = load_observation_factor_lineage(args.factor_bundle_manifest)
        if args.command == "validate-factor-bundle":
            result = validate_twin_belief_observation_factor_lineage(
                artifact,
                lineage,
                require_bound=args.require_bound,
            )
        else:
            if not args.confirm_factor_bundle_was_consumed:
                raise ValueError(
                    "binding requires "
                    "--confirm-factor-bundle-was-consumed; validation alone "
                    "does not establish estimator provenance"
                )
            bound = bind_twin_belief_observation_factor_lineage(
                artifact,
                lineage,
            )
            save_contract(args.output_twin_belief, bound)
            result = validate_twin_belief_observation_factor_lineage(
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
