"""Convert one explicitly trusted legacy target pickle into a safe artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def _load_runtime_dependencies() -> None:
    global PhysicalPosterior
    global import_legacy_physical_target
    global load_contract
    global save_held_out_physical_target

    from causal4d.contracts import PhysicalPosterior, load_contract
    from causal4d.held_out_target import save_held_out_physical_target
    from causal4d.legacy_physical_target import import_legacy_physical_target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an explicitly trusted, SHA-256-verified final_data.pkl into "
            "a non-pickled held-out physical target. This is a migration command; "
            "the claim-bearing evaluator never opens pickle inputs."
        )
    )
    parser.add_argument("physical_posterior_npz")
    parser.add_argument("final_data_pickle")
    parser.add_argument("output_target_npz")
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help="explicitly acknowledge that unpickling can execute arbitrary code",
    )
    parser.add_argument(
        "--expected-sha256",
        required=True,
        help="lowercase SHA-256 digest of the exact trusted pickle bytes",
    )
    parser.add_argument(
        "--source-revision",
        default="legacy-final-data-v1",
        help="dataset, acquisition, or producer revision bound into the target",
    )
    parser.add_argument(
        "--source-artifact-id",
        help="optional upstream dataset or acquisition artifact identity",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="fail rather than replacing an existing target artifact",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_runtime_dependencies()
    artifact = load_contract(args.physical_posterior_npz)
    if not isinstance(artifact, PhysicalPosterior):
        raise TypeError("physical_posterior_npz must contain a PhysicalPosterior")
    target = import_legacy_physical_target(
        artifact,
        args.final_data_pickle,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
        expected_sha256=args.expected_sha256,
        source_revision=args.source_revision,
        source_artifact_id=args.source_artifact_id,
    )
    save_held_out_physical_target(
        args.output_target_npz,
        target,
        overwrite=not args.no_overwrite,
    )
    summary = target.summary()
    summary["output"] = str(Path(args.output_target_npz).resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
