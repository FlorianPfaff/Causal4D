"""Scaffold, seal, and verify pre-acquisition readiness evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from causal4d.preacquisition_readiness import (
    GATE_PATHS,
    build_preacquisition_readiness,
    scaffold_preacquisition_readiness,
    seal_preacquisition_gate,
    write_preacquisition_readiness,
)


_VALID_BUT_INCOMPLETE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scaffold = subparsers.add_parser(
        "scaffold",
        help="write incomplete evidence templates without overwriting existing files",
    )
    scaffold.add_argument("repository_root")
    scaffold.add_argument("dataset_root")

    seal = subparsers.add_parser(
        "seal-gate",
        help="validate and atomically seal one completed operational gate",
    )
    seal.add_argument("repository_root")
    seal.add_argument("dataset_root")
    seal.add_argument("gate_id", choices=tuple(GATE_PATHS))
    seal.add_argument("--approved-by", required=True)
    seal.add_argument("--approved-at-utc")

    status = subparsers.add_parser(
        "status",
        help="derive whether the first confirmatory execution is permitted",
    )
    status.add_argument("repository_root")
    status.add_argument("dataset_root")
    status.add_argument("--output-json")
    status.add_argument("--verify-file-hashes", action="store_true")
    status.add_argument(
        "--require-ready",
        action="store_true",
        help="return exit code 3 when evidence is valid but incomplete",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "scaffold":
            result = scaffold_preacquisition_readiness(
                args.repository_root,
                args.dataset_root,
            )
        elif args.command == "seal-gate":
            result = seal_preacquisition_gate(
                args.repository_root,
                args.dataset_root,
                args.gate_id,
                approved_by=args.approved_by,
                approved_at_utc=args.approved_at_utc,
            )
        else:
            result = build_preacquisition_readiness(
                args.repository_root,
                args.dataset_root,
                verify_file_hashes=args.verify_file_hashes,
            )
            if args.output_json:
                write_preacquisition_readiness(args.output_json, result)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "valid": False,
                    "ready": False,
                    "passed": False,
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if args.command != "status":
        return 0
    if not result["valid"]:
        return 2
    if args.require_ready and not result["ready"]:
        return _VALID_BUT_INCOMPLETE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
