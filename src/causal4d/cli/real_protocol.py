"""Generate, inspect, and validate the Causal4D multi-action real protocol."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from causal4d.operator_bound_real_evidence import (
    build_operator_bound_real_evidence_status as build_real_evidence_status,
    validate_operator_bound_real_dataset as validate_real_dataset_v2,
)
from causal4d.real_evidence_contract_v2 import (
    scaffold_real_evidence_v2_templates,
    write_real_evidence_status,
)
from causal4d.real_protocol import (
    build_same_object_real_protocol,
    load_protocol,
    scaffold_dataset,
    validate_protocol,
    write_acquisition_schedule,
    write_protocol,
)

INCOMPLETE_EVIDENCE_EXIT_CODE = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="write the deterministic preregistered protocol",
    )
    generate.add_argument("output_json")
    generate.add_argument(
        "--schedule-csv",
        help="also write the locked operator acquisition order",
    )

    validate = subparsers.add_parser(
        "validate-protocol",
        help="validate design balance, hashes, and split boundaries",
    )
    validate.add_argument("protocol_json")

    scaffold = subparsers.add_parser(
        "scaffold",
        help="create non-overwriting session, execution, and evidence templates",
    )
    scaffold.add_argument("protocol_json")
    scaffold.add_argument("output_root")

    status = subparsers.add_parser(
        "status",
        help="report acquisition, evidence, analysis, and claim readiness",
    )
    status.add_argument("protocol_json")
    status.add_argument("dataset_root")
    status.add_argument(
        "--repository-root",
        help=(
            "clean checkout at the sealed Causal4D commit; required to verify "
            "method_freeze.json and operator_registry.json before claim readiness"
        ),
    )
    status.add_argument(
        "--verify-file-hashes",
        action="store_true",
        help="rehash every registered artifact before declaring claim readiness",
    )
    status.add_argument(
        "--output-json",
        help="atomically write the complete machine-readable status report",
    )
    status.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "return exit code 3 until identities, freeze, registration, timebase, "
            "sessions, executions, and hashes are claim-ready"
        ),
    )

    dataset = subparsers.add_parser(
        "validate-dataset",
        help="validate a completed version-2 acquisition evidence tree",
    )
    dataset.add_argument("protocol_json")
    dataset.add_argument("dataset_root")
    dataset.add_argument(
        "--repository-root",
        help="clean checkout at the method-freeze Causal4D commit",
    )
    dataset.add_argument(
        "--skip-file-hashes",
        action="store_true",
        help="validate structure without declaring the evidence claim-ready",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exit_code = 0
    try:
        if args.command == "generate":
            protocol = build_same_object_real_protocol()
            output = write_protocol(args.output_json, protocol)
            result = {**validate_protocol(protocol), "output": str(output.resolve())}
            if args.schedule_csv:
                schedule = write_acquisition_schedule(args.schedule_csv, protocol)
                result["schedule_csv"] = str(schedule.resolve())
        elif args.command == "validate-protocol":
            result = validate_protocol(load_protocol(args.protocol_json))
        elif args.command == "scaffold":
            protocol = load_protocol(args.protocol_json)
            result = scaffold_dataset(protocol, args.output_root)
            result = {
                **result,
                **scaffold_real_evidence_v2_templates(protocol, args.output_root),
            }
        elif args.command == "status":
            result = build_real_evidence_status(
                load_protocol(args.protocol_json),
                args.dataset_root,
                repository_root=args.repository_root,
                verify_file_hashes=args.verify_file_hashes,
            )
            if args.output_json:
                output = write_real_evidence_status(args.output_json, result)
                result = {**result, "output": str(output.resolve())}
            if args.require_complete and not result["claim_ready"]:
                exit_code = INCOMPLETE_EVIDENCE_EXIT_CODE
        else:
            result = validate_real_dataset_v2(
                load_protocol(args.protocol_json),
                args.dataset_root,
                repository_root=args.repository_root,
                verify_files=not args.skip_file_hashes,
            )
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
