"""Scaffold, seal, and verify pre-acquisition readiness evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from causal4d import preacquisition_operator_flow as _operator_flow
from causal4d.operator_registry import (
    scaffold_operator_registry,
    seal_operator_registry,
)
from causal4d.preacquisition_next_action_validation import (
    validate_preacquisition_next_action_report,
    write_preacquisition_next_action_validation,
)
from causal4d.preacquisition_readiness import (
    GATE_PATHS,
    build_preacquisition_readiness,
    scaffold_preacquisition_readiness,
    seal_preacquisition_gate,
    write_preacquisition_readiness,
)
from causal4d.preacquisition_source_panel_control import (
    build_source_panel_status,
    publish_source_panel_manifest,
    write_source_panel_status,
)
from causal4d.preacquisition_source_panel_staging import (
    verify_source_panel_manifest_staging,
    write_source_panel_staging_preflight,
)

build_preacquisition_next_action = (
    _operator_flow.build_preacquisition_operator_next_action
)
write_preacquisition_next_action = (
    _operator_flow.write_preacquisition_operator_next_action
)
write_preacquisition_next_action_markdown = (
    _operator_flow.write_preacquisition_operator_next_action_markdown
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

    registry_scaffold = subparsers.add_parser(
        "scaffold-operator-registry",
        help="write the protocol-bound operator registry template once",
    )
    registry_scaffold.add_argument("repository_root")
    registry_scaffold.add_argument("dataset_root")

    registry_seal = subparsers.add_parser(
        "seal-operator-registry",
        help="validate and atomically seal the operator identity roster",
    )
    registry_seal.add_argument("repository_root")
    registry_seal.add_argument("dataset_root")
    registry_seal.add_argument("source_json")
    registry_seal.add_argument("--sealed-by", required=True)
    registry_seal.add_argument("--sealed-at-utc")

    seal = subparsers.add_parser(
        "seal-gate",
        help="validate and atomically seal one completed operational gate",
    )
    seal.add_argument("repository_root")
    seal.add_argument("dataset_root")
    seal.add_argument("gate_id", choices=tuple(GATE_PATHS))
    seal.add_argument("--approved-by", required=True)
    seal.add_argument("--approved-at-utc")

    source_status = subparsers.add_parser(
        "source-panel-status",
        help="validate ordered progress through the 12 physical source executions",
    )
    source_status.add_argument("repository_root")
    source_status.add_argument("dataset_root")
    source_status.add_argument("--output-json")
    source_status.add_argument("--verify-file-hashes", action="store_true")
    source_status.add_argument(
        "--require-complete",
        action="store_true",
        help="return exit code 3 while the valid source panel is incomplete",
    )

    source_verify = subparsers.add_parser(
        "source-panel-verify-staged",
        help=(
            "hash-verify exactly the next staged source manifest without publishing it"
        ),
    )
    source_verify.add_argument("repository_root")
    source_verify.add_argument("dataset_root")
    source_verify.add_argument("source_json")
    source_verify.add_argument("--output-json")

    source_publish = subparsers.add_parser(
        "source-panel-publish",
        help="hash-verify and publish exactly the next source execution manifest",
    )
    source_publish.add_argument("repository_root")
    source_publish.add_argument("dataset_root")
    source_publish.add_argument("source_json")

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

    next_action = subparsers.add_parser(
        "next-action",
        help="derive exactly one admissible operator action from current evidence",
    )
    next_action.add_argument("repository_root")
    next_action.add_argument("dataset_root")
    next_action.add_argument("--output-json")
    next_action.add_argument("--output-markdown")
    next_action.add_argument(
        "--skip-file-hashes",
        action="store_true",
        help="inspect structure only; the suggested action cannot authorize collection",
    )

    next_action_validate = subparsers.add_parser(
        "next-action-validate",
        help="require a persisted action to equal the current hash-verified decision",
    )
    next_action_validate.add_argument("repository_root")
    next_action_validate.add_argument("dataset_root")
    next_action_validate.add_argument("decision_json")
    next_action_validate.add_argument("--output-json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "scaffold":
            result = scaffold_preacquisition_readiness(
                args.repository_root,
                args.dataset_root,
            )
        elif args.command == "scaffold-operator-registry":
            result = scaffold_operator_registry(
                args.repository_root,
                args.dataset_root,
            )
        elif args.command == "seal-operator-registry":
            result = seal_operator_registry(
                args.repository_root,
                args.dataset_root,
                args.source_json,
                sealed_by=args.sealed_by,
                sealed_at_utc=args.sealed_at_utc,
            )
        elif args.command == "seal-gate":
            result = seal_preacquisition_gate(
                args.repository_root,
                args.dataset_root,
                args.gate_id,
                approved_by=args.approved_by,
                approved_at_utc=args.approved_at_utc,
            )
        elif args.command == "source-panel-status":
            result = build_source_panel_status(
                args.repository_root,
                args.dataset_root,
                verify_file_hashes=args.verify_file_hashes,
            )
            if args.output_json:
                write_source_panel_status(args.output_json, result)
        elif args.command == "source-panel-verify-staged":
            result = verify_source_panel_manifest_staging(
                args.repository_root,
                args.dataset_root,
                args.source_json,
            )
            if args.output_json:
                write_source_panel_staging_preflight(args.output_json, result)
        elif args.command == "source-panel-publish":
            result = publish_source_panel_manifest(
                args.repository_root,
                args.dataset_root,
                args.source_json,
            )
        elif args.command == "next-action":
            result = build_preacquisition_next_action(
                args.repository_root,
                args.dataset_root,
                verify_file_hashes=not args.skip_file_hashes,
            )
            if args.output_json:
                write_preacquisition_next_action(args.output_json, result)
            if args.output_markdown:
                write_preacquisition_next_action_markdown(
                    args.output_markdown,
                    result,
                )
        elif args.command == "next-action-validate":
            result = validate_preacquisition_next_action_report(
                args.repository_root,
                args.dataset_root,
                args.decision_json,
            )
            if args.output_json:
                write_preacquisition_next_action_validation(
                    args.output_json,
                    result,
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
                    "complete": False,
                    "passed": False,
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if args.command == "status":
        if not result["valid"]:
            return 2
        if args.require_ready and not result["ready"]:
            return _VALID_BUT_INCOMPLETE
    if args.command == "source-panel-status":
        if not result["valid"]:
            return 2
        if args.require_complete and not result["complete"]:
            return _VALID_BUT_INCOMPLETE
    if args.command == "next-action":
        if not result["valid"]:
            return 2
        if not result["ready"]:
            return _VALID_BUT_INCOMPLETE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
