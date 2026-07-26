"""Seal or validate the frozen Causal4D confirmatory real-experiment method."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from causal4d.real_experiment_freeze import (
    build_method_freeze_manifest,
    load_method_freeze_manifest,
    repository_git_state,
    validate_method_freeze_manifest,
    validate_repository_checkout,
    write_method_freeze_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal = subparsers.add_parser(
        "seal",
        help="freeze a clean Git checkout before the first confirmatory execution",
    )
    seal.add_argument("repository_root")
    seal.add_argument("output_json")
    seal.add_argument("--frozen-by", required=True)
    seal.add_argument("--frozen-at-utc")

    validate = subparsers.add_parser(
        "validate",
        help="verify the frozen commit, dependency pin, contracts, and file hashes",
    )
    validate.add_argument("manifest_json")
    validate.add_argument("repository_root")
    validate.add_argument("--expected-causal4d-commit")
    validate.add_argument("--skip-file-hashes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "seal":
            git_state = repository_git_state(args.repository_root)
            if git_state["dirty_worktree"]:
                raise ValueError("refusing to freeze a dirty acquisition checkout")
            manifest = build_method_freeze_manifest(
                args.repository_root,
                causal4d_commit_sha=git_state["commit_sha"],
                frozen_by=args.frozen_by,
                frozen_at_utc=args.frozen_at_utc,
            )
            output = write_method_freeze_manifest(args.output_json, manifest)
            result = {
                **validate_method_freeze_manifest(
                    manifest,
                    args.repository_root,
                    expected_causal4d_commit_sha=git_state["commit_sha"],
                ),
                "output": str(output.resolve()),
            }
        else:
            manifest = load_method_freeze_manifest(args.manifest_json)
            checkout = validate_repository_checkout(manifest, args.repository_root)
            if (
                args.expected_causal4d_commit is not None
                and args.expected_causal4d_commit != checkout["commit_sha"]
            ):
                raise ValueError(
                    "--expected-causal4d-commit does not match the checkout"
                )
            result = {
                **validate_method_freeze_manifest(
                    manifest,
                    args.repository_root,
                    expected_causal4d_commit_sha=checkout["commit_sha"],
                    verify_files=not args.skip_file_hashes,
                ),
                "checkout_clean": True,
            }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
