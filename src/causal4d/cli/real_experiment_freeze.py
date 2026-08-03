"""Seal, attest, or validate the frozen Causal4D real-experiment method."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from causal4d.operator_identity_integration import (
    validate_method_freeze_identity_evidence,
)
from causal4d.operator_registry import load_registered_operator_registry
from causal4d.real_evidence_contract_v2 import (
    build_method_freeze_validation_attestation,
    write_method_freeze_validation_attestation,
)
from causal4d.real_experiment_freeze import (
    build_method_freeze_manifest,
    load_method_freeze_manifest,
    repository_git_state,
    validate_method_freeze_manifest,
    validate_repository_checkout,
    write_method_freeze_manifest,
)
from causal4d.real_protocol import load_protocol


def _load_registry(
    repository_root: str,
    dataset_root: str | Path,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    result, registry = load_registered_operator_registry(
        repository_root,
        dataset_root,
    )
    if result.get("valid") is not True or registry is None:
        raise ValueError(str(result.get("error") or "operator registry is invalid"))
    return result, registry


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

    attest = subparsers.add_parser(
        "attest",
        help="independently verify and bind the exact method-freeze file",
    )
    attest.add_argument("manifest_json")
    attest.add_argument("protocol_json")
    attest.add_argument("repository_root")
    attest.add_argument("output_json")
    attest.add_argument("--verified-by", required=True)
    attest.add_argument("--verified-at-utc")

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
            dataset_root = Path(args.output_json).parent
            registry_result, registry = _load_registry(
                args.repository_root,
                dataset_root,
            )
            git_state = repository_git_state(args.repository_root)
            if git_state["dirty_worktree"]:
                raise ValueError("refusing to freeze a dirty acquisition checkout")
            manifest = build_method_freeze_manifest(
                args.repository_root,
                causal4d_commit_sha=git_state["commit_sha"],
                frozen_by=args.frozen_by,
                frozen_at_utc=args.frozen_at_utc,
            )
            identity = validate_method_freeze_identity_evidence(
                manifest,
                None,
                registry,
            )
            output = write_method_freeze_manifest(args.output_json, manifest)
            result = {
                **validate_method_freeze_manifest(
                    manifest,
                    args.repository_root,
                    expected_causal4d_commit_sha=git_state["commit_sha"],
                ),
                **identity,
                "operator_registry_artifact_sha256": registry_result["artifact_sha256"],
                "output": str(output.resolve()),
            }
        elif args.command == "attest":
            dataset_root = Path(args.manifest_json).parent
            registry_result, registry = _load_registry(
                args.repository_root,
                dataset_root,
            )
            method_freeze = load_method_freeze_manifest(args.manifest_json)
            attestation = build_method_freeze_validation_attestation(
                load_protocol(args.protocol_json),
                args.manifest_json,
                args.repository_root,
                verified_by=args.verified_by,
                verified_at_utc=args.verified_at_utc,
            )
            identity = validate_method_freeze_identity_evidence(
                method_freeze,
                attestation,
                registry,
            )
            output = write_method_freeze_validation_attestation(
                args.output_json,
                attestation,
            )
            result = {
                "passed": True,
                "method_freeze_sha256": attestation["method_freeze_sha256"],
                "causal4d_commit_sha": attestation["causal4d_commit_sha"],
                "bayesian_phystwin_commit_sha": attestation[
                    "bayesian_phystwin_commit_sha"
                ],
                "verifier_id": attestation["verifier_id"],
                **identity,
                "operator_registry_artifact_sha256": registry_result["artifact_sha256"],
                "output": str(output.resolve()),
            }
        else:
            dataset_root = Path(args.manifest_json).parent
            registry_result, registry = _load_registry(
                args.repository_root,
                dataset_root,
            )
            manifest = load_method_freeze_manifest(args.manifest_json)
            checkout = validate_repository_checkout(manifest, args.repository_root)
            if (
                args.expected_causal4d_commit is not None
                and args.expected_causal4d_commit != checkout["commit_sha"]
            ):
                raise ValueError(
                    "--expected-causal4d-commit does not match the checkout"
                )
            identity = validate_method_freeze_identity_evidence(
                manifest,
                None,
                registry,
            )
            result = {
                **validate_method_freeze_manifest(
                    manifest,
                    args.repository_root,
                    expected_causal4d_commit_sha=checkout["commit_sha"],
                    verify_files=not args.skip_file_hashes,
                ),
                **identity,
                "operator_registry_artifact_sha256": registry_result["artifact_sha256"],
                "checkout_clean": True,
            }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
