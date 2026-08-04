#!/usr/bin/env python3
"""Write a runtime-bound sidecar for a verified Causal4D result bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from result_bundle_reproducibility import (
    RESULT_MANIFEST_NAME,
    build_reproduction_manifest,
    sha256_file,
    verify_result_manifest,
    write_reproduction_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository")
    parser.add_argument("--commit-sha")
    parser.add_argument("--workflow-run-id")
    parser.add_argument("--runner-name")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    raw_bundle_directory = arguments.bundle_directory
    if raw_bundle_directory.is_symlink():
        raise ValueError(
            f"result bundle directory must not be a symlink: {raw_bundle_directory}"
        )
    bundle_directory = raw_bundle_directory.absolute()
    bundle = verify_result_manifest(
        bundle_directory / RESULT_MANIFEST_NAME,
        bundle_directory,
    )
    raw_output = arguments.output
    if raw_output.is_symlink():
        raise ValueError(
            f"the reproduction sidecar must not be a symlink: {raw_output}"
        )
    output = raw_output.resolve()
    if output.is_relative_to(bundle_directory):
        raise ValueError(
            "the reproduction sidecar must be written outside the verified "
            "bundle directory so it cannot change the embedded file inventory"
        )
    document = build_reproduction_manifest(
        bundle,
        repository=arguments.repository,
        commit_sha=arguments.commit_sha,
        workflow_run_id=arguments.workflow_run_id,
        runner_name=arguments.runner_name,
    )
    write_reproduction_manifest(output, document)
    summary = {
        "bundle": str(bundle_directory),
        "benchmark": bundle.benchmark,
        "artifact_count": len(bundle.artifacts),
        "output": str(output),
        "sha256": sha256_file(output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
