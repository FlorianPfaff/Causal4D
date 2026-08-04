#!/usr/bin/env python3
"""Verify a Causal4D result bundle against its embedded checksum manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from result_bundle_reproducibility import verify_result_manifest


def verify_result_bundle(manifest_path: Path, bundle_directory: Path) -> dict[str, Any]:
    """Verify exact inventory, byte counts, digests, and strict JSON metadata."""

    bundle = verify_result_manifest(manifest_path, bundle_directory)
    return {
        "benchmark": bundle.benchmark,
        "artifact_count": len(bundle.artifacts),
        "manifest": str(bundle.manifest_path),
        "bundle": str(bundle.directory),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("bundle_directory", type=Path, nargs="?")
    arguments = parser.parse_args(argv)
    directory = arguments.bundle_directory or arguments.manifest.parent
    summary = verify_result_bundle(arguments.manifest, directory)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
