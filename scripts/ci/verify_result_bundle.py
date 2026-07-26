#!/usr/bin/env python3
"""Verify a Causal4D result bundle against its embedded checksum manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise ValueError(f"unsafe artifact name: {value!r}")
    return path.name


def verify_result_bundle(manifest_path: Path, bundle_directory: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("result manifest schema_version must equal 1")
    benchmark = manifest.get("benchmark")
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("result manifest benchmark must be nonempty")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("result manifest artifacts must be a nonempty object")

    expected_names: set[str] = set()
    for raw_name, raw_record in artifacts.items():
        name = _safe_name(str(raw_name))
        expected_names.add(name)
        if not isinstance(raw_record, dict):
            raise ValueError(f"artifact record for {name!r} must be an object")
        expected_hash = raw_record.get("sha256")
        expected_bytes = raw_record.get("bytes")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or any(
            character not in "0123456789abcdef" for character in expected_hash
        ):
            raise ValueError(f"artifact {name!r} has an invalid SHA-256 digest")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ValueError(f"artifact {name!r} has an invalid byte count")
        path = bundle_directory / name
        if not path.is_file():
            raise FileNotFoundError(f"bundle artifact is missing: {path}")
        actual_bytes = path.stat().st_size
        actual_hash = _sha256(path)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"artifact {name!r} byte count changed: {actual_bytes} != {expected_bytes}"
            )
        if actual_hash != expected_hash:
            raise ValueError(
                f"artifact {name!r} checksum changed: {actual_hash} != {expected_hash}"
            )

    actual_names = {
        path.name
        for path in bundle_directory.iterdir()
        if path.is_file() and path.name != manifest_path.name
    }
    unexpected = sorted(actual_names - expected_names)
    missing = sorted(expected_names - actual_names)
    if unexpected or missing:
        raise ValueError(
            f"bundle file set differs from the manifest: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return {
        "benchmark": benchmark,
        "artifact_count": len(expected_names),
        "manifest": str(manifest_path.resolve()),
        "bundle": str(bundle_directory.resolve()),
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
