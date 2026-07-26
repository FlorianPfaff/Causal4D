#!/usr/bin/env python3
"""Validate the frozen Causal4D milestone manifests and checked-in archive subset."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePath
import re
import sys
from typing import Any, Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _entries(document: Mapping[str, Any], *, name: str) -> dict[str, dict[str, Any]]:
    raw_entries = document.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(f"{name} entries must be a nonempty array")
    result: dict[str, dict[str, Any]] = {}
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError(f"{name} entries must be objects")
        identifier = raw_entry.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{name} entry id must be nonempty")
        if identifier in result:
            raise ValueError(f"duplicate {name} entry id: {identifier}")
        result[identifier] = raw_entry
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_revision_digests(value: Any, *, path: str = "source-revisions") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in {"commit", "commit_before_release_archive"} and child is not None:
                if not isinstance(child, str) or _GIT_SHA.fullmatch(child) is None:
                    raise ValueError(f"{child_path} must be a lowercase 40-hex commit")
            if key.endswith("_sha256") and child is not None:
                if not isinstance(child, str) or _SHA256.fullmatch(child) is None:
                    raise ValueError(f"{child_path} must be a lowercase SHA-256 digest")
            _validate_revision_digests(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_revision_digests(child, path=f"{path}[{index}]")


def verify_frozen_manifest(milestone_root: Path) -> dict[str, Any]:
    artifact_manifest = _load_object(milestone_root / "artifact-manifest.json")
    artifact_paths = _load_object(milestone_root / "artifact-paths.json")
    revisions = _load_object(milestone_root / "source-revisions.json")
    milestone = str(artifact_paths.get("milestone", ""))
    if not milestone or milestone != milestone_root.name:
        raise ValueError("artifact-paths milestone must match its directory name")
    if revisions.get("milestone") != milestone:
        raise ValueError("source-revisions milestone differs from artifact-paths")
    if artifact_paths.get("schema_version") != 1:
        raise ValueError("artifact-paths schema_version must equal 1")
    if revisions.get("schema_version") != 1:
        raise ValueError("source-revisions schema_version must equal 1")
    if artifact_manifest.get("captured_at") != artifact_paths.get("captured_at"):
        raise ValueError(
            "artifact manifest and path inventory have different capture times"
        )

    manifest_entries = _entries(artifact_manifest, name="artifact manifest")
    path_entries = _entries(artifact_paths, name="artifact paths")
    if set(manifest_entries) != set(path_entries):
        raise ValueError("artifact manifest IDs differ from artifact-path IDs")

    checked_files = 0
    external_files = 0
    category_counts: Counter[str] = Counter()
    for identifier, entry in manifest_entries.items():
        path_entry = path_entries[identifier]
        for field in ("category", "source_path", "archive_path"):
            if entry.get(field) != path_entry.get(field):
                raise ValueError(f"{identifier}: {field} differs between inventories")
        category = entry.get("category")
        if not isinstance(category, str) or not category:
            raise ValueError(f"{identifier}: category must be nonempty")
        category_counts[category] += 1
        digest = entry.get("sha256")
        byte_count = entry.get("bytes")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"{identifier}: invalid SHA-256 digest")
        if not isinstance(byte_count, int) or byte_count < 0:
            raise ValueError(f"{identifier}: invalid byte count")
        archive_path = entry.get("archive_path")
        if not isinstance(archive_path, str):
            continue
        archive_parts = PurePath(archive_path).parts
        if milestone not in archive_parts:
            continue
        milestone_index = archive_parts.index(milestone)
        relative_parts = archive_parts[milestone_index + 1 :]
        if not relative_parts:
            raise ValueError(f"{identifier}: archive_path names no artifact")
        local_path = milestone_root.joinpath(*relative_parts)
        if not local_path.is_file():
            external_files += 1
            continue
        checked_files += 1
        if local_path.stat().st_size != byte_count:
            raise ValueError(f"{identifier}: checked-in archive byte count changed")
        if _sha256(local_path) != digest:
            raise ValueError(f"{identifier}: checked-in archive checksum changed")

    _validate_revision_digests(revisions)
    if checked_files == 0:
        raise ValueError(
            "no checked-in frozen artifact was available for checksum verification"
        )
    return {
        "milestone": milestone,
        "entry_count": len(manifest_entries),
        "checked_in_artifact_count": checked_files,
        "external_artifact_count": external_files,
        "categories": dict(sorted(category_counts.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("milestone_root", type=Path)
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            verify_frozen_manifest(arguments.milestone_root),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
