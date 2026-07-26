"""Regression tests for standalone milestone verification utilities."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_VERIFY_RESULT_BUNDLE = _REPO_ROOT / "scripts/release/verify_result_bundle.py"
_CAPTURE_FILE_MANIFEST = _REPO_ROOT / "scripts/release/capture_file_manifest.py"


def _run_script(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_result_bundle_verifier_accepts_only_bounded_numeric_drift(
    tmp_path: Path,
) -> None:
    expected_dir = tmp_path / "expected"
    actual_dir = tmp_path / "actual"
    expected_dir.mkdir()
    actual_dir.mkdir()

    artifact_name = "summary.json"
    expected_artifact = expected_dir / artifact_name
    actual_artifact = actual_dir / artifact_name
    expected_artifact.write_text(
        json.dumps({"label": "frozen", "value": 1.0}) + "\n",
        encoding="utf-8",
    )
    actual_artifact.write_text(
        json.dumps({"label": "frozen", "value": 1.0000000000005}) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "benchmark": "test-benchmark",
        "artifacts": {
            artifact_name: {
                "bytes": expected_artifact.stat().st_size,
                "sha256": _sha256(expected_artifact),
            }
        },
    }
    manifest_path = expected_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    exact = _run_script(
        _VERIFY_RESULT_BUNDLE,
        str(manifest_path),
        str(actual_dir),
    )
    assert exact.returncode == 1, exact.stderr
    assert json.loads(exact.stdout)["passed"] is False

    tolerant = _run_script(
        _VERIFY_RESULT_BUNDLE,
        str(manifest_path),
        str(actual_dir),
        "--numeric-atol",
        "1e-12",
    )
    assert tolerant.returncode == 0, tolerant.stderr
    result = json.loads(tolerant.stdout)
    assert result["passed"] is True
    assert result["tolerance_matches"] == [artifact_name]


def test_file_manifest_capture_and_verification_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    archive = tmp_path / "archive.bin"
    source.write_bytes(b"frozen milestone payload\n")
    archive.write_bytes(source.read_bytes())

    specification = {
        "schema_version": 1,
        "milestone": "test-milestone",
        "captured_at": "2026-07-12T00:00:00Z",
        "host": "test-host",
        "entries": [
            {
                "id": "payload",
                "category": "test",
                "source_path": str(source),
                "archive_path": str(archive),
            }
        ],
    }
    specification_path = tmp_path / "specification.json"
    manifest_path = tmp_path / "manifest.json"
    specification_path.write_text(json.dumps(specification), encoding="utf-8")

    captured = _run_script(
        _CAPTURE_FILE_MANIFEST,
        "capture",
        str(specification_path),
        str(manifest_path),
    )
    assert captured.returncode == 0, captured.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["entry_count"] == 1
    assert manifest["entries"][0]["archive_verified"] is True

    for location in ("source", "archive"):
        verified = _run_script(
            _CAPTURE_FILE_MANIFEST,
            "verify",
            str(manifest_path),
            "--location",
            location,
        )
        assert verified.returncode == 0, verified.stderr
        assert json.loads(verified.stdout)["passed"] is True
