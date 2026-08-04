from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from result_bundle_test_support import (
    ROOT,
    WRITE_REPRODUCTION,
    _copy_bundle,
    _read_json,
    _refresh_result_manifest,
    _run_comparison,
    _write_bundle,
    _write_json,
)

def test_reproduction_manifest_records_runtime_and_binds_exact_bytes(
    tmp_path: Path,
) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    sidecar = tmp_path / "actual.reproduction.json"
    _write_bundle(expected)
    _copy_bundle(expected, actual)
    writer = subprocess.run(
        [
            sys.executable,
            str(WRITE_REPRODUCTION),
            str(actual),
            "--output",
            str(sidecar),
            "--repository",
            "IPS-Stuttgart/Causal4D",
            "--commit-sha",
            "a" * 40,
            "--workflow-run-id",
            "123",
            "--runner-name",
            "workstation2",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert writer.returncode == 0, writer.stderr
    sidecar_document = _read_json(sidecar)
    assert sidecar_document["comparison_contract_version"] == 2
    assert sidecar_document["runtime"]["distributions"]["numpy"]
    assert sidecar_document["source"]["runner_name"] == "workstation2"

    process, report = _run_comparison(
        expected,
        actual,
        tmp_path,
        "--actual-reproduction-manifest",
        str(sidecar),
        "--require-actual-reproduction-manifest",
    )
    assert process.returncode == 0
    assert report["reproduction_manifests"]["actual"]["valid"] is True

    summary = _read_json(actual / "summary.json")
    summary["metric"] = 1.0 + 1e-13
    _write_json(actual / "summary.json", summary)
    _refresh_result_manifest(actual)
    stale_process, stale_report = _run_comparison(
        expected,
        actual,
        tmp_path,
        "--actual-reproduction-manifest",
        str(sidecar),
        "--require-actual-reproduction-manifest",
    )
    assert stale_process.returncode == 2
    assert stale_report["semantic_match"] is False
    assert any(
        "actual reproduction manifest is invalid" in item
        for item in stale_report["mismatches"]
    )


def test_reproduction_manifest_runtime_schema_is_exact(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    sidecar = tmp_path / "actual.reproduction.json"
    _write_bundle(expected)
    _copy_bundle(expected, actual)
    writer = subprocess.run(
        [
            sys.executable,
            str(WRITE_REPRODUCTION),
            str(actual),
            "--output",
            str(sidecar),
            "--commit-sha",
            "a" * 40,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert writer.returncode == 0, writer.stderr
    document = _read_json(sidecar)
    document["runtime"] = {"invented": "environment"}
    _write_json(sidecar, document)

    process, report = _run_comparison(
        expected,
        actual,
        tmp_path,
        "--actual-reproduction-manifest",
        str(sidecar),
        "--require-actual-reproduction-manifest",
    )

    assert process.returncode == 2
    assert report["semantic_match"] is False
    assert any(
        "reproduction manifest runtime keys differ" in item
        for item in report["mismatches"]
    )


def test_symlinked_bundle_directory_is_rejected(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual_target = tmp_path / "actual-target"
    actual_link = tmp_path / "actual-link"
    _write_bundle(expected)
    _copy_bundle(expected, actual_target)
    actual_link.symlink_to(actual_target, target_is_directory=True)

    process, report = _run_comparison(expected, actual_link, tmp_path)

    assert process.returncode == 2
    assert report["semantic_match"] is False
    assert any(
        "result bundle directory must not be a symlink" in item
        for item in report["mismatches"]
    )


def test_undeclared_subdirectory_is_rejected(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _write_bundle(expected)
    _copy_bundle(expected, actual)
    hidden = actual / "undeclared"
    hidden.mkdir()
    (hidden / "payload.txt").write_text("not in manifest\n", encoding="utf-8")

    process, report = _run_comparison(expected, actual, tmp_path)

    assert process.returncode == 2
    assert report["semantic_match"] is False
    assert any(
        "undeclared non-file entry" in item for item in report["mismatches"]
    )


