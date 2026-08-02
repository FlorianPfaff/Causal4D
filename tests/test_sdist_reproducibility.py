from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_PATHS = frozenset(
    {
        ".github/workflows/ci.yml",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "ci/project_status_v1.json",
        "ci/three_repository_golden_path.py",
        "ci/three_repository_status.py",
        "configs/causal4d/sloth_multi_action_v1.json",
        "configs/causal4d/sloth_multi_action_v1_schedule.csv",
        "docs/causal4d_paper_scope.md",
        "milestones/v0.3.0-causal4d-aip/README.md",
        "runs/causal4d-real-undercoverage-v1/manifest.json",
        "scripts/release/capture_file_manifest.py",
        "scripts/release/verify_result_bundle.py",
        "tests/conftest.py",
        "tests/fixtures/prob4d_joint_observation_v1.json",
    }
)


def _build_sdist(tmp_path: Path) -> Path:
    output = tmp_path / "dist"
    output.mkdir()
    script = (
        "from setuptools.build_meta import build_sdist; "
        f"print(build_sdist({str(output)!r}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    archives = tuple(output.glob("*.tar.gz"))
    assert len(archives) == 1, result.stdout + result.stderr
    return archives[0]


def _relative_archive_paths(archive: Path) -> tuple[str, set[str]]:
    with tarfile.open(archive, "r:gz") as handle:
        members = tuple(handle.getmembers())
    assert members
    roots: set[str] = set()
    relative: set[str] = set()
    for member in members:
        path = PurePosixPath(member.name)
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert path.parts
        roots.add(path.parts[0])
        if len(path.parts) > 1:
            relative.add(PurePosixPath(*path.parts[1:]).as_posix())
    assert len(roots) == 1
    return roots.pop(), relative


def test_sdist_contains_repository_validation_assets(tmp_path: Path) -> None:
    archive = _build_sdist(tmp_path)
    _, relative = _relative_archive_paths(archive)
    missing = sorted(_REQUIRED_PATHS - relative)
    assert not missing, f"source distribution omitted required assets: {missing}"


def _extract_regular_files(archive: Path, destination: Path) -> None:
    """Extract a locally built archive without following links or path escapes."""

    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            relative = PurePosixPath(member.name)
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            assert member.isfile(), (
                f"source distribution contains a link: {member.name}"
            )
            source = handle.extractfile(member)
            assert source is not None
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())


def test_extracted_sdist_runs_core_contract_tests(tmp_path: Path) -> None:
    archive = _build_sdist(tmp_path)
    root_name, _ = _relative_archive_paths(archive)
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    _extract_regular_files(archive, extracted)
    source_root = extracted / root_name
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_causal4d_contracts.py",
            "tests/test_probability_support_invariants.py",
        ],
        cwd=source_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
