import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from causal4d.real_experiment_freeze import (
    REQUIRED_LOCKED_PATHS,
    build_method_freeze_manifest,
    validate_method_freeze_manifest,
    validate_repository_checkout,
)


BPT_SHA = "c7ad36aad7e592ce8a391c9ca2d4db7389dee3ac"
CAUSAL4D_SHA = "a" * 40


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    for relative in REQUIRED_LOCKED_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"locked:{relative}\n", encoding="utf-8")
    (root / "configs/causal4d/sloth_multi_action_v1.json").write_text(
        json.dumps({"design_sha256": "b" * 64}), encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(
        "[project.optional-dependencies]\n"
        "phystwin = [\n"
        f'  "bayesian-phystwin @ git+https://github.com/FlorianPfaff/'
        f'Bayesian-PhysTwin.git@{BPT_SHA}",\n'
        "]\n",
        encoding="utf-8",
    )
    return root


def test_freeze_binds_method_files_and_dependency_commit(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    manifest = build_method_freeze_manifest(
        root,
        causal4d_commit_sha=CAUSAL4D_SHA,
        frozen_by="operator-1",
        frozen_at_utc="2026-07-26T18:00:00+00:00",
    )
    result = validate_method_freeze_manifest(
        manifest,
        root,
        expected_causal4d_commit_sha=CAUSAL4D_SHA,
    )
    assert result["locked_files_checked"] == len(REQUIRED_LOCKED_PATHS)
    assert result["bayesian_phystwin_commit_sha"] == BPT_SHA
    assert result["passed"]


def test_freeze_rejects_file_drift_and_target_informed_selection(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    manifest = build_method_freeze_manifest(
        root,
        causal4d_commit_sha=CAUSAL4D_SHA,
        frozen_by="operator-1",
    )
    (root / "docs/causal4d_paper_scope.md").write_text("changed after freeze\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_method_freeze_manifest(manifest, root)

    root = _repository(tmp_path / "second")
    manifest = build_method_freeze_manifest(
        root,
        causal4d_commit_sha=CAUSAL4D_SHA,
        frozen_by="operator-1",
    )
    changed = deepcopy(manifest)
    changed["target_outcomes_observed_at_freeze"] = True
    with pytest.raises(ValueError, match="target outcomes"):
        validate_method_freeze_manifest(changed, root)

    changed = deepcopy(manifest)
    changed["analysis_contract"]["optional_branches_may_change_primary_analysis"] = True
    with pytest.raises(ValueError, match="analysis contract"):
        validate_method_freeze_manifest(changed, root)


def test_freeze_rejects_checkout_or_bpt_pin_mismatch(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    manifest = build_method_freeze_manifest(
        root,
        causal4d_commit_sha=CAUSAL4D_SHA,
        frozen_by="operator-1",
    )
    with pytest.raises(ValueError, match="checkout does not match"):
        validate_method_freeze_manifest(
            manifest,
            root,
            expected_causal4d_commit_sha="c" * 40,
        )

    pyproject = root / "pyproject.toml"
    pyproject.write_text(pyproject.read_text().replace(BPT_SHA, "d" * 40))
    with pytest.raises(ValueError, match="Bayesian-PhysTwin pin"):
        validate_method_freeze_manifest(manifest, root, verify_files=False)


def test_checkout_validation_rejects_tracked_or_untracked_drift(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=root, check=True
    )
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze"], cwd=root, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = build_method_freeze_manifest(
        root, causal4d_commit_sha=commit, frozen_by="operator-1"
    )
    assert validate_repository_checkout(manifest, root)["commit_sha"] == commit

    (root / "untracked-analysis.py").write_text("print('drift')\n")
    with pytest.raises(ValueError, match="checkout is dirty"):
        validate_repository_checkout(manifest, root)
