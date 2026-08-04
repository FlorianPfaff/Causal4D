from __future__ import annotations

from pathlib import Path

import pytest

from causal4d.preacquisition_readiness import (
    SOURCE_PANEL_MANIFEST_PATH,
    scaffold_preacquisition_readiness,
)
from causal4d.preacquisition_source_panel_control import build_source_panel_status


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIRST_EXECUTION_ID = "sloth-pre-v2-lift_high-r1"


def _manifest_path(dataset_root: Path) -> Path:
    return dataset_root / SOURCE_PANEL_MANIFEST_PATH.format(
        execution_id=FIRST_EXECUTION_ID
    )


def test_status_rejects_directory_at_manifest_path(tmp_path: Path) -> None:
    scaffold_preacquisition_readiness(REPOSITORY_ROOT, tmp_path)
    manifest = _manifest_path(tmp_path)
    manifest.mkdir()

    status = build_source_panel_status(
        REPOSITORY_ROOT,
        tmp_path,
        verify_file_hashes=True,
    )

    assert status["valid"] is False
    assert status["complete"] is False
    assert status["invalid_execution_ids"] == [FIRST_EXECUTION_ID]
    assert status["executions"][0]["manifest_present"] is True
    assert status["executions"][0]["valid"] is False


def test_status_rejects_dangling_symlink_at_manifest_path(
    tmp_path: Path,
) -> None:
    scaffold_preacquisition_readiness(REPOSITORY_ROOT, tmp_path)
    manifest = _manifest_path(tmp_path)
    try:
        manifest.symlink_to(tmp_path / "missing-source-panel-manifest.json")
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    status = build_source_panel_status(
        REPOSITORY_ROOT,
        tmp_path,
        verify_file_hashes=True,
    )

    assert status["valid"] is False
    assert status["complete"] is False
    assert status["invalid_execution_ids"] == [FIRST_EXECUTION_ID]
    assert status["executions"][0]["manifest_present"] is True
    assert status["executions"][0]["valid"] is False
