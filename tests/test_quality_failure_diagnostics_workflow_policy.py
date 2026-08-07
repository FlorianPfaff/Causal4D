from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
LEGACY_WORKFLOW = ROOT / ".github" / "workflows" / "quality-failure-diagnostics.yml"


def test_ruff_diagnostics_stay_in_the_read_only_pull_request_workflow() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")

    assert not LEGACY_WORKFLOW.exists()
    assert "pull_request:" in text
    assert "permissions:\n  contents: read" in text
    assert "Run Ruff and capture diagnostics" in text
    assert "if: ${{ failure() && steps.ruff.outcome == 'failure' }}" in text
    assert "quality-failure-diagnostics-${{ github.run_id }}" in text
    assert "ruff-version.txt" in text
    assert "ruff-output.txt" in text
    assert "ruff-exit-status.txt" in text
    assert "workflow_run:" not in text
