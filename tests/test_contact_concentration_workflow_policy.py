from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "contact-posterior-concentration.yml"


def test_concentration_workflow_is_read_only_and_self_hosted() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "permissions:\n  contents: read\n" in text
    assert "contents: write" not in text
    assert "git push" not in text
    assert "persist-credentials: false" in text
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in text
    assert "github.event.pull_request.head.repo.full_name == github.repository" in text
    assert "workflow_dispatch:" in text
    assert "  push:" not in text


def test_concentration_workflow_uses_fresh_registered_panel() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "DIAGNOSTIC_SEEDS: ${{ inputs.seeds || '200:220' }}" in text
    assert "0.25,0.50,0.75" in text
    assert '--seeds "${DIAGNOSTIC_SEEDS}"' in text
    assert '--softening-logit-scales "${SOFTENING_LOGIT_SCALES}"' in text
    assert "retention-days: 30" in text
