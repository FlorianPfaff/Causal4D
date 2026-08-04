from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "contact-posterior-diagnostics.yml"


def test_contact_diagnostic_workflow_is_read_only_and_uses_grouped_cli() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "permissions:\n  contents: read\n" in text
    assert "\n  push:\n" not in text
    assert ".contact-diagnostic-venv/bin/causal4d benchmark latent-contact" in text
    assert "causal4d.cli.latent_contact_benchmark" not in text
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in text
