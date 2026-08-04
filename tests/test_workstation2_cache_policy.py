from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "workstation2-evaluation.yml"


def test_workstation2_uses_isolated_grouped_reproduction_path() -> None:
    """Keep the self-hosted evaluation isolated and compatible with 0.5+ CLI."""

    text = WORKFLOW.read_text(encoding="utf-8")

    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in text
    assert "cache: pip" not in text
    assert ".workstation2-venv/bin/causal4d benchmark latent-contact" in text
    assert "causal4d-latent-contact-benchmark" not in text
    assert text.count("scripts/ci/write_reproduction_manifest.py") >= 2
    assert "--actual-reproduction-manifest" in text
    assert "--require-actual-reproduction-manifest" in text
