from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "workstation2-evaluation.yml"


def test_workstation2_does_not_restore_the_actions_pip_cache() -> None:
    """The self-hosted runner keeps a local cache; remote restore can be huge."""

    text = WORKFLOW.read_text(encoding="utf-8")

    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in text
    assert "cache: pip" not in text
