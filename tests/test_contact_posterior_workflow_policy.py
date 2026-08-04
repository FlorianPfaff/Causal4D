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
    assert "default: github-hosted" in text
    assert "'ubuntu-latest'" in text
    assert (
        "fromJSON('[\"self-hosted\",\"Linux\",\"X64\",\"nvidia-smi\"]')"
        in text
    )


def test_contact_diagnostic_caches_only_on_hosted_runners() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    hosted = "      - name: Set up Python 3.12 with pip cache\n"
    self_hosted = "      - name: Set up Python 3.12 without Actions cache\n"
    install = "      - name: Install isolated diagnostic environment\n"
    assert hosted in text
    assert self_hosted in text
    assert "        if: inputs.runner != 'self-hosted'\n" in text
    assert "        if: inputs.runner == 'self-hosted'\n" in text
    assert text.count("          cache: pip\n") == 1
    assert text.index(hosted) < text.index(self_hosted) < text.index(install)
