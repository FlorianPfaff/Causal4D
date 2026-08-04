from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "contact-posterior-concentration.yml"


def test_concentration_workflow_is_read_only_and_runner_selectable() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "permissions:\n  contents: read\n" in text
    assert "contents: write" not in text
    assert "git push" not in text
    assert "persist-credentials: false" in text
    assert "default: github-hosted" in text
    assert "'ubuntu-latest'" in text
    assert (
        "fromJSON('[\"self-hosted\",\"Linux\",\"X64\",\"nvidia-smi\"]')"
        in text
    )
    assert "github.event.pull_request.head.repo.full_name == github.repository" in text
    assert "workflow_dispatch:" in text
    assert "  push:" not in text


def test_concentration_workflow_caches_only_on_hosted_runners() -> None:
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


def test_concentration_workflow_uses_fresh_registered_panel() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "DIAGNOSTIC_SEEDS: ${{ inputs.seeds || '200:220' }}" in text
    assert "0.25,0.50,0.75" in text
    assert '--seeds "${DIAGNOSTIC_SEEDS}"' in text
    assert '--softening-logit-scales "${SOFTENING_LOGIT_SCALES}"' in text
    assert "retention-days: 30" in text
