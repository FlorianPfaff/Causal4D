from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "contact-prefix-correlation.yml"


def test_correlation_workflow_is_read_only_and_uses_hosted_cache() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "permissions:\n  contents: read\n" in text
    assert "contents: write" not in text
    assert "git push" not in text
    assert "persist-credentials: false" in text
    assert "runs-on: ubuntu-latest" in text
    assert "runs-on: [self-hosted" not in text
    assert "cache: pip" in text
    assert "cache-dependency-path: pyproject.toml" in text
    assert "github.event.pull_request.head.repo.full_name == github.repository" in text
    assert "workflow_dispatch:" in text
    assert "  push:" not in text


def test_correlation_workflow_locks_the_fresh_panel_and_candidate_grids() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "DIAGNOSTIC_SEEDS: ${{ inputs.seeds || '300:320' }}" in text
    assert "FRAME_BLOCK_SIZES: ${{ inputs.frame_block_sizes || '2,3,4' }}" in text
    assert "0.10,0.25,0.50,0.75" in text
    assert "0.25,0.50,0.75,1.00" in text
    assert '--seeds "${DIAGNOSTIC_SEEDS}"' in text
    assert '--frame-block-sizes "${FRAME_BLOCK_SIZES}"' in text
    assert '--whitening-shrinkages "${WHITENING_SHRINKAGES}"' in text
    assert '--generalized-bayes-rates "${GENERALIZED_BAYES_RATES}"' in text
    assert "retention-days: 30" in text
