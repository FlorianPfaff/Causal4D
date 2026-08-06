from pathlib import Path

import pytest


def test_tag_release_requires_public_provider_integration() -> None:
    workflow = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
    if not workflow.is_file():
        pytest.skip("GitHub workflow is not included in the source distribution")
    text = workflow.read_text(encoding="utf-8")
    assert "Pinned Bayesian-PhysTwin installed-wheel integration" in text
    assert "repository: IPS-Stuttgart/BayesianPhysTwin" in text
    assert "python -m build --wheel" in text
    assert "artifact-venv/bin/python scripts/ci/run_bpt_integration_tests.py" in text
    assert (
        "    needs:\n"
        "      - lint\n"
        "      - format\n"
        "      - types\n"
        "      - core\n"
        "      - pinned-bpt\n"
        "      - build\n"
        "      - installed-artifact\n"
        "      - bundles\n"
        "      - frozen-manifest\n"
        in text
    )
    assert "BPT_READ_SSH_KEY" not in text
    assert "ssh-key:" not in text


def test_quality_failures_are_reported_by_independent_jobs() -> None:
    workflow = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
    if not workflow.is_file():
        pytest.skip("GitHub workflow is not included in the source distribution")
    text = workflow.read_text(encoding="utf-8")
    quality_section, _ = text.split("\n  core:\n", maxsplit=1)

    assert "\n  quality:\n" not in quality_section
    assert "\n  lint:\n    name: Ruff lint\n" in quality_section
    assert "\n  format:\n    name: Ruff formatting\n" in quality_section
    assert "\n  types:\n    name: Mypy stable contracts\n" in quality_section
    assert quality_section.count("python -m ruff check .") == 1
    assert quality_section.count("python -m ruff format --check") == 1
    assert quality_section.count("python -m mypy --python-version 3.12") == 1
    assert "src/causal4d/artifact_io.py" in quality_section
    assert "src/causal4d/discrepancy_belief.py" in quality_section
