from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "contact-topology-covariance.yml"


def test_topology_covariance_workflow_is_read_only_and_reproduced() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "permissions:\n  contents: read\n" in text
    assert "contents: write" not in text
    assert "git push" not in text
    assert "persist-credentials: false" in text
    assert "lane: hosted" in text
    assert 'runs_on: \'"ubuntu-latest"\'' in text
    assert "lane: workstation2" in text
    assert 'runs_on: \'["self-hosted","Linux","X64","nvidia-smi"]\'' in text
    assert "cache: pip" in text
    assert "cache-dependency-path: pyproject.toml" in text
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository" in text
    )
    assert "workflow_dispatch:" in text
    assert "  push:" not in text


def test_topology_covariance_workflow_locks_panels_grid_and_runtime() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "DEVELOPMENT_SEEDS: ${{ inputs.development_seeds || '300:320' }}" in text
    assert "EVALUATION_SEEDS: ${{ inputs.evaluation_seeds || '400:420' }}" in text
    assert "0,0.25,0.50,0.75,1.00" in text
    assert "0.10,0.25,0.50,0.75,1.00" in text
    assert '--development-seeds "${DEVELOPMENT_SEEDS}"' in text
    assert '--evaluation-seeds "${EVALUATION_SEEDS}"' in text
    assert '--shared-correlation-weights "${SHARED_CORRELATION_WEIGHTS}"' in text
    assert '--identity-shrinkages "${IDENTITY_SHRINKAGES}"' in text
    for requirement in (
        'python-version: "3.12.13"',
        '"numpy==2.2.6"',
        '"scipy==1.17.1"',
        '"pyrecest==2.4.1"',
        '"pytest==9.1.1"',
        '"ruff==0.16.1"',
    ):
        assert requirement in text
    assert "retention-days: 30" in text
    assert (
        "contact-topology-covariance-${{ matrix.lane }}-${{ github.run_id }}"
        in text
    )


def test_topology_covariance_workflow_verifies_frozen_source_before_target() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "sha256sum -c <<'EOF'" in text
    for path in (
        "src/causal4d/contact_topology_covariance_diagnostic.py",
        "scripts/ci/run_contact_topology_covariance_diagnostic.py",
        "tests/test_contact_topology_covariance_diagnostic.py",
        "tests/test_contact_topology_covariance_workflow_policy.py",
        "docs/contact_topology_covariance.md",
    ):
        assert path in text
    assert text.count("python -m pytest -q") == 2
    assert text.index("sha256sum -c") < text.index(
        "Run fresh target topology-conditioned comparison"
    )
