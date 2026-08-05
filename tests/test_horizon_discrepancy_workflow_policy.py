from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(".github/workflows/horizon-discrepancy-integration.yml")
BPT_HORIZON_PROVIDER_REVISION = "bfa844798f0ab3ddbc67a0744ae14a221324e504"


def test_horizon_workflow_uses_exact_public_provider_revision() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "repository: IPS-Stuttgart/BayesianPhysTwin" in text
    assert f"ref: {BPT_HORIZON_PROVIDER_REVISION}" in text
    assert "persist-credentials: false" in text
    assert "BPT_READ_SSH_KEY" not in text


def test_horizon_workflow_exercises_installed_wheels() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "python -m build --wheel" in text
    assert "horizon-discrepancy-wheelhouse" in text
    assert "horizon-discrepancy-venv" in text
    assert "--import-mode=importlib" in text
    assert "env -u PYTHONPATH" in text
    assert "test_belief_provider_v2_contract.py" in text
    assert "test_horizon_discrepancy.py" in text
    assert "test_belief_provider_contract.py" in text


def test_horizon_workflow_pins_external_actions() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert text.count(
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
    ) == 2
    assert (
        "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97"
        in text
    )
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("uses:"):
            assert "@" in stripped
            reference = stripped.rsplit("@", 1)[1].split()[0]
            assert len(reference) == 40
            assert all(character in "0123456789abcdef" for character in reference)
