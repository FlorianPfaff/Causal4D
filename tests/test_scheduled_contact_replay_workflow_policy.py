from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(".github/workflows/scheduled-contact-replay-integration.yml")
BPT_CONTRACT_REVISION = "58bf6a6f06ad27fce525060190cff787cde58fa4"


def test_scheduled_replay_workflow_uses_exact_public_provider_revision() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "repository: IPS-Stuttgart/BayesianPhysTwin" in text
    assert f"ref: {BPT_CONTRACT_REVISION}" in text
    assert "persist-credentials: false" in text
    assert "BPT_READ_SSH_KEY" not in text


def test_scheduled_replay_workflow_exercises_installed_wheels() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "python -m build --wheel" in text
    assert "scheduled-contact-wheelhouse" in text
    assert "scheduled-contact-venv" in text
    assert "--import-mode=importlib" in text
    assert "env -u PYTHONPATH" in text
    assert "test_scheduled_contact_replay_contract.py" in text
    assert "test_scheduled_contact_replay_adapter.py" in text
    assert "test_multi_contact_hardening.py" in text


def test_scheduled_replay_workflow_pins_external_actions() -> None:
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
