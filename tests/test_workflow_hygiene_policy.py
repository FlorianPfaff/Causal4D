from __future__ import annotations

from pathlib import Path


WORKFLOW_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"
PUBLIC_PROVIDER_WORKFLOWS = (
    "optional-integrations.yml",
    "self-hosted-evaluation.yml",
    "workstation2-evaluation.yml",
    "deform360-prefix-kinematics.yml",
    "deform360-contact-support.yml",
    "deform360-reset-mechanics.yml",
)


def test_no_temporary_self_modifying_workflows_are_shipped() -> None:
    temporary = sorted(path.name for path in WORKFLOW_DIR.glob("temporary-*.yml"))
    assert temporary == []


def test_public_provider_workflows_are_secret_free_and_read_only() -> None:
    for name in PUBLIC_PROVIDER_WORKFLOWS:
        text = (WORKFLOW_DIR / name).read_text(encoding="utf-8")
        assert "BPT_READ_SSH_KEY" not in text
        assert "ssh-key:" not in text
        assert "contents: write" not in text
        assert "issues: write" not in text
        assert "pull-requests: write" not in text
