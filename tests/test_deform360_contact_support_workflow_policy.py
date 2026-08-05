from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "deform360-contact-support.yml"
TEMPORARY_WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "temporary-deform360-contact-support-evidence.yml"
)
SHELL = ROOT / "scripts" / "remote" / "run_deform360_contact_support_workflow.sh"
LOCK = ROOT / "configs" / "causal4d_public" / "deform360_contact_support_v1.json"


def test_permanent_gpu_evidence_requires_explicit_dispatch() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text
    assert "github.event_name == 'workflow_dispatch'" in text
    assert "inputs.run_source_diagnostic" in text
    assert "needs: contract" in text
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in text
    assert "permissions:\n  contents: read" in text
    assert "contents: write" not in text
    assert "continue-on-error: true" not in text
    assert "target" not in text.lower().split("information boundary", maxsplit=1)[0]


def test_temporary_evidence_is_pr_only_and_exact_head_bound() -> None:
    text = TEMPORARY_WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in text
    assert "workflow_dispatch:" not in text
    assert "push:" not in text
    assert "PR_HEAD_SHA: ${{ github.event.pull_request.head.sha }}" in text
    assert "ref: ${{ env.PR_HEAD_SHA }}" in text
    assert "github.event.pull_request.head.repo.full_name == github.repository" in text
    assert "needs: contract" in text
    assert "permissions:\n  contents: read" in text
    assert "contents: write" not in text
    assert "persist-credentials: false" in text
    assert "if: always()" in text


def test_gpu_paths_reuse_the_locked_conditional_runtime() -> None:
    permanent = WORKFLOW.read_text(encoding="utf-8")
    temporary = TEMPORARY_WORKFLOW.read_text(encoding="utf-8")
    shell = SHELL.read_text(encoding="utf-8")

    for text in (permanent, temporary):
        assert "select_deform360_prefix_kinematics_python.py" in text
        assert "python-selection.json" in text
        assert "CONTACT_SUPPORT_PYTHON=" in text
        assert "BPT_READ_SSH_KEY" in text
        assert "IPS-Stuttgart/BayesianPhysTwin" in text
        assert "lhy0807/deform360" in text
        assert "Jianghanxiao/PhysTwin" in text
    assert 'python_bin="${CONTACT_SUPPORT_PYTHON:-python3}"' in shell
    assert "select_deform360_prefix_kinematics_python.py" in shell
    assert '"$python_bin" -m pytest' in shell
    assert '"$python_bin" "$repository_root/scripts/remote/' in shell
    assert LOCK.is_file()


def test_workflows_run_the_locked_source_only_entrypoint() -> None:
    permanent = WORKFLOW.read_text(encoding="utf-8")
    temporary = TEMPORARY_WORKFLOW.read_text(encoding="utf-8")
    shell = SHELL.read_text(encoding="utf-8")

    for text in (permanent, temporary):
        assert "run_deform360_contact_support_workflow.sh" in text
        assert "deform360-contact-support" in text
        assert "Upload" in text
    assert "run_deform360_contact_support.py" in shell
    assert "--runtime-selection" in shell
    assert "--device cuda:0" in shell
    assert "target" not in shell.lower()
