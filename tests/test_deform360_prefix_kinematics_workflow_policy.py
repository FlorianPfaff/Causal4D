from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "deform360-prefix-kinematics.yml"


def test_prefix_kinematics_workflow_is_read_only_and_review_safe() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "permissions:\n  contents: read\n" in text
    assert "contents: write" not in text
    assert "pull_request_target" not in text
    assert "git push" not in text
    assert text.count("persist-credentials: false") >= 4
    assert "runs-on: ubuntu-latest" in text
    assert "cache: pip" in text


def test_prefix_kinematics_gpu_evidence_requires_explicit_dispatch() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text
    assert "github.event_name == 'workflow_dispatch'" in text
    assert "inputs.run_source_diagnostic" in text
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in text
    assert "BPT_READ_SSH_KEY: ${{ secrets.BPT_READ_SSH_KEY }}" in text
    assert "ssh-key: ${{ secrets.BPT_READ_SSH_KEY }}" in text
    assert "continue-on-error: true" not in text


def test_prefix_kinematics_workflow_pins_inputs_and_archives_evidence() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "ref: 7fea8e20231a47641d1d2bc8791920ec4e62ec5e" in text
    assert "ref: 2b6630528141b9cba5a7677c8b88b2129b4a8390" in text
    assert "read_bpt_pin.py" in text
    assert "run_deform360_prefix_kinematics_workflow.sh" in text
    shell = (
        ROOT / "scripts" / "remote" / "run_deform360_prefix_kinematics_workflow.sh"
    ).read_text(encoding="utf-8")
    assert "--bayesian-phystwin-repo" in shell
    assert "--deform360-repo" in shell
    runner = (
        ROOT / "scripts" / "remote" / "run_deform360_prefix_kinematics.py"
    ).read_text(encoding="utf-8")
    for repository_name in (
        "causal4d",
        "bayesian_phystwin",
        "deform360",
        "official_phystwin",
    ):
        assert f'("{repository_name}",' in runner
    assert "result.runtime.json" in (
        ROOT / "scripts" / "remote" / "run_deform360_prefix_kinematics_workflow.sh"
    ).read_text(encoding="utf-8")
    assert "retention-days: 30" in text
