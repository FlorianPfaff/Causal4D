from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "deform360-prefix-kinematics.yml"
PROTOCOL = ROOT / "configs" / "causal4d_public" / "deform360_replication_v1.json"


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
    assert text.count("Set up Python 3.12") >= 2
    assert "continue-on-error: true" not in text


def test_prefix_kinematics_workflow_separates_code_and_dataset_pins() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))["config"]

    assert protocol["deform360_code_commit"] == (
        "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
    )
    assert protocol["dataset_revision"] == ("7fea8e20231a47641d1d2bc8791920ec4e62ec5e")
    assert protocol["deform360_code_commit"] != protocol["dataset_revision"]
    assert protocol["official_phystwin_commit"] == (
        "2b6630528141b9cba5a7677c8b88b2129b4a8390"
    )
    assert 'protocol["deform360_code_commit"]' in text
    assert 'protocol["official_phystwin_commit"]' in text
    assert 'protocol["dataset_revision"]' not in text
    assert "steps.repository-pins.outputs.bpt_sha" in text
    assert "steps.repository-pins.outputs.deform360_sha" in text
    assert "steps.repository-pins.outputs.official_phystwin_sha" in text


def test_prefix_kinematics_workflow_archives_exact_runtime_evidence() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

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
    assert "result.runtime.json" in shell
    assert "retention-days: 30" in text
