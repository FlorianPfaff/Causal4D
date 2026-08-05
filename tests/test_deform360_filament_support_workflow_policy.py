from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/deform360-filament-support.yml"


def test_filament_support_workflow_is_read_only_and_source_scoped() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "permissions:\n  contents: read" in text
    assert "runs-on: [self-hosted, Linux, X64, nvidia-smi]" in text
    assert "github.event.pull_request.head.repo.full_name == github.repository" in text
    assert "workflow_dispatch:" in text
    assert "run_source_diagnostic:" in text
    assert "target" not in text.lower()


def test_filament_support_workflow_runs_the_locked_surface() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    required_paths = (
        "configs/causal4d_public/deform360_filament_support_v1.json",
        "docs/causal4d_deform360_filament_support.md",
        "milestones/deform360-reset-mechanics-v1/README.md",
        "milestones/deform360-reset-mechanics-v1/summary.json",
        "scripts/remote/run_deform360_filament_support.py",
        "scripts/remote/run_deform360_filament_support_workflow.sh",
        "src/causal4d_public/deform360_filament_support.py",
        "src/causal4d_public/deform360_replication_graph.py",
        "src/causal4d_public/deform360_rope_graph.py",
        "tests/test_deform360_filament_support.py",
        "tests/test_deform360_filament_support_workflow_policy.py",
    )
    for path in required_paths:
        assert f'      - "{path}"' in text
    assert "python -m ruff check" in text
    assert "python -m ruff format --check" in text
    assert "python -m mypy --no-site-packages" in text
    assert "bash -n scripts/remote/run_deform360_filament_support_workflow.sh" in text
    assert "bash scripts/remote/run_deform360_filament_support_workflow.sh" in text
    assert "if-no-files-found: error" in text


def test_self_hosted_lane_does_not_enable_actions_pip_cache() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    source_job = text.split("  source-diagnostic:", maxsplit=1)[1]
    assert "actions/setup-python" not in source_job
    assert "cache: pip" not in source_job
