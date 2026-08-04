from __future__ import annotations

from pathlib import Path

import pytest

from causal4d.cli.dynamic_contact_benchmark import (
    delayed_contact_case,
    delayed_contact_trace,
)
from causal4d.cli.dynamic_contact_demo import aggregate_demo_cases


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "controlled-demo-video.yml"
)


def test_delayed_contact_trace_preserves_registered_summary() -> None:
    trace = delayed_contact_trace(seed=3, prefix_frame_count=8)

    assert trace.summary == delayed_contact_case(seed=3, prefix_frame_count=8)
    assert trace.summary["future_observations_read"] == 0
    assert trace.truth_m.shape == trace.posterior.mean_m.shape
    assert trace.static_persistence_m.shape == trace.truth_m.shape


def test_demo_aggregate_matches_registered_smoke_cases() -> None:
    aggregate = aggregate_demo_cases()

    assert aggregate["case_count"] == 40
    assert aggregate["gate_pass_count"] == 40
    assert aggregate["all_gates_passed"] is True
    assert aggregate["future_observations_read"] == 0
    assert aggregate["mean_static_persistence_rmse_mm"] == pytest.approx(
        35.496421304347976
    )
    assert aggregate["mean_dynamic_contact_rmse_mm"] == pytest.approx(
        0.14398398253490627
    )
    assert aggregate["mean_relative_rmse_improvement_percent"] == pytest.approx(
        99.58777530093099
    )


def test_demo_workflow_uploads_all_presentation_artifacts() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text
    assert "default: github-hosted" in text
    assert "'ubuntu-latest'" in text
    assert (
        "fromJSON('[\"self-hosted\",\"Linux\",\"X64\",\"collaborator-demo\"]')"
        in text
    )
    assert "causal4d.cli.dynamic_contact_demo" in text
    assert "--require-gates" in text
    assert "causal4d_dynamic_contact_demo.mp4" in text
    assert "causal4d_dynamic_contact_demo.gif" in text
    assert "causal4d_dynamic_contact_poster.png" in text
    assert "summary.json" in text
    assert "SHA256SUMS" in text
    assert "actions/upload-artifact@" in text


def test_demo_workflow_caches_only_on_hosted_runners() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    hosted = "      - name: Set up Python 3.12 with pip cache\n"
    self_hosted = "      - name: Set up Python 3.12 without Actions cache\n"
    install = "      - name: Install controlled-demo renderer\n"
    assert hosted in text
    assert self_hosted in text
    assert "        if: inputs.runner != 'self-hosted'\n" in text
    assert "        if: inputs.runner == 'self-hosted'\n" in text
    assert text.count("          cache: pip\n") == 1
    assert text.index(hosted) < text.index(self_hosted) < text.index(install)
