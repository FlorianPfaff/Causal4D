from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "controlled-demo-video.yml"


def _render_job(text: str) -> str:
    return text.split("  render:", maxsplit=1)[1]


def test_controlled_demo_workflow_is_read_only_and_runner_selectable() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "permissions:\n  contents: read\n" in text
    assert "workflow_dispatch:" in text
    assert "default: github-hosted" in text
    assert "'ubuntu-latest'" in text
    assert (
        "fromJSON('[\"self-hosted\",\"Linux\",\"X64\",\"collaborator-demo\"]')"
        in text
    )
    assert "persist-credentials: false" in text
    assert "contents: write" not in text
    assert "git push" not in text


def test_controlled_demo_self_hosted_dispatch_is_main_only() -> None:
    job = _render_job(WORKFLOW.read_text(encoding="utf-8"))

    assert "inputs.runner != 'self-hosted'" in job
    assert "github.ref == 'refs/heads/main'" in job
    assert "runs-on:" in job


def test_controlled_demo_checks_out_and_verifies_exact_dispatch_revision() -> None:
    job = _render_job(WORKFLOW.read_text(encoding="utf-8"))

    assert "ref: ${{ github.sha }}" in job
    assert "clean: true" in job
    assert 'test "$(git rev-parse HEAD)" = "${GITHUB_SHA}"' in job
    checkout = job.index("- name: Check out exact reviewed Causal4D revision")
    verify = job.index("- name: Verify reviewed dispatch revision")
    install = job.index("- name: Install controlled-demo renderer")
    assert checkout < verify < install


def test_controlled_demo_installs_non_editable_distribution() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "python -m pip install . matplotlib pillow imageio-ffmpeg" in text
    assert "python -m pip install -e ." not in text
    assert "python -m pip check" in text


def test_controlled_demo_verifies_and_uploads_complete_bundle() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    required = (
        "causal4d_dynamic_contact_demo.mp4",
        "causal4d_dynamic_contact_demo.gif",
        "causal4d_dynamic_contact_poster.png",
        "summary.json",
        "summary.md",
        "SHA256SUMS",
    )
    for name in required:
        assert name in text
    assert "if-no-files-found: error" in text
    assert "retention-days: 30" in text
