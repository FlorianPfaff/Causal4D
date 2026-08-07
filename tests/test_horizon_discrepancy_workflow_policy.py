from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(".github/workflows/horizon-discrepancy-integration.yml")
REDUNDANT_SELF_HOSTED_WORKFLOW = Path(
    ".github/workflows/horizon-discrepancy-self-hosted.yml"
)
BPT_HORIZON_PROVIDER_REVISION = "bfa844798f0ab3ddbc67a0744ae14a221324e504"
CAUSAL4D_HEAD_REF = "${{ github.event.pull_request.head.sha || github.sha }}"


def test_horizon_workflow_uses_exact_public_provider_revision() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "repository: IPS-Stuttgart/BayesianPhysTwin" in text
    assert f"ref: {BPT_HORIZON_PROVIDER_REVISION}" in text
    assert "persist-credentials: false" in text
    assert "BPT_READ_SSH_KEY" not in text


def test_horizon_workflow_records_exact_causal4d_revision() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert f"ref: {CAUSAL4D_HEAD_REF}" in text
    assert f"EXPECTED_CAUSAL4D_SHA: {CAUSAL4D_HEAD_REF}" in text
    assert 'actual_sha="$(git rev-parse HEAD)"' in text
    assert 'test "$actual_sha" = "$EXPECTED_CAUSAL4D_SHA"' in text


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


def test_horizon_pull_request_validation_is_hosted_read_only_and_unique() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in text
    assert "permissions:\n  contents: read\n" in text
    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text
    assert not REDUNDANT_SELF_HOSTED_WORKFLOW.exists(), (
        "horizon provider contracts already run as installed wheels on the hosted "
        "workflow; a second PR-head workflow must not allocate workstation2"
    )


def test_horizon_workflow_runs_contract_before_local_quality() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    contract_index = text.index(
        "- name: Exercise the installed cross-repository contract"
    )
    quality_index = text.index("- name: Check Causal4D source quality")

    assert contract_index < quality_index
    assert "if: always()" in text[quality_index:]


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
