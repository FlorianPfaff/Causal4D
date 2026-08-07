from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT / ".github"
    / "workflows"
    / "bayesian-phystwin-provider-compatibility.yml"
)
BPT_PIN = ROOT / "requirements" / "ci" / "bayesian-phystwin-three-repository.sha"
PROB4D_PIN = ROOT / "requirements" / "ci" / "prob4d-three-repository.sha"


def test_public_provider_workflow_is_mandatory_and_secret_free() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Prob4D -> BPT -> Causal4D installed wheels" in text
    assert "repository: IPS-Stuttgart/BayesianPhysTwin" in text
    assert "repository: IPS-Stuttgart/Prob4D" in text
    assert "credential-gate:" not in text
    assert "external-pull-request:" not in text
    assert "BPT_READ_SSH_KEY" not in text
    assert "PROB4D_READ_TOKEN" not in text
    assert "ssh-key:" not in text
    assert "needs: credential-gate" not in text
    assert "steps.prob4d-access" not in text


def test_external_forks_use_the_same_public_installed_wheel_path() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in text
    assert "github.event.pull_request.head.repo.full_name" not in text
    assert "External PR cannot access private providers" not in text
    assert "private golden path" not in text


def test_provider_checkouts_use_non_configurable_immutable_refs() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    bpt_pin = BPT_PIN.read_text(encoding="utf-8").strip()
    prob4d_pin = PROB4D_PIN.read_text(encoding="utf-8").strip()

    assert "workflow_dispatch:" in text
    assert len(bpt_pin) == 40
    assert len(prob4d_pin) == 40
    assert all(character in "0123456789abcdef" for character in bpt_pin)
    assert all(character in "0123456789abcdef" for character in prob4d_pin)
    assert text.count(f"ref: {bpt_pin}") == 1
    assert text.count(f"ref: {prob4d_pin}") == 1
    assert text.count(f'"{BPT_PIN.relative_to(ROOT).as_posix()}"') == 2
    assert text.count(f'"{PROB4D_PIN.relative_to(ROOT).as_posix()}"') == 2
    assert "ref: main" not in text
    assert "ref: ${{" not in text
    assert "inputs.bpt_ref" not in text
    assert "inputs.prob4d_ref" not in text
    assert "Bayesian-PhysTwin revision to test" not in text
    assert "Prob4D revision to test" not in text


def test_strict_claim_bearing_path_is_mandatory() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Run strict claim-bearing provider-v2 admission path" in text
    assert "three_repository_provider_v2_attestation.py" in text
    assert "three-repository-provider-v2-summary.json" in text
    assert "prob4d/tests/test_claim_bearing_observation.py" in text
    assert "bayesian-phystwin/tests/test_prob4d_causal_lineage.py" in text
    assert "causal4d/tests/test_prob4d_stream_contract_version.py" in text
    assert text.count("set -o pipefail") >= 2
    assert text.count('python" -m json.tool') >= 2
    assert text.count('test -s "$RUNNER_TEMP/three-repository-') >= 2
    assert "steps.prob4d-access.outputs.available" not in text


def test_built_wheels_receive_persistent_content_identities() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "Record wheel SHA-256 identities" in text
    assert "sha256sum ./*.whl | sort" in text
    assert "three-repository-wheel-sha256.txt" in text
    assert "Wheel SHA-256 identities" in text


def test_rollout_bank_contract_changes_trigger_installed_wheel_path() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    for path in (
        "src/causal4d/rollout_bank.py",
        "src/causal4d/rollout_bank_io.py",
        "tests/test_causal4d_rollout_bank.py",
        "tests/test_rollout_bank_io.py",
    ):
        assert text.count(f'"{path}"') == 2
