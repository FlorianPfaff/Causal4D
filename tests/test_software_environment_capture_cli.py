from __future__ import annotations

import json

import pytest

from causal4d.cli import preacquisition_readiness as cli


BASE = [
    "capture-software-environment",
    "/opt/causal4d-frozen",
    "/data/causal4d",
    "/artifacts/stack-lock.json",
    "--wheel",
    "/artifacts/prob4d.whl",
    "--wheel",
    "/artifacts/bayesian_phystwin.whl",
    "--wheel",
    "/artifacts/causal4d.whl",
    "--execution-backend",
    "numpy_cpu",
    "--observation-producer-name",
    "registered-rgbd-prefix",
    "--observation-producer-version",
    "1",
    "--observation-artifact-contract",
    "phys4d.observation_belief.v1",
]


def test_capture_parser_requires_explicit_prob4d_declaration() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(BASE)


def test_capture_parser_keeps_used_and_unused_declarations_disjoint() -> None:
    used = cli.build_parser().parse_args(
        [
            *BASE,
            "--prob4d-used",
            "--prob4d-observation-contract-version",
            "phys4d.observation_belief.v1",
        ]
    )
    assert used.prob4d_used is True
    assert used.prob4d_unused_reason is None

    unused = cli.build_parser().parse_args(
        [*BASE, "--prob4d-unused-reason", "fresh real provider is not admitted"]
    )
    assert unused.prob4d_used is False
    assert unused.prob4d_unused_reason == "fresh real provider is not admitted"


def test_capture_cli_forwards_exact_stack_and_runtime_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: dict = {}

    def fake_capture(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {
            "passed": True,
            "ready_to_seal": True,
            "approved": False,
        }

    monkeypatch.setattr(cli, "capture_software_environment_template", fake_capture)
    exit_code = cli.main(
        [
            *BASE,
            "--prob4d-unused-reason",
            "fresh real provider is not admitted",
            "--container-image-digest",
            "sha256:" + "e" * 64,
            "--completed-at-utc",
            "2026-08-08T12:00:00+00:00",
        ]
    )

    assert exit_code == 0
    assert observed["args"] == (
        "/opt/causal4d-frozen",
        "/data/causal4d",
        "/artifacts/stack-lock.json",
        [
            "/artifacts/prob4d.whl",
            "/artifacts/bayesian_phystwin.whl",
            "/artifacts/causal4d.whl",
        ],
    )
    assert observed["kwargs"] == {
        "execution_backend": "numpy_cpu",
        "observation_producer_name": "registered-rgbd-prefix",
        "observation_producer_version": "1",
        "observation_artifact_contract": "phys4d.observation_belief.v1",
        "prob4d_used": False,
        "prob4d_unused_reason": "fresh real provider is not admitted",
        "prob4d_observation_contract_version": None,
        "container_image_digest": "sha256:" + "e" * 64,
        "completed_at_utc": "2026-08-08T12:00:00+00:00",
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready_to_seal"] is True
    assert payload["approved"] is False
