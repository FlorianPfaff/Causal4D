from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "scripts" / "remote" / "select_deform360_prefix_kinematics_python.py"


def _load_selector():
    spec = importlib.util.spec_from_file_location(
        "deform360_prefix_kinematics_python_selector",
        SELECTOR,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_mismatches_requires_exact_versions_and_cuda() -> None:
    selector = _load_selector()
    expected = {key: "locked" for key in selector._EXPECTED_KEYS}
    observed = {
        **expected,
        "pytest": "9.1.1",
        "torch_cuda_available": True,
        "warp_cuda_device_count": 1,
    }

    assert selector.runtime_mismatches(expected, observed) == []

    observed["numpy"] = "different"
    observed["torch_cuda_available"] = False
    observed["warp_cuda_device_count"] = 0
    mismatches = selector.runtime_mismatches(expected, observed)
    assert any(message.startswith("numpy:") for message in mismatches)
    assert "PyTorch cannot see CUDA" in mismatches
    assert "Warp cannot see a CUDA device" in mismatches


def test_candidate_paths_prioritize_configured_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = _load_selector()
    configured = tmp_path / "configured-python"
    configured.write_text("#!/bin/sh\n", encoding="utf-8")
    configured.chmod(0o755)
    monkeypatch.setenv("PREFIX_KINEMATICS_PYTHON", str(configured))

    candidates = selector._candidate_paths([str(configured)])

    assert candidates[0] == configured.absolute()
    assert candidates.count(configured.absolute()) == 1


def test_reproduction_runtime_deviation_is_conditional_and_exact() -> None:
    selector = _load_selector()
    original_path = ROOT / selector._ENVIRONMENT_PATH
    original = json.loads(original_path.read_text(encoding="utf-8"))

    expected, provenance = selector._load_runtime_lock(ROOT)

    assert original["numpy"] == "2.5.1"
    assert expected == {
        "python": "3.12.3",
        "numpy": "1.26.4",
        "scipy": "1.13.1",
        "torch": "2.4.0+cu121",
        "torch_cuda": "12.1",
        "warp": "1.15.0",
    }
    assert provenance["status"] == "conditional-reproduction-runtime-deviation"
    assert provenance["recorded_runtime"]["numpy"] == "2.5.1"
    assert provenance["candidate_runtime"]["numpy"] == "1.26.4"
    assert provenance["deviation"] == {
        "numpy": {"recorded": "2.5.1", "candidate": "1.26.4"}
    }
    assert provenance["zero_baseline_reproduction_required"] is True
    assert (
        provenance[
            "interpretation_permitted_only_after_zero_baseline_reproduction"
        ]
        is True
    )
    assert provenance["recorded_environment_sha256"] == (
        "2274f2a38e5b49a9e1fc5e4c49c80910d2095cf43e8b1e84928c6cc3d99b2d8c"
    )
    assert provenance["reproduction_runtime_contract_sha256"] == (
        "144ea36f828703a713ff3bc3afe49ff0518926d73544bf7b477bdf9eb5f17f98"
    )


def test_reproduction_runtime_rejects_content_identity_drift(
    tmp_path: Path,
) -> None:
    selector = _load_selector()
    for relative in (
        selector._ENVIRONMENT_PATH,
        selector._REPRODUCTION_RUNTIME_PATH,
        Path(
            "milestones/deform360-replication-source-backend-v1/verification/"
            "test-and-lint.txt"
        ),
        Path("milestones/v0.3.0-causal4d-aip/environment/bpt-gpu-pip-freeze.txt"),
    ):
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    contract_path = tmp_path / selector._REPRODUCTION_RUNTIME_PATH
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["candidate_runtime"]["numpy"] = "2.0.0"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="content checksum changed"):
        selector._load_runtime_lock(tmp_path)


def test_expected_runtime_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    selector = _load_selector()
    environment = tmp_path / selector._ENVIRONMENT_PATH
    environment.parent.mkdir(parents=True)
    environment.write_text(
        '{"python":"3.12.3","python":"3.12.4"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        selector._expected_runtime(tmp_path)
