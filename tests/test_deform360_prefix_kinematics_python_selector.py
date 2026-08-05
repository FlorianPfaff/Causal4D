from __future__ import annotations

import importlib.util
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


def test_expected_runtime_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    selector = _load_selector()
    environment = (
        tmp_path
        / "milestones"
        / "deform360-replication-source-backend-v1"
        / "verification"
        / "environment.json"
    )
    environment.parent.mkdir(parents=True)
    environment.write_text(
        '{"python":"3.12.3","python":"3.12.4"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        selector._expected_runtime(tmp_path)
