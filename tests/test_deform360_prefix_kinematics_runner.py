from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "remote" / "run_deform360_prefix_kinematics.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "deform360_prefix_kinematics_runner",
        RUNNER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _selection_payload() -> dict[str, object]:
    candidate = {
        "python": "3.12.3",
        "numpy": "1.26.4",
        "scipy": "1.13.1",
        "torch": "2.4.0+cu121",
        "torch_cuda": "12.1",
        "warp": "1.15.0",
    }
    recorded = {**candidate, "numpy": "2.5.1"}
    return {
        "schema_version": 1,
        "artifact_kind": "Deform360PrefixKinematicsPythonSelection",
        "expected": candidate,
        "runtime_provenance": {
            "status": "conditional-reproduction-runtime-deviation",
            "recorded_runtime": recorded,
            "candidate_runtime": candidate,
            "deviation": {
                "numpy": {"recorded": "2.5.1", "candidate": "1.26.4"}
            },
            "interpretation_permitted_only_after_zero_baseline_reproduction": True,
            "zero_baseline_reproduction_required": True,
        },
        "candidates": [],
        "selected": sys.executable,
        "selected_runtime": {
            **candidate,
            "pytest": "9.1.1",
            "torch_cuda_available": True,
            "warp_cuda_device_count": 1,
        },
    }


def test_runner_accepts_exact_runtime_selection(tmp_path: Path) -> None:
    runner = _load_runner()
    selection = tmp_path / "python-selection.json"
    payload = _selection_payload()
    selection.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    restored = runner._load_runtime_selection(selection)

    assert restored == payload


def test_runner_rejects_another_selected_interpreter(tmp_path: Path) -> None:
    runner = _load_runner()
    selection = tmp_path / "python-selection.json"
    payload = _selection_payload()
    payload["selected"] = "/another/python"
    selection.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="another interpreter"):
        runner._load_runtime_selection(selection)


def test_runner_rejects_relaxed_zero_baseline_gate(tmp_path: Path) -> None:
    runner = _load_runner()
    selection = tmp_path / "python-selection.json"
    payload = _selection_payload()
    provenance = payload["runtime_provenance"]
    assert isinstance(provenance, dict)
    provenance["zero_baseline_reproduction_required"] = False
    selection.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relaxed zero-baseline"):
        runner._load_runtime_selection(selection)


def test_runner_rejects_premature_interpretation(tmp_path: Path) -> None:
    runner = _load_runner()
    selection = tmp_path / "python-selection.json"
    payload = _selection_payload()
    provenance = payload["runtime_provenance"]
    assert isinstance(provenance, dict)
    provenance[
        "interpretation_permitted_only_after_zero_baseline_reproduction"
    ] = False
    selection.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="premature interpretation"):
        runner._load_runtime_selection(selection)


def test_runner_rejects_unregistered_runtime_deviation(tmp_path: Path) -> None:
    runner = _load_runner()
    selection = tmp_path / "python-selection.json"
    payload = _selection_payload()
    provenance = payload["runtime_provenance"]
    assert isinstance(provenance, dict)
    deviation = provenance["deviation"]
    assert isinstance(deviation, dict)
    deviation["numpy"] = {"recorded": "2.5.1", "candidate": "2.0.0"}
    selection.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="declared deviation"):
        runner._load_runtime_selection(selection)
