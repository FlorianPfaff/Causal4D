from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from causal4d.cli.external_forecast_import import main as import_main
from causal4d.cli.molmo_task_posterior import main as task_posterior_main
from causal4d.contracts import (
    PhysicalPosterior,
    TaskPosterior,
    build_causal_context,
    load_contract,
    save_contract,
)
from causal4d.external_forecast import import_external_forecast


EXAMPLE_DIR = (
    Path(__file__).resolve().parents[1] / "examples" / "molmomotion_physics_bridge"
)


def _load_example_module(filename: str) -> ModuleType:
    path = EXAMPLE_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load example module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matching_physical_posterior(bundle) -> PhysicalPosterior:
    horizon = int(np.ceil(bundle.physical_frame_indices[-1]))
    node_count = int(np.max(bundle.node_indices)) + 1
    observations = np.zeros((horizon + 1, node_count, 3), dtype=float)
    actions = np.zeros((horizon + 1, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="molmomotion_bridge_example",
        case_id=bundle.case_id,
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=bundle.anchor_physical_frame,
    )

    states = np.zeros((2, horizon + 1, node_count, 3), dtype=float)
    states[:, :, bundle.node_indices] = bundle.anchor_positions_m[None, None]
    forecast = bundle.future_positions_m[bundle.forecast_index("instruction")]
    for frame, positions in zip(
        bundle.physical_frame_indices.astype(int),
        forecast,
        strict=True,
    ):
        states[1, frame, bundle.node_indices] = positions

    return PhysicalPosterior(
        context=context,
        component_ids=("stationary", "matching"),
        state_trajectories_m=states,
        readout_trajectories_m=states,
        readout_variance_m2=np.full((2, node_count, 3), 1e-5),
        weights=np.asarray([0.6, 0.4]),
        phi=np.asarray([[1.0], [1.0]]),
        kappa_cf=np.asarray([[0.0], [1.0]]),
        hypothesis_indices=np.asarray([0, 1]),
        twin_particle_indices=np.asarray([0, 0]),
        phi_names=("gain",),
        kappa_names=("contact",),
        source_twin_belief_id="1" * 64,
        source_factual_intervention_id="2" * 64,
        source_query_id="3" * 64,
    )


def test_runnable_bridge_export_import_and_exact_fallback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    demo = _load_example_module("make_demo_input.py")
    exporter = _load_example_module("export_molmo_forecast.py")

    raw = tmp_path / "molmo_raw.npz"
    producer = tmp_path / "producer_forecast.npz"
    manifest = tmp_path / "external_forecast_manifest.json"
    canonical = tmp_path / "canonical_forecast.npz"

    assert demo.main([str(raw), "--future-count", "4"]) == 0
    demo_summary = json.loads(capsys.readouterr().out)
    assert demo_summary["forecast_shape_kpfc"] == [3, 8, 4, 3]

    assert (
        exporter.main(
            [
                str(raw),
                str(producer),
                str(manifest),
                "--case-id",
                "single_lift_cloth",
                "--source-revision",
                "demo-checkpoint",
                "--anchor-physical-frame",
                "70",
                "--physical-fps",
                "30",
                "--forecast-fps",
                "15",
                "--forecast",
                "instruction=Lift the cloth upward.",
                "--forecast",
                "paraphrase=Raise the cloth vertically with one hand.",
                "--forecast",
                "shuffled=Push the cloth sideways.",
            ]
        )
        == 0
    )
    export_summary = json.loads(capsys.readouterr().out)
    assert export_summary["physical_frame_indices"] == [72.0, 74.0, 76.0, 78.0]

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["layout"] == "KPFC"
    assert manifest_payload["forecast_ids"] == [
        "instruction",
        "paraphrase",
        "shuffled",
    ]

    imported = import_external_forecast(producer, manifest)
    assert imported.future_positions_m.shape == (3, 4, 8, 3)
    assert imported.source_model == "MolmoMotion"
    assert imported.source_revision == "demo-checkpoint"

    assert import_main([str(producer), str(manifest), str(canonical)]) == 0
    import_summary = json.loads(capsys.readouterr().out)
    assert import_summary["forecast_ids"] == [
        "instruction",
        "paraphrase",
        "shuffled",
    ]

    physical = _matching_physical_posterior(imported)
    physical_path = tmp_path / "physical.npz"
    task_zero_path = tmp_path / "task_beta0.npz"
    task_positive_path = tmp_path / "task_beta20.npz"
    save_contract(physical_path, physical)

    assert (
        task_posterior_main(
            [
                str(physical_path),
                str(canonical),
                "instruction",
                str(task_zero_path),
                "--beta",
                "0",
                "--scale-m",
                "0.005",
            ]
        )
        == 0
    )
    zero_summary = json.loads(capsys.readouterr().out)
    assert zero_summary["forecast_kind"] == "external"
    assert zero_summary["weights_bit_identical"] is True
    zero_task = load_contract(task_zero_path)
    assert isinstance(zero_task, TaskPosterior)
    assert zero_task.task_weights.tobytes() == physical.weights.tobytes()

    assert (
        task_posterior_main(
            [
                str(physical_path),
                str(canonical),
                "instruction",
                str(task_positive_path),
                "--beta",
                "20",
                "--scale-m",
                "0.005",
            ]
        )
        == 0
    )
    capsys.readouterr()
    positive_task = load_contract(task_positive_path)
    assert isinstance(positive_task, TaskPosterior)
    assert positive_task.task_weights[1] > 0.999


def test_export_helper_rejects_forecast_count_mismatch(tmp_path: Path) -> None:
    demo = _load_example_module("make_demo_input.py")
    exporter = _load_example_module("export_molmo_forecast.py")
    raw = tmp_path / "molmo_raw.npz"
    demo.main([str(raw)])

    with pytest.raises(ValueError, match="forecast count"):
        exporter.export_bridge_package(
            raw,
            tmp_path / "producer.npz",
            tmp_path / "manifest.json",
            case_id="single_lift_cloth",
            source_model="MolmoMotion",
            source_revision="demo",
            source_artifact_id=None,
            forecasts=(("instruction", "Lift the cloth upward."),),
            anchor_physical_frame=70,
            physical_fps=30.0,
            forecast_fps=15.0,
            producer_environment="test",
        )
