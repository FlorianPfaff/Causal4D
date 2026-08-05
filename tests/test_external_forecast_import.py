from __future__ import annotations

import json
from pathlib import Path

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
from causal4d.external_forecast import (
    EXTERNAL_FORECAST_IMPORT_SCHEMA,
    ExternalForecastBundle,
    import_external_forecast,
    is_external_forecast_artifact,
    load_external_forecast,
    save_external_forecast,
)
from causal4d.semantic_posterior import (
    build_task_posterior,
    external_forecast_evidence,
)


def _write_manifest(path: Path, **overrides: object) -> None:
    manifest: dict[str, object] = {
        "schema": EXTERNAL_FORECAST_IMPORT_SCHEMA,
        "schema_version": 1,
        "case_id": "synthetic",
        "source": {
            "model": "ExampleForecast",
            "revision": "abc123",
            "artifact_id": "checkpoint-v1",
        },
        "arrays": {
            "node_indices": "nodes",
            "anchor_positions": "anchor",
            "future_positions": "future",
            "future_times_s": "times",
            "validity_mask": "valid",
        },
        "layout": "FPC",
        "coordinate_frame": "world",
        "position_unit": "m",
        "forecast_ids": ["instruction"],
        "physical_fps": 1.0,
        "forecast_metadata": {"instruction": {"caption": "Lift the cloth upward."}},
        "metadata": {"producer": "unit-test"},
    }
    manifest.update(overrides)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _physical() -> PhysicalPosterior:
    observations = np.zeros((7, 2, 3), dtype=float)
    actions = np.zeros((7, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="external_forecast_unit",
        case_id="synthetic",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=3,
    )
    states = np.zeros((2, 5, 2, 3), dtype=float)
    states[0, :, 0, 0] = -np.arange(5) * 0.01
    states[1, :, 0, 0] = np.arange(5) * 0.01
    return PhysicalPosterior(
        context=context,
        component_ids=("left", "right"),
        state_trajectories_m=states,
        readout_trajectories_m=states,
        readout_variance_m2=np.full((2, 2, 3), 1e-5),
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


def test_import_round_trip_and_task_posterior_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    physical = _physical()
    source = tmp_path / "producer.npz"
    future = physical.readout_trajectories_m[1, 1:4][:, [0]]
    np.savez(
        source,
        nodes=np.asarray([0], dtype=np.int32),
        anchor=np.zeros((1, 3)),
        future=future,
        times=np.asarray([1.0, 2.0, 3.0]),
        valid=np.ones((3, 1), dtype=bool),
    )
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    canonical = tmp_path / "external_forecast.npz"

    assert import_main([str(source), str(manifest), str(canonical)]) == 0
    import_summary = json.loads(capsys.readouterr().out)
    assert import_summary["forecast_ids"] == ["instruction"]
    assert is_external_forecast_artifact(canonical)

    loaded = load_external_forecast(canonical)
    evidence = external_forecast_evidence(
        loaded,
        "instruction",
        physical,
        scale_m=0.002,
    )
    fallback = build_task_posterior(physical, evidence, beta=0.0)
    assert fallback.task_weights.tobytes() == physical.weights.tobytes()
    task = build_task_posterior(physical, evidence, beta=20.0)
    assert task.task_weights[1] > 0.999
    assert task.metadata["semantic_interface"].startswith("q_external")

    physical_path = tmp_path / "physical.npz"
    task_path = tmp_path / "task.npz"
    save_contract(physical_path, physical)
    assert (
        task_posterior_main(
            [
                str(physical_path),
                str(canonical),
                "instruction",
                str(task_path),
                "--beta",
                "20",
                "--scale-m",
                "0.002",
            ]
        )
        == 0
    )
    task_summary = json.loads(capsys.readouterr().out)
    assert task_summary["forecast_kind"] == "external"
    reloaded_task = load_contract(task_path)
    assert isinstance(reloaded_task, TaskPosterior)
    assert reloaded_task.task_weights[1] > 0.999


def test_camera_mm_import_converts_to_metric_world(tmp_path: Path) -> None:
    source = tmp_path / "camera.npz"
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    np.savez(
        source,
        nodes=np.asarray([0]),
        anchor=np.asarray([[1000.0, 0.0, 0.0]]),
        future=np.asarray([[[2000.0, 0.0, 0.0]]]),
        frames=np.asarray([1.0]),
        c2w=transform,
    )
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        arrays={
            "node_indices": "nodes",
            "anchor_positions": "anchor",
            "future_positions": "future",
            "physical_frame_indices": "frames",
            "camera_to_world": "c2w",
        },
        coordinate_frame="camera",
        position_unit="mm",
        physical_fps=30.0,
    )

    imported = import_external_forecast(source, manifest)
    assert np.allclose(imported.anchor_positions_m, [[2.0, 2.0, 3.0]])
    assert np.allclose(imported.future_positions_m[0, 0], [[3.0, 2.0, 3.0]])


def test_import_rejects_inconsistent_time_and_frame_mapping(tmp_path: Path) -> None:
    source = tmp_path / "producer.npz"
    np.savez(
        source,
        nodes=np.asarray([0]),
        anchor=np.zeros((1, 3)),
        future=np.zeros((1, 1, 3)),
        times=np.asarray([1.0]),
        frames=np.asarray([2.0]),
    )
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        arrays={
            "node_indices": "nodes",
            "anchor_positions": "anchor",
            "future_positions": "future",
            "future_times_s": "times",
            "physical_frame_indices": "frames",
        },
        physical_fps=30.0,
    )
    with pytest.raises(ValueError, match="disagree"):
        import_external_forecast(source, manifest)


def test_loader_rejects_payload_tampering(tmp_path: Path) -> None:
    bundle = ExternalForecastBundle(
        case_id="synthetic",
        source_model="producer",
        forecast_ids=("f",),
        node_indices=np.asarray([0]),
        anchor_positions_m=np.zeros((1, 3)),
        future_positions_m=np.zeros((1, 1, 1, 3)),
        physical_frame_indices=np.asarray([1.0]),
    )
    original = tmp_path / "original.npz"
    save_external_forecast(original, bundle)
    with np.load(original, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    payload["future_positions_m"] = payload["future_positions_m"].copy()
    payload["future_positions_m"][0, 0, 0, 0] = 0.5
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **payload)
    with pytest.raises(ValueError, match="artifact_id|payload hashes"):
        load_external_forecast(tampered)


def test_kpfc_validity_is_unambiguous_when_frame_count_is_three(
    tmp_path: Path,
) -> None:
    source = tmp_path / "producer.npz"
    np.savez(
        source,
        nodes=np.asarray([0, 1]),
        anchor=np.zeros((2, 3)),
        future=np.zeros((1, 2, 3, 3)),
        valid=np.ones((1, 2, 3), dtype=bool),
        frames=np.asarray([1.0, 2.0, 3.0]),
    )
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        arrays={
            "node_indices": "nodes",
            "anchor_positions": "anchor",
            "future_positions": "future",
            "validity_mask": "valid",
            "physical_frame_indices": "frames",
        },
        layout="KPFC",
    )
    imported = import_external_forecast(source, manifest)
    assert imported.coordinate_validity.shape == (1, 3, 2, 3)
