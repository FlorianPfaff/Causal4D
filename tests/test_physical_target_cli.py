from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from causal4d.atomic_io import atomic_write_binary
from causal4d.cli import evaluate_physical_counterfactual as evaluate_cli
from causal4d.cli import import_physical_target as import_cli
from causal4d.contracts import (
    PhysicalPosterior,
    build_causal_context,
    load_contract,
    save_contract,
)
from causal4d.physical_target import (
    build_physical_target,
    load_physical_target,
    save_physical_target,
)
from causal4d.physical_validation import evaluate_beta_zero_physical_posterior
from causal4d.trusted_pickle import load_trusted_pickle


def _posterior() -> tuple[PhysicalPosterior, np.ndarray]:
    observations = np.zeros((7, 1, 3), dtype=np.float32)
    actions = np.zeros((7, 1, 3), dtype=np.float64)
    context = build_causal_context(
        protocol_id="physical-target-cli",
        case_id="synthetic",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=3,
    )
    states = np.zeros((1, 5, 1, 3), dtype=np.float32)
    posterior = PhysicalPosterior(
        context=context,
        component_ids=("component",),
        state_trajectories_m=states,
        readout_trajectories_m=states,
        readout_variance_m2=np.full((1, 1, 3), 1e-5, dtype=np.float32),
        weights=np.asarray([1.0]),
        phi=np.asarray([[1.0]]),
        kappa_cf=np.asarray([[0.0]]),
        hypothesis_indices=np.asarray([0]),
        twin_particle_indices=np.asarray([0]),
        phi_names=("gain",),
        kappa_names=("contact",),
        source_twin_belief_id="1" * 64,
        source_factual_intervention_id="2" * 64,
        source_query_id="3" * 64,
    )
    return posterior, observations


def _install_import_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    def load_dependencies() -> None:
        import_cli.target_validity = lambda visible, motion: np.asarray(
            visible, dtype=bool
        )
        import_cli.PhysicalPosterior = PhysicalPosterior
        import_cli.load_contract = load_contract
        import_cli.build_physical_target = build_physical_target
        import_cli.save_physical_target = save_physical_target
        import_cli.load_trusted_pickle = load_trusted_pickle

    monkeypatch.setattr(import_cli, "_load_runtime_dependencies", load_dependencies)


def _install_evaluate_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    def load_dependencies() -> None:
        evaluate_cli.atomic_write_binary = atomic_write_binary
        evaluate_cli.PhysicalPosterior = PhysicalPosterior
        evaluate_cli.load_contract = load_contract
        evaluate_cli.load_physical_target = load_physical_target
        evaluate_cli.evaluate_beta_zero_physical_posterior = (
            evaluate_beta_zero_physical_posterior
        )

    monkeypatch.setattr(evaluate_cli, "_load_runtime_dependencies", load_dependencies)


def test_legacy_import_requires_digest_and_produces_safe_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posterior, observations = _posterior()
    posterior_path = tmp_path / "posterior.npz"
    pickle_path = tmp_path / "final_data.pkl"
    target_path = tmp_path / "target.npz"
    save_contract(posterior_path, posterior)
    pickle_path.write_bytes(
        pickle.dumps(
            {
                "object_points": observations.astype(np.float64),
                "object_visibilities": np.ones(observations.shape[:2], dtype=bool),
                "object_motions_valid": np.ones(observations.shape[:2], dtype=bool),
            }
        )
    )
    digest = hashlib.sha256(pickle_path.read_bytes()).hexdigest()
    _install_import_dependencies(monkeypatch)

    assert (
        import_cli.main(
            [
                str(posterior_path),
                str(pickle_path),
                str(target_path),
                "--allow-unsafe-pickle",
                "--expected-sha256",
                digest,
            ]
        )
        == 0
    )
    target = load_physical_target(target_path)
    assert target.source_final_data_sha256 == digest
    assert target.object_points.dtype == np.dtype(np.float32)
    assert target.context == posterior.context


def test_safe_evaluation_binds_target_and_publishes_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posterior, observations = _posterior()
    posterior_path = tmp_path / "posterior.npz"
    target_path = tmp_path / "target.npz"
    output_path = tmp_path / "evaluation.json"
    save_contract(posterior_path, posterior)
    target = build_physical_target(
        posterior.context,
        observations,
        np.ones(observations.shape[:2], dtype=bool),
        source_final_data_sha256="a" * 64,
    )
    save_physical_target(target_path, target)
    _install_evaluate_dependencies(monkeypatch)

    assert (
        evaluate_cli.main(
            [
                str(posterior_path),
                str(target_path),
                str(output_path),
                "--start-frame",
                "2",
            ]
        )
        == 0
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["physical_target_id"] == target.artifact_id
    assert result["physical_posterior_id"] == posterior.artifact_id
    assert (
        result["evaluation_target_id"]
        == target.evaluation_target(start_frame=2).artifact_id
    )
    assert len(result["evaluation_id"]) == 64

    with pytest.raises(FileExistsError):
        evaluate_cli.main(
            [
                str(posterior_path),
                str(target_path),
                str(output_path),
                "--start-frame",
                "2",
            ]
        )


def test_safe_evaluation_rejects_symlinked_output_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posterior, observations = _posterior()
    posterior_path = tmp_path / "posterior.npz"
    target_path = tmp_path / "target.npz"
    real_output = tmp_path / "real-output"
    linked_output = tmp_path / "linked-output"
    save_contract(posterior_path, posterior)
    target = build_physical_target(
        posterior.context,
        observations,
        np.ones(observations.shape[:2], dtype=bool),
        source_final_data_sha256="a" * 64,
    )
    save_physical_target(target_path, target)
    real_output.mkdir()
    try:
        linked_output.symlink_to(real_output, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    _install_evaluate_dependencies(monkeypatch)

    with pytest.raises(ValueError, match="output path contains a symlink"):
        evaluate_cli.main(
            [
                str(posterior_path),
                str(target_path),
                str(linked_output / "evaluation.json"),
                "--start-frame",
                "2",
            ]
        )
