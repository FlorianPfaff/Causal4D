from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from causal4d.cli import evaluate_physical_counterfactual
from causal4d.contracts import PhysicalPosterior, build_causal_context, save_contract
from causal4d.held_out_target import (
    HeldOutPhysicalTarget,
    load_held_out_physical_target,
    save_held_out_physical_target,
)
from causal4d.legacy_physical_target import import_legacy_physical_target
from causal4d.physical_evaluation_record import (
    build_physical_counterfactual_evaluation_record,
    load_physical_counterfactual_evaluation_record,
    validate_physical_counterfactual_evaluation_record,
)
from causal4d.physical_validation import (
    evaluate_beta_zero_physical_posterior,
    physical_posterior_moments,
)


def _posterior(*, source_query_id: str = "3" * 64) -> PhysicalPosterior:
    observations = np.zeros((7, 1, 3), dtype=float)
    actions = np.zeros((7, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="held_out_target",
        case_id="synthetic",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=3,
    )
    states = np.zeros((2, 5, 1, 3), dtype=float)
    states[0, :, 0, 0] = np.arange(5) * 0.01
    states[1, :, 0, 0] = np.arange(5) * 0.02
    return PhysicalPosterior(
        context=context,
        component_ids=("a", "b"),
        state_trajectories_m=states,
        readout_trajectories_m=states + np.asarray([0.001, 0.0, 0.0]),
        readout_variance_m2=np.full((2, 1, 3), 1e-5),
        weights=np.asarray([0.75, 0.25]),
        phi=np.asarray([[1.0], [1.0]]),
        kappa_cf=np.asarray([[0.0], [1.0]]),
        hypothesis_indices=np.asarray([0, 1]),
        twin_particle_indices=np.asarray([0, 0]),
        phi_names=("gain",),
        kappa_names=("contact",),
        source_twin_belief_id="1" * 64,
        source_factual_intervention_id="2" * 64,
        source_query_id=source_query_id,
    )


def _target(posterior: PhysicalPosterior) -> HeldOutPhysicalTarget:
    mean, _ = physical_posterior_moments(posterior)
    return HeldOutPhysicalTarget(
        context=posterior.context,
        source_query_id=posterior.source_query_id,
        trajectory_frame_start=posterior.context.o_minus.frame_stop - 1,
        node_indices=np.arange(mean.shape[1], dtype=np.int64),
        positions_m=mean,
        validity_mask=np.ones(mean.shape[:2], dtype=bool),
        source_kind="synthetic_test_target",
        source_revision="test-v1",
        source_content_sha256="4" * 64,
        metadata={"split": "held-out"},
    )


def test_held_out_target_round_trip_is_content_addressed_and_immutable(
    tmp_path: Path,
) -> None:
    posterior = _posterior()
    target = _target(posterior)
    path = tmp_path / "target.npz"
    save_held_out_physical_target(path, target)
    restored = load_held_out_physical_target(path)

    assert restored.artifact_id == target.artifact_id
    assert restored.summary()["source_query_id"] == posterior.source_query_id
    assert restored.positions_m.flags.writeable is False
    assert restored.validity_mask.flags.writeable is False
    assert restored.node_indices.flags.writeable is False
    with pytest.raises(ValueError):
        restored.positions_m[0, 0, 0] = 1.0
    restored.require_compatible_physical_posterior(posterior)


def test_held_out_target_rejects_payload_tampering(tmp_path: Path) -> None:
    target = _target(_posterior())
    path = tmp_path / "target.npz"
    save_held_out_physical_target(path, target)
    with np.load(path, allow_pickle=False) as archive:
        descriptor = np.asarray(archive["descriptor_json"]).copy()
        nodes = np.asarray(archive["node_indices"]).copy()
        positions = np.asarray(archive["positions_m"]).copy()
        validity = np.asarray(archive["validity_mask"]).copy()
    positions[1, 0, 0] += 0.5
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            descriptor_json=descriptor,
            node_indices=nodes,
            positions_m=positions,
            validity_mask=validity,
        )
    with pytest.raises(ValueError, match="artifact_id|payload hashes"):
        load_held_out_physical_target(path)


def test_held_out_target_binds_exact_query_and_shape() -> None:
    posterior = _posterior()
    target = _target(posterior)
    with pytest.raises(ValueError, match="source_query_id"):
        target.require_compatible_physical_posterior(
            _posterior(source_query_id="5" * 64)
        )

    mean, _ = physical_posterior_moments(posterior)
    with pytest.raises(ValueError, match="counterfactual action stop"):
        HeldOutPhysicalTarget(
            context=posterior.context,
            source_query_id=posterior.source_query_id,
            trajectory_frame_start=posterior.context.o_minus.frame_stop - 1,
            node_indices=np.arange(mean.shape[1], dtype=np.int64),
            positions_m=mean[:-1],
            validity_mask=np.ones(mean.shape[:2], dtype=bool)[:-1],
            source_kind="synthetic_test_target",
            source_revision="test-v1",
            source_content_sha256="4" * 64,
        )


def test_legacy_import_requires_explicit_consent_and_exact_digest(
    tmp_path: Path,
) -> None:
    posterior = _posterior()
    mean, _ = physical_posterior_moments(posterior)
    points = np.zeros((7, 1, 3), dtype=float)
    points[2:] = mean
    payload = {
        "object_points": points,
        "object_visibilities": np.ones((7, 1), dtype=bool),
        "object_motions_valid": np.ones((6, 1), dtype=bool),
    }
    source = tmp_path / "final_data.pkl"
    source.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(PermissionError, match="pickle loading is disabled"):
        import_legacy_physical_target(
            posterior,
            source,
            expected_sha256=digest,
        )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        import_legacy_physical_target(
            posterior,
            source,
            allow_unsafe_pickle=True,
            expected_sha256="0" * 64,
        )

    target = import_legacy_physical_target(
        posterior,
        source,
        allow_unsafe_pickle=True,
        expected_sha256=digest,
        source_revision="fixture-v1",
    )
    assert target.source_content_sha256 == digest
    assert target.positions_m.shape == posterior.readout_trajectories_m.shape[1:]
    assert target.metadata["legacy_import"]["format"] == "python-pickle"
    target.require_compatible_physical_posterior(posterior)


def test_evaluation_record_and_cli_bind_both_input_artifacts(tmp_path: Path) -> None:
    posterior = _posterior()
    target = _target(posterior)
    metrics = evaluate_beta_zero_physical_posterior(
        posterior,
        target.positions_m,
        mask=target.validity_mask,
        start_frame=1,
    )
    record = build_physical_counterfactual_evaluation_record(
        posterior,
        target,
        metrics,
        start_frame=1,
        confidence_level=0.90,
    )
    assert len(record["evaluation_id"]) == 64
    assert record["physical_posterior_id"] == posterior.artifact_id
    assert record["held_out_target_id"] == target.artifact_id
    assert record["source_query_id"] == posterior.source_query_id
    assert record["evaluation_frame_interval_absolute"] == [3, 7]

    posterior_path = tmp_path / "physical.npz"
    target_path = tmp_path / "target.npz"
    output_path = tmp_path / "evaluation.json"
    save_contract(posterior_path, posterior)
    save_held_out_physical_target(target_path, target)
    assert (
        evaluate_physical_counterfactual.main(
            [str(posterior_path), str(target_path), str(output_path)]
        )
        == 0
    )
    written = load_physical_counterfactual_evaluation_record(output_path)
    assert written["held_out_target_id"] == target.artifact_id
    assert written["physical_posterior_id"] == posterior.artifact_id
    assert (
        written["held_out_target_descriptor"]["artifact_id"]
        == target.artifact_id
    )
    malformed_target = json.loads(json.dumps(written))
    malformed_target["held_out_target_descriptor"]["metadata"] = {
        "split": "changed"
    }
    with pytest.raises(ValueError, match="held-out target artifact_id"):
        validate_physical_counterfactual_evaluation_record(malformed_target)
    malformed = dict(written)
    malformed["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        validate_physical_counterfactual_evaluation_record(malformed)

    tampered = dict(written)
    tampered["coverage"] = 0.5
    tampered_path = tmp_path / "tampered-evaluation.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation_id"):
        load_physical_counterfactual_evaluation_record(tampered_path)
    with pytest.raises(FileExistsError):
        evaluate_physical_counterfactual.main(
            [
                str(posterior_path),
                str(target_path),
                str(output_path),
                "--no-overwrite",
            ]
        )
