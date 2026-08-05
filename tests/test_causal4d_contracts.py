import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from causal4d.contracts import (
    ActionWindow,
    CausalContext,
    CounterfactualQuery,
    FactualIntervention,
    ObservationWindow,
    PhysicalPosterior,
    TaskPosterior,
    TwinBelief,
    array_sha256,
    build_causal_context,
    load_contract,
    save_contract,
)


def _context() -> tuple[CausalContext, np.ndarray]:
    observations = np.arange(8 * 2 * 3, dtype=np.float64).reshape(8, 2, 3)
    actions = np.arange(8 * 1 * 3, dtype=np.float64).reshape(8, 1, 3)
    counterfactual = actions.copy()
    counterfactual[4:, :, 0] *= -1.0
    context = build_causal_context(
        protocol_id="unit_protocol",
        case_id="unit_case",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=counterfactual,
        intervention_frame=4,
    )
    return context, counterfactual[4:]


def _belief(context: CausalContext) -> TwinBelief:
    position = np.zeros((2, 3, 3), dtype=float)
    position[1, :, 0] = 0.01
    return TwinBelief(
        context=context,
        endpoint_frame=3,
        particle_ids=("theta_0", "theta_1"),
        theta_names=("object_log_scale", "controller_log_scale"),
        endpoint_position_m=position,
        endpoint_velocity_mps=np.zeros_like(position),
        theta=np.asarray([[0.0, 0.0], [0.1, -0.1]]),
        discrepancy_mean_m=np.zeros_like(position),
        discrepancy_variance_m2=np.full_like(position, 1e-5),
        weights=np.asarray([0.6, 0.4]),
        metadata={"fit": "O- only"},
    )


def _factual(context: CausalContext, belief: TwinBelief) -> FactualIntervention:
    return FactualIntervention(
        context=context,
        component_ids=("z0", "z1"),
        phi_names=("gain", "delay_s"),
        kappa_names=("contact_node", "slip"),
        phi=np.asarray([[1.0, 0.0], [0.9, 0.03]]),
        kappa_obs=np.asarray([[2.0, 0.0], [1.0, 0.1]]),
        hypothesis_indices=np.asarray([0, 1]),
        twin_particle_indices=np.asarray([0, 1]),
        weights=np.asarray([0.7, 0.3]),
        evidence_frame_stop=6,
        source_twin_belief_id=belief.artifact_id,
    )


def _rewrite_contract_descriptor(
    source: Path,
    target: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    with np.load(source, allow_pickle=False) as archive:
        descriptor = json.loads(str(archive["descriptor_json"]))
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "descriptor_json"
        }
    mutate(descriptor)
    np.savez_compressed(
        target,
        descriptor_json=np.asarray(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
        ),
        **arrays,
    )


def test_context_rejects_overlapping_observation_windows() -> None:
    digest = "0" * 64
    with pytest.raises(ValueError, match="must not overlap"):
        CausalContext(
            protocol_id="bad",
            o_minus=ObservationWindow("case", "points", 0, 5, digest),
            o_plus=ObservationWindow("case", "points", 4, 8, digest),
            u_obs=ActionWindow("observed", "case", 0, 8, digest, "recorded"),
            u_cf=ActionWindow("future", "case", 5, 8, digest, "counterfactual"),
        )


def test_context_hashes_declared_windows_independently() -> None:
    context, _ = _context()
    observations = np.arange(8 * 2 * 3, dtype=np.float64).reshape(8, 2, 3)
    changed = observations.copy()
    changed[4:] += 1000.0
    actions = np.arange(8 * 1 * 3, dtype=np.float64).reshape(8, 1, 3)
    changed_context = build_causal_context(
        protocol_id="unit_protocol",
        case_id="unit_case",
        observations=changed,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=4,
    )
    assert context.o_minus.content_sha256 == changed_context.o_minus.content_sha256
    assert context.o_plus.content_sha256 != changed_context.o_plus.content_sha256


def test_all_contracts_round_trip_with_complete_causal_context(tmp_path: Path) -> None:
    context, future_controls = _context()
    belief = _belief(context)
    factual = _factual(context, belief)
    query = CounterfactualQuery(
        context=context,
        controller_points_m=future_controls,
        horizon_frames=4,
        contact_policy="new_contact",
        source_factual_intervention_id=factual.artifact_id,
        language="move the left endpoint",
        query_node_indices=np.asarray([0, 2]),
    )
    physical = PhysicalPosterior(
        context=context,
        component_ids=("rollout_0", "rollout_1"),
        state_trajectories_m=np.zeros((2, 4, 3, 3)),
        readout_trajectories_m=np.zeros((2, 4, 3, 3)),
        readout_variance_m2=np.full((2, 3, 3), 1e-5),
        weights=np.asarray([0.75, 0.25]),
        phi=np.asarray([[1.0, 0.0], [0.9, 0.03]]),
        kappa_cf=np.asarray([[0.0, 0.0], [2.0, 0.1]]),
        hypothesis_indices=np.asarray([0, 1]),
        twin_particle_indices=np.asarray([0, 1]),
        phi_names=("gain", "delay_s"),
        kappa_names=("contact_node", "slip"),
        source_twin_belief_id=belief.artifact_id,
        source_factual_intervention_id=factual.artifact_id,
        source_query_id=query.artifact_id,
    )
    task = TaskPosterior(
        context=context,
        physical_posterior_id=physical.artifact_id,
        component_ids=physical.component_ids,
        physical_weights=physical.weights,
        task_weights=physical.weights.copy(),
        semantic_log_scores=np.asarray([0.2, -0.2]),
        beta=0.0,
        query_node_indices=np.asarray([0, 2]),
        semantic_source="unit semantic model",
    )

    for index, artifact in enumerate((belief, factual, query, physical, task)):
        path = tmp_path / f"artifact_{index}.npz"
        save_contract(path, artifact)
        restored = load_contract(path)
        assert type(restored) is type(artifact)
        assert restored.artifact_id == artifact.artifact_id
        context_dict = restored.context.as_dict()
        assert set(context_dict) == {
            "protocol_id",
            "o_minus",
            "o_plus",
            "u_obs",
            "u_cf",
        }


def test_twin_belief_keeps_particle_specific_endpoint_states() -> None:
    context, _ = _context()
    belief = _belief(context)
    assert not np.array_equal(
        belief.endpoint_position_m[0], belief.endpoint_position_m[1]
    )
    with pytest.raises(ValueError, match="read-only"):
        belief.endpoint_position_m[0, 0, 0] = 1.0


def test_task_posterior_beta_zero_requires_bit_identical_weights() -> None:
    context, _ = _context()
    physical_id = array_sha256(np.zeros(1))
    physical = np.asarray([0.6, 0.4], dtype=np.float64)
    altered = np.nextafter(physical, np.asarray([1.0, 0.0]))
    altered /= np.sum(altered)
    with pytest.raises(ValueError, match="bit-for-bit"):
        TaskPosterior(
            context=context,
            physical_posterior_id=physical_id,
            component_ids=("a", "b"),
            physical_weights=physical,
            task_weights=altered,
            semantic_log_scores=np.zeros(2),
            beta=0.0,
            query_node_indices=np.asarray([0]),
            semantic_source="unit",
        )


def test_new_contact_query_does_not_carry_factual_contact_identity() -> None:
    context, future_controls = _context()
    factual_id = array_sha256(np.ones(1))
    query = CounterfactualQuery(
        context=context,
        controller_points_m=future_controls,
        horizon_frames=4,
        contact_policy="new_contact",
        source_factual_intervention_id=factual_id,
    )
    assert query.source_factual_intervention_id == factual_id
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        CounterfactualQuery(
            context=context,
            controller_points_m=future_controls,
            horizon_frames=4,
            contact_policy="same_grasp",
            source_factual_intervention_id="not-an-artifact-id",
        )


def test_contract_metadata_is_deeply_immutable_and_artifact_id_stable(
    tmp_path: Path,
) -> None:
    context, _ = _context()
    metadata = {
        "nested": {
            "items": [1, {"accepted": True}],
            "tuple_items": (2, 3),
        }
    }
    belief = TwinBelief(**{**_belief(context).__dict__, "metadata": metadata})
    artifact_id = belief.artifact_id

    metadata["nested"]["items"][1]["accepted"] = False
    assert belief.metadata["nested"]["items"][1]["accepted"] is True
    assert belief.artifact_id == artifact_id
    assert isinstance(belief.metadata, Mapping)
    assert isinstance(belief.metadata["nested"]["items"], Sequence)
    assert not isinstance(belief.metadata, dict)
    assert not isinstance(belief.metadata["nested"]["items"], list)

    with pytest.raises(TypeError, match="immutable"):
        belief.metadata["nested"]["items"][1]["accepted"] = False
    with pytest.raises(TypeError, match="immutable"):
        belief.metadata["nested"]["items"].append("mutated")

    copied = copy.deepcopy(belief.metadata)
    copied["nested"]["items"].append("copy-only")
    assert "copy-only" not in belief.metadata["nested"]["items"]

    path = tmp_path / "immutable-belief.npz"
    save_contract(path, belief)
    restored = load_contract(path)
    assert restored.artifact_id == artifact_id
    assert restored.metadata == belief.metadata


def test_contract_publication_can_refuse_overwrite(tmp_path: Path) -> None:
    context, _ = _context()
    first = _belief(context)
    second = TwinBelief(**{**first.__dict__, "metadata": {"revision": 2}})
    target = tmp_path / "belief.npz"

    save_contract(target, first, overwrite=False)
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        save_contract(target, second, overwrite=False)

    assert target.read_bytes() == original
    assert load_contract(target).artifact_id == first.artifact_id


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda descriptor: descriptor.__setitem__("contract_version", True),
            "contract_version must be an integer",
        ),
        (
            lambda descriptor: descriptor["payload"].__setitem__("endpoint_frame", 3.9),
            "endpoint_frame must be an integer",
        ),
        (
            lambda descriptor: descriptor["context"]["o_minus"].__setitem__(
                "frame_start", 0.0
            ),
            "observation frame_start must be an integer",
        ),
    ),
)
def test_contract_loader_rejects_coercible_identity_fields(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    context, _ = _context()
    valid = tmp_path / "valid.npz"
    malformed = tmp_path / "malformed.npz"
    save_contract(valid, _belief(context))
    _rewrite_contract_descriptor(valid, malformed, mutate)

    with pytest.raises(ValueError, match=message):
        load_contract(malformed)


def test_contract_loader_rejects_unknown_descriptor_and_array_fields(
    tmp_path: Path,
) -> None:
    context, _ = _context()
    valid = tmp_path / "valid.npz"
    unknown_descriptor = tmp_path / "unknown-descriptor.npz"
    save_contract(valid, _belief(context))
    _rewrite_contract_descriptor(
        valid,
        unknown_descriptor,
        lambda descriptor: descriptor.__setitem__("ignored", "ambiguous"),
    )
    with pytest.raises(ValueError, match="descriptor fields do not match schema"):
        load_contract(unknown_descriptor)

    with np.load(valid, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    payload["ignored_array"] = np.asarray([1])
    unknown_array = tmp_path / "unknown-array.npz"
    np.savez_compressed(unknown_array, **payload)
    with pytest.raises(ValueError, match="array fields do not match schema"):
        load_contract(unknown_array)


def test_contract_loader_rejects_noncanonical_float_array_dtype(
    tmp_path: Path,
) -> None:
    context, _ = _context()
    valid = tmp_path / "valid.npz"
    malformed = tmp_path / "wrong-dtype.npz"
    save_contract(valid, _belief(context))

    with np.load(valid, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    payload["endpoint_velocity_mps"] = payload["endpoint_velocity_mps"].astype(
        np.float32
    )
    np.savez_compressed(malformed, **payload)

    with pytest.raises(
        ValueError,
        match=r"endpoint_velocity_mps.*dtype float64.*float32",
    ):
        load_contract(malformed)


@pytest.mark.parametrize(
    "indices",
    (
        np.asarray([0.0, 1.0]),
        np.asarray([False, True]),
        np.asarray(["0", "1"]),
    ),
)
def test_contract_support_indices_reject_coercible_arrays(indices: np.ndarray) -> None:
    context, _ = _context()
    belief = _belief(context)
    factual = _factual(context, belief)
    with pytest.raises(ValueError, match="hypothesis_indices must contain integers"):
        FactualIntervention(
            **{
                **factual.__dict__,
                "hypothesis_indices": indices,
            }
        )


def test_integer_zero_semantic_beta_round_trips_without_identity_drift(
    tmp_path: Path,
) -> None:
    context, _ = _context()
    weights = np.asarray([0.6, 0.4], dtype=np.float64)
    task = TaskPosterior(
        context=context,
        physical_posterior_id=array_sha256(np.zeros(1)),
        component_ids=("a", "b"),
        physical_weights=weights,
        task_weights=weights.copy(),
        semantic_log_scores=np.zeros(2),
        beta=0,
        query_node_indices=np.asarray([0]),
        semantic_source="unit",
    )
    target = tmp_path / "integer-beta.npz"

    save_contract(target, task)
    restored = load_contract(target)

    assert restored.artifact_id == task.artifact_id
    assert restored.beta == 0
    assert type(restored.beta) is int
