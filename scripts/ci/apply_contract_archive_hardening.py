from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "src" / "causal4d" / "contracts.py"
TESTS = ROOT / "tests" / "test_contract_archive_strict_loading.py"
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "tmp-contract-archive-standalone-apply.yml"
)


def replace_once(text: str, old: str, new: str, *, name: str) -> str:
    if old not in text:
        raise RuntimeError(f"expected patch anchor is missing: {name}")
    return text.replace(old, new, 1)


def replace_member(
    text: str,
    *,
    signature: str,
    next_marker: str,
    replacement: str,
) -> str:
    start = text.index("    @classmethod\n" + signature)
    stop = text.index(next_marker, start)
    return text[:start] + replacement.rstrip() + "\n\n\n" + text[stop:]


def patch_contracts() -> None:
    text = CONTRACTS.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from dataclasses import dataclass, field\n"
        "from pathlib import Path\n"
        "from typing import Any, BinaryIO, ClassVar, Literal, Mapping\n",
        "from collections.abc import Mapping\n"
        "from dataclasses import dataclass, field\n"
        "from pathlib import Path\n"
        "from typing import Any, BinaryIO, ClassVar, Literal, cast\n",
        name="contracts imports",
    )
    text = replace_once(
        text,
        "def _validate_sha256(value: str, *, name: str) -> None:\n"
        "    if len(value) != 64 or any(\n",
        "def _validate_sha256(value: str, *, name: str) -> None:\n"
        "    if type(value) is not str or len(value) != 64 or any(\n",
        name="SHA-256 type validation",
    )

    helper_marker = "\n\n@dataclass(frozen=True)\nclass ObservationWindow:"
    helpers = r'''

_OBSERVATION_WINDOW_FIELDS = frozenset(
    {"case_id", "stream_id", "frame_start", "frame_stop", "content_sha256"}
)
_ACTION_WINDOW_FIELDS = frozenset(
    {
        "action_id",
        "case_id",
        "frame_start",
        "frame_stop",
        "trajectory_sha256",
        "provenance",
    }
)
_CAUSAL_CONTEXT_FIELDS = frozenset(
    {"protocol_id", "o_minus", "o_plus", "u_obs", "u_cf"}
)
_CONTRACT_DESCRIPTOR_FIELDS = frozenset(
    {"contract_version", "contract_type", "artifact_id", "context", "payload"}
)


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _require_exact_fields(
    value: Any,
    *,
    fields: frozenset[str],
    name: str,
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, name=name)
    actual = set(mapping)
    missing = sorted(fields - actual)
    unexpected = sorted(actual - fields)
    if missing or unexpected:
        raise ValueError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return mapping


def _require_string(value: Any, *, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be a string")
    return value


def _require_optional_string(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, name=name)


def _require_integer(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _require_finite_float(value: Any, *, name: str) -> float:
    if type(value) is not float or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def _require_string_list(value: Any, *, name: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array of strings")
    return tuple(
        _require_string(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )


def _json_object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _strict_json_mapping(text: str, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_json_object_from_pairs,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be strict finite JSON") from error
    return _require_mapping(value, name=name)


def _read_descriptor_text(archive: Any) -> str:
    files = tuple(archive.files)
    if files.count("descriptor_json") != 1:
        raise ValueError("contract archive must contain descriptor_json exactly once")
    descriptor = np.asarray(archive["descriptor_json"])
    if descriptor.shape != () or descriptor.dtype.kind != "U":
        raise ValueError("descriptor_json must be a scalar Unicode array")
    value = descriptor.item()
    if type(value) is not str:
        raise ValueError("descriptor_json must contain a string")
    return value
'''
    text = replace_once(
        text,
        helper_marker,
        helpers + helper_marker,
        name="strict archive helpers",
    )

    text = replace_member(
        text,
        signature=(
            "    def from_dict(cls, values: Mapping[str, Any]) "
            "-> ObservationWindow:\n"
        ),
        next_marker="@dataclass(frozen=True)\nclass ActionWindow:",
        replacement='''    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ObservationWindow:
        payload = _require_exact_fields(
            values,
            fields=_OBSERVATION_WINDOW_FIELDS,
            name="observation window",
        )
        return cls(
            case_id=_require_string(payload["case_id"], name="case_id"),
            stream_id=_require_string(payload["stream_id"], name="stream_id"),
            frame_start=_require_integer(
                payload["frame_start"], name="frame_start"
            ),
            frame_stop=_require_integer(payload["frame_stop"], name="frame_stop"),
            content_sha256=_require_string(
                payload["content_sha256"], name="content_sha256"
            ),
        )''',
    )
    text = replace_member(
        text,
        signature=(
            "    def from_dict(cls, values: Mapping[str, Any]) "
            "-> ActionWindow:\n"
        ),
        next_marker="@dataclass(frozen=True)\nclass CausalContext:",
        replacement='''    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ActionWindow:
        payload = _require_exact_fields(
            values,
            fields=_ACTION_WINDOW_FIELDS,
            name="action window",
        )
        return cls(
            action_id=_require_string(payload["action_id"], name="action_id"),
            case_id=_require_string(payload["case_id"], name="case_id"),
            frame_start=_require_integer(
                payload["frame_start"], name="frame_start"
            ),
            frame_stop=_require_integer(payload["frame_stop"], name="frame_stop"),
            trajectory_sha256=_require_string(
                payload["trajectory_sha256"], name="trajectory_sha256"
            ),
            provenance=_require_string(payload["provenance"], name="provenance"),
        )''',
    )
    text = replace_member(
        text,
        signature=(
            "    def from_dict(cls, values: Mapping[str, Any]) "
            "-> CausalContext:\n"
        ),
        next_marker="def build_causal_context(",
        replacement='''    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> CausalContext:
        payload = _require_exact_fields(
            values,
            fields=_CAUSAL_CONTEXT_FIELDS,
            name="causal context",
        )
        return cls(
            protocol_id=_require_string(
                payload["protocol_id"], name="protocol_id"
            ),
            o_minus=ObservationWindow.from_dict(payload["o_minus"]),
            o_plus=ObservationWindow.from_dict(payload["o_plus"]),
            u_obs=ActionWindow.from_dict(payload["u_obs"]),
            u_cf=ActionWindow.from_dict(payload["u_cf"]),
        )''',
    )

    contract_marker = '''Contract = (
    TwinBelief
    | FactualIntervention
    | CounterfactualQuery
    | PhysicalPosterior
    | TaskPosterior
)


def save_contract(
'''
    contract_schema = r'''Contract = (
    TwinBelief
    | FactualIntervention
    | CounterfactualQuery
    | PhysicalPosterior
    | TaskPosterior
)

_CONTRACT_TYPES = frozenset(
    {
        TwinBelief.contract_type,
        FactualIntervention.contract_type,
        CounterfactualQuery.contract_type,
        PhysicalPosterior.contract_type,
        TaskPosterior.contract_type,
    }
)
_CONTRACT_PAYLOAD_FIELDS = {
    TwinBelief.contract_type: frozenset(
        {"endpoint_frame", "particle_ids", "theta_names", "metadata"}
    ),
    FactualIntervention.contract_type: frozenset(
        {
            "component_ids",
            "phi_names",
            "kappa_names",
            "evidence_frame_stop",
            "source_twin_belief_id",
            "metadata",
        }
    ),
    CounterfactualQuery.contract_type: frozenset(
        {
            "horizon_frames",
            "contact_policy",
            "language",
            "source_factual_intervention_id",
            "metadata",
        }
    ),
    PhysicalPosterior.contract_type: frozenset(
        {
            "component_ids",
            "phi_names",
            "kappa_names",
            "source_twin_belief_id",
            "source_factual_intervention_id",
            "source_query_id",
            "metadata",
        }
    ),
    TaskPosterior.contract_type: frozenset(
        {
            "physical_posterior_id",
            "component_ids",
            "beta",
            "semantic_source",
            "metadata",
        }
    ),
}
_CONTRACT_ARRAY_DTYPES = {
    TwinBelief.contract_type: {
        "endpoint_position_m": np.dtype(np.float64),
        "endpoint_velocity_mps": np.dtype(np.float64),
        "theta": np.dtype(np.float64),
        "discrepancy_mean_m": np.dtype(np.float64),
        "discrepancy_variance_m2": np.dtype(np.float64),
        "weights": np.dtype(np.float64),
    },
    FactualIntervention.contract_type: {
        "phi": np.dtype(np.float64),
        "kappa_obs": np.dtype(np.float64),
        "hypothesis_indices": np.dtype(np.int64),
        "twin_particle_indices": np.dtype(np.int64),
        "weights": np.dtype(np.float64),
    },
    CounterfactualQuery.contract_type: {
        "controller_points_m": np.dtype(np.float64),
        "query_node_indices": np.dtype(np.int64),
    },
    PhysicalPosterior.contract_type: {
        "state_trajectories_m": np.dtype(np.float32),
        "readout_trajectories_m": np.dtype(np.float32),
        "readout_variance_m2": np.dtype(np.float32),
        "weights": np.dtype(np.float64),
        "phi": np.dtype(np.float64),
        "kappa_cf": np.dtype(np.float64),
        "hypothesis_indices": np.dtype(np.int64),
        "twin_particle_indices": np.dtype(np.int64),
    },
    TaskPosterior.contract_type: {
        "physical_weights": np.dtype(np.float64),
        "task_weights": np.dtype(np.float64),
        "semantic_log_scores": np.dtype(np.float64),
        "query_node_indices": np.dtype(np.int64),
    },
}
_CONTRACT_OPTIONAL_ARRAYS = {
    CounterfactualQuery.contract_type: frozenset({"query_node_indices"}),
}


def _validate_archive_inventory(archive: Any, *, kind: str) -> None:
    files = tuple(archive.files)
    if len(files) != len(set(files)):
        raise ValueError("contract archive contains duplicate member names")
    actual = set(files) - {"descriptor_json"}
    expected_dtypes = _CONTRACT_ARRAY_DTYPES[kind]
    expected = set(expected_dtypes)
    optional = _CONTRACT_OPTIONAL_ARRAYS.get(kind, frozenset())
    required = expected - optional
    missing = sorted(required - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"{kind} array inventory does not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name in sorted(actual):
        actual_dtype = np.asarray(archive[name]).dtype
        expected_dtype = expected_dtypes[name]
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"{kind} array {name!r} must use dtype {expected_dtype}; "
                f"got {actual_dtype}"
            )


def save_contract(
'''
    text = replace_once(
        text,
        contract_marker,
        contract_schema,
        name="contract archive schemas",
    )
    text = replace_once(
        text,
        'json.dumps(descriptor, sort_keys=True, separators=(",", ":"))',
        '''json.dumps(
                    descriptor,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )''',
        name="finite descriptor serialization",
    )

    load_start = text.index("def load_contract(path: str | Path) -> Contract:\n")
    strict_loader = r'''def load_contract(path: str | Path) -> Contract:
    """Load and strictly revalidate any Causal4D contract artifact."""

    with np.load(path, allow_pickle=False) as archive:
        descriptor = _require_exact_fields(
            _strict_json_mapping(
                _read_descriptor_text(archive),
                name="contract descriptor",
            ),
            fields=_CONTRACT_DESCRIPTOR_FIELDS,
            name="contract descriptor",
        )
        version = _require_integer(
            descriptor["contract_version"],
            name="contract_version",
        )
        if version != CONTRACT_VERSION:
            raise ValueError("unsupported Causal4D contract version")
        kind = _require_string(descriptor["contract_type"], name="contract_type")
        if kind not in _CONTRACT_TYPES:
            raise ValueError(f"unknown Causal4D contract type {kind!r}")
        expected_artifact_id = _require_string(
            descriptor["artifact_id"],
            name="artifact_id",
        )
        _validate_sha256(expected_artifact_id, name="artifact_id")
        context = CausalContext.from_dict(descriptor["context"])
        payload = _require_exact_fields(
            descriptor["payload"],
            fields=_CONTRACT_PAYLOAD_FIELDS[kind],
            name=f"{kind} payload",
        )
        _validate_archive_inventory(archive, kind=kind)
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "descriptor_json"
        }

    if kind == TwinBelief.contract_type:
        artifact: Contract = TwinBelief(
            context=context,
            endpoint_frame=_require_integer(
                payload["endpoint_frame"],
                name="endpoint_frame",
            ),
            particle_ids=_require_string_list(
                payload["particle_ids"],
                name="particle_ids",
            ),
            theta_names=_require_string_list(
                payload["theta_names"],
                name="theta_names",
            ),
            metadata=_require_mapping(payload["metadata"], name="metadata"),
            **arrays,
        )
    elif kind == FactualIntervention.contract_type:
        artifact = FactualIntervention(
            context=context,
            component_ids=_require_string_list(
                payload["component_ids"],
                name="component_ids",
            ),
            phi_names=_require_string_list(payload["phi_names"], name="phi_names"),
            kappa_names=_require_string_list(
                payload["kappa_names"],
                name="kappa_names",
            ),
            evidence_frame_stop=_require_integer(
                payload["evidence_frame_stop"],
                name="evidence_frame_stop",
            ),
            source_twin_belief_id=_require_string(
                payload["source_twin_belief_id"],
                name="source_twin_belief_id",
            ),
            metadata=_require_mapping(payload["metadata"], name="metadata"),
            **arrays,
        )
    elif kind == CounterfactualQuery.contract_type:
        artifact = CounterfactualQuery(
            context=context,
            horizon_frames=_require_integer(
                payload["horizon_frames"],
                name="horizon_frames",
            ),
            contact_policy=cast(
                Literal["same_grasp", "new_contact"],
                _require_string(payload["contact_policy"], name="contact_policy"),
            ),
            language=_require_optional_string(payload["language"], name="language"),
            source_factual_intervention_id=_require_string(
                payload["source_factual_intervention_id"],
                name="source_factual_intervention_id",
            ),
            metadata=_require_mapping(payload["metadata"], name="metadata"),
            query_node_indices=arrays.pop("query_node_indices", None),
            **arrays,
        )
    elif kind == PhysicalPosterior.contract_type:
        artifact = PhysicalPosterior(
            context=context,
            component_ids=_require_string_list(
                payload["component_ids"],
                name="component_ids",
            ),
            phi_names=_require_string_list(payload["phi_names"], name="phi_names"),
            kappa_names=_require_string_list(
                payload["kappa_names"],
                name="kappa_names",
            ),
            source_twin_belief_id=_require_string(
                payload["source_twin_belief_id"],
                name="source_twin_belief_id",
            ),
            source_factual_intervention_id=_require_string(
                payload["source_factual_intervention_id"],
                name="source_factual_intervention_id",
            ),
            source_query_id=_require_string(
                payload["source_query_id"],
                name="source_query_id",
            ),
            metadata=_require_mapping(payload["metadata"], name="metadata"),
            **arrays,
        )
    else:
        artifact = TaskPosterior(
            context=context,
            physical_posterior_id=_require_string(
                payload["physical_posterior_id"],
                name="physical_posterior_id",
            ),
            component_ids=_require_string_list(
                payload["component_ids"],
                name="component_ids",
            ),
            beta=_require_finite_float(payload["beta"], name="beta"),
            semantic_source=_require_string(
                payload["semantic_source"],
                name="semantic_source",
            ),
            metadata=_require_mapping(payload["metadata"], name="metadata"),
            **arrays,
        )
    if artifact.artifact_id != expected_artifact_id:
        raise ValueError("Causal4D artifact digest does not match its payload")
    return artifact
'''
    text = text[:load_start] + strict_loader
    CONTRACTS.write_text(text, encoding="utf-8")


def write_tests() -> None:
    TESTS.write_text(
        r'''from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from causal4d.contracts import (
    CausalContext,
    TaskPosterior,
    TwinBelief,
    array_sha256,
    build_causal_context,
    load_contract,
    save_contract,
)


def _context() -> CausalContext:
    observations = np.zeros((4, 1, 3), dtype=np.float64)
    actions = np.zeros((4, 1, 3), dtype=np.float64)
    return build_causal_context(
        protocol_id="strict-archive-test",
        case_id="case-1",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=2,
    )


def _belief() -> TwinBelief:
    context = _context()
    state = np.zeros((2, 1, 3), dtype=np.float64)
    return TwinBelief(
        context=context,
        endpoint_frame=1,
        particle_ids=("1", "theta-1"),
        theta_names=("scale",),
        endpoint_position_m=state,
        endpoint_velocity_mps=state,
        theta=np.zeros((2, 1), dtype=np.float64),
        discrepancy_mean_m=state,
        discrepancy_variance_m2=np.ones_like(state),
        weights=np.asarray([0.5, 0.5], dtype=np.float64),
    )


def _task() -> TaskPosterior:
    return TaskPosterior(
        context=_context(),
        physical_posterior_id=array_sha256(np.zeros(1)),
        component_ids=("component",),
        physical_weights=np.asarray([1.0], dtype=np.float64),
        task_weights=np.asarray([1.0], dtype=np.float64),
        semantic_log_scores=np.asarray([0.0], dtype=np.float64),
        beta=1.0,
        query_node_indices=np.asarray([0], dtype=np.int64),
        semantic_source="strict-test",
    )


def _read_archive(path: Path) -> tuple[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        descriptor = np.asarray(archive["descriptor_json"]).item()
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "descriptor_json"
        }
    assert type(descriptor) is str
    return descriptor, arrays


def _write_archive(
    path: Path,
    descriptor: str,
    arrays: dict[str, np.ndarray],
) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            descriptor_json=np.asarray(descriptor),
            **arrays,
        )


def _mutate_descriptor(
    path: Path,
    operation: Callable[[dict[str, Any]], None],
) -> None:
    text, arrays = _read_archive(path)
    descriptor = json.loads(text)
    operation(descriptor)
    _write_archive(
        path,
        json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        arrays,
    )


def test_valid_contract_archive_keeps_its_content_identity(tmp_path: Path) -> None:
    artifact = _belief()
    path = tmp_path / "belief.npz"
    save_contract(path, artifact)

    restored = load_contract(path)

    assert isinstance(restored, TwinBelief)
    assert restored.artifact_id == artifact.artifact_id


def test_context_loader_rejects_coercible_and_unknown_fields() -> None:
    values = _context().as_dict()
    coercible = copy.deepcopy(values)
    coercible["o_minus"]["frame_start"] = "0"
    with pytest.raises(ValueError, match="frame_start must be an integer"):
        CausalContext.from_dict(coercible)

    unknown = copy.deepcopy(values)
    unknown["o_minus"]["ignored"] = True
    with pytest.raises(ValueError, match="observation window fields"):
        CausalContext.from_dict(unknown)

    bad_protocol = copy.deepcopy(values)
    bad_protocol["protocol_id"] = None
    with pytest.raises(ValueError, match="protocol_id must be a string"):
        CausalContext.from_dict(bad_protocol)


def test_archive_rejects_coercible_version_and_nested_context(tmp_path: Path) -> None:
    path = tmp_path / "belief.npz"
    save_contract(path, _belief())
    _mutate_descriptor(path, lambda value: value.__setitem__("contract_version", "1"))
    with pytest.raises(ValueError, match="contract_version must be an integer"):
        load_contract(path)

    save_contract(path, _belief())
    _mutate_descriptor(
        path,
        lambda value: value["context"]["o_minus"].__setitem__(
            "frame_start", 0.0
        ),
    )
    with pytest.raises(ValueError, match="frame_start must be an integer"):
        load_contract(path)


def test_archive_rejects_unknown_descriptor_and_payload_fields(tmp_path: Path) -> None:
    path = tmp_path / "belief.npz"
    save_contract(path, _belief())
    _mutate_descriptor(path, lambda value: value.__setitem__("ignored", 1))
    with pytest.raises(ValueError, match="contract descriptor fields"):
        load_contract(path)

    save_contract(path, _belief())
    _mutate_descriptor(
        path,
        lambda value: value["payload"].__setitem__("ignored", 1),
    )
    with pytest.raises(ValueError, match="TwinBelief payload fields"):
        load_contract(path)


def test_archive_rejects_string_list_coercion(tmp_path: Path) -> None:
    path = tmp_path / "belief.npz"
    save_contract(path, _belief())
    _mutate_descriptor(
        path,
        lambda value: value["payload"].__setitem__(
            "particle_ids", [1, "theta-1"]
        ),
    )

    with pytest.raises(ValueError, match="particle_ids"):
        load_contract(path)


def test_archive_rejects_duplicate_and_nonfinite_json_members(tmp_path: Path) -> None:
    path = tmp_path / "belief.npz"
    save_contract(path, _belief())
    text, arrays = _read_archive(path)
    duplicate = text.replace(
        '"contract_version":1',
        '"contract_version":1,"contract_version":1',
        1,
    )
    _write_archive(path, duplicate, arrays)
    with pytest.raises(ValueError, match="strict finite JSON"):
        load_contract(path)

    save_contract(path, _belief())
    text, arrays = _read_archive(path)
    nonfinite = text[:-1] + ',"ignored":NaN}'
    _write_archive(path, nonfinite, arrays)
    with pytest.raises(ValueError, match="strict finite JSON"):
        load_contract(path)


def test_archive_rejects_dtype_normalization(tmp_path: Path) -> None:
    path = tmp_path / "belief.npz"
    save_contract(path, _belief())
    text, arrays = _read_archive(path)
    arrays["weights"] = arrays["weights"].astype(np.float32)
    _write_archive(path, text, arrays)

    with pytest.raises(ValueError, match="weights.*dtype float64"):
        load_contract(path)


def test_task_archive_rejects_boolean_beta_coercion(tmp_path: Path) -> None:
    path = tmp_path / "task.npz"
    save_contract(path, _task())
    _mutate_descriptor(
        path,
        lambda value: value["payload"].__setitem__("beta", True),
    )

    with pytest.raises(ValueError, match="beta must be a finite float"):
        load_contract(path)


def test_archive_requires_exact_array_inventory(tmp_path: Path) -> None:
    path = tmp_path / "belief.npz"
    save_contract(path, _belief())
    text, arrays = _read_archive(path)
    arrays["invented"] = np.asarray([1.0], dtype=np.float64)
    _write_archive(path, text, arrays)
    with pytest.raises(ValueError, match="array inventory"):
        load_contract(path)

    save_contract(path, _belief())
    text, arrays = _read_archive(path)
    del arrays["weights"]
    _write_archive(path, text, arrays)
    with pytest.raises(ValueError, match="array inventory"):
        load_contract(path)
''',
        encoding="utf-8",
    )


def cleanup_bootstrap() -> None:
    TESTS.parent.mkdir(parents=True, exist_ok=True)
    for path in (WORKFLOW, Path(__file__)):
        path.unlink()


if __name__ == "__main__":
    patch_contracts()
    write_tests()
    cleanup_bootstrap()
