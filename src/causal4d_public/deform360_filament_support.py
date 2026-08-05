"""Source-only structural admission for Deform360 filament reset graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, cast

import numpy as np


FILAMENT_SUPPORT_SCHEMA_VERSION = 1
FILAMENT_SUPPORT_KIND = "Deform360SourceFilamentSupportDiagnostic"
FILAMENT_SUPPORT_CONFIG_KIND = "Deform360SourceFilamentSupportConfig"
FILAMENT_SUPPORT_POLICIES = ("registered_v1", "component_bridge_v1")
REGISTERED_POLICY = FILAMENT_SUPPORT_POLICIES[0]
PRIMARY_POLICY = FILAMENT_SUPPORT_POLICIES[1]
SOURCE_MILESTONE = Path("milestones/deform360-replication-source-backend-v1")
RESET_MECHANICS_SUMMARY = Path("milestones/deform360-reset-mechanics-v1/summary.json")
RESET_MECHANICS_LOCK = Path("configs/causal4d_public/deform360_reset_mechanics_v1.json")
_DISCONNECTED_ROPE_MESSAGE = (
    "rope point cloud remains disconnected at the maximum neighbor count"
)
_TECHNICAL_EXCEPTIONS = (
    ValueError,
    RuntimeError,
    FloatingPointError,
    np.linalg.LinAlgError,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_mapping(value: Any, *, message: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(message)
    return cast(Mapping[str, Any], value)


def _require_nonempty_list(value: Any, *, message: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(message)
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filament_support_result_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def filament_support_config_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _finite_float(value: Any, *, name: str) -> float:
    _require(
        type(value) in {int, float} and type(value) is not bool and np.isfinite(value),
        f"{name} must be a finite number",
    )
    return float(value)


def _nonnegative_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    _require(result >= 0.0, f"{name} must be nonnegative")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    _require(type(value) is int and value >= 1, f"{name} must be a positive integer")
    return value


def _require_string_sequence(value: Any, *, message: str) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or any(type(item) is not str or not item for item in value)
    ):
        raise ValueError(message)
    return tuple(value)


@dataclass(frozen=True)
class FilamentSupportConfig:
    """Predeclared structural support and geometry-admission controls."""

    node_count: int = 21
    reset_count: int = 3
    maximum_horizon_observation_count: int = 6
    expected_total_reset_count: int = 36
    expected_registered_success_reset_count: int = 28
    expected_registered_failure_episode_count: int = 7
    expected_registered_failure_reset_count: int = 8
    maximum_component_count_before_bridge: int = 4
    maximum_bridge_to_local_scale_ratio: float = 12.0
    maximum_single_bridge_fraction_of_centerline_length: float = 0.25
    maximum_total_bridge_fraction_of_centerline_length: float = 0.35
    maximum_repaired_p95_ratio_to_object_q95: float = 1.5
    minimum_repaired_length_ratio_to_object_median: float = 0.75
    maximum_repaired_length_ratio_to_object_median: float = 1.25
    maximum_edge_length_coefficient_of_variation: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "node_count",
            "reset_count",
            "maximum_horizon_observation_count",
            "expected_total_reset_count",
            "expected_registered_success_reset_count",
            "expected_registered_failure_episode_count",
            "expected_registered_failure_reset_count",
            "maximum_component_count_before_bridge",
        ):
            object.__setattr__(
                self,
                name,
                _positive_integer(getattr(self, name), name=name),
            )
        _require(self.node_count >= 4, "filament graph needs at least four nodes")
        _require(
            self.expected_registered_success_reset_count
            + self.expected_registered_failure_reset_count
            == self.expected_total_reset_count,
            "registered reset support counts do not sum to the total",
        )
        for name in (
            "maximum_bridge_to_local_scale_ratio",
            "maximum_single_bridge_fraction_of_centerline_length",
            "maximum_total_bridge_fraction_of_centerline_length",
            "maximum_repaired_p95_ratio_to_object_q95",
            "minimum_repaired_length_ratio_to_object_median",
            "maximum_repaired_length_ratio_to_object_median",
            "maximum_edge_length_coefficient_of_variation",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_float(getattr(self, name), name=name),
            )
        _require(
            0.0
            < self.minimum_repaired_length_ratio_to_object_median
            <= 1.0
            <= self.maximum_repaired_length_ratio_to_object_median,
            "repaired centerline-length ratios must bracket one",
        )
        _require(
            self.maximum_single_bridge_fraction_of_centerline_length
            <= self.maximum_total_bridge_fraction_of_centerline_length,
            "single-bridge fraction cannot exceed total-bridge fraction",
        )
        _require(
            self.maximum_repaired_p95_ratio_to_object_q95 >= 1.0,
            "repaired p95 ratio must admit the registered reference boundary",
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_filament_support_lock(path: str | Path) -> dict[str, Any]:
    """Load and validate the immutable structure-only filament-support lock."""

    lock_path = Path(path).resolve()
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    _require(
        payload.get("schema_version") == FILAMENT_SUPPORT_SCHEMA_VERSION,
        "filament-support config schema changed",
    )
    _require(
        payload.get("artifact_kind") == FILAMENT_SUPPORT_CONFIG_KIND,
        "filament-support config kind changed",
    )
    _require(
        payload.get("config_sha256") == filament_support_config_sha256(payload),
        "filament-support config checksum mismatch",
    )
    controls = _require_mapping(
        payload.get("config"),
        message="filament-support controls are missing",
    )
    selected = _require_nonempty_list(
        controls.get("selected_object_ids"),
        message="filament-support object set is invalid",
    )
    _require(
        all(type(value) is str and value for value in selected)
        and len(selected) == len(set(selected)),
        "filament-support object set is invalid",
    )
    _require(
        _require_string_sequence(
            controls.get("policy_order"),
            message="filament-support policy order is invalid",
        )
        == FILAMENT_SUPPORT_POLICIES,
        "filament-support policy set or ordering changed",
    )
    _require(
        controls.get("registered_policy") == REGISTERED_POLICY
        and controls.get("primary_policy") == PRIMARY_POLICY,
        "filament-support policy roles changed",
    )
    _require(
        controls.get("reset_selection")
        == "availability_only_evenly_spaced_including_prefix_and_latest_eligible",
        "filament-support reset-selection policy changed",
    )
    _require(
        controls.get("primary_construction")
        == "maximum_knn_component_mst_bridge_then_registered_refinement_v1",
        "filament-support construction policy changed",
    )
    boundary = _require_mapping(
        payload.get("information_boundary"),
        message="filament-support information boundary is missing",
    )
    _require(
        boundary.get("source_only") is True
        and boundary.get("source_future_geometry_allowed_for_structure_scoring") is True
        and boundary.get("source_candidate_outcomes_allowed_for_identity_only") is True
        and boundary.get("calibration_outcomes_allowed") is False
        and boundary.get("target_prefix_allowed") is False
        and boundary.get("target_future_allowed") is False
        and boundary.get("registered_reset_result_mutable") is False
        and boundary.get("registered_method_mutable") is False,
        "filament-support lock opens a forbidden information or method boundary",
    )
    required_parent = payload.get("required_parent_commit")
    _require(
        type(required_parent) is str
        and len(required_parent) == 40
        and all(character in "0123456789abcdef" for character in required_parent),
        "filament-support required parent is invalid",
    )
    config = FilamentSupportConfig(
        node_count=controls["node_count"],
        reset_count=controls["reset_count"],
        maximum_horizon_observation_count=(
            controls["maximum_horizon_observation_count"]
        ),
        expected_total_reset_count=controls["expected_total_reset_count"],
        expected_registered_success_reset_count=(
            controls["expected_registered_success_reset_count"]
        ),
        expected_registered_failure_episode_count=(
            controls["expected_registered_failure_episode_count"]
        ),
        expected_registered_failure_reset_count=(
            controls["expected_registered_failure_reset_count"]
        ),
        maximum_component_count_before_bridge=(
            controls["maximum_component_count_before_bridge"]
        ),
        maximum_bridge_to_local_scale_ratio=(
            controls["maximum_bridge_to_local_scale_ratio"]
        ),
        maximum_single_bridge_fraction_of_centerline_length=(
            controls["maximum_single_bridge_fraction_of_centerline_length"]
        ),
        maximum_total_bridge_fraction_of_centerline_length=(
            controls["maximum_total_bridge_fraction_of_centerline_length"]
        ),
        maximum_repaired_p95_ratio_to_object_q95=(
            controls["maximum_repaired_p95_ratio_to_object_q95"]
        ),
        minimum_repaired_length_ratio_to_object_median=(
            controls["minimum_repaired_length_ratio_to_object_median"]
        ),
        maximum_repaired_length_ratio_to_object_median=(
            controls["maximum_repaired_length_ratio_to_object_median"]
        ),
        maximum_edge_length_coefficient_of_variation=(
            controls["maximum_edge_length_coefficient_of_variation"]
        ),
    )
    return {
        "path": lock_path,
        "payload": payload,
        "config": config,
        "selected_object_ids": tuple(selected),
        "required_parent_commit": required_parent,
    }


def _require_parent_ancestor(repository_root: Path, revision: str) -> None:
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "merge-base",
                "--is-ancestor",
                revision,
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            "filament-support run is not based on the required parent commit"
        ) from error


def _graph_content_sha256(graph: Any) -> str:
    digest = hashlib.sha256()
    for name in ("positions_m", "spring_edges", "spring_families", "masses"):
        values = np.ascontiguousarray(getattr(graph, name))
        digest.update(name.encode("ascii"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    digest.update(str(graph.stratum).encode("utf-8"))
    return digest.hexdigest()


def _graph_metrics(graph: Any, hull: np.ndarray) -> dict[str, Any]:
    from .deform360_replication_warp import symmetric_chamfer_distance_m

    centerline = _require_mapping(
        graph.diagnostics.get("centerline"),
        message="filament graph omitted centerline diagnostics",
    )
    point_distance = _require_mapping(
        centerline.get("point_to_centerline_node_distance_m"),
        message="filament graph omitted point-distance diagnostics",
    )
    bridge_raw = centerline.get("component_bridge")
    bridge = (
        _require_mapping(
            bridge_raw,
            message="filament graph bridge diagnostics are invalid",
        )
        if bridge_raw is not None
        else None
    )
    result: dict[str, Any] = {
        "graph_content_sha256": _graph_content_sha256(graph),
        "node_count": len(graph.positions_m),
        "spring_count": len(graph.spring_edges),
        "centerline_length_m": _finite_float(
            centerline["centerline_length_m"],
            name="centerline_length_m",
        ),
        "edge_length_coefficient_of_variation": _finite_float(
            centerline["edge_length_coefficient_of_variation"],
            name="edge_length_coefficient_of_variation",
        ),
        "point_to_centerline_node_distance_m": {
            key: _finite_float(point_distance[key], name=f"point_distance_{key}")
            for key in ("median", "p95", "maximum")
        },
        "symmetric_chamfer_m": symmetric_chamfer_distance_m(
            np.asarray(hull, dtype=np.float64),
            graph.positions_m,
        ),
        "connectivity_policy": centerline.get("connectivity_policy", REGISTERED_POLICY),
        "component_bridge": dict(bridge) if bridge is not None else None,
    }
    return result


def _technical_failure(stage: str, error: BaseException) -> dict[str, Any]:
    return {
        "status": "technical_failure",
        "technical_failure": {
            "stage": stage,
            "exception_type": type(error).__name__,
            "message": str(error) or repr(error),
        },
    }


def _evaluate_graph_policy(
    hull: np.ndarray,
    *,
    policy: str,
    node_count: int,
) -> dict[str, Any]:
    from .deform360_replication_graph import (
        build_filament_sparse_graph,
        build_filament_sparse_graph_component_bridge,
    )

    if policy == REGISTERED_POLICY:
        builder = build_filament_sparse_graph
    elif policy == PRIMARY_POLICY:
        builder = build_filament_sparse_graph_component_bridge
    else:
        raise ValueError("unknown filament graph policy")
    try:
        graph = builder(hull, node_count=node_count)
        metrics = _graph_metrics(graph, hull)
    except _TECHNICAL_EXCEPTIONS as error:
        return _technical_failure("build_graph", error)
    return {"status": "completed", "metrics": metrics}


def _load_reset_summary(
    repository_root: Path,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    summary_path = repository_root / RESET_MECHANICS_SUMMARY
    reset_lock_path = repository_root / RESET_MECHANICS_LOCK
    _require(summary_path.is_file(), "reset-mechanics summary is missing")
    _require(reset_lock_path.is_file(), "reset-mechanics lock is missing")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    source = _require_mapping(
        payload.get("source"),
        message="reset-mechanics summary source is missing",
    )
    decision = _require_mapping(
        payload.get("decision"),
        message="reset-mechanics summary decision is missing",
    )
    prior = _require_mapping(
        lock.get("prior_reset_mechanics"),
        message="filament-support prior-result lock is missing",
    )
    _require(
        _sha256_file(summary_path) == prior.get("summary_file_sha256"),
        "reset-mechanics summary file identity changed",
    )
    _require(
        payload.get("summary_sha256") == prior.get("summary_content_sha256"),
        "reset-mechanics summary content identity changed",
    )
    _require(
        source.get("result_content_sha256") == prior.get("result_content_sha256"),
        "reset-mechanics result identity changed",
    )
    _require(
        _sha256_file(reset_lock_path) == prior.get("config_file_sha256"),
        "reset-mechanics config file identity changed",
    )
    reset_lock = json.loads(reset_lock_path.read_text(encoding="utf-8"))
    _require(
        reset_lock.get("config_sha256") == prior.get("config_sha256"),
        "reset-mechanics config content identity changed",
    )
    _require(
        decision.get("classification") == "insufficient_common_episode_support"
        and decision.get("technical_failure_episode_count") == 7
        and decision.get("technical_failure_reset_count") == 8,
        "reset-mechanics prior boundary changed",
    )
    failures = _require_nonempty_list(
        payload.get("technical_failures"),
        message="reset-mechanics summary has no technical failures",
    )
    return {
        "path": str(RESET_MECHANICS_SUMMARY),
        "file_sha256": _sha256_file(summary_path),
        "summary_sha256": payload["summary_sha256"],
        "result_content_sha256": source["result_content_sha256"],
        "classification": decision["classification"],
        "technical_failure_episode_count": decision["technical_failure_episode_count"],
        "technical_failure_reset_count": decision["technical_failure_reset_count"],
        "technical_failures": failures,
    }


def _prior_failure_keys(summary: Mapping[str, Any]) -> set[tuple[str, int, int, str]]:
    result: set[tuple[str, int, int, str]] = set()
    for raw in summary["technical_failures"]:
        failure = _require_mapping(raw, message="prior technical failure is invalid")
        key = (
            str(failure["episode_id"]),
            int(failure["reset_ordinal"]),
            int(failure["reset_raw_frame"]),
            str(failure["message"]),
        )
        _require(key not in result, "prior technical failure is repeated")
        result.add(key)
    return result


def _record_registered_failure_key(
    reset: Mapping[str, Any],
) -> tuple[str, int, int, str] | None:
    registered = _require_mapping(
        reset.get("registered"),
        message="registered reset result is missing",
    )
    if registered.get("status") != "technical_failure":
        return None
    failure = _require_mapping(
        registered.get("technical_failure"),
        message="registered technical failure is missing",
    )
    return (
        str(reset["episode_id"]),
        int(reset["reset_ordinal"]),
        int(reset["reset_raw_frame"]),
        str(failure["message"]),
    )


def _object_reference_statistics(
    reset_records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for reset in reset_records:
        registered = _require_mapping(
            reset.get("registered"),
            message="registered reset result is missing",
        )
        if registered.get("status") != "completed":
            continue
        metrics = _require_mapping(
            registered.get("metrics"),
            message="registered metrics are missing",
        )
        point_distance = _require_mapping(
            metrics.get("point_to_centerline_node_distance_m"),
            message="registered point-distance metrics are missing",
        )
        bucket = grouped.setdefault(
            str(reset["object_id"]),
            {"p95": [], "length": []},
        )
        bucket["p95"].append(_finite_float(point_distance["p95"], name="p95"))
        bucket["length"].append(
            _finite_float(metrics["centerline_length_m"], name="centerline_length_m")
        )
    result: dict[str, dict[str, Any]] = {}
    for object_id, values in grouped.items():
        _require(
            bool(values["p95"]) and bool(values["length"]),
            "object reference is empty",
        )
        result[object_id] = {
            "registered_success_reset_count": len(values["p95"]),
            "registered_p95_distance_q95_m": float(np.quantile(values["p95"], 0.95)),
            "registered_centerline_length_median_m": float(np.median(values["length"])),
        }
    return result


def _repaired_reset_admission(
    reset: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    config: FilamentSupportConfig,
) -> dict[str, Any]:
    primary = _require_mapping(
        reset.get("primary"),
        message="primary reset result is missing",
    )
    if primary.get("status") != "completed":
        return {
            "completed": False,
            "passed": False,
            "reason": "primary graph construction failed",
        }
    metrics = _require_mapping(
        primary.get("metrics"),
        message="primary metrics are missing",
    )
    point_distance = _require_mapping(
        metrics.get("point_to_centerline_node_distance_m"),
        message="primary point-distance metrics are missing",
    )
    bridge = _require_mapping(
        metrics.get("component_bridge"),
        message="primary bridge diagnostics are missing",
    )
    p95 = _finite_float(point_distance["p95"], name="primary_p95")
    p95_reference = _finite_float(
        reference["registered_p95_distance_q95_m"],
        name="registered_p95_distance_q95_m",
    )
    length = _finite_float(
        metrics["centerline_length_m"],
        name="primary_centerline_length_m",
    )
    length_reference = _finite_float(
        reference["registered_centerline_length_median_m"],
        name="registered_centerline_length_median_m",
    )
    _require(p95_reference > 0.0 and length_reference > 0.0, "invalid reference")
    p95_ratio = p95 / p95_reference
    length_ratio = length / length_reference
    component_count = int(bridge["component_count_before_bridge"])
    bridge_count = int(bridge["bridge_count"])
    maximum_local_ratio = _finite_float(
        bridge["maximum_bridge_to_local_scale_ratio"],
        name="maximum_bridge_to_local_scale_ratio",
    )
    maximum_fraction = _finite_float(
        bridge["maximum_bridge_fraction_of_centerline_length"],
        name="maximum_bridge_fraction_of_centerline_length",
    )
    total_fraction = _finite_float(
        bridge["total_bridge_fraction_of_centerline_length"],
        name="total_bridge_fraction_of_centerline_length",
    )
    edge_cv = _finite_float(
        metrics["edge_length_coefficient_of_variation"],
        name="edge_length_coefficient_of_variation",
    )
    structure_passed = bool(
        2 <= component_count <= config.maximum_component_count_before_bridge
        and bridge_count == component_count - 1
        and maximum_local_ratio <= config.maximum_bridge_to_local_scale_ratio
        and maximum_fraction
        <= config.maximum_single_bridge_fraction_of_centerline_length
        and total_fraction <= config.maximum_total_bridge_fraction_of_centerline_length
    )
    geometry_passed = bool(
        p95_ratio <= config.maximum_repaired_p95_ratio_to_object_q95
        and config.minimum_repaired_length_ratio_to_object_median
        <= length_ratio
        <= config.maximum_repaired_length_ratio_to_object_median
        and edge_cv <= config.maximum_edge_length_coefficient_of_variation
    )
    return {
        "completed": True,
        "component_count_before_bridge": component_count,
        "bridge_count": bridge_count,
        "maximum_bridge_to_local_scale_ratio": maximum_local_ratio,
        "maximum_bridge_fraction_of_centerline_length": maximum_fraction,
        "total_bridge_fraction_of_centerline_length": total_fraction,
        "point_distance_p95_m": p95,
        "object_registered_p95_q95_m": p95_reference,
        "point_distance_p95_ratio_to_object_q95": p95_ratio,
        "centerline_length_m": length,
        "object_registered_centerline_length_median_m": length_reference,
        "centerline_length_ratio_to_object_median": length_ratio,
        "edge_length_coefficient_of_variation": edge_cv,
        "structure_passed": structure_passed,
        "geometry_passed": geometry_passed,
        "passed": bool(structure_passed and geometry_passed),
    }


def build_filament_support_decision(
    reset_records: Sequence[Mapping[str, Any]],
    prior_summary: Mapping[str, Any],
    *,
    config: FilamentSupportConfig,
) -> dict[str, Any]:
    """Apply the locked structural support and geometry-admission boundary."""

    actual_failures = {
        key
        for reset in reset_records
        if (key := _record_registered_failure_key(reset)) is not None
    }
    prior_failures = _prior_failure_keys(prior_summary)
    failure_episode_ids = {key[0] for key in actual_failures}
    registered_completed = sum(
        _require_mapping(
            reset.get("registered"),
            message="registered reset result is missing",
        ).get("status")
        == "completed"
        for reset in reset_records
    )
    primary_completed = sum(
        _require_mapping(
            reset.get("primary"),
            message="primary reset result is missing",
        ).get("status")
        == "completed"
        for reset in reset_records
    )
    common_records = [
        reset
        for reset in reset_records
        if _require_mapping(
            reset.get("registered"),
            message="registered reset result is missing",
        ).get("status")
        == "completed"
    ]
    exact_common_parity_count = sum(
        reset.get("common_case_exact_graph_parity") is True for reset in common_records
    )
    references = _object_reference_statistics(reset_records)
    repaired_records = [
        reset
        for reset in reset_records
        if _require_mapping(
            reset.get("registered"),
            message="registered reset result is missing",
        ).get("status")
        == "technical_failure"
    ]
    repaired_admissions = []
    for reset in repaired_records:
        object_id = str(reset["object_id"])
        _require(object_id in references, "repaired object lacks registered reference")
        repaired_admissions.append(
            {
                "episode_id": str(reset["episode_id"]),
                "object_id": object_id,
                "reset_ordinal": int(reset["reset_ordinal"]),
                "reset_raw_frame": int(reset["reset_raw_frame"]),
                **_repaired_reset_admission(
                    reset,
                    references[object_id],
                    config=config,
                ),
            }
        )

    reset_count_matches = len(reset_records) == config.expected_total_reset_count
    registered_boundary_matches = bool(
        reset_count_matches
        and actual_failures == prior_failures
        and registered_completed == config.expected_registered_success_reset_count
        and len(actual_failures) == config.expected_registered_failure_reset_count
        and len(failure_episode_ids) == config.expected_registered_failure_episode_count
    )
    primary_support_complete = bool(
        reset_count_matches and primary_completed == config.expected_total_reset_count
    )
    common_case_parity_passed = bool(
        len(common_records) == config.expected_registered_success_reset_count
        and exact_common_parity_count == len(common_records)
    )
    repaired_structure_passed = bool(
        len(repaired_admissions) == config.expected_registered_failure_reset_count
        and all(record["structure_passed"] for record in repaired_admissions)
    )
    repaired_geometry_passed = bool(
        len(repaired_admissions) == config.expected_registered_failure_reset_count
        and all(record["geometry_passed"] for record in repaired_admissions)
    )
    passed = bool(
        registered_boundary_matches
        and primary_support_complete
        and common_case_parity_passed
        and repaired_structure_passed
        and repaired_geometry_passed
    )
    if not registered_boundary_matches:
        classification = "registered_failure_boundary_changed"
        interpretation = (
            "the structure-only study cannot be interpreted because the registered "
            "reset failure set no longer matches the frozen prior result"
        )
    elif not primary_support_complete:
        classification = "component_bridge_incomplete_support"
        interpretation = (
            "the component-bridge construction does not produce a graph for every "
            "registered filament reset"
        )
    elif not common_case_parity_passed:
        classification = "component_bridge_common_case_parity_failure"
        interpretation = (
            "the additive construction changes at least one reset that the registered "
            "extractor already supported"
        )
    elif not repaired_structure_passed:
        classification = "component_bridge_nonlocal_structure_failure"
        interpretation = (
            "the candidate closes the technical gap only through bridges outside the "
            "registered locality and component-count boundary"
        )
    elif not repaired_geometry_passed:
        classification = "component_bridge_geometry_admission_failure"
        interpretation = (
            "the candidate connects every failed reset but does not preserve the "
            "registered same-object geometry envelope"
        )
    else:
        classification = "component_bridge_filament_support_admitted"
        interpretation = (
            "the minimal component-bridge construction is admitted only as a "
            "separately versioned source-side filament graph candidate"
        )
    return {
        "classification": classification,
        "passed": passed,
        "registered_failure_boundary_matches": registered_boundary_matches,
        "registered_success_reset_count": registered_completed,
        "registered_failure_reset_count": len(actual_failures),
        "registered_failure_episode_count": len(failure_episode_ids),
        "primary_completed_reset_count": primary_completed,
        "common_case_reset_count": len(common_records),
        "exact_common_case_parity_count": exact_common_parity_count,
        "common_case_parity_passed": common_case_parity_passed,
        "repaired_reset_count": len(repaired_admissions),
        "repaired_structure_passed": repaired_structure_passed,
        "repaired_geometry_passed": repaired_geometry_passed,
        "object_reference_statistics": references,
        "repaired_reset_admissions": repaired_admissions,
        "interpretation": interpretation,
        "registered_reset_result_changed": False,
        "registered_method_changed": False,
        "mechanics_rescoring_permitted": False,
        "target_prefix_access_permitted": False,
        "target_future_access_permitted": False,
    }


def _episode_records(
    *,
    repository_root: Path,
    data_root: Path,
    source_grid_path: Path,
    cohort: Mapping[str, Any],
    config: FilamentSupportConfig,
) -> list[dict[str, Any]]:
    from .deform360_replication_fit import validate_source_warp_candidate_grid
    from .deform360_replication_geometry import load_replication_hull_archive
    from .deform360_reset_mechanics import select_reset_positions

    grid = json.loads(source_grid_path.read_text(encoding="utf-8"))
    validate_source_warp_candidate_grid(grid)
    object_id = str(cohort["object_id"])
    episode_id = str(grid["episode_id"])
    _require(str(grid["stratum"]) == "filament", "selected object is not filament")
    _require(
        episode_id.startswith(f"{object_id}/episode_"),
        "source grid and cohort object differ",
    )
    raw_episode_index = episode_id.rsplit("episode_", maxsplit=1)[1]
    _require(
        len(raw_episode_index) == 4 and raw_episode_index.isdigit(),
        "source episode identity is malformed",
    )
    episode_index = int(raw_episode_index)
    _require(
        episode_index in cohort["source_episode_ids"],
        "filament study was given a non-source episode",
    )
    hull_json = (
        data_root
        / "observations"
        / object_id
        / f"episode_{episode_index:04d}"
        / "sampled_hulls.json"
    )
    hull_payload = json.loads(hull_json.read_text(encoding="utf-8"))
    _require(
        hull_payload["result_sha256"] == grid["reference_geometry_result_sha256"],
        "source hull identity changed",
    )
    frames, hulls = load_replication_hull_archive(hull_payload)
    total_frame_count = len(hulls)
    available = np.asarray([len(hull) > 0 for hull in hulls], dtype=bool)
    frames = frames[available]
    hulls = tuple(hull for hull, keep in zip(hulls, available, strict=True) if keep)
    _require(
        total_frame_count == grid["reference_geometry_total_frame_count"]
        and len(hulls) == grid["reference_geometry_available_frame_count"],
        "source hull availability changed",
    )
    _require(
        frames.astype(int).tolist() == grid["raw_hull_frame_indices"],
        "source hull frame indices changed",
    )
    positions = select_reset_positions(
        frames,
        reset_count=config.reset_count,
        maximum_horizon_observation_count=(config.maximum_horizon_observation_count),
    )
    result = []
    for reset_ordinal, reset_position in enumerate(positions):
        hull = np.asarray(hulls[reset_position], dtype=np.float64)
        registered = _evaluate_graph_policy(
            hull,
            policy=REGISTERED_POLICY,
            node_count=config.node_count,
        )
        primary = _evaluate_graph_policy(
            hull,
            policy=PRIMARY_POLICY,
            node_count=config.node_count,
        )
        exact_parity: bool | None = None
        if registered["status"] == primary["status"] == "completed":
            registered_metrics = _require_mapping(
                registered["metrics"],
                message="registered metrics are missing",
            )
            primary_metrics = _require_mapping(
                primary["metrics"],
                message="primary metrics are missing",
            )
            exact_parity = (
                registered_metrics["graph_content_sha256"]
                == primary_metrics["graph_content_sha256"]
            )
        result.append(
            {
                "object_id": object_id,
                "episode_id": episode_id,
                "source_grid_path": str(source_grid_path.relative_to(repository_root)),
                "source_grid_result_sha256": grid["result_sha256"],
                "reset_ordinal": reset_ordinal,
                "reset_hull_position": reset_position,
                "reset_raw_frame": int(frames[reset_position]),
                "available_hull_count": len(frames),
                "registered": registered,
                "primary": primary,
                "common_case_exact_graph_parity": exact_parity,
                "information_boundary": {
                    "source_episode_only": True,
                    "reset_selection_uses_availability_only": True,
                    "current_reset_hull_only_used_for_graph_construction": True,
                    "future_geometry_used_for_graph_construction": False,
                    "mechanics_scores_read": False,
                    "calibration_outcomes_read": False,
                    "target_prefix_read": False,
                    "target_future_read": False,
                },
            }
        )
    return result


def run_source_filament_support_diagnostic(
    repository_root: str | Path,
    protocol_path: str | Path,
    data_root: str | Path,
    output_path: str | Path,
    *,
    lock_path: str | Path,
) -> dict[str, Any]:
    """Run the locked source-only filament graph support boundary."""

    from causal4d.atomic_io import atomic_write_json

    from .deform360_prefix_kinematics_diagnostic import verify_source_milestone
    from .deform360_replication import load_deform360_replication_protocol

    lock = load_filament_support_lock(lock_path)
    config: FilamentSupportConfig = lock["config"]
    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    data = Path(data_root).resolve()
    _require(root.is_dir(), "repository root is missing")
    _require(protocol_file.is_file(), "replication protocol is missing")
    _require(data.is_dir(), "Deform360 derived-data root is missing")
    milestone = verify_source_milestone(root)
    protocol = load_deform360_replication_protocol(protocol_file)
    lock_payload = lock["payload"]
    _require_parent_ancestor(root, lock["required_parent_commit"])
    _require(
        lock_payload["protocol_config_sha256"] == protocol["config_sha256"],
        "locked replication protocol identity changed",
    )
    _require(
        lock_payload["source_milestone_manifest_sha256"]
        == milestone["manifest_sha256"],
        "locked source milestone identity changed",
    )
    prior_summary = _load_reset_summary(root, lock_payload)
    cohorts = {
        str(record["object_id"]): record for record in protocol["config"]["cohort"]
    }
    grid_root = root / SOURCE_MILESTONE / "artifacts" / "source-grids"
    records: list[dict[str, Any]] = []
    for object_id in lock["selected_object_ids"]:
        _require(object_id in cohorts, f"locked object is absent: {object_id}")
        _require(
            str(cohorts[object_id]["stratum"]) == "filament",
            f"locked object is not filament: {object_id}",
        )
        paths = sorted((grid_root / object_id).glob("source_episode_*_grid.json"))
        expected = len(cohorts[object_id]["source_episode_ids"])
        _require(len(paths) == expected, f"{object_id} source-grid set is incomplete")
        for path in paths:
            records.extend(
                _episode_records(
                    repository_root=root,
                    data_root=data,
                    source_grid_path=path,
                    cohort=cohorts[object_id],
                    config=config,
                )
            )
    decision = build_filament_support_decision(
        records,
        prior_summary,
        config=config,
    )
    payload = {
        "schema_version": FILAMENT_SUPPORT_SCHEMA_VERSION,
        "artifact_kind": FILAMENT_SUPPORT_KIND,
        "config": config.as_dict(),
        "protocol": {
            "path": (
                str(protocol_file.relative_to(root))
                if protocol_file.is_relative_to(root)
                else str(protocol_file)
            ),
            "file_sha256": _sha256_file(protocol_file),
            "config_sha256": protocol["config_sha256"],
        },
        "source_milestone": milestone,
        "prior_reset_mechanics": prior_summary,
        "diagnostic_lock": {
            "path": (
                str(lock["path"].relative_to(root))
                if lock["path"].is_relative_to(root)
                else str(lock["path"])
            ),
            "file_sha256": _sha256_file(lock["path"]),
            "config_sha256": lock_payload["config_sha256"],
        },
        "selected_object_ids": list(lock["selected_object_ids"]),
        "reset_records": records,
        "decision": decision,
        "information_boundary": {
            "source_candidate_outcomes_read_for_identity_only": True,
            "source_reset_geometry_read_for_structure_scoring": True,
            "source_future_mechanics_outcomes_read": False,
            "calibration_outcomes_read": False,
            "target_prefix_read": False,
            "target_future_geometry_read": False,
            "target_future_tactile_read": False,
            "registered_reset_result_changed": False,
            "registered_replication_result_changed": False,
            "registered_36_execution_method_changed": False,
        },
    }
    payload["result_sha256"] = filament_support_result_sha256(payload)
    atomic_write_json(output_path, payload)
    return payload


def validate_source_filament_support_diagnostic(
    payload: Mapping[str, Any],
) -> None:
    """Validate result identity, prior boundary, and structure-only semantics."""

    _require(
        payload.get("schema_version") == FILAMENT_SUPPORT_SCHEMA_VERSION,
        "filament-support diagnostic schema changed",
    )
    _require(
        payload.get("artifact_kind") == FILAMENT_SUPPORT_KIND,
        "filament-support diagnostic kind changed",
    )
    _require(
        payload.get("result_sha256") == filament_support_result_sha256(payload),
        "filament-support diagnostic checksum mismatch",
    )
    config_mapping = _require_mapping(
        payload.get("config"),
        message="filament-support result config is missing",
    )
    config = FilamentSupportConfig(**dict(config_mapping))
    boundary = _require_mapping(
        payload.get("information_boundary"),
        message="filament-support result boundary is missing",
    )
    _require(
        boundary.get("source_future_mechanics_outcomes_read") is False
        and boundary.get("calibration_outcomes_read") is False
        and boundary.get("target_prefix_read") is False
        and boundary.get("target_future_geometry_read") is False
        and boundary.get("target_future_tactile_read") is False
        and boundary.get("registered_reset_result_changed") is False
        and boundary.get("registered_replication_result_changed") is False
        and boundary.get("registered_36_execution_method_changed") is False,
        "filament-support result crossed its information or method boundary",
    )
    prior = _require_mapping(
        payload.get("prior_reset_mechanics"),
        message="filament-support prior summary is missing",
    )
    _require(
        prior.get("classification") == "insufficient_common_episode_support"
        and prior.get("technical_failure_episode_count")
        == config.expected_registered_failure_episode_count
        and prior.get("technical_failure_reset_count")
        == config.expected_registered_failure_reset_count,
        "filament-support prior reset boundary changed",
    )
    records = _require_nonempty_list(
        payload.get("reset_records"),
        message="filament-support result has no reset records",
    )
    _require(
        len(records) == config.expected_total_reset_count,
        "filament-support reset count changed",
    )
    identities: set[tuple[str, int]] = set()
    for raw in records:
        record = _require_mapping(raw, message="filament reset record is invalid")
        identity = (str(record["episode_id"]), int(record["reset_ordinal"]))
        _require(identity not in identities, "filament reset identity is repeated")
        identities.add(identity)
        registered = _require_mapping(
            record.get("registered"),
            message="registered reset result is missing",
        )
        primary = _require_mapping(
            record.get("primary"),
            message="primary reset result is missing",
        )
        for name, result in (
            (REGISTERED_POLICY, registered),
            (PRIMARY_POLICY, primary),
        ):
            _require(
                result.get("status") in {"completed", "technical_failure"},
                f"{name} reset status is invalid",
            )
            if result["status"] == "completed":
                metrics = _require_mapping(
                    result.get("metrics"),
                    message=f"{name} completed reset omitted metrics",
                )
                _require(
                    type(metrics.get("graph_content_sha256")) is str
                    and len(metrics["graph_content_sha256"]) == 64,
                    f"{name} graph identity is invalid",
                )
            else:
                _require(
                    "metrics" not in result,
                    f"{name} technical failure contains graph metrics",
                )
                failure = _require_mapping(
                    result.get("technical_failure"),
                    message=f"{name} technical failure metadata is missing",
                )
                _require(
                    set(failure) == {"stage", "exception_type", "message"}
                    and all(
                        type(failure[key]) is str and failure[key]
                        for key in ("stage", "exception_type", "message")
                    ),
                    f"{name} technical failure metadata is invalid",
                )
        if registered["status"] == primary["status"] == "completed":
            registered_metrics = _require_mapping(
                registered["metrics"],
                message="registered metrics are missing",
            )
            primary_metrics = _require_mapping(
                primary["metrics"],
                message="primary metrics are missing",
            )
            expected_parity = (
                registered_metrics["graph_content_sha256"]
                == primary_metrics["graph_content_sha256"]
            )
            _require(
                record.get("common_case_exact_graph_parity") is expected_parity,
                "filament common-case parity accounting changed",
            )
        else:
            _require(
                record.get("common_case_exact_graph_parity") is None,
                "filament non-common reset reports graph parity",
            )
        episode_boundary = _require_mapping(
            record.get("information_boundary"),
            message="filament reset boundary is missing",
        )
        _require(
            episode_boundary.get("source_episode_only") is True
            and episode_boundary.get("reset_selection_uses_availability_only") is True
            and episode_boundary.get(
                "current_reset_hull_only_used_for_graph_construction"
            )
            is True
            and episode_boundary.get("future_geometry_used_for_graph_construction")
            is False
            and episode_boundary.get("mechanics_scores_read") is False
            and episode_boundary.get("calibration_outcomes_read") is False
            and episode_boundary.get("target_prefix_read") is False
            and episode_boundary.get("target_future_read") is False,
            "filament reset crossed its structure-only boundary",
        )
    expected_decision = build_filament_support_decision(
        cast(Sequence[Mapping[str, Any]], records),
        prior,
        config=config,
    )
    _require(
        payload.get("decision") == expected_decision,
        "filament-support decision does not match its reset records",
    )


__all__ = [
    "FILAMENT_SUPPORT_CONFIG_KIND",
    "FILAMENT_SUPPORT_KIND",
    "FILAMENT_SUPPORT_POLICIES",
    "FILAMENT_SUPPORT_SCHEMA_VERSION",
    "FilamentSupportConfig",
    "build_filament_support_decision",
    "filament_support_config_sha256",
    "filament_support_result_sha256",
    "load_filament_support_lock",
    "run_source_filament_support_diagnostic",
    "validate_source_filament_support_diagnostic",
]
