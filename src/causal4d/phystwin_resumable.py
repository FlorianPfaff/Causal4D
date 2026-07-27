"""Resumable content-addressed execution for official PhysTwin rollout banks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import platform
import threading
from typing import Any

import numpy as np

import causal4d.phystwin_backend as phystwin_backend_module
from causal4d.contracts import TwinBelief, array_sha256
from causal4d.phystwin_backend import (
    OfficialPhysTwinBackend,
    PhysTwinActionProposal,
    PhysTwinHypothesisConfig,
    build_contact_states,
    build_rollout_hypotheses,
)
from causal4d.rollout_bank import JointRolloutBank
from causal4d.rollout_cache import (
    ContentAddressedRolloutCache,
    RolloutCacheResult,
    file_sha256,
    repository_source_identity,
)

_PROVIDER_PATCH_LOCK = threading.RLock()


def _json_sha256(values: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(values),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_descriptor(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    return {
        "sha256": array_sha256(array),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
    }


def _graph_descriptor(graph: Any) -> dict[str, Any]:
    descriptor = {
        "vertices": _array_descriptor(np.asarray(graph.vertices)),
        "springs": _array_descriptor(np.asarray(graph.springs)),
        "rest_lengths": _array_descriptor(np.asarray(graph.rest_lengths)),
        "masses": _array_descriptor(np.asarray(graph.masses)),
        "num_object_springs": int(graph.num_object_springs),
        "num_object_points": int(
            getattr(graph, "num_object_points", len(np.asarray(graph.vertices)))
        ),
    }
    descriptor["graph_sha256"] = _json_sha256(descriptor)
    return descriptor


def _factory_value(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    name: str,
    position: int | None = None,
) -> Any:
    if name in kwargs:
        return kwargs[name]
    if position is not None and position < len(args):
        return args[position]
    raise TypeError(f"provider factory did not supply {name!r}")


def _installed_version(distribution_name: str) -> str | None:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return None


def _numerical_runtime_descriptor() -> dict[str, Any]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "distributions": {
            name: _installed_version(name)
            for name in ("bayesian-phystwin", "numpy", "torch", "warp-lang")
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _provider_source_identity() -> dict[str, Any]:
    specification = find_spec("bayesian_phystwin")
    if specification is None or specification.origin is None:
        return {"kind": "unavailable", "fingerprint": "unavailable"}
    return repository_source_identity(Path(specification.origin).resolve().parent)


def _source_artifact_digests(backend: OfficialPhysTwinBackend) -> dict[str, str]:
    sources = {
        "final_data": backend.final_data_path,
        "optimal_params": backend.optimal_params_path,
        "checkpoint": backend.checkpoint_path,
        "baseline_trajectory": backend.baseline_trajectory_path,
        "parameter_profile": backend.profile_path,
    }
    return {name: file_sha256(path) for name, path in sources.items()}


@dataclass
class _RolloutCacheSession:
    cache: ContentAddressedRolloutCache
    provider_manifest: dict[str, Any]
    provider_source: dict[str, Any]
    official_source: dict[str, Any]
    numerical_runtime: dict[str, Any]
    source_artifacts_sha256: dict[str, str]
    case_name: str
    twin_belief_id: str
    records: list[dict[str, Any]] = field(default_factory=list)
    provider_proxy_count: int = 0
    provider_instance_count: int = 0
    deterministic_modes: set[bool] = field(default_factory=set)

    @classmethod
    def from_backend(
        cls,
        backend: OfficialPhysTwinBackend,
        twin_belief: TwinBelief,
        rollout_cache_dir: str | Path,
    ) -> _RolloutCacheSession:
        return cls(
            cache=ContentAddressedRolloutCache(rollout_cache_dir),
            provider_manifest=backend.provider_manifest.as_dict(),
            provider_source=_provider_source_identity(),
            official_source=repository_source_identity(backend.official_repo),
            numerical_runtime=_numerical_runtime_descriptor(),
            source_artifacts_sha256=_source_artifact_digests(backend),
            case_name=backend.case_name,
            twin_belief_id=twin_belief.artifact_id,
        )

    def wrap_factory(self, factory: Callable[..., Any]) -> Callable[..., Any]:
        def cached_factory(*args: Any, **kwargs: Any) -> _LazyCachedReplayProvider:
            self.provider_proxy_count += 1
            graph = _factory_value(args, kwargs, name="graph", position=4)
            original_count = int(_factory_value(args, kwargs, name="original_count"))
            deterministic = bool(
                _factory_value(args, kwargs, name="deterministic_spring_forces")
            )
            self.deterministic_modes.add(deterministic)
            static_descriptor = {
                "provider": self.provider_manifest,
                "provider_source": self.provider_source,
                "official_phystwin_source": self.official_source,
                "numerical_runtime": self.numerical_runtime,
                "source_artifacts_sha256": self.source_artifacts_sha256,
                "case": self.case_name,
                "spring_graph": _graph_descriptor(graph),
                "provider_factory": {
                    "num_surface_points": int(
                        _factory_value(args, kwargs, name="num_surface_points")
                    ),
                    "original_count": original_count,
                    "dt": float(_factory_value(args, kwargs, name="dt")),
                    "num_substeps": int(
                        _factory_value(args, kwargs, name="num_substeps")
                    ),
                    "self_collision": bool(
                        _factory_value(args, kwargs, name="self_collision")
                    ),
                    "deterministic_spring_forces": deterministic,
                    "spring_parameterization": str(
                        _factory_value(
                            args,
                            kwargs,
                            name="spring_parameterization",
                        )
                    ),
                    "device": str(_factory_value(args, kwargs, name="device")),
                },
            }
            return _LazyCachedReplayProvider(
                session=self,
                factory=factory,
                factory_args=args,
                factory_kwargs=dict(kwargs),
                static_descriptor=static_descriptor,
                original_count=original_count,
            )

        setattr(cached_factory, "__causal4d_rollout_cache_proxy__", True)
        return cached_factory

    def note_provider_instantiation(self) -> None:
        self.provider_instance_count += 1

    def record_call(
        self,
        result: RolloutCacheResult,
        *,
        controller_points: np.ndarray,
        group_log_scales: np.ndarray,
        endpoint_position: np.ndarray,
        endpoint_velocity: np.ndarray,
        start_frame: int,
        stop_frame: int,
    ) -> None:
        self.records.append(
            {
                "ordinal": len(self.records),
                "cache_key": result.cache_key,
                "cache_status": result.status,
                "cache_hit": result.cache_hit,
                "record_path": result.relative_path,
                "trajectory_sha256": result.trajectory_sha256,
                "controller_points_sha256": array_sha256(controller_points),
                "group_log_scales_sha256": array_sha256(group_log_scales),
                "group_log_scales": np.asarray(group_log_scales, dtype=float).tolist(),
                "endpoint_position_sha256": array_sha256(endpoint_position),
                "endpoint_velocity_sha256": array_sha256(endpoint_velocity),
                "frame_interval": [int(start_frame), int(stop_frame)],
            }
        )

    def bind_rollout_components(
        self,
        backend: OfficialPhysTwinBackend,
        action_proposals: Sequence[PhysTwinActionProposal],
        hypothesis_config: PhysTwinHypothesisConfig | None,
        bank: JointRolloutBank,
    ) -> None:
        contact_states = build_contact_states(backend.hand_count, hypothesis_config)
        hypotheses = build_rollout_hypotheses(action_proposals, contact_states)
        expected_ids = tuple(value.hypothesis_id for value in hypotheses)
        if bank.hypothesis_ids != expected_ids:
            raise RuntimeError("rollout-bank hypothesis order changed during caching")
        expected: list[tuple[int, str, int]] = []
        unique_shifts = tuple(
            dict.fromkeys(
                hypothesis.contact.attachment_shifts for hypothesis in hypotheses
            )
        )
        for shifts in unique_shifts:
            for hypothesis_index, hypothesis in enumerate(hypotheses):
                if hypothesis.contact.attachment_shifts != shifts:
                    continue
                for particle_index in range(len(backend.particles.weights)):
                    expected.append(
                        (
                            hypothesis_index,
                            hypothesis.hypothesis_id,
                            particle_index,
                        )
                    )
        if len(expected) != len(self.records):
            raise RuntimeError(
                "cached replay call count does not match the rollout-bank support"
            )
        for record, component in zip(self.records, expected, strict=True):
            hypothesis_index, hypothesis_id, particle_index = component
            record.update(
                {
                    "hypothesis_index": hypothesis_index,
                    "hypothesis_id": hypothesis_id,
                    "particle_index": particle_index,
                }
            )

    def manifest(self) -> dict[str, Any]:
        counts = Counter(record["cache_status"] for record in self.records)
        deterministic = self.deterministic_modes == {True}
        return {
            "enabled": True,
            "schema_name": "causal4d.phystwin-rollout-cache-manifest",
            "schema_version": 1,
            "root": str(self.cache.root),
            "record_count": len(self.records),
            "validated_record_count": len(self.records),
            "hit_count": counts["hit"] + counts["race_hit"],
            "miss_count": counts["miss"],
            "repaired_count": counts["repaired"],
            "race_hit_count": counts["race_hit"],
            "provider_proxy_count": self.provider_proxy_count,
            "provider_instance_count": self.provider_instance_count,
            "all_records_validated": True,
            "replay_semantics": (
                "deterministic"
                if deterministic
                else "frozen_sample_for_nondeterministic_provider"
            ),
            "provider": self.provider_manifest,
            "provider_source": self.provider_source,
            "official_phystwin_source": self.official_source,
            "numerical_runtime": self.numerical_runtime,
            "source_artifacts_sha256": self.source_artifacts_sha256,
            "twin_belief_id": self.twin_belief_id,
            "records": self.records,
        }


class _LazyCachedReplayProvider:
    """Instantiate the real provider only when one requested record is absent."""

    def __init__(
        self,
        *,
        session: _RolloutCacheSession,
        factory: Callable[..., Any],
        factory_args: tuple[Any, ...],
        factory_kwargs: dict[str, Any],
        static_descriptor: dict[str, Any],
        original_count: int,
    ) -> None:
        self._session = session
        self._factory = factory
        self._factory_args = factory_args
        self._factory_kwargs = factory_kwargs
        self._static_descriptor = static_descriptor
        self._original_count = original_count
        self._provider: Any | None = None
        self._controller_points: np.ndarray | None = None
        self._group_log_scales: np.ndarray | None = None
        self.device = str(factory_kwargs.get("device", "unknown"))

    def _ensure_provider(self) -> Any:
        if self._provider is None:
            self._provider = self._factory(
                *self._factory_args,
                **self._factory_kwargs,
            )
            self._session.note_provider_instantiation()
            if self._controller_points is not None:
                self._provider.set_controller_points(self._controller_points.copy())
            if self._group_log_scales is not None:
                self._provider.set_group_log_scales(self._group_log_scales.copy())
        return self._provider

    def set_controller_points(self, values: np.ndarray) -> None:
        self._controller_points = np.asarray(values, dtype=np.float32).copy()
        if self._provider is not None:
            self._provider.set_controller_points(self._controller_points.copy())

    def set_group_log_scales(self, values: np.ndarray) -> None:
        self._group_log_scales = np.asarray(values, dtype=np.float32).copy()
        if self._provider is not None:
            self._provider.set_group_log_scales(self._group_log_scales.copy())

    def replay_initial(self, *, frame_count: int) -> Any:
        return self._ensure_provider().replay_initial(frame_count=frame_count)

    def replay_restart(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        *,
        start_frame: int,
        stop_frame: int,
    ) -> np.ndarray:
        if self._controller_points is None or self._group_log_scales is None:
            raise RuntimeError(
                "controller points and group scales must be set before replay"
            )
        position = np.asarray(position_m).copy()
        velocity = np.asarray(velocity_mps).copy()
        call_descriptor = {
            **self._static_descriptor,
            "controller_points": _array_descriptor(self._controller_points),
            "group_log_scales": {
                **_array_descriptor(self._group_log_scales),
                "values": np.asarray(self._group_log_scales, dtype=float).tolist(),
            },
            "endpoint_position": _array_descriptor(position),
            "endpoint_velocity": _array_descriptor(velocity),
            "frame_interval": {
                "start": int(start_frame),
                "stop": int(stop_frame),
            },
        }

        def compute() -> np.ndarray:
            provider = self._ensure_provider()
            return np.asarray(
                provider.replay_restart(
                    position.copy(),
                    velocity.copy(),
                    start_frame=start_frame,
                    stop_frame=stop_frame,
                ),
                dtype=np.float32,
            )

        result = self._session.cache.get_or_compute(
            call_descriptor,
            compute,
            expected_frame_count=stop_frame - start_frame,
            minimum_node_count=self._original_count,
        )
        self._session.record_call(
            result,
            controller_points=self._controller_points,
            group_log_scales=self._group_log_scales,
            endpoint_position=position,
            endpoint_velocity=velocity,
            start_frame=start_frame,
            stop_frame=stop_frame,
        )
        return result.trajectory

    def close(self) -> None:
        if self._provider is not None:
            self._provider.close()
            self._provider = None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._ensure_provider(), name)


def build_resumable_rollout_bank(
    backend: OfficialPhysTwinBackend,
    action_proposals: Sequence[PhysTwinActionProposal],
    *,
    twin_belief: TwinBelief,
    hypothesis_config: PhysTwinHypothesisConfig | None = None,
    rollout_cache_dir: str | Path | None,
) -> tuple[JointRolloutBank, dict[str, Any]]:
    """Build a bank while atomically persisting each individual Warp replay."""

    if rollout_cache_dir is None:
        bank, manifest = backend.build_rollout_bank(
            action_proposals,
            twin_belief=twin_belief,
            hypothesis_config=hypothesis_config,
        )
        result_manifest = dict(manifest)
        result_manifest["rollout_cache"] = {
            "enabled": False,
            "schema_name": "causal4d.phystwin-rollout-cache-manifest",
            "schema_version": 1,
        }
        return bank, result_manifest

    session = _RolloutCacheSession.from_backend(
        backend,
        twin_belief,
        rollout_cache_dir,
    )
    with _PROVIDER_PATCH_LOCK:
        original_factory = phystwin_backend_module.create_official_replay_provider
        if getattr(
            original_factory,
            "__causal4d_rollout_cache_proxy__",
            False,
        ):
            raise RuntimeError("nested PhysTwin rollout-cache sessions are unsupported")
        phystwin_backend_module.create_official_replay_provider = session.wrap_factory(
            original_factory
        )
        try:
            bank, manifest = backend.build_rollout_bank(
                action_proposals,
                twin_belief=twin_belief,
                hypothesis_config=hypothesis_config,
            )
        finally:
            phystwin_backend_module.create_official_replay_provider = original_factory

    session.bind_rollout_components(
        backend,
        action_proposals,
        hypothesis_config,
        bank,
    )
    result_manifest = dict(manifest)
    result_manifest["rollout_cache"] = session.manifest()
    return bank, result_manifest
