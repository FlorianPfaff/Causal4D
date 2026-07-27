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

from bayesian_phystwin.causal4d_provider_v2 import (
    InitialReplayRequestV1,
    ReplayRequestV1,
    ReplayTrajectoryV1,
    RestartReplayRequestV1,
)

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
    ContentAddressedReplayCache,
    ReplayCacheResult,
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
    recorded = getattr(backend, "source_artifacts_sha256", None)
    if recorded is not None:
        return dict(recorded)
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
    cache: ContentAddressedReplayCache
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
            cache=ContentAddressedReplayCache(rollout_cache_dir),
            provider_manifest=backend.replay_provider_manifest.as_dict(),
            provider_source=_provider_source_identity(),
            official_source=repository_source_identity(backend.official_repo),
            numerical_runtime=_numerical_runtime_descriptor(),
            source_artifacts_sha256=_source_artifact_digests(backend),
            case_name=backend.case_name,
            twin_belief_id=twin_belief.artifact_id,
        )

    def wrap_factory(self, factory: Callable[..., Any]) -> Callable[..., Any]:
        def cached_factory(*args: Any, **kwargs: Any) -> _LazyCachedReplayProviderV2:
            self.provider_proxy_count += 1
            graph = _factory_value(args, kwargs, name="graph", position=4)
            original_count = int(_factory_value(args, kwargs, name="original_count"))
            deterministic = bool(
                _factory_value(args, kwargs, name="deterministic_spring_forces")
            )
            simulator_configuration_id = str(
                _factory_value(args, kwargs, name="simulator_configuration_id")
            )
            released_initial_state_id = str(
                _factory_value(args, kwargs, name="released_initial_state_id")
            )
            dt = float(_factory_value(args, kwargs, name="dt"))
            num_substeps = int(_factory_value(args, kwargs, name="num_substeps"))
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
                    "dt": dt,
                    "num_substeps": num_substeps,
                    "frame_dt_s": dt * num_substeps,
                    "self_collision": bool(
                        _factory_value(args, kwargs, name="self_collision")
                    ),
                    "simulator_configuration_id": simulator_configuration_id,
                    "released_initial_state_id": released_initial_state_id,
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
            return _LazyCachedReplayProviderV2(
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
        result: ReplayCacheResult,
        *,
        request: ReplayRequestV1,
    ) -> None:
        record: dict[str, Any] = {
            "ordinal": len(self.records),
            "cache_key": result.cache_key,
            "cache_status": result.status,
            "cache_hit": result.cache_hit,
            "record_path": result.relative_path,
            "request_type": type(request).__name__,
            "request_id": request.request_id,
            "simulator_configuration_id": request.simulator_configuration_id,
            "initial_state_id": request.initial_state_id,
            "positions_sha256": result.positions_sha256,
            "velocities_sha256": result.velocities_sha256,
            "frame_ids": result.replay.frame_ids.tolist(),
            "dt_s": result.replay.dt_s,
            "controller_points_sha256": array_sha256(request.controller_points_m),
            "group_log_scales_sha256": array_sha256(request.group_log_scales),
            "group_log_scales": np.asarray(
                request.group_log_scales, dtype=float
            ).tolist(),
        }
        if isinstance(request, RestartReplayRequestV1):
            record.update(
                {
                    "endpoint_position_sha256": array_sha256(request.position_m),
                    "endpoint_velocity_sha256": array_sha256(request.velocity_mps),
                    "frame_interval": [request.start_frame, request.stop_frame],
                }
            )
        elif isinstance(request, InitialReplayRequestV1):
            record["frame_interval"] = [0, request.frame_count]
        else:  # pragma: no cover - typed providers admit only the two requests
            raise TypeError("unsupported replay request type")
        self.records.append(record)

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
            "schema_version": 2,
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
            "cached_payload": "positions_velocities_and_provenance",
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


class _LazyCachedReplayProviderV2:
    """Instantiate the real replay-v2 provider only on a cache miss."""

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
        self._device = str(factory_kwargs.get("device", "unknown"))
        self._frame_dt_s = float(factory_kwargs["dt"]) * int(
            factory_kwargs["num_substeps"]
        )
        self._simulator_configuration_id = str(
            factory_kwargs["simulator_configuration_id"]
        )
        self._released_initial_state_id = str(
            factory_kwargs["released_initial_state_id"]
        )

    @property
    def device(self) -> str:
        return self._device

    @property
    def frame_dt_s(self) -> float:
        return self._frame_dt_s

    @property
    def simulator_configuration_id(self) -> str:
        return self._simulator_configuration_id

    @property
    def released_initial_state_id(self) -> str:
        return self._released_initial_state_id

    def _ensure_provider(self) -> Any:
        if self._provider is None:
            self._provider = self._factory(
                *self._factory_args,
                **self._factory_kwargs,
            )
            self._session.note_provider_instantiation()
        return self._provider

    @staticmethod
    def _request_descriptor(request: ReplayRequestV1) -> dict[str, Any]:
        descriptor: dict[str, Any] = {
            "request_type": type(request).__name__,
            "request_id": request.request_id,
            "simulator_configuration_id": request.simulator_configuration_id,
            "initial_state_id": request.initial_state_id,
            "controller_points": _array_descriptor(request.controller_points_m),
            "group_log_scales": {
                **_array_descriptor(request.group_log_scales),
                "values": np.asarray(request.group_log_scales, dtype=float).tolist(),
            },
        }
        if isinstance(request, InitialReplayRequestV1):
            descriptor["frame_interval"] = {"start": 0, "stop": request.frame_count}
        elif isinstance(request, RestartReplayRequestV1):
            descriptor.update(
                {
                    "endpoint_position": _array_descriptor(request.position_m),
                    "endpoint_velocity": _array_descriptor(request.velocity_mps),
                    "frame_interval": {
                        "start": request.start_frame,
                        "stop": request.stop_frame,
                    },
                }
            )
        else:  # pragma: no cover - typed providers admit only the two requests
            raise TypeError("unsupported replay request type")
        return descriptor

    def replay(self, request: ReplayRequestV1) -> ReplayTrajectoryV1:
        if request.simulator_configuration_id != self._simulator_configuration_id:
            raise ValueError("request configuration does not match cached provider")
        if (
            isinstance(request, InitialReplayRequestV1)
            and request.initial_state_id != self._released_initial_state_id
        ):
            raise ValueError("initial request does not identify the released state")
        if isinstance(request, InitialReplayRequestV1):
            expected_frame_ids = np.arange(request.frame_count, dtype=np.int64)
        elif isinstance(request, RestartReplayRequestV1):
            expected_frame_ids = np.arange(
                request.start_frame,
                request.stop_frame,
                dtype=np.int64,
            )
        else:  # pragma: no cover - typed providers admit only the two requests
            raise TypeError("unsupported replay request type")
        call_descriptor = {
            **self._static_descriptor,
            "request": self._request_descriptor(request),
        }

        def compute() -> Any:
            return self._ensure_provider().replay(request)

        result = self._session.cache.get_or_compute(
            call_descriptor,
            compute,
            expected_frame_ids=expected_frame_ids,
            minimum_node_count=self._original_count,
            expected_dt_s=self._frame_dt_s,
            request_id=request.request_id,
            simulator_configuration_id=request.simulator_configuration_id,
            initial_state_id=request.initial_state_id,
        )
        self._session.record_call(result, request=request)
        replay = result.replay
        return ReplayTrajectoryV1(
            positions_m=replay.positions_m,
            velocities_mps=replay.velocities_mps,
            frame_ids=replay.frame_ids,
            dt_s=replay.dt_s,
            request_id=replay.request_id,
            simulator_configuration_id=replay.simulator_configuration_id,
            initial_state_id=replay.initial_state_id,
        )

    def close(self) -> None:
        if self._provider is not None:
            self._provider.close()
            self._provider = None


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
            "schema_version": 2,
            "cached_payload": "positions_velocities_and_provenance",
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
