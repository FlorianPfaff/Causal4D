"""Fail-closed contract for Bayesian-PhysTwin replay provider API v2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.provider_contract import (
    BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
    PhysicalBeliefProviderManifest,
    ProviderCompatibilityResult,
    validate_provider_compatibility,
)

BAYESIAN_PHYSTWIN_REPLAY_PROVIDER_SCHEMA_VERSION = 2
BAYESIAN_PHYSTWIN_REPLAY_PROVIDER_CAPABILITIES = (
    "artifact_checksums",
    "immutable_replay_trajectories",
    "particle_endpoint_position",
    "particle_endpoint_velocity",
    "physical_parameter_particles",
    "phystwin_replay",
    "residual_lifting",
    "restart_velocity_history",
    "stateless_replay_requests",
    "target_validity",
    "typed_replay_requests",
)
BAYESIAN_PHYSTWIN_REPLAY_ARTIFACT_SCHEMA_VERSIONS = {
    "GraphBelief": 1,
    "TwinBelief": 1,
    "ReplayRequest": 1,
    "ReplayTrajectory": 1,
}


def stable_replay_identifier(namespace: str, payload: Mapping[str, Any]) -> str:
    """Return a deterministic nonempty identifier for one replay-owned object."""

    if type(namespace) is not str:
        raise TypeError("replay identifier namespace must be a string")
    prefix = namespace.strip()
    if not prefix:
        raise ValueError("replay identifier namespace must be nonempty")
    if not isinstance(payload, Mapping):
        raise TypeError("replay identifier payload must be a mapping")
    normalized = validated_json_mapping(
        payload,
        error_message=(
            "replay identifier payload must contain finite JSON data with string keys"
        ),
    )
    encoded = json.dumps(
        plain_json(normalized),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


def load_bayesian_phystwin_replay_provider_manifest(
    *,
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Load BPT's immutable replay descriptor from the versioned v2 module."""

    if provider_revision is not None and (
        type(provider_revision) is not str or not provider_revision
    ):
        raise ValueError("provider_revision must be a nonempty string")
    from bayesian_phystwin.causal4d_provider_v2 import causal4d_provider_manifest

    values = causal4d_provider_manifest(provider_revision=provider_revision)
    manifest = PhysicalBeliefProviderManifest.from_provider_descriptor(values)
    if (
        provider_revision is not None
        and manifest.provider_revision != provider_revision
    ):
        raise ValueError(
            "replay provider descriptor revision does not match requested revision"
        )
    return manifest


def validate_bayesian_phystwin_replay_provider(
    manifest: PhysicalBeliefProviderManifest | None = None,
    *,
    provider_revision: str | None = None,
) -> ProviderCompatibilityResult:
    """Validate identity, version, capabilities, and replay artifact schemas."""

    candidate = manifest or load_bayesian_phystwin_replay_provider_manifest(
        provider_revision=provider_revision
    )
    if candidate.provider_name != "bayesian-phystwin":
        raise ValueError("expected the bayesian-phystwin replay provider")
    provider_api = candidate.metadata.get("provider_api")
    if type(provider_api) is not str or provider_api != (
        "bayesian_phystwin.causal4d_provider_v2"
    ):
        raise ValueError("replay provider must identify causal4d_provider_v2")
    provider_api_version = candidate.metadata.get("provider_api_version")
    if type(provider_api_version) is not int or provider_api_version != 2:
        raise ValueError("replay provider metadata must identify API version 2")
    return validate_provider_compatibility(
        candidate,
        required_capabilities=BAYESIAN_PHYSTWIN_REPLAY_PROVIDER_CAPABILITIES,
        supported_schema_versions=(BAYESIAN_PHYSTWIN_REPLAY_PROVIDER_SCHEMA_VERSION,),
        supported_provider_versions=BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
        required_artifact_versions=BAYESIAN_PHYSTWIN_REPLAY_ARTIFACT_SCHEMA_VERSIONS,
    )


def require_bayesian_phystwin_replay_provider(
    *,
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Return a compatible replay-v2 manifest or fail before simulation."""

    manifest = load_bayesian_phystwin_replay_provider_manifest(
        provider_revision=provider_revision
    )
    result = validate_bayesian_phystwin_replay_provider(manifest)
    if not result.compatible:
        raise RuntimeError(
            "incompatible Bayesian-PhysTwin replay provider: "
            + json.dumps(result.as_dict(), sort_keys=True)
        )
    return manifest


def validate_replay_trajectory(
    request: Any,
    trajectory: Any,
    *,
    expected_dt_s: float,
) -> None:
    """Independently bind one provider response to its immutable request."""

    from bayesian_phystwin.causal4d_provider_v2 import (
        InitialReplayRequestV1,
        RestartReplayRequestV1,
    )

    expected_dt = float(expected_dt_s)
    if not np.isfinite(expected_dt) or expected_dt <= 0.0:
        raise ValueError("expected_dt_s must be positive and finite")
    expected_frames: np.ndarray
    if isinstance(request, InitialReplayRequestV1):
        expected_frames = np.arange(request.frame_count, dtype=np.int64)
    elif isinstance(request, RestartReplayRequestV1):
        expected_frames = np.arange(
            request.start_frame,
            request.stop_frame,
            dtype=np.int64,
        )
    else:
        raise TypeError("unsupported replay request type")

    if trajectory.request_id != request.request_id:
        raise ValueError("replay trajectory request_id does not match its request")
    if trajectory.simulator_configuration_id != request.simulator_configuration_id:
        raise ValueError(
            "replay trajectory simulator_configuration_id does not match its request"
        )
    if trajectory.initial_state_id != request.initial_state_id:
        raise ValueError(
            "replay trajectory initial_state_id does not match its request"
        )
    if not np.array_equal(np.asarray(trajectory.frame_ids), expected_frames):
        raise ValueError(
            "replay trajectory frame provenance does not match its request"
        )
    if not np.isclose(float(trajectory.dt_s), expected_dt, rtol=0.0, atol=1e-15):
        raise ValueError("replay trajectory timestep does not match the backend")

    positions = np.asarray(trajectory.positions_m)
    velocities = np.asarray(trajectory.velocities_mps)
    if positions.shape != velocities.shape:
        raise ValueError("replay trajectory positions and velocities differ in shape")
    if positions.shape[0] != len(expected_frames):
        raise ValueError("replay trajectory frame count does not match its request")
    if positions.ndim != 3 or positions.shape[1] < 1 or positions.shape[2] != 3:
        raise ValueError("replay trajectory must have shape (T, N>=1, 3)")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
        raise ValueError("replay trajectory must contain only finite values")
