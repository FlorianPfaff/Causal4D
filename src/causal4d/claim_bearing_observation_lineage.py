"""Strict loading boundary for prospective claim-bearing Prob4D observations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .observation_lineage import ObservationLineage, load_observation_lineage


def require_claim_bearing_prob4d_lineage(
    lineage: ObservationLineage,
) -> ObservationLineage:
    """Reject provider-v1, exploratory, or approximate Prob4D lineage."""

    validation = lineage.provider_validation
    if not isinstance(validation, Mapping):
        raise ValueError("a validated Prob4D provider boundary is required")
    if validation.get("strict_causal_stream_contract") is not True:
        raise ValueError(
            "claim-bearing Prob4D observation requires a strict causal stream contract"
        )
    provider = validation.get("provider_attestation")
    if not isinstance(provider, Mapping):
        raise ValueError("a claim-bearing Prob4D provider-v2 attestation is required")
    if provider.get("claim_bearing") is not True:
        raise ValueError("exploratory Prob4D provider-v2 artifacts are not claim-bearing")
    if provider.get("calibration_compatibility_validated") is not True:
        raise ValueError("Prob4D calibration compatibility was not validated")
    if provider.get("runtime_revision_independently_verified") is not True:
        raise ValueError("Prob4D runtime revision was not independently verified")
    return lineage


def load_claim_bearing_prob4d_observation_lineage(
    path: str | Path,
) -> ObservationLineage:
    """Load an observation and require the prospective Prob4D provider contract."""

    return require_claim_bearing_prob4d_lineage(load_observation_lineage(path))


__all__ = [
    "load_claim_bearing_prob4d_observation_lineage",
    "require_claim_bearing_prob4d_lineage",
]
