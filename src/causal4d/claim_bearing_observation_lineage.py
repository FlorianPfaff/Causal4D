"""Strict loading boundary for prospective claim-bearing Prob4D observations."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .observation_lineage import ObservationLineage, load_observation_lineage
from .prob4d_observation_lineage import (
    PROB4D_CAUSAL_STREAM_CONTRACT_VERSION,
    PROB4D_JOINT_GAUGE_MODEL,
)


def require_claim_bearing_prob4d_lineage(
    lineage: ObservationLineage,
) -> ObservationLineage:
    """Reject provider-v1, exploratory, approximate, or fallback-bearing lineage."""

    validation = lineage.provider_validation
    if not isinstance(validation, Mapping):
        raise ValueError("a validated Prob4D provider boundary is required")
    if validation.get("claim_bearing_provider_v2_validated") is not True:
        raise ValueError(
            "the complete claim-bearing Prob4D provider-v2 boundary was not validated"
        )
    if (
        validation.get("stream_contract_version")
        != PROB4D_CAUSAL_STREAM_CONTRACT_VERSION
        or validation.get("stream_contract_version_inferred") is not False
    ):
        raise ValueError(
            "claim-bearing Prob4D observation requires explicit causal stream "
            "contract version 2"
        )
    stream_contract = validation.get("claim_bearing_stream_contract")
    if not isinstance(stream_contract, Mapping):
        raise ValueError(
            "claim-bearing Prob4D causal stream contract was not validated"
        )
    if (
        stream_contract.get("version") != PROB4D_CAUSAL_STREAM_CONTRACT_VERSION
        or stream_contract.get("causal_frame_stop_convention") != "exclusive"
    ):
        raise ValueError("claim-bearing Prob4D causal stream contract is invalid")

    if (
        validation.get("covariance_semantics") != PROB4D_JOINT_GAUGE_MODEL
        or validation.get("cross_window_covariance_preserved") is not True
    ):
        raise ValueError(
            "claim-bearing Prob4D observation requires full joint cross-window "
            "gauge covariance"
        )

    provider = validation.get("provider_attestation")
    if not isinstance(provider, Mapping):
        raise ValueError("a claim-bearing Prob4D provider-v2 attestation is required")
    if provider.get("claim_bearing") is not True:
        raise ValueError(
            "exploratory Prob4D provider-v2 artifacts are not claim-bearing"
        )
    if provider.get("calibration_compatibility_validated") is not True:
        raise ValueError("Prob4D calibration compatibility was not validated")
    if provider.get("runtime_revision_independently_verified") is not True:
        raise ValueError("Prob4D runtime revision was not independently verified")

    calibration = validation.get("claim_bearing_covariance_calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError(
            "claim-bearing Prob4D covariance calibration was not validated"
        )
    if calibration.get("status") != "calibrated":
        raise ValueError("claim-bearing Prob4D covariance calibration is incomplete")
    if calibration.get("covariance_fallback_counts") != {}:
        raise ValueError("claim-bearing Prob4D covariance fallback use is not allowed")
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
