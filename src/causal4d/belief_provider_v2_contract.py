"""Compatibility contract for Bayesian-PhysTwin horizon discrepancy provider v2."""

from __future__ import annotations

import json

from causal4d.provider_contract import (
    BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
    PhysicalBeliefProviderManifest,
    ProviderCompatibilityResult,
    validate_provider_compatibility,
)

BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_API = (
    "bayesian_phystwin.causal4d_belief_provider_v2"
)
BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_API_VERSION = 2
BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_CAPABILITIES = (
    "causal_prefix_endpoint_inference",
    "evidence_weighted_endpoint_model_average",
    "horizon_dependent_predictive_covariance",
    "source_calibrated_horizon_discrepancy",
    "mean_reverting_discrepancy_prediction",
    "immutable_endpoint_posterior",
    "numpy_only_endpoint_inference",
    "per_track_component_evidence",
    "residual_finite_preflight",
)
BAYESIAN_PHYSTWIN_BELIEF_V2_ARTIFACT_SCHEMA_VERSIONS = {
    "ModelAveragedEndpointConfig": 1,
    "ModelAveragedEndpointPosterior": 1,
    "ModelAveragedEndpointPrediction": 1,
    "HorizonDiscrepancyCalibration": 1,
    "HorizonConditionedEndpointPrediction": 1,
}
BAYESIAN_PHYSTWIN_BELIEF_V2_INFERENCE_ROLE = (
    "model-averaged robust readout-discrepancy endpoint"
)
BAYESIAN_PHYSTWIN_BELIEF_V2_COMPATIBILITY = (
    "additive provider; causal4d_belief_provider_v1 is unchanged"
)
BAYESIAN_PHYSTWIN_BELIEF_V2_RAW_COVARIANCE_CLAIM = (
    "model-based predictive covariance; source-calibrated horizon dynamics and "
    "interval calibration remain separate gates"
)


def load_bayesian_phystwin_belief_provider_v2_manifest(
    *,
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Load BPT's additive model-averaged horizon provider descriptor."""

    if provider_revision is not None and (
        type(provider_revision) is not str or not provider_revision
    ):
        raise ValueError("provider_revision must be a nonempty string")
    from bayesian_phystwin.causal4d_belief_provider_v2 import (
        causal4d_belief_provider_v2_manifest,
    )

    values = causal4d_belief_provider_v2_manifest(
        provider_revision=provider_revision
    )
    manifest = PhysicalBeliefProviderManifest.from_provider_descriptor(values)
    if (
        provider_revision is not None
        and manifest.provider_revision != provider_revision
    ):
        raise ValueError(
            "belief provider v2 descriptor revision does not match requested revision"
        )
    return manifest


def _validate_belief_provider_v2_metadata(
    manifest: PhysicalBeliefProviderManifest,
) -> None:
    expected = {
        "provider_api": BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_API,
        "provider_api_version": BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_API_VERSION,
        "inference_role": BAYESIAN_PHYSTWIN_BELIEF_V2_INFERENCE_ROLE,
        "compatibility": BAYESIAN_PHYSTWIN_BELIEF_V2_COMPATIBILITY,
        "raw_covariance_claim": BAYESIAN_PHYSTWIN_BELIEF_V2_RAW_COVARIANCE_CLAIM,
    }
    mismatches = {}
    for name, value in expected.items():
        actual = manifest.metadata.get(name)
        if type(actual) is not type(value) or actual != value:
            mismatches[name] = (value, actual)
    if mismatches:
        raise ValueError(
            "unexpected Bayesian-PhysTwin belief provider v2 metadata: "
            + json.dumps(mismatches, sort_keys=True)
        )


def validate_bayesian_phystwin_belief_provider_v2(
    manifest: PhysicalBeliefProviderManifest | None = None,
    *,
    provider_revision: str | None = None,
) -> ProviderCompatibilityResult:
    """Validate the additive horizon-discrepancy provider contract."""

    candidate = manifest or load_bayesian_phystwin_belief_provider_v2_manifest(
        provider_revision=provider_revision
    )
    if candidate.provider_name != "bayesian-phystwin":
        raise ValueError("expected the bayesian-phystwin belief provider v2")
    _validate_belief_provider_v2_metadata(candidate)
    return validate_provider_compatibility(
        candidate,
        required_capabilities=BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_CAPABILITIES,
        supported_provider_versions=BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
        required_artifact_versions=(
            BAYESIAN_PHYSTWIN_BELIEF_V2_ARTIFACT_SCHEMA_VERSIONS
        ),
    )


def require_bayesian_phystwin_belief_provider_v2(
    *,
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Return the installed v2 manifest or fail before opening residual inputs."""

    manifest = load_bayesian_phystwin_belief_provider_v2_manifest(
        provider_revision=provider_revision
    )
    result = validate_bayesian_phystwin_belief_provider_v2(manifest)
    if not result.compatible:
        raise RuntimeError(
            "incompatible Bayesian-PhysTwin belief provider v2: "
            + json.dumps(result.as_dict(), sort_keys=True)
        )
    return manifest


__all__ = [
    "BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_API",
    "BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_API_VERSION",
    "BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_CAPABILITIES",
    "BAYESIAN_PHYSTWIN_BELIEF_V2_ARTIFACT_SCHEMA_VERSIONS",
    "BAYESIAN_PHYSTWIN_BELIEF_V2_COMPATIBILITY",
    "BAYESIAN_PHYSTWIN_BELIEF_V2_INFERENCE_ROLE",
    "BAYESIAN_PHYSTWIN_BELIEF_V2_RAW_COVARIANCE_CLAIM",
    "load_bayesian_phystwin_belief_provider_v2_manifest",
    "require_bayesian_phystwin_belief_provider_v2",
    "validate_bayesian_phystwin_belief_provider_v2",
]
