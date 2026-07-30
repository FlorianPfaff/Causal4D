"""Versioned compatibility contract for Bayesian-PhysTwin belief providers."""

from __future__ import annotations

import json

from causal4d.provider_contract import (
    BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
    PhysicalBeliefProviderManifest,
    ProviderCompatibilityResult,
    validate_provider_compatibility,
)

BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_API = (
    "bayesian_phystwin.causal4d_belief_provider_v1"
)
BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_API_VERSION = 1
BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_CAPABILITIES = (
    "causal_prefix_endpoint_inference",
    "fixed_bayesian_anchor_endpoint",
    "immutable_endpoint_posterior",
    "numpy_only_endpoint_inference",
    "residual_finite_preflight",
)
BAYESIAN_PHYSTWIN_BELIEF_ARTIFACT_SCHEMA_VERSIONS = {
    "FixedBayesianAnchorConfig": 1,
    "RobustEndpointPosterior": 1,
}
BAYESIAN_PHYSTWIN_BELIEF_INFERENCE_ROLE = (
    "fixed robust readout-discrepancy endpoint"
)


def load_bayesian_phystwin_belief_provider_manifest(
    *,
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Load BPT's NumPy-only fixed-anchor provider descriptor."""

    from bayesian_phystwin.causal4d_belief_provider_v1 import (
        causal4d_belief_provider_manifest,
    )

    values = causal4d_belief_provider_manifest(
        provider_revision=provider_revision
    )
    return PhysicalBeliefProviderManifest(
        provider_name=str(values["provider_name"]),
        provider_version=str(values["provider_version"]),
        provider_revision=str(values["provider_revision"]),
        schema_version=int(values["schema_version"]),
        capabilities=tuple(map(str, values["capabilities"])),
        artifact_schema_versions=dict(values["artifact_schema_versions"]),
        metadata=dict(values.get("metadata", {})),
    )


def _validate_belief_provider_metadata(
    manifest: PhysicalBeliefProviderManifest,
) -> None:
    expected = {
        "provider_api": BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_API,
        "provider_api_version": BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_API_VERSION,
        "inference_role": BAYESIAN_PHYSTWIN_BELIEF_INFERENCE_ROLE,
    }
    mismatches = {
        name: (value, manifest.metadata.get(name))
        for name, value in expected.items()
        if manifest.metadata.get(name) != value
    }
    if mismatches:
        raise ValueError(
            "unexpected Bayesian-PhysTwin belief provider metadata: "
            + json.dumps(mismatches, sort_keys=True)
        )


def validate_bayesian_phystwin_belief_provider(
    manifest: PhysicalBeliefProviderManifest | None = None,
    *,
    provider_revision: str | None = None,
) -> ProviderCompatibilityResult:
    """Validate the fixed-anchor child contract before endpoint inference."""

    candidate = manifest or load_bayesian_phystwin_belief_provider_manifest(
        provider_revision=provider_revision
    )
    if candidate.provider_name != "bayesian-phystwin":
        raise ValueError("expected the bayesian-phystwin belief provider")
    _validate_belief_provider_metadata(candidate)
    return validate_provider_compatibility(
        candidate,
        required_capabilities=BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_CAPABILITIES,
        supported_provider_versions=BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
        required_artifact_versions=(
            BAYESIAN_PHYSTWIN_BELIEF_ARTIFACT_SCHEMA_VERSIONS
        ),
    )


def require_bayesian_phystwin_belief_provider(
    *,
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Return the installed belief manifest or fail before reading residuals."""

    manifest = load_bayesian_phystwin_belief_provider_manifest(
        provider_revision=provider_revision
    )
    result = validate_bayesian_phystwin_belief_provider(manifest)
    if not result.compatible:
        raise RuntimeError(
            "incompatible Bayesian-PhysTwin belief provider: "
            + json.dumps(result.as_dict(), sort_keys=True)
        )
    return manifest


__all__ = [
    "BAYESIAN_PHYSTWIN_BELIEF_ARTIFACT_SCHEMA_VERSIONS",
    "BAYESIAN_PHYSTWIN_BELIEF_INFERENCE_ROLE",
    "BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_API",
    "BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_API_VERSION",
    "BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_CAPABILITIES",
    "load_bayesian_phystwin_belief_provider_manifest",
    "require_bayesian_phystwin_belief_provider",
    "validate_bayesian_phystwin_belief_provider",
]
