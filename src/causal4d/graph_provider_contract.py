"""Versioned compatibility contract for Bayesian-PhysTwin graph providers."""

from __future__ import annotations

import json

from causal4d.provider_contract import (
    BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
    PhysicalBeliefProviderManifest,
    ProviderCompatibilityResult,
    validate_provider_compatibility,
)

BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_CAPABILITIES = (
    "controller_grouping",
    "phystwin_spring_graph",
)
BAYESIAN_PHYSTWIN_GRAPH_ARTIFACT_SCHEMA_VERSIONS = {
    "PhysTwinSpringGraph": 1,
}


def load_bayesian_phystwin_graph_provider_manifest(
    *,
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Load BPT's public graph-provider descriptor without experiment imports."""

    from bayesian_phystwin.causal4d_graph_provider_v1 import (
        causal4d_graph_provider_manifest,
    )

    values = causal4d_graph_provider_manifest(provider_revision=provider_revision)
    return PhysicalBeliefProviderManifest(
        provider_name=str(values["provider_name"]),
        provider_version=str(values["provider_version"]),
        provider_revision=str(values["provider_revision"]),
        schema_version=int(values["schema_version"]),
        capabilities=tuple(map(str, values["capabilities"])),
        artifact_schema_versions=dict(values["artifact_schema_versions"]),
        metadata=dict(values.get("metadata", {})),
    )


def validate_bayesian_phystwin_graph_provider(
    manifest: PhysicalBeliefProviderManifest | None = None,
    *,
    provider_revision: str | None = None,
) -> ProviderCompatibilityResult:
    """Validate the installed BPT graph provider against Causal4D's v1 contract."""

    candidate = manifest or load_bayesian_phystwin_graph_provider_manifest(
        provider_revision=provider_revision
    )
    if candidate.provider_name != "bayesian-phystwin":
        raise ValueError("expected the bayesian-phystwin graph provider")
    if candidate.metadata.get("provider_api") != (
        "bayesian_phystwin.causal4d_graph_provider_v1"
    ):
        raise ValueError("unexpected Bayesian-PhysTwin graph provider API")
    return validate_provider_compatibility(
        candidate,
        required_capabilities=BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_CAPABILITIES,
        supported_provider_versions=BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
        required_artifact_versions=(
            BAYESIAN_PHYSTWIN_GRAPH_ARTIFACT_SCHEMA_VERSIONS
        ),
    )


def require_bayesian_phystwin_graph_provider(
    *,
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Return the installed graph manifest or raise before PhysTwin execution."""

    manifest = load_bayesian_phystwin_graph_provider_manifest(
        provider_revision=provider_revision
    )
    result = validate_bayesian_phystwin_graph_provider(manifest)
    if not result.compatible:
        raise RuntimeError(
            "incompatible Bayesian-PhysTwin graph provider: "
            + json.dumps(result.as_dict(), sort_keys=True)
        )
    return manifest


__all__ = [
    "BAYESIAN_PHYSTWIN_GRAPH_ARTIFACT_SCHEMA_VERSIONS",
    "BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_CAPABILITIES",
    "load_bayesian_phystwin_graph_provider_manifest",
    "require_bayesian_phystwin_graph_provider",
    "validate_bayesian_phystwin_graph_provider",
]
