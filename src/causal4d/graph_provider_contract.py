"""Versioned compatibility contract for Bayesian-PhysTwin graph providers."""

from __future__ import annotations

import json

from causal4d.provider_contract import (
    BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
    PhysicalBeliefProviderManifest,
    ProviderCompatibilityResult,
    validate_provider_compatibility,
)

BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_API = "bayesian_phystwin.causal4d_graph_provider_v1"
BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_API_VERSION = 1
BAYESIAN_PHYSTWIN_GRAPH_PARENT_PROVIDER_API = "bayesian_phystwin.causal4d_provider_v2"
BAYESIAN_PHYSTWIN_GRAPH_PARENT_PROVIDER_API_VERSION = 2
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

    if provider_revision is not None and (
        type(provider_revision) is not str or not provider_revision
    ):
        raise ValueError("provider_revision must be a nonempty string")
    from bayesian_phystwin.causal4d_graph_provider_v1 import (
        causal4d_graph_provider_manifest,
    )

    values = causal4d_graph_provider_manifest(provider_revision=provider_revision)
    manifest = PhysicalBeliefProviderManifest.from_provider_descriptor(values)
    if (
        provider_revision is not None
        and manifest.provider_revision != provider_revision
    ):
        raise ValueError(
            "graph provider descriptor revision does not match requested revision"
        )
    return manifest


def _validate_graph_provider_metadata(
    manifest: PhysicalBeliefProviderManifest,
) -> None:
    expected = {
        "provider_api": BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_API,
        "provider_api_version": BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_API_VERSION,
        "parent_provider_api": BAYESIAN_PHYSTWIN_GRAPH_PARENT_PROVIDER_API,
        "parent_provider_api_version": (
            BAYESIAN_PHYSTWIN_GRAPH_PARENT_PROVIDER_API_VERSION
        ),
    }
    mismatches = {}
    for name, value in expected.items():
        actual = manifest.metadata.get(name)
        if type(actual) is not type(value) or actual != value:
            mismatches[name] = (value, actual)
    if mismatches:
        raise ValueError(
            "unexpected Bayesian-PhysTwin graph provider metadata: "
            + json.dumps(mismatches, sort_keys=True)
        )


def validate_bayesian_phystwin_graph_provider(
    manifest: PhysicalBeliefProviderManifest | None = None,
    *,
    provider_revision: str | None = None,
) -> ProviderCompatibilityResult:
    """Validate BPT's graph child contract against immutable provider v2."""

    candidate = manifest or load_bayesian_phystwin_graph_provider_manifest(
        provider_revision=provider_revision
    )
    if candidate.provider_name != "bayesian-phystwin":
        raise ValueError("expected the bayesian-phystwin graph provider")
    _validate_graph_provider_metadata(candidate)
    return validate_provider_compatibility(
        candidate,
        required_capabilities=BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_CAPABILITIES,
        supported_provider_versions=BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
        required_artifact_versions=BAYESIAN_PHYSTWIN_GRAPH_ARTIFACT_SCHEMA_VERSIONS,
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
    "BAYESIAN_PHYSTWIN_GRAPH_PARENT_PROVIDER_API",
    "BAYESIAN_PHYSTWIN_GRAPH_PARENT_PROVIDER_API_VERSION",
    "BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_API",
    "BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_API_VERSION",
    "BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_CAPABILITIES",
    "load_bayesian_phystwin_graph_provider_manifest",
    "require_bayesian_phystwin_graph_provider",
    "validate_bayesian_phystwin_graph_provider",
]
