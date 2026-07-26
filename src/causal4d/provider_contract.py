"""Versioned capability contract for Bayesian-PhysTwin providers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version


PHYSICAL_BELIEF_PROVIDER_SCHEMA_VERSION = 1
BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE = ">=0.4,<0.5"
BASE_CAUSAL4D_PROVIDER_CAPABILITIES = (
    "artifact_checksums",
    "particle_endpoint_position",
    "particle_endpoint_velocity",
    "physical_parameter_particles",
)
BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES = (
    *BASE_CAUSAL4D_PROVIDER_CAPABILITIES,
    "diagnostic_compatibility",
    "phystwin_replay",
    "residual_lifting",
    "target_validity",
)
BAYESIAN_PHYSTWIN_ARTIFACT_SCHEMA_VERSIONS = {
    "GraphBelief": 1,
    "TwinBelief": 1,
}


def _json_metadata(values: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(values), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must contain finite JSON values") from error


@dataclass(frozen=True)
class PhysicalBeliefProviderManifest:
    """Provider identity, artifact schemas, and explicit capabilities."""

    provider_name: str
    provider_version: str
    provider_revision: str
    schema_version: int
    capabilities: tuple[str, ...]
    artifact_schema_versions: Mapping[str, int]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not self.provider_name
            or not self.provider_version
            or not self.provider_revision
        ):
            raise ValueError("provider identity fields must be nonempty")
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        capabilities = tuple(sorted(map(str, self.capabilities)))
        if not capabilities or any(not value for value in capabilities):
            raise ValueError("capabilities must contain nonempty names")
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("capabilities must be unique")
        artifact_versions = {
            str(name): int(version)
            for name, version in dict(self.artifact_schema_versions).items()
        }
        if not artifact_versions or any(
            not name or version < 1 for name, version in artifact_versions.items()
        ):
            raise ValueError("artifact schema versions must be positive and named")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "artifact_schema_versions", artifact_versions)
        object.__setattr__(self, "metadata", _json_metadata(self.metadata))

    @property
    def manifest_id(self) -> str:
        descriptor = {
            "provider_name": self.provider_name,
            "provider_version": self.provider_version,
            "provider_revision": self.provider_revision,
            "schema_version": self.schema_version,
            "capabilities": list(self.capabilities),
            "artifact_schema_versions": self.artifact_schema_versions,
            "metadata": self.metadata,
        }
        return hashlib.sha256(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict[str, object]:
        return {
            "manifest_id": self.manifest_id,
            "provider_name": self.provider_name,
            "provider_version": self.provider_version,
            "provider_revision": self.provider_revision,
            "schema_version": self.schema_version,
            "capabilities": list(self.capabilities),
            "artifact_schema_versions": dict(self.artifact_schema_versions),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProviderCompatibilityResult:
    """Fail-closed compatibility decision for one provider manifest."""

    compatible: bool
    missing_capabilities: tuple[str, ...]
    unsupported_schema_version: int | None
    unsupported_provider_version: str | None
    supported_provider_versions: str | None
    artifact_version_mismatches: tuple[str, ...]
    provider_manifest_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "compatible": self.compatible,
            "missing_capabilities": list(self.missing_capabilities),
            "unsupported_schema_version": self.unsupported_schema_version,
            "unsupported_provider_version": self.unsupported_provider_version,
            "supported_provider_versions": self.supported_provider_versions,
            "artifact_version_mismatches": list(
                self.artifact_version_mismatches
            ),
            "provider_manifest_id": self.provider_manifest_id,
        }


def validate_provider_compatibility(
    manifest: PhysicalBeliefProviderManifest,
    *,
    required_capabilities: Sequence[str] = BASE_CAUSAL4D_PROVIDER_CAPABILITIES,
    supported_schema_versions: Sequence[int] = (
        PHYSICAL_BELIEF_PROVIDER_SCHEMA_VERSION,
    ),
    supported_provider_versions: str | None = None,
    required_artifact_versions: Mapping[str, int] | None = None,
) -> ProviderCompatibilityResult:
    """Validate a provider without relying on a hard-coded implementation pin."""

    required = tuple(sorted(map(str, required_capabilities)))
    if any(not value for value in required) or len(set(required)) != len(required):
        raise ValueError("required_capabilities must contain unique names")
    supported = tuple(int(value) for value in supported_schema_versions)
    if not supported or any(value < 1 for value in supported):
        raise ValueError("supported_schema_versions must be positive")
    missing = tuple(
        value for value in required if value not in set(manifest.capabilities)
    )
    unsupported_schema = (
        None if manifest.schema_version in supported else manifest.schema_version
    )

    unsupported_provider_version = None
    normalized_version_range = None
    if supported_provider_versions is not None:
        normalized_version_range = str(supported_provider_versions).strip()
        if not normalized_version_range:
            raise ValueError("supported_provider_versions must be nonempty")
        try:
            version_range = SpecifierSet(normalized_version_range)
        except InvalidSpecifier as error:
            raise ValueError("supported_provider_versions is invalid") from error
        try:
            provider_version = Version(manifest.provider_version)
        except InvalidVersion:
            unsupported_provider_version = manifest.provider_version
        else:
            if provider_version not in version_range:
                unsupported_provider_version = manifest.provider_version

    mismatches = []
    for name, expected in dict(required_artifact_versions or {}).items():
        actual = manifest.artifact_schema_versions.get(str(name))
        if actual != int(expected):
            mismatches.append(f"{name}:expected={int(expected)}:actual={actual}")
    return ProviderCompatibilityResult(
        compatible=(
            not missing
            and unsupported_schema is None
            and unsupported_provider_version is None
            and not mismatches
        ),
        missing_capabilities=missing,
        unsupported_schema_version=unsupported_schema,
        unsupported_provider_version=unsupported_provider_version,
        supported_provider_versions=normalized_version_range,
        artifact_version_mismatches=tuple(sorted(mismatches)),
        provider_manifest_id=manifest.manifest_id,
    )


def load_bayesian_phystwin_provider_manifest(
    *,
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Load BPT's public Causal4D provider descriptor without importing internals."""

    from bayesian_phystwin.causal4d_provider_v1 import causal4d_provider_manifest

    values = causal4d_provider_manifest(provider_revision=provider_revision)
    return PhysicalBeliefProviderManifest(
        provider_name=str(values["provider_name"]),
        provider_version=str(values["provider_version"]),
        provider_revision=str(values["provider_revision"]),
        schema_version=int(values["schema_version"]),
        capabilities=tuple(map(str, values["capabilities"])),
        artifact_schema_versions=dict(values["artifact_schema_versions"]),
        metadata=dict(values.get("metadata", {})),
    )


def validate_bayesian_phystwin_provider(
    manifest: PhysicalBeliefProviderManifest | None = None,
    *,
    provider_revision: str | None = None,
) -> ProviderCompatibilityResult:
    """Validate an installed BPT provider against Causal4D's supported v1 range."""

    candidate = manifest or load_bayesian_phystwin_provider_manifest(
        provider_revision=provider_revision
    )
    if candidate.provider_name != "bayesian-phystwin":
        raise ValueError("expected the bayesian-phystwin provider")
    return validate_provider_compatibility(
        candidate,
        required_capabilities=BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES,
        supported_provider_versions=BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
        required_artifact_versions=BAYESIAN_PHYSTWIN_ARTIFACT_SCHEMA_VERSIONS,
    )


def require_bayesian_phystwin_provider(
    *,
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Return the installed BPT manifest or raise before provider execution."""

    manifest = load_bayesian_phystwin_provider_manifest(
        provider_revision=provider_revision
    )
    result = validate_bayesian_phystwin_provider(manifest)
    if not result.compatible:
        raise RuntimeError(
            "incompatible Bayesian-PhysTwin provider: "
            + json.dumps(result.as_dict(), sort_keys=True)
        )
    return manifest
