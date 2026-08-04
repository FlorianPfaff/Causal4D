"""Versioned capability contract for Bayesian-PhysTwin providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from causal4d.immutable_json import validated_json_mapping


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
    "bayesian_anchor_endpoint",
    "diagnostic_comparison",
    "diagnostic_compatibility",
    "diagnostic_discrepancy",
    "diagnostic_observation_audit",
    "diagnostic_propagated_state",
    "diagnostic_rest_geometry",
    "phystwin_replay",
    "residual_lifting",
    "target_validity",
)
BAYESIAN_PHYSTWIN_ARTIFACT_SCHEMA_VERSIONS = {
    "GraphBelief": 1,
    "TwinBelief": 1,
}
_PROVIDER_DESCRIPTOR_REQUIRED_FIELDS = frozenset(
    {
        "provider_name",
        "provider_version",
        "provider_revision",
        "schema_version",
        "capabilities",
        "artifact_schema_versions",
    }
)
_PROVIDER_DESCRIPTOR_OPTIONAL_FIELDS = frozenset({"metadata"})


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_positive_integer(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_capabilities(
    values: Any,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    capabilities = tuple(
        sorted(
            _require_nonempty_string(value, name=f"{name}[{index}]")
            for index, value in enumerate(values)
        )
    )
    if not capabilities and not allow_empty:
        raise ValueError(f"{name} must be nonempty")
    if len(set(capabilities)) != len(capabilities):
        raise ValueError(f"{name} must contain unique names")
    return capabilities


def _validated_artifact_versions(
    values: Any,
    *,
    name: str,
    allow_empty: bool = False,
) -> dict[str, int]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    artifact_versions: dict[str, int] = {}
    for raw_artifact, raw_version in values.items():
        artifact = _require_nonempty_string(raw_artifact, name=f"{name} key")
        if artifact in artifact_versions:
            raise ValueError(f"{name} must contain unique artifact names")
        artifact_versions[artifact] = _require_positive_integer(
            raw_version,
            name=f"{name}[{artifact!r}]",
        )
    if not artifact_versions and not allow_empty:
        raise ValueError(f"{name} must be nonempty")
    return artifact_versions


def _require_exact_provider_descriptor_fields(values: Mapping[Any, Any]) -> None:
    if any(type(key) is not str for key in values):
        raise ValueError("provider descriptor keys must be strings")
    actual = set(values)
    allowed = (
        _PROVIDER_DESCRIPTOR_REQUIRED_FIELDS | _PROVIDER_DESCRIPTOR_OPTIONAL_FIELDS
    )
    missing = sorted(_PROVIDER_DESCRIPTOR_REQUIRED_FIELDS - actual)
    unexpected = sorted(actual - allowed)
    if missing or unexpected:
        raise ValueError(
            "provider descriptor fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )


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
        provider_name = _require_nonempty_string(
            self.provider_name,
            name="provider_name",
        )
        provider_version = _require_nonempty_string(
            self.provider_version,
            name="provider_version",
        )
        provider_revision = _require_nonempty_string(
            self.provider_revision,
            name="provider_revision",
        )
        schema_version = _require_positive_integer(
            self.schema_version,
            name="schema_version",
        )
        capabilities = _validated_capabilities(
            self.capabilities,
            name="capabilities",
        )
        artifact_versions = _validated_artifact_versions(
            self.artifact_schema_versions,
            name="artifact_schema_versions",
        )
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "provider_name", provider_name)
        object.__setattr__(self, "provider_version", provider_version)
        object.__setattr__(self, "provider_revision", provider_revision)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(
            self,
            "artifact_schema_versions",
            validated_json_mapping(
                artifact_versions,
                error_message="artifact schema versions must be finite JSON data",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="metadata must contain finite JSON values",
            ),
        )

    @classmethod
    def from_provider_descriptor(
        cls,
        values: Mapping[Any, Any],
    ) -> PhysicalBeliefProviderManifest:
        """Construct a manifest from the exact provider-v1 descriptor schema."""

        if not isinstance(values, Mapping):
            raise ValueError("provider descriptor must be a mapping")
        _require_exact_provider_descriptor_fields(values)
        metadata = values.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("provider descriptor metadata must be a mapping")
        return cls(
            provider_name=values["provider_name"],
            provider_version=values["provider_version"],
            provider_revision=values["provider_revision"],
            schema_version=values["schema_version"],
            capabilities=values["capabilities"],
            artifact_schema_versions=values["artifact_schema_versions"],
            metadata=metadata,
        )

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
            "artifact_version_mismatches": list(self.artifact_version_mismatches),
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

    required = _validated_capabilities(
        required_capabilities,
        name="required_capabilities",
        allow_empty=True,
    )
    if isinstance(supported_schema_versions, (str, bytes)) or not isinstance(
        supported_schema_versions,
        Sequence,
    ):
        raise ValueError("supported_schema_versions must be a sequence of integers")
    supported = tuple(
        _require_positive_integer(value, name=f"supported_schema_versions[{index}]")
        for index, value in enumerate(supported_schema_versions)
    )
    if not supported:
        raise ValueError("supported_schema_versions must be nonempty")
    if len(set(supported)) != len(supported):
        raise ValueError("supported_schema_versions must be unique")
    missing = tuple(
        value for value in required if value not in set(manifest.capabilities)
    )
    unsupported_schema = (
        None if manifest.schema_version in supported else manifest.schema_version
    )

    unsupported_provider_version = None
    normalized_version_range = None
    if supported_provider_versions is not None:
        if type(supported_provider_versions) is not str:
            raise ValueError("supported_provider_versions must be a string")
        normalized_version_range = supported_provider_versions.strip()
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

    required_versions = _validated_artifact_versions(
        ({} if required_artifact_versions is None else required_artifact_versions),
        name="required_artifact_versions",
        allow_empty=True,
    )
    mismatches = []
    for name, expected in required_versions.items():
        actual = manifest.artifact_schema_versions.get(name)
        if actual != expected:
            mismatches.append(f"{name}:expected={expected}:actual={actual}")
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

    if provider_revision is not None:
        _require_nonempty_string(provider_revision, name="provider_revision")
    from bayesian_phystwin.causal4d_provider_v1 import causal4d_provider_manifest

    values = causal4d_provider_manifest(provider_revision=provider_revision)
    return PhysicalBeliefProviderManifest.from_provider_descriptor(values)


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
