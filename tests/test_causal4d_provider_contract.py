from causal4d.provider_contract import (
    BASE_CAUSAL4D_PROVIDER_CAPABILITIES,
    PHYSICAL_BELIEF_PROVIDER_SCHEMA_VERSION,
    PhysicalBeliefProviderManifest,
    validate_provider_compatibility,
)


def _manifest(**overrides):
    values = {
        "provider_name": "bayesian-phystwin",
        "provider_version": "0.4.0",
        "provider_revision": "c7ad36aad7e592ce8a391c9ca2d4db7389dee3ac",
        "schema_version": PHYSICAL_BELIEF_PROVIDER_SCHEMA_VERSION,
        "capabilities": (*BASE_CAUSAL4D_PROVIDER_CAPABILITIES, "graph_covariance"),
        "artifact_schema_versions": {"TwinBelief": 1, "GraphBelief": 1},
    }
    values.update(overrides)
    return PhysicalBeliefProviderManifest(**values)


def test_compatible_provider_passes_explicit_contract() -> None:
    manifest = _manifest()
    result = validate_provider_compatibility(
        manifest,
        required_artifact_versions={"TwinBelief": 1},
    )
    assert result.compatible
    assert not result.missing_capabilities
    assert result.provider_manifest_id == manifest.manifest_id


def test_missing_capability_fails_closed() -> None:
    manifest = _manifest(capabilities=("artifact_checksums",))
    result = validate_provider_compatibility(manifest)
    assert not result.compatible
    assert "particle_endpoint_velocity" in result.missing_capabilities


def test_schema_and_artifact_mismatch_are_reported() -> None:
    manifest = _manifest(schema_version=2)
    result = validate_provider_compatibility(
        manifest,
        required_artifact_versions={"TwinBelief": 2},
    )
    assert not result.compatible
    assert result.unsupported_schema_version == 2
    assert result.artifact_version_mismatches == (
        "TwinBelief:expected=2:actual=1",
    )


def test_manifest_identifier_is_order_invariant() -> None:
    first = _manifest(
        capabilities=("graph_covariance", *BASE_CAUSAL4D_PROVIDER_CAPABILITIES),
        artifact_schema_versions={"GraphBelief": 1, "TwinBelief": 1},
    )
    second = _manifest()
    assert first.manifest_id == second.manifest_id
