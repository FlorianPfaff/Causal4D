import pytest

import causal4d.provider_contract as provider_contract
from causal4d.provider_contract import (
    BASE_CAUSAL4D_PROVIDER_CAPABILITIES,
    BAYESIAN_PHYSTWIN_ARTIFACT_SCHEMA_VERSIONS,
    BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
    BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES,
    PHYSICAL_BELIEF_PROVIDER_SCHEMA_VERSION,
    PhysicalBeliefProviderManifest,
    require_bayesian_phystwin_provider,
    validate_bayesian_phystwin_provider,
    validate_provider_compatibility,
)


def _manifest(**overrides):
    values = {
        "provider_name": "bayesian-phystwin",
        "provider_version": "0.4.0",
        "provider_revision": "c7ad36aad7e592ce8a391c9ca2d4db7389dee3ac",
        "schema_version": PHYSICAL_BELIEF_PROVIDER_SCHEMA_VERSION,
        "capabilities": (*BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES, "graph_covariance"),
        "artifact_schema_versions": {"TwinBelief": 1, "GraphBelief": 1},
    }
    values.update(overrides)
    return PhysicalBeliefProviderManifest(**values)


def test_compatible_provider_passes_explicit_contract() -> None:
    manifest = _manifest()
    result = validate_provider_compatibility(
        manifest,
        required_capabilities=BASE_CAUSAL4D_PROVIDER_CAPABILITIES,
        required_artifact_versions={"TwinBelief": 1},
    )
    assert result.compatible
    assert not result.missing_capabilities
    assert result.unsupported_provider_version is None
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


def test_provider_version_range_is_enforced() -> None:
    compatible = validate_provider_compatibility(
        _manifest(provider_version="0.4.7"),
        supported_provider_versions=BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
    )
    assert compatible.compatible
    assert compatible.supported_provider_versions == ">=0.4,<0.5"
    assert compatible.unsupported_provider_version is None

    incompatible = validate_provider_compatibility(
        _manifest(provider_version="0.5.0"),
        supported_provider_versions=BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
    )
    assert not incompatible.compatible
    assert incompatible.unsupported_provider_version == "0.5.0"

    invalid = validate_provider_compatibility(
        _manifest(provider_version="working-tree"),
        supported_provider_versions=BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
    )
    assert not invalid.compatible
    assert invalid.unsupported_provider_version == "working-tree"


def test_bayesian_phystwin_policy_requires_execution_and_artifact_contracts() -> None:
    result = validate_bayesian_phystwin_provider(_manifest())
    assert result.compatible
    assert result.supported_provider_versions == BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE

    missing_replay = _manifest(
        capabilities=tuple(
            value
            for value in BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES
            if value != "phystwin_replay"
        )
    )
    result = validate_bayesian_phystwin_provider(missing_replay)
    assert not result.compatible
    assert result.missing_capabilities == ("phystwin_replay",)

    wrong_artifact = _manifest(
        artifact_schema_versions={
            **BAYESIAN_PHYSTWIN_ARTIFACT_SCHEMA_VERSIONS,
            "TwinBelief": 2,
        }
    )
    result = validate_bayesian_phystwin_provider(wrong_artifact)
    assert not result.compatible
    assert result.artifact_version_mismatches == (
        "TwinBelief:expected=1:actual=2",
    )



def test_runtime_requirement_rejects_an_incompatible_installed_provider(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        provider_contract,
        "load_bayesian_phystwin_provider_manifest",
        lambda **kwargs: _manifest(provider_version="0.5.0"),
    )
    with pytest.raises(RuntimeError, match="incompatible Bayesian-PhysTwin provider"):
        require_bayesian_phystwin_provider()


def test_wrong_provider_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected the bayesian-phystwin provider"):
        validate_bayesian_phystwin_provider(_manifest(provider_name="other-provider"))


def test_manifest_identifier_is_order_invariant() -> None:
    first = _manifest(
        capabilities=("graph_covariance", *BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES),
        artifact_schema_versions={"GraphBelief": 1, "TwinBelief": 1},
    )
    second = _manifest()
    assert first.manifest_id == second.manifest_id
