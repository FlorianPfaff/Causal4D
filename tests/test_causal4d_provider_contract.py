from typing import Any

import pytest

import causal4d.provider_contract as provider_contract
from causal4d.provider_contract import (
    BASE_CAUSAL4D_PROVIDER_CAPABILITIES,
    BAYESIAN_PHYSTWIN_ARTIFACT_SCHEMA_VERSIONS,
    BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
    BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES,
    PHYSICAL_BELIEF_PROVIDER_SCHEMA_VERSION,
    PhysicalBeliefProviderManifest,
    load_bayesian_phystwin_provider_manifest,
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


def _descriptor(**overrides: Any) -> dict[Any, Any]:
    values: dict[Any, Any] = {
        "provider_name": "bayesian-phystwin",
        "provider_version": "0.4.0",
        "provider_revision": "c7ad36aad7e592ce8a391c9ca2d4db7389dee3ac",
        "schema_version": PHYSICAL_BELIEF_PROVIDER_SCHEMA_VERSION,
        "capabilities": list(BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES),
        "artifact_schema_versions": {"TwinBelief": 1, "GraphBelief": 1},
        "metadata": {
            "provider_api": "bayesian_phystwin.causal4d_provider_v1",
            "provider_api_version": 1,
        },
    }
    values.update(overrides)
    return values


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
    assert result.artifact_version_mismatches == ("TwinBelief:expected=2:actual=1",)


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
    assert result.artifact_version_mismatches == ("TwinBelief:expected=1:actual=2",)


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


def test_manifest_content_address_cannot_be_changed_by_nested_mutation() -> None:
    artifact_versions = {"TwinBelief": 1, "GraphBelief": 1}
    metadata = {"nested": {"items": [1, {"accepted": True}]}}
    manifest = _manifest(
        artifact_schema_versions=artifact_versions,
        metadata=metadata,
    )
    manifest_id = manifest.manifest_id

    artifact_versions["TwinBelief"] = 99
    metadata["nested"]["items"][1]["accepted"] = False
    assert manifest.artifact_schema_versions["TwinBelief"] == 1
    assert manifest.metadata["nested"]["items"][1]["accepted"] is True
    assert manifest.manifest_id == manifest_id

    with pytest.raises(TypeError, match="immutable"):
        manifest.artifact_schema_versions["TwinBelief"] = 2
    with pytest.raises(TypeError, match="immutable"):
        manifest.metadata["nested"]["items"].append("mutated")


def test_exact_provider_descriptor_constructs_the_same_manifest() -> None:
    from_descriptor = PhysicalBeliefProviderManifest.from_provider_descriptor(
        _descriptor()
    )
    direct = _manifest(metadata=_descriptor()["metadata"])

    assert from_descriptor.manifest_id == direct.manifest_id
    assert from_descriptor.capabilities == tuple(
        sorted(BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES)
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"provider_name": None}, "provider_name must be a nonempty string"),
        ({"schema_version": True}, "schema_version must be a positive integer"),
        ({"capabilities": ("valid", 3)}, "capabilities\[1\]"),
        (
            {"artifact_schema_versions": {1: 1}},
            "artifact_schema_versions key must be a nonempty string",
        ),
        (
            {"artifact_schema_versions": {"TwinBelief": True}},
            "must be a positive integer",
        ),
        ({"metadata": []}, "metadata must be a mapping"),
    ),
)
def test_manifest_constructor_rejects_coercible_fields(
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _manifest(**overrides)


def test_provider_descriptor_rejects_schema_and_type_drift() -> None:
    valid = _descriptor()
    assert PhysicalBeliefProviderManifest.from_provider_descriptor(valid)

    malformed: list[tuple[dict[Any, Any], str]] = []

    missing = _descriptor()
    del missing["provider_name"]
    malformed.append((missing, "fields do not match schema"))

    unknown = _descriptor(unregistered="value")
    malformed.append((unknown, "unexpected=\['unregistered'\]"))

    non_string_key = _descriptor()
    non_string_key[1] = "value"
    malformed.append((non_string_key, "keys must be strings"))

    boolean_schema = _descriptor(schema_version=True)
    malformed.append((boolean_schema, "schema_version must be a positive integer"))

    string_capabilities = _descriptor(capabilities="artifact_checksums")
    malformed.append((string_capabilities, "must be a sequence of strings"))

    fractional_artifact = _descriptor(
        artifact_schema_versions={"TwinBelief": 1.5}
    )
    malformed.append((fractional_artifact, "must be a positive integer"))

    malformed_metadata = _descriptor(metadata=[])
    malformed.append((malformed_metadata, "metadata must be a mapping"))

    for descriptor, message in malformed:
        with pytest.raises(ValueError, match=message):
            PhysicalBeliefProviderManifest.from_provider_descriptor(descriptor)


def test_compatibility_requirements_reject_coercible_values() -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match="required_capabilities\[0\]"):
        validate_provider_compatibility(manifest, required_capabilities=(1,))
    with pytest.raises(ValueError, match="supported_schema_versions\[0\]"):
        validate_provider_compatibility(manifest, supported_schema_versions=(True,))
    with pytest.raises(ValueError, match="supported_provider_versions must be a string"):
        validate_provider_compatibility(manifest, supported_provider_versions=1)
    with pytest.raises(ValueError, match="required_artifact_versions key"):
        validate_provider_compatibility(
            manifest,
            required_artifact_versions={1: 1},
        )
    with pytest.raises(ValueError, match="must be a positive integer"):
        validate_provider_compatibility(
            manifest,
            required_artifact_versions={"TwinBelief": 1.5},
        )


def test_loader_rejects_coercible_revision_before_import() -> None:
    with pytest.raises(ValueError, match="provider_revision must be a nonempty string"):
        load_bayesian_phystwin_provider_manifest(provider_revision=1)
