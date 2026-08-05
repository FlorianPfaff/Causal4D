from __future__ import annotations

import pytest

from causal4d.belief_provider_v2_contract import (
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_API,
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_API_VERSION,
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_CAPABILITIES,
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_SCHEMA_VERSIONS,
    BAYESIAN_PHYSTWIN_BELIEF_V2_ARTIFACT_SCHEMA_VERSIONS,
    BAYESIAN_PHYSTWIN_BELIEF_V2_COMPATIBILITY,
    BAYESIAN_PHYSTWIN_BELIEF_V2_INFERENCE_ROLE,
    BAYESIAN_PHYSTWIN_BELIEF_V2_RAW_COVARIANCE_CLAIM,
    validate_bayesian_phystwin_belief_provider_v2,
)
from causal4d.provider_contract import PhysicalBeliefProviderManifest


def _manifest(
    *,
    schema_version: int = 2,
    capabilities: tuple[str, ...] = (
        BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_CAPABILITIES
    ),
    schemas: dict[str, int] | None = None,
    metadata: dict[str, object] | None = None,
) -> PhysicalBeliefProviderManifest:
    return PhysicalBeliefProviderManifest(
        provider_name="bayesian-phystwin",
        provider_version="0.4.0",
        provider_revision="a" * 40,
        schema_version=schema_version,
        capabilities=capabilities,
        artifact_schema_versions=(
            BAYESIAN_PHYSTWIN_BELIEF_V2_ARTIFACT_SCHEMA_VERSIONS
            if schemas is None
            else schemas
        ),
        metadata=(
            {
                "provider_api": BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_API,
                "provider_api_version": (
                    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_API_VERSION
                ),
                "inference_role": BAYESIAN_PHYSTWIN_BELIEF_V2_INFERENCE_ROLE,
                "compatibility": BAYESIAN_PHYSTWIN_BELIEF_V2_COMPATIBILITY,
                "raw_covariance_claim": (
                    BAYESIAN_PHYSTWIN_BELIEF_V2_RAW_COVARIANCE_CLAIM
                ),
            }
            if metadata is None
            else metadata
        ),
    )


def test_belief_provider_v2_accepts_complete_additive_manifest() -> None:
    result = validate_bayesian_phystwin_belief_provider_v2(_manifest())

    assert BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_SCHEMA_VERSIONS == (2,)
    assert result.compatible
    assert result.unsupported_schema_version is None
    assert result.missing_capabilities == ()
    assert result.artifact_version_mismatches == ()


@pytest.mark.parametrize("schema_version", [1, 3])
def test_belief_provider_v2_rejects_other_manifest_schemas(
    schema_version: int,
) -> None:
    result = validate_bayesian_phystwin_belief_provider_v2(
        _manifest(schema_version=schema_version)
    )

    assert not result.compatible
    assert result.unsupported_schema_version == schema_version


def test_belief_provider_v2_rejects_missing_horizon_capability() -> None:
    capabilities = tuple(
        value
        for value in BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_V2_CAPABILITIES
        if value != "source_calibrated_horizon_discrepancy"
    )

    result = validate_bayesian_phystwin_belief_provider_v2(
        _manifest(capabilities=capabilities)
    )

    assert not result.compatible
    assert result.missing_capabilities == (
        "source_calibrated_horizon_discrepancy",
    )


def test_belief_provider_v2_rejects_horizon_schema_drift() -> None:
    schemas = dict(BAYESIAN_PHYSTWIN_BELIEF_V2_ARTIFACT_SCHEMA_VERSIONS)
    schemas["HorizonConditionedEndpointPrediction"] = 2

    result = validate_bayesian_phystwin_belief_provider_v2(
        _manifest(schemas=schemas)
    )

    assert not result.compatible
    assert result.artifact_version_mismatches == (
        "HorizonConditionedEndpointPrediction:expected=1:actual=2",
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("provider_api", "bayesian_phystwin.causal4d_belief_provider_v1"),
        ("provider_api_version", 1),
        ("inference_role", "physical state correction"),
        ("compatibility", "replaces provider v1"),
        ("raw_covariance_claim", "calibrated deployment intervals"),
    ],
)
def test_belief_provider_v2_rejects_metadata_drift(
    name: str,
    value: object,
) -> None:
    metadata = dict(_manifest().metadata)
    metadata[name] = value

    with pytest.raises(ValueError, match="unexpected"):
        validate_bayesian_phystwin_belief_provider_v2(
            _manifest(metadata=metadata)
        )


def test_belief_provider_v2_rejects_wrong_provider_name() -> None:
    manifest = _manifest()
    wrong = PhysicalBeliefProviderManifest(
        provider_name="other-provider",
        provider_version=manifest.provider_version,
        provider_revision=manifest.provider_revision,
        schema_version=manifest.schema_version,
        capabilities=manifest.capabilities,
        artifact_schema_versions=manifest.artifact_schema_versions,
        metadata=manifest.metadata,
    )

    with pytest.raises(ValueError, match="bayesian-phystwin"):
        validate_bayesian_phystwin_belief_provider_v2(wrong)
