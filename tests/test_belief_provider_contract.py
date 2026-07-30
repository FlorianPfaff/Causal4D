from __future__ import annotations

import pytest

from causal4d.belief_provider_contract import (
    BAYESIAN_PHYSTWIN_BELIEF_INFERENCE_ROLE,
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_API,
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_API_VERSION,
    validate_bayesian_phystwin_belief_provider,
)
from causal4d.provider_contract import PhysicalBeliefProviderManifest


def _manifest(
    *,
    capabilities: tuple[str, ...] = (
        "causal_prefix_endpoint_inference",
        "fixed_bayesian_anchor_endpoint",
        "immutable_endpoint_posterior",
        "numpy_only_endpoint_inference",
        "residual_finite_preflight",
    ),
    config_schema: int = 1,
    posterior_schema: int = 1,
    metadata: dict[str, object] | None = None,
) -> PhysicalBeliefProviderManifest:
    return PhysicalBeliefProviderManifest(
        provider_name="bayesian-phystwin",
        provider_version="0.4.0",
        provider_revision="a" * 40,
        schema_version=1,
        capabilities=capabilities,
        artifact_schema_versions={
            "FixedBayesianAnchorConfig": config_schema,
            "RobustEndpointPosterior": posterior_schema,
        },
        metadata=(
            metadata
            if metadata is not None
            else {
                "provider_api": BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_API,
                "provider_api_version": (
                    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_API_VERSION
                ),
                "inference_role": BAYESIAN_PHYSTWIN_BELIEF_INFERENCE_ROLE,
            }
        ),
    )


def test_belief_provider_contract_accepts_complete_v1_manifest() -> None:
    result = validate_bayesian_phystwin_belief_provider(_manifest())
    assert result.compatible
    assert result.missing_capabilities == ()
    assert result.artifact_version_mismatches == ()


def test_belief_provider_contract_rejects_missing_capability() -> None:
    capabilities = tuple(
        value
        for value in _manifest().capabilities
        if value != "residual_finite_preflight"
    )
    result = validate_bayesian_phystwin_belief_provider(
        _manifest(capabilities=capabilities)
    )
    assert not result.compatible
    assert result.missing_capabilities == ("residual_finite_preflight",)


@pytest.mark.parametrize(
    ("config_schema", "posterior_schema", "expected"),
    [
        (2, 1, "FixedBayesianAnchorConfig:expected=1:actual=2"),
        (1, 2, "RobustEndpointPosterior:expected=1:actual=2"),
    ],
)
def test_belief_provider_contract_rejects_schema_drift(
    config_schema: int,
    posterior_schema: int,
    expected: str,
) -> None:
    result = validate_bayesian_phystwin_belief_provider(
        _manifest(
            config_schema=config_schema,
            posterior_schema=posterior_schema,
        )
    )
    assert not result.compatible
    assert result.artifact_version_mismatches == (expected,)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("provider_api", "bayesian_phystwin.fixed_anchor"),
        ("provider_api_version", 2),
        ("inference_role", "physical state correction"),
    ],
)
def test_belief_provider_contract_rejects_metadata_drift(
    name: str,
    value: object,
) -> None:
    metadata = dict(_manifest().metadata)
    metadata[name] = value
    with pytest.raises(ValueError, match="unexpected"):
        validate_bayesian_phystwin_belief_provider(
            _manifest(metadata=metadata)
        )


def test_belief_provider_contract_rejects_wrong_provider_name() -> None:
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
        validate_bayesian_phystwin_belief_provider(wrong)
