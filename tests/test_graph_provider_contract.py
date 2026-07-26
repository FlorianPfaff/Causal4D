from __future__ import annotations

from causal4d.graph_provider_contract import (
    validate_bayesian_phystwin_graph_provider,
)
from causal4d.provider_contract import PhysicalBeliefProviderManifest


def _manifest(
    *,
    capabilities: tuple[str, ...] = (
        "controller_grouping",
        "phystwin_spring_graph",
    ),
    graph_schema: int = 1,
) -> PhysicalBeliefProviderManifest:
    return PhysicalBeliefProviderManifest(
        provider_name="bayesian-phystwin",
        provider_version="0.4.0",
        provider_revision="a" * 40,
        schema_version=1,
        capabilities=capabilities,
        artifact_schema_versions={"PhysTwinSpringGraph": graph_schema},
        metadata={
            "provider_api": "bayesian_phystwin.causal4d_graph_provider_v1",
            "provider_api_version": 1,
        },
    )


def test_graph_provider_contract_accepts_complete_v1_manifest() -> None:
    result = validate_bayesian_phystwin_graph_provider(_manifest())
    assert result.compatible
    assert result.missing_capabilities == ()
    assert result.artifact_version_mismatches == ()


def test_graph_provider_contract_rejects_missing_capability() -> None:
    result = validate_bayesian_phystwin_graph_provider(
        _manifest(capabilities=("phystwin_spring_graph",))
    )
    assert not result.compatible
    assert result.missing_capabilities == ("controller_grouping",)


def test_graph_provider_contract_rejects_graph_schema_drift() -> None:
    result = validate_bayesian_phystwin_graph_provider(_manifest(graph_schema=2))
    assert not result.compatible
    assert result.artifact_version_mismatches == (
        "PhysTwinSpringGraph:expected=1:actual=2",
    )


def test_graph_provider_contract_rejects_unexpected_api_path() -> None:
    manifest = _manifest()
    wrong = PhysicalBeliefProviderManifest(
        provider_name=manifest.provider_name,
        provider_version=manifest.provider_version,
        provider_revision=manifest.provider_revision,
        schema_version=manifest.schema_version,
        capabilities=manifest.capabilities,
        artifact_schema_versions=manifest.artifact_schema_versions,
        metadata={"provider_api": "bayesian_phystwin.phystwin_graph"},
    )
    try:
        validate_bayesian_phystwin_graph_provider(wrong)
    except ValueError as error:
        assert "unexpected" in str(error)
    else:
        raise AssertionError("unexpected graph provider API was accepted")
