from __future__ import annotations

import pytest

from causal4d.graph_provider_contract import (
    BAYESIAN_PHYSTWIN_GRAPH_PARENT_PROVIDER_API,
    BAYESIAN_PHYSTWIN_GRAPH_PARENT_PROVIDER_API_VERSION,
    BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_API,
    BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_API_VERSION,
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
    metadata: dict[str, object] | None = None,
) -> PhysicalBeliefProviderManifest:
    return PhysicalBeliefProviderManifest(
        provider_name="bayesian-phystwin",
        provider_version="0.4.0",
        provider_revision="a" * 40,
        schema_version=1,
        capabilities=capabilities,
        artifact_schema_versions={"PhysTwinSpringGraph": graph_schema},
        metadata=(
            metadata
            if metadata is not None
            else {
                "provider_api": BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_API,
                "provider_api_version": (BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_API_VERSION),
                "parent_provider_api": (BAYESIAN_PHYSTWIN_GRAPH_PARENT_PROVIDER_API),
                "parent_provider_api_version": (
                    BAYESIAN_PHYSTWIN_GRAPH_PARENT_PROVIDER_API_VERSION
                ),
            }
        ),
    )


def test_graph_provider_contract_accepts_complete_v1_child_manifest() -> None:
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


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("provider_api", "bayesian_phystwin.phystwin_graph"),
        ("provider_api_version", 2),
        ("parent_provider_api", "bayesian_phystwin.causal4d_provider_v1"),
        ("parent_provider_api_version", 1),
    ],
)
def test_graph_provider_contract_rejects_metadata_drift(
    name: str,
    value: object,
) -> None:
    metadata = dict(_manifest().metadata)
    metadata[name] = value
    with pytest.raises(ValueError, match="unexpected"):
        validate_bayesian_phystwin_graph_provider(_manifest(metadata=metadata))


def test_graph_provider_contract_rejects_missing_parent_metadata() -> None:
    metadata = dict(_manifest().metadata)
    del metadata["parent_provider_api"]
    with pytest.raises(ValueError, match="parent_provider_api"):
        validate_bayesian_phystwin_graph_provider(_manifest(metadata=metadata))
