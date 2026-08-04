from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from causal4d.provider_contract import (
    BAYESIAN_PHYSTWIN_ARTIFACT_SCHEMA_VERSIONS,
    BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES,
    PhysicalBeliefProviderManifest,
    load_bayesian_phystwin_provider_manifest,
    validate_provider_compatibility,
)


def test_falsey_non_mapping_artifact_requirements_are_rejected() -> None:
    manifest = PhysicalBeliefProviderManifest(
        provider_name="bayesian-phystwin",
        provider_version="0.4.0",
        provider_revision="a" * 40,
        schema_version=1,
        capabilities=BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES,
        artifact_schema_versions={"GraphBelief": 1, "TwinBelief": 1},
    )

    with pytest.raises(
        ValueError,
        match="required_artifact_versions must be a mapping",
    ):
        validate_provider_compatibility(
            manifest,
            required_artifact_versions=[],  # type: ignore[arg-type]
        )


def test_root_provider_loader_binds_the_requested_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = ModuleType("bayesian_phystwin")
    package.__path__ = []  # type: ignore[attr-defined]
    provider = ModuleType("bayesian_phystwin.causal4d_provider_v1")
    descriptor = {
        "provider_name": "bayesian-phystwin",
        "provider_version": "0.4.0",
        "provider_revision": "different-revision",
        "schema_version": 1,
        "capabilities": list(BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES),
        "artifact_schema_versions": dict(
            BAYESIAN_PHYSTWIN_ARTIFACT_SCHEMA_VERSIONS
        ),
        "metadata": {
            "provider_api": "bayesian_phystwin.causal4d_provider_v1",
            "provider_api_version": 1,
        },
    }

    def manifest(**kwargs: Any) -> dict[str, object]:
        del kwargs
        return descriptor

    provider.causal4d_provider_manifest = manifest  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bayesian_phystwin", package)
    monkeypatch.setitem(
        sys.modules,
        "bayesian_phystwin.causal4d_provider_v1",
        provider,
    )

    with pytest.raises(ValueError, match="revision does not match"):
        load_bayesian_phystwin_provider_manifest(
            provider_revision="requested-revision"
        )
