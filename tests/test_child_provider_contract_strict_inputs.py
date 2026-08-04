from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Callable

import pytest

from causal4d.belief_provider_contract import (
    BAYESIAN_PHYSTWIN_BELIEF_ARTIFACT_SCHEMA_VERSIONS,
    BAYESIAN_PHYSTWIN_BELIEF_INFERENCE_ROLE,
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_API,
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_API_VERSION,
    BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_CAPABILITIES,
    load_bayesian_phystwin_belief_provider_manifest,
    validate_bayesian_phystwin_belief_provider,
)
from causal4d.graph_provider_contract import (
    BAYESIAN_PHYSTWIN_GRAPH_ARTIFACT_SCHEMA_VERSIONS,
    BAYESIAN_PHYSTWIN_GRAPH_PARENT_PROVIDER_API,
    BAYESIAN_PHYSTWIN_GRAPH_PARENT_PROVIDER_API_VERSION,
    BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_API,
    BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_API_VERSION,
    BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_CAPABILITIES,
    load_bayesian_phystwin_graph_provider_manifest,
    validate_bayesian_phystwin_graph_provider,
)
from causal4d.provider_contract import PhysicalBeliefProviderManifest


Loader = Callable[..., PhysicalBeliefProviderManifest]
Validator = Callable[[PhysicalBeliefProviderManifest], object]


def _belief_descriptor(**overrides: Any) -> dict[Any, Any]:
    values: dict[Any, Any] = {
        "provider_name": "bayesian-phystwin",
        "provider_version": "0.4.0",
        "provider_revision": "belief-provider-test",
        "schema_version": 1,
        "capabilities": list(BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_CAPABILITIES),
        "artifact_schema_versions": dict(
            BAYESIAN_PHYSTWIN_BELIEF_ARTIFACT_SCHEMA_VERSIONS
        ),
        "metadata": {
            "provider_api": BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_API,
            "provider_api_version": BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_API_VERSION,
            "inference_role": BAYESIAN_PHYSTWIN_BELIEF_INFERENCE_ROLE,
        },
    }
    values.update(overrides)
    return values


def _graph_descriptor(**overrides: Any) -> dict[Any, Any]:
    values: dict[Any, Any] = {
        "provider_name": "bayesian-phystwin",
        "provider_version": "0.4.0",
        "provider_revision": "graph-provider-test",
        "schema_version": 1,
        "capabilities": list(BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_CAPABILITIES),
        "artifact_schema_versions": dict(
            BAYESIAN_PHYSTWIN_GRAPH_ARTIFACT_SCHEMA_VERSIONS
        ),
        "metadata": {
            "provider_api": BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_API,
            "provider_api_version": BAYESIAN_PHYSTWIN_GRAPH_PROVIDER_API_VERSION,
            "parent_provider_api": BAYESIAN_PHYSTWIN_GRAPH_PARENT_PROVIDER_API,
            "parent_provider_api_version": (
                BAYESIAN_PHYSTWIN_GRAPH_PARENT_PROVIDER_API_VERSION
            ),
        },
    }
    values.update(overrides)
    return values


def _install_fake_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module_name: str,
    function_name: str,
    descriptor: Any,
) -> None:
    package = ModuleType("bayesian_phystwin")
    package.__path__ = []  # type: ignore[attr-defined]
    provider = ModuleType(module_name)

    def manifest(**kwargs: Any) -> Any:
        del kwargs
        return descriptor

    setattr(provider, function_name, manifest)
    monkeypatch.setitem(sys.modules, "bayesian_phystwin", package)
    monkeypatch.setitem(sys.modules, module_name, provider)


@pytest.mark.parametrize(
    "loader",
    (
        load_bayesian_phystwin_belief_provider_manifest,
        load_bayesian_phystwin_graph_provider_manifest,
    ),
)
@pytest.mark.parametrize("revision", (1, True, 1.0, ""))
def test_child_provider_loaders_reject_coercible_revision_before_import(
    loader: Loader,
    revision: Any,
) -> None:
    with pytest.raises(ValueError, match="provider_revision must be a nonempty string"):
        loader(provider_revision=revision)


@pytest.mark.parametrize(
    ("module_name", "function_name", "loader", "descriptor"),
    (
        (
            "bayesian_phystwin.causal4d_belief_provider_v1",
            "causal4d_belief_provider_manifest",
            load_bayesian_phystwin_belief_provider_manifest,
            _belief_descriptor(schema_version=True),
        ),
        (
            "bayesian_phystwin.causal4d_graph_provider_v1",
            "causal4d_graph_provider_manifest",
            load_bayesian_phystwin_graph_provider_manifest,
            _graph_descriptor(capabilities=["controller_grouping", 3]),
        ),
    ),
)
def test_child_provider_loaders_reject_coercible_descriptors(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    function_name: str,
    loader: Loader,
    descriptor: Any,
) -> None:
    _install_fake_provider(
        monkeypatch,
        module_name=module_name,
        function_name=function_name,
        descriptor=descriptor,
    )

    with pytest.raises(ValueError):
        loader(provider_revision="provider-test")


@pytest.mark.parametrize(
    ("module_name", "function_name", "loader", "descriptor", "validator"),
    (
        (
            "bayesian_phystwin.causal4d_belief_provider_v1",
            "causal4d_belief_provider_manifest",
            load_bayesian_phystwin_belief_provider_manifest,
            _belief_descriptor(),
            validate_bayesian_phystwin_belief_provider,
        ),
        (
            "bayesian_phystwin.causal4d_graph_provider_v1",
            "causal4d_graph_provider_manifest",
            load_bayesian_phystwin_graph_provider_manifest,
            _graph_descriptor(),
            validate_bayesian_phystwin_graph_provider,
        ),
    ),
)
def test_child_provider_loaders_preserve_valid_descriptor_identity(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    function_name: str,
    loader: Loader,
    descriptor: dict[Any, Any],
    validator: Validator,
) -> None:
    _install_fake_provider(
        monkeypatch,
        module_name=module_name,
        function_name=function_name,
        descriptor=descriptor,
    )

    revision = descriptor["provider_revision"]
    loaded = loader(provider_revision=revision)
    direct = PhysicalBeliefProviderManifest.from_provider_descriptor(descriptor)

    assert loaded.manifest_id == direct.manifest_id
    result = validator(loaded)
    assert getattr(result, "compatible") is True


@pytest.mark.parametrize(
    ("module_name", "function_name", "loader", "descriptor"),
    (
        (
            "bayesian_phystwin.causal4d_belief_provider_v1",
            "causal4d_belief_provider_manifest",
            load_bayesian_phystwin_belief_provider_manifest,
            _belief_descriptor(provider_revision="different-belief-revision"),
        ),
        (
            "bayesian_phystwin.causal4d_graph_provider_v1",
            "causal4d_graph_provider_manifest",
            load_bayesian_phystwin_graph_provider_manifest,
            _graph_descriptor(provider_revision="different-graph-revision"),
        ),
    ),
)
def test_child_provider_loaders_bind_the_requested_revision(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    function_name: str,
    loader: Loader,
    descriptor: dict[Any, Any],
) -> None:
    _install_fake_provider(
        monkeypatch,
        module_name=module_name,
        function_name=function_name,
        descriptor=descriptor,
    )

    with pytest.raises(ValueError, match="revision does not match"):
        loader(provider_revision="requested-revision")


@pytest.mark.parametrize("api_version", (True, 1.0, "1"))
def test_belief_provider_requires_exact_integer_api_metadata(
    api_version: Any,
) -> None:
    descriptor = _belief_descriptor(
        metadata={
            "provider_api": BAYESIAN_PHYSTWIN_BELIEF_PROVIDER_API,
            "provider_api_version": api_version,
            "inference_role": BAYESIAN_PHYSTWIN_BELIEF_INFERENCE_ROLE,
        }
    )
    manifest = PhysicalBeliefProviderManifest.from_provider_descriptor(descriptor)

    with pytest.raises(ValueError, match="unexpected"):
        validate_bayesian_phystwin_belief_provider(manifest)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("provider_api_version", 1.0),
        ("parent_provider_api_version", 2.0),
        ("parent_provider_api_version", True),
    ),
)
def test_graph_provider_requires_exact_integer_api_metadata(
    name: str,
    value: Any,
) -> None:
    metadata = dict(_graph_descriptor()["metadata"])
    metadata[name] = value
    manifest = PhysicalBeliefProviderManifest.from_provider_descriptor(
        _graph_descriptor(metadata=metadata)
    )

    with pytest.raises(ValueError, match="unexpected"):
        validate_bayesian_phystwin_graph_provider(manifest)
