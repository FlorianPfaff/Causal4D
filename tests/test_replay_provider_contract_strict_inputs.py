from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from causal4d.provider_contract import PhysicalBeliefProviderManifest
from causal4d.replay_provider_contract import (
    BAYESIAN_PHYSTWIN_REPLAY_ARTIFACT_SCHEMA_VERSIONS,
    BAYESIAN_PHYSTWIN_REPLAY_PROVIDER_CAPABILITIES,
    load_bayesian_phystwin_replay_provider_manifest,
    stable_replay_identifier,
    validate_bayesian_phystwin_replay_provider,
)


def _descriptor(**overrides: Any) -> dict[Any, Any]:
    values: dict[Any, Any] = {
        "provider_name": "bayesian-phystwin",
        "provider_version": "0.4.0",
        "provider_revision": "replay-provider-test",
        "schema_version": 2,
        "capabilities": list(BAYESIAN_PHYSTWIN_REPLAY_PROVIDER_CAPABILITIES),
        "artifact_schema_versions": dict(
            BAYESIAN_PHYSTWIN_REPLAY_ARTIFACT_SCHEMA_VERSIONS
        ),
        "metadata": {
            "provider_api": "bayesian_phystwin.causal4d_provider_v2",
            "provider_api_version": 2,
        },
    }
    values.update(overrides)
    return values


def _install_fake_provider(monkeypatch: pytest.MonkeyPatch, descriptor: Any) -> None:
    package = ModuleType("bayesian_phystwin")
    package.__path__ = []  # type: ignore[attr-defined]
    provider = ModuleType("bayesian_phystwin.causal4d_provider_v2")

    def manifest(**kwargs: Any) -> Any:
        del kwargs
        return descriptor

    provider.causal4d_provider_manifest = manifest  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bayesian_phystwin", package)
    monkeypatch.setitem(
        sys.modules,
        "bayesian_phystwin.causal4d_provider_v2",
        provider,
    )


@pytest.mark.parametrize("revision", (1, True, 1.0, ""))
def test_replay_loader_rejects_coercible_revision_before_import(
    revision: Any,
) -> None:
    with pytest.raises(ValueError, match="provider_revision must be a nonempty string"):
        load_bayesian_phystwin_replay_provider_manifest(
            provider_revision=revision,
        )


@pytest.mark.parametrize(
    ("descriptor", "message"),
    (
        (
            _descriptor(schema_version=True),
            "schema_version must be a positive integer",
        ),
        (
            _descriptor(capabilities=["artifact_checksums", 3]),
            r"capabilities\[1\]",
        ),
        (
            _descriptor(artifact_schema_versions={1: 1}),
            "artifact_schema_versions key must be a nonempty string",
        ),
        (_descriptor(metadata=[]), "metadata must be a mapping"),
        (_descriptor(unregistered="value"), r"unexpected=\['unregistered'\]"),
    ),
)
def test_replay_loader_rejects_coercible_provider_descriptors(
    monkeypatch: pytest.MonkeyPatch,
    descriptor: Any,
    message: str,
) -> None:
    _install_fake_provider(monkeypatch, descriptor)

    with pytest.raises(ValueError, match=message):
        load_bayesian_phystwin_replay_provider_manifest(
            provider_revision="replay-provider-test"
        )


def test_replay_loader_preserves_exact_valid_descriptor_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _descriptor()
    _install_fake_provider(monkeypatch, descriptor)

    loaded = load_bayesian_phystwin_replay_provider_manifest(
        provider_revision="replay-provider-test"
    )
    direct = PhysicalBeliefProviderManifest.from_provider_descriptor(descriptor)

    assert loaded.manifest_id == direct.manifest_id
    assert validate_bayesian_phystwin_replay_provider(loaded).compatible


def test_replay_loader_binds_the_requested_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _descriptor(provider_revision="different-revision")
    _install_fake_provider(monkeypatch, descriptor)

    with pytest.raises(ValueError, match="revision does not match"):
        load_bayesian_phystwin_replay_provider_manifest(
            provider_revision="requested-revision"
        )


@pytest.mark.parametrize("api_version", (True, 2.0, "2"))
def test_replay_provider_requires_exact_integer_api_metadata(
    api_version: Any,
) -> None:
    descriptor = _descriptor(
        metadata={
            "provider_api": "bayesian_phystwin.causal4d_provider_v2",
            "provider_api_version": api_version,
        }
    )
    manifest = PhysicalBeliefProviderManifest.from_provider_descriptor(descriptor)

    with pytest.raises(ValueError, match="API version 2"):
        validate_bayesian_phystwin_replay_provider(manifest)


def test_replay_identifier_preserves_registered_content_identity() -> None:
    identifier = stable_replay_identifier(
        "request",
        {"frame": 3, "mode": "restart"},
    )

    assert identifier == (
        "request:c8064e2fee453ab7b7e1813fd3220f5ebd274a1a158bb27998c3207e9766131d"
    )


def test_replay_identifier_requires_a_mapping_payload() -> None:
    with pytest.raises(TypeError, match="payload must be a mapping"):
        stable_replay_identifier("request", [])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "payload",
    (
        {1: "coerced"},
        {"nested": {1: "coerced"}},
        {"nested": [{1: "coerced"}]},
    ),
)
def test_replay_identifier_rejects_non_string_json_keys(payload: Any) -> None:
    with pytest.raises(ValueError, match="string keys"):
        stable_replay_identifier("request", payload)


def test_replay_identifier_rejects_string_subclass_namespace() -> None:
    class Namespace(str):
        pass

    with pytest.raises(TypeError, match="namespace must be a string"):
        stable_replay_identifier(Namespace("request"), {"frame": 1})
