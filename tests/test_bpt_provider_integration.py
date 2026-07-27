"""Cross-repository contract tests for the public Bayesian-PhysTwin provider."""

from __future__ import annotations

import ast
from importlib import import_module
import os
from pathlib import Path

import pytest

from causal4d.provider_contract import (
    BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
    load_bayesian_phystwin_provider_manifest,
    require_bayesian_phystwin_provider,
    validate_bayesian_phystwin_provider,
)
from causal4d.replay_provider_contract import (
    load_bayesian_phystwin_replay_provider_manifest,
    require_bayesian_phystwin_replay_provider,
    validate_bayesian_phystwin_replay_provider,
)


def _provider_api():
    try:
        return import_module("bayesian_phystwin.causal4d_provider_v1")
    except ModuleNotFoundError:
        if os.environ.get("CAUSAL4D_REQUIRE_BPT_PROVIDER") == "1":
            raise
        pytest.skip("Bayesian-PhysTwin provider is an optional integration")


def _replay_provider_api():
    _provider_api()
    return import_module("bayesian_phystwin.causal4d_provider_v2")


def _provider_import_names(module_name: str) -> set[str]:
    repository_root = Path(__file__).resolve().parents[1]
    names: set[str] = set()
    for directory in (repository_root / "src", repository_root / "scripts"):
        if not directory.exists():
            continue
        for path in directory.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    == module_name
                ):
                    names.update(alias.name for alias in node.names)
    return names


def test_installed_provider_manifest_matches_supported_range() -> None:
    _provider_api()
    manifest = load_bayesian_phystwin_provider_manifest(
        provider_revision="cross-repository-test"
    )
    result = validate_bayesian_phystwin_provider(manifest)
    assert result.compatible, result.as_dict()
    assert require_bayesian_phystwin_provider(
        provider_revision="cross-repository-test"
    ).manifest_id == manifest.manifest_id
    assert result.supported_provider_versions == BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE


def test_provider_exports_every_name_consumed_by_causal4d() -> None:
    provider_api = _provider_api()
    missing = sorted(
        name
        for name in _provider_import_names(
            "bayesian_phystwin.causal4d_provider_v1"
        )
        if not hasattr(provider_api, name)
    )
    assert not missing, f"provider API is missing Causal4D imports: {missing}"


def test_replay_protocol_is_runtime_checkable() -> None:
    provider_api = _provider_api()

    class MinimalProvider:
        device = "cpu"

        def set_group_log_scales(self, values):
            del values

        def set_controller_points(self, values):
            del values

        def replay_initial(self, *, frame_count):
            del frame_count
            return (), ()

        def replay_restart(
            self,
            position_m,
            velocity_mps,
            *,
            start_frame,
            stop_frame,
        ):
            del position_m, velocity_mps, start_frame, stop_frame
            return ()

        def close(self):
            return None

    assert isinstance(MinimalProvider(), provider_api.PhysTwinReplayProvider)


def test_installed_replay_provider_manifest_matches_supported_range() -> None:
    _replay_provider_api()
    manifest = load_bayesian_phystwin_replay_provider_manifest(
        provider_revision="cross-repository-replay-test"
    )
    result = validate_bayesian_phystwin_replay_provider(manifest)
    assert result.compatible, result.as_dict()
    assert require_bayesian_phystwin_replay_provider(
        provider_revision="cross-repository-replay-test"
    ).manifest_id == manifest.manifest_id
    assert result.supported_provider_versions == BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE


def test_replay_provider_exports_every_v2_name_consumed_by_causal4d() -> None:
    provider_api = _replay_provider_api()
    missing = sorted(
        name
        for name in _provider_import_names(
            "bayesian_phystwin.causal4d_provider_v2"
        )
        if not hasattr(provider_api, name)
    )
    assert not missing, f"replay provider API is missing Causal4D imports: {missing}"


def test_replay_v2_protocol_is_runtime_checkable() -> None:
    provider_api = _replay_provider_api()

    class MinimalProviderV2:
        device = "cpu"
        frame_dt_s = 0.03
        simulator_configuration_id = "configuration-v2"
        released_initial_state_id = "released-state-v1"

        def replay(self, request):
            return request

        def close(self):
            return None

    assert isinstance(MinimalProviderV2(), provider_api.PhysTwinReplayProvider)
