from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

provider_api = pytest.importorskip("bayesian_phystwin.causal4d_provider_v2")
if not hasattr(provider_api, "ScheduledContactReplayProviderV1"):
    pytest.skip(
        "installed Bayesian-PhysTwin lacks scheduled replay contracts",
        allow_module_level=True,
    )

from causal4d.multi_contact import MultiContactPathPrior
from causal4d.scheduled_contact_replay import (
    ScheduledContactReplayRejectedError,
    ScheduledContactReplayUnavailableError,
    replay_multi_contact_prior,
)
import causal4d.scheduled_contact_replay as scheduled_replay


def _prior() -> MultiContactPathPrior:
    return MultiContactPathPrior(
        contact_ids=("left", "right"),
        path_ids=("path-a", "path-b"),
        regime_paths=np.asarray(
            [
                [[0, 1, 1, 3], [0, 0, 2, 3]],
                [[0, 0, 1, 3], [0, 1, 1, 3]],
            ],
            dtype=np.int8,
        ),
        weights=np.asarray((0.6, 0.4)),
        retained_prior_mass=0.9,
        marginal_retained_prior_mass=np.asarray((1.0, 1.0)),
        marginal_path_indices=np.asarray(((0, 0), (1, 1))),
        marginal_path_counts=np.asarray((2, 2)),
    )


def _contact_geometry(
    prior: MultiContactPathPrior,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.full((*prior.regime_paths.shape, 2), -1, dtype=np.int64)
    weights = np.zeros(indices.shape, dtype=float)
    active = (prior.regime_paths == 1) | (prior.regime_paths == 2)
    for path_index, contact_index, frame_index in np.argwhere(active):
        indices[path_index, contact_index, frame_index] = (0, 1)
        weights[path_index, contact_index, frame_index] = (0.25, 0.75)
    return indices, weights


def _call(provider):
    prior = _prior()
    indices, weights = _contact_geometry(prior)
    return replay_multi_contact_prior(
        prior,
        provider,
        request_id="scheduled-001",
        simulator_configuration_id="sim-config-001",
        initial_state_id="endpoint-001",
        group_log_scales=np.asarray((0.1, -0.2)),
        controller_points_m=np.zeros((4, 2, 3)),
        position_m=np.zeros((3, 3)),
        velocity_mps=np.zeros((3, 3)),
        frame_times_s=np.asarray((0.0, 0.04, 0.08, 0.12)),
        contact_node_indices=indices,
        contact_node_weights=weights,
        normal_stiffness_npm=10.0,
        tangential_stiffness_npm=np.asarray((2.0, 3.0)),
        friction_coefficient=0.5,
    )


class _FakeProvider:
    simulator_configuration_id = "sim-config-001"
    provider_revision = "provider-revision-001"

    def __init__(self, *, mode: str = "valid") -> None:
        self.mode = mode
        self.calls = 0
        self.request = None

    def replay_scheduled_contacts(self, request):
        self.calls += 1
        self.request = request
        if self.mode == "wrong-type":
            return object()
        result = provider_api.ScheduledContactReplayResultV1.from_request(
            request,
            positions_m=np.zeros((2, 4, 3, 3)),
            velocities_mps=np.zeros((2, 4, 3, 3)),
            conditional_variance_m2=1e-6,
            provider_name="fake-phystwin",
            provider_version="0.4.0",
            provider_revision=(
                "other-revision"
                if self.mode == "revision-drift"
                else self.provider_revision
            ),
        )
        if self.mode == "request-drift":
            result = replace(result, request_identity="other-request")
        return result

    def close(self) -> None:
        return None


def test_adapter_constructs_a_verified_replay_bound_bank() -> None:
    provider = _FakeProvider()

    evidence = _call(provider)

    assert provider.calls == 1
    assert provider.request.schedule_identity == _prior().schedule_identity
    assert provider.request.contact_ids == ("left", "right")
    assert evidence.bank.replay_bound
    assert evidence.bank.schedule_identity == _prior().schedule_identity
    assert evidence.bank.replay_result_identity == evidence.replay_result_identity
    assert evidence.request_identity == provider.request.request_identity
    assert evidence.provider_revision == provider.provider_revision
    assert evidence.bank.trajectories_m.shape == (2, 4, 3, 3)
    assert evidence.bank.base_variance_m2.shape == ()


def test_configuration_mismatch_fails_before_provider_execution() -> None:
    provider = _FakeProvider()
    provider.simulator_configuration_id = "other-config"

    with pytest.raises(ScheduledContactReplayRejectedError, match="configuration"):
        _call(provider)

    assert provider.calls == 0


@pytest.mark.parametrize(
    ("mode", "message"),
    (
        ("wrong-type", "wrong scheduled replay result type"),
        ("revision-drift", "revision does not match"),
        ("request-drift", "failed scheduled replay validation"),
    ),
)
def test_provider_result_drift_is_rejected(mode: str, message: str) -> None:
    provider = _FakeProvider(mode=mode)

    with pytest.raises(ScheduledContactReplayRejectedError, match=message):
        _call(provider)


def test_missing_provider_contract_has_a_clear_additive_compatibility_error(
    monkeypatch,
) -> None:
    class _LegacyProviderModule:
        pass

    monkeypatch.setattr(
        scheduled_replay,
        "import_module",
        lambda name: _LegacyProviderModule(),
    )

    with pytest.raises(ScheduledContactReplayUnavailableError, match="lacks"):
        scheduled_replay._provider_contracts()
