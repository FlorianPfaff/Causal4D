"""Consume verified Bayesian-PhysTwin joint contact schedule replays."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType

import numpy as np

from causal4d._multi_contact_inference import MultiContactPathBank
from causal4d._multi_contact_prior import MultiContactPathPrior


class ScheduledContactReplayUnavailableError(ImportError):
    """Raised when the installed Bayesian-PhysTwin lacks the additive contract."""


class ScheduledContactReplayRejectedError(ValueError):
    """Raised when a provider or result drifts from the requested replay boundary."""


def _identifier(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ScheduledContactReplayRejectedError(
            f"{name} must be a nonempty canonical string"
        )
    return value


def _provider_contracts() -> ModuleType:
    try:
        module = import_module("bayesian_phystwin.causal4d_provider_v2")
    except ImportError as error:
        raise ScheduledContactReplayUnavailableError(
            "Bayesian-PhysTwin provider v2 is not installed"
        ) from error
    required = (
        "ScheduledContactReplayProviderV1",
        "ScheduledContactReplayRequestV1",
        "ScheduledContactReplayResultV1",
        "validate_scheduled_contact_replay_result",
    )
    missing = tuple(name for name in required if not hasattr(module, name))
    if missing:
        raise ScheduledContactReplayUnavailableError(
            "installed Bayesian-PhysTwin lacks scheduled-contact replay contracts: "
            + ", ".join(missing)
        )
    return module


@dataclass(frozen=True)
class ScheduledContactReplayEvidence:
    """Verified replay bank plus portable identities for run-manifest binding."""

    bank: MultiContactPathBank
    request_identity: str
    replay_result_identity: str
    provider_name: str
    provider_version: str
    provider_revision: str
    simulator_configuration_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.bank, MultiContactPathBank):
            raise TypeError("bank must be a MultiContactPathBank")
        for name in (
            "request_identity",
            "replay_result_identity",
            "provider_name",
            "provider_version",
            "provider_revision",
            "simulator_configuration_id",
        ):
            object.__setattr__(
                self,
                name,
                _identifier(getattr(self, name), name=name),
            )
        if self.bank.replay_result_identity != self.replay_result_identity:
            raise ScheduledContactReplayRejectedError(
                "bank replay identity does not match the verified provider result"
            )
        if not self.bank.replay_bound:
            raise ScheduledContactReplayRejectedError(
                "scheduled replay bank must bind a result identity and timebase"
            )


def replay_multi_contact_prior(
    prior: MultiContactPathPrior,
    provider: object,
    *,
    request_id: str,
    simulator_configuration_id: str,
    initial_state_id: str,
    group_log_scales: np.ndarray,
    controller_points_m: np.ndarray,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    frame_times_s: np.ndarray,
    contact_node_indices: np.ndarray,
    contact_node_weights: np.ndarray,
    normal_stiffness_npm: np.ndarray | float,
    tangential_stiffness_npm: np.ndarray | float,
    friction_coefficient: np.ndarray | float,
) -> ScheduledContactReplayEvidence:
    """Replay every retained schedule and construct a provider-bound path bank.

    The installed Bayesian-PhysTwin contract validates finite-area contact
    geometry, physical parameters, endpoint state, schedule support, and timebase
    before the provider runs. The result is revalidated before its trajectories
    can enter ``MultiContactPathBank``.
    """

    if not isinstance(prior, MultiContactPathPrior):
        raise TypeError("prior must be a MultiContactPathPrior")
    contracts = _provider_contracts()
    provider_protocol = contracts.ScheduledContactReplayProviderV1
    if not isinstance(provider, provider_protocol):
        raise TypeError("provider must implement ScheduledContactReplayProviderV1")

    requested_configuration = _identifier(
        simulator_configuration_id,
        name="simulator_configuration_id",
    )
    provider_configuration = _identifier(
        provider.simulator_configuration_id,
        name="provider.simulator_configuration_id",
    )
    if provider_configuration != requested_configuration:
        raise ScheduledContactReplayRejectedError(
            "provider simulator configuration does not match the request"
        )
    provider_revision = _identifier(
        provider.provider_revision,
        name="provider.provider_revision",
    )

    request = contracts.ScheduledContactReplayRequestV1(
        request_id=request_id,
        schedule_identity=prior.schedule_identity,
        simulator_configuration_id=requested_configuration,
        initial_state_id=initial_state_id,
        contact_ids=prior.contact_ids,
        path_ids=prior.path_ids,
        regime_paths=prior.regime_paths,
        prior_weights=prior.weights,
        retained_prior_mass=prior.retained_prior_mass,
        group_log_scales=group_log_scales,
        controller_points_m=controller_points_m,
        position_m=position_m,
        velocity_mps=velocity_mps,
        frame_times_s=frame_times_s,
        contact_node_indices=contact_node_indices,
        contact_node_weights=contact_node_weights,
        normal_stiffness_npm=normal_stiffness_npm,
        tangential_stiffness_npm=tangential_stiffness_npm,
        friction_coefficient=friction_coefficient,
    )
    result = provider.replay_scheduled_contacts(request)
    if not isinstance(result, contracts.ScheduledContactReplayResultV1):
        raise ScheduledContactReplayRejectedError(
            "provider returned the wrong scheduled replay result type"
        )
    try:
        validated = contracts.validate_scheduled_contact_replay_result(
            request,
            result,
        )
    except (TypeError, ValueError) as error:
        raise ScheduledContactReplayRejectedError(
            "provider result failed scheduled replay validation"
        ) from error
    if validated.provider_revision != provider_revision:
        raise ScheduledContactReplayRejectedError(
            "provider result revision does not match the runtime provider"
        )

    bank = MultiContactPathBank.from_prior(
        prior,
        validated.positions_m,
        base_variance_m2=validated.conditional_variance_m2,
        replay_result_identity=validated.replay_result_identity,
        frame_times_s=validated.frame_times_s,
    )
    if bank.schedule_identity != prior.schedule_identity:
        raise ScheduledContactReplayRejectedError(
            "constructed bank changed the requested schedule identity"
        )
    return ScheduledContactReplayEvidence(
        bank=bank,
        request_identity=validated.request_identity,
        replay_result_identity=validated.replay_result_identity,
        provider_name=validated.provider_name,
        provider_version=validated.provider_version,
        provider_revision=validated.provider_revision,
        simulator_configuration_id=validated.simulator_configuration_id,
    )


__all__ = [
    "ScheduledContactReplayEvidence",
    "ScheduledContactReplayRejectedError",
    "ScheduledContactReplayUnavailableError",
    "replay_multi_contact_prior",
]
