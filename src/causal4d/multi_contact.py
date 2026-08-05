"""Factorized dynamic contact-path inference for multiple contact channels.

The single-contact dynamic model represents one regime path per rollout. This
module composes independently parameterized contact channels into a
deterministic top-k joint path support, then performs prefix-only Bayesian
reweighting over trajectories simulated continuously for those schedules.
"""

from __future__ import annotations

from causal4d._multi_contact_common import MULTI_CONTACT_SCHEDULE_SCHEMA_VERSION
from causal4d._multi_contact_inference import (
    MULTI_CONTACT_ROLLOUT_SCHEMA_VERSION,
    MultiContactInferencePolicy,
    MultiContactInferenceRejectedError,
    MultiContactPathBank,
    MultiContactPosterior,
    infer_multi_contact_posterior,
    multi_contact_conditioned_variance,
)
from causal4d._multi_contact_prior import (
    MultiContactEnumerationConfig,
    MultiContactPathPrior,
    enumerate_multi_contact_paths,
)
from causal4d.scheduled_contact_replay import (
    ScheduledContactReplayEvidence,
    ScheduledContactReplayRejectedError,
    ScheduledContactReplayUnavailableError,
    replay_multi_contact_prior,
)


__all__ = [
    "MULTI_CONTACT_ROLLOUT_SCHEMA_VERSION",
    "MULTI_CONTACT_SCHEDULE_SCHEMA_VERSION",
    "MultiContactEnumerationConfig",
    "MultiContactInferencePolicy",
    "MultiContactInferenceRejectedError",
    "MultiContactPathBank",
    "MultiContactPathPrior",
    "MultiContactPosterior",
    "ScheduledContactReplayEvidence",
    "ScheduledContactReplayRejectedError",
    "ScheduledContactReplayUnavailableError",
    "enumerate_multi_contact_paths",
    "infer_multi_contact_posterior",
    "multi_contact_conditioned_variance",
    "replay_multi_contact_prior",
]
