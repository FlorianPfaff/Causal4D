from __future__ import annotations

import numpy as np
import pytest

from causal4d.functional_support_v1 import (
    FunctionalSupportActionV1,
    FunctionalSupportCertificateV1,
    FunctionalSupportPolicyV1,
    certify_functional_support_v1,
)


def _action(**overrides) -> FunctionalSupportActionV1:
    values = {
        "action_id": "source-session-1/action-1",
        "full_trajectories_m": np.zeros((1, 2, 1, 1)),
        "full_weights": np.ones(1),
        "reduced_trajectories_m": np.zeros((1, 2, 1, 1)),
        "reduced_weights": np.ones(1),
    }
    values.update(overrides)
    return FunctionalSupportActionV1(**values)


def _policy() -> FunctionalSupportPolicyV1:
    return FunctionalSupportPolicyV1(
        maximum_normalized_mean_error=0.0,
        maximum_variance_trace_relative_error=0.0,
        maximum_interval_endpoint_error_m=0.0,
        maximum_energy_distance_m=0.0,
    )


def test_action_rejects_target_outcome_access() -> None:
    with pytest.raises(ValueError, match="must be false"):
        _action(target_outcomes_used=True)
    with pytest.raises(ValueError, match="must be a boolean"):
        _action(target_outcomes_used=0)


def test_certificate_binds_no_target_access_declaration() -> None:
    certificate = certify_functional_support_v1(
        (_action(),),
        policy=_policy(),
        source_artifact_ids=("source-freeze",),
    )

    assert certificate.accepted is True
    assert certificate.target_outcomes_used is False
    assert certificate.as_dict()["target_outcomes_used"] is False
    assert certificate.source_artifact_ids[0] == "source-freeze"


def test_direct_certificate_construction_rejects_target_access() -> None:
    certificate = certify_functional_support_v1(
        (_action(),),
        policy=_policy(),
        source_artifact_ids=("source-freeze",),
    )

    with pytest.raises(ValueError, match="must be false"):
        FunctionalSupportCertificateV1(
            accepted=certificate.accepted,
            reasons=certificate.reasons,
            action_metrics=certificate.action_metrics,
            policy=certificate.policy,
            source_artifact_ids=certificate.source_artifact_ids,
            target_outcomes_used=True,
        )
