from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from causal4d.functional_support_v1 import (
    FunctionalSupportActionV1,
    FunctionalSupportCertificateV1,
    FunctionalSupportPolicyV1,
    certify_functional_support_v1,
)


def _policy(**overrides: float | int) -> FunctionalSupportPolicyV1:
    values: dict[str, float | int] = {
        "maximum_normalized_mean_error": 0.05,
        "maximum_variance_trace_relative_error": 0.05,
        "maximum_interval_endpoint_error_m": 0.05,
        "maximum_energy_distance_m": 0.05,
        "variance_floor_m2": 1e-4,
    }
    values.update(overrides)
    return FunctionalSupportPolicyV1(**values)


def test_identical_predictive_support_is_certified() -> None:
    trajectories = np.asarray(
        (
            (((-1.0, 0.0),),),
            (((1.0, 0.0),),),
        )
    )
    action = FunctionalSupportActionV1(
        action_id="source-action",
        full_trajectories_m=trajectories,
        full_weights=np.asarray((0.5, 0.5)),
        reduced_trajectories_m=trajectories.copy(),
        reduced_weights=np.asarray((0.5, 0.5)),
    )
    certificate = certify_functional_support_v1(
        (action,),
        policy=_policy(),
        source_artifact_ids=("source-freeze",),
    )
    assert certificate.accepted
    assert certificate.reasons == ()
    metrics = certificate.action_metrics[0]
    assert metrics.normalized_mean_error == pytest.approx(0.0)
    assert metrics.energy_distance_m == pytest.approx(0.0)


def test_mean_preserving_support_collapse_is_rejected() -> None:
    full = np.asarray(
        (
            (((-1.0, 0.0),),),
            (((1.0, 0.0),),),
        )
    )
    reduced = np.asarray(((((0.0, 0.0),),),))
    action = FunctionalSupportActionV1(
        action_id="source-action",
        full_trajectories_m=full,
        full_weights=np.asarray((0.5, 0.5)),
        reduced_trajectories_m=reduced,
        reduced_weights=np.asarray((1.0,)),
    )
    certificate = certify_functional_support_v1(
        (action,),
        policy=_policy(),
        source_artifact_ids=("source-freeze",),
    )
    assert not certificate.accepted
    metrics = certificate.action_metrics[0]
    assert metrics.mean_rmse_m == pytest.approx(0.0)
    assert "source-action:variance_trace_relative_error_exceeds_limit" in (
        certificate.reasons
    )
    assert "source-action:interval_endpoint_error_exceeds_limit" in (
        certificate.reasons
    )
    assert "source-action:energy_distance_exceeds_limit" in certificate.reasons


def test_certificate_fails_on_the_worst_source_action() -> None:
    exact = np.asarray(((((0.0, 0.0),),),))
    shifted = np.asarray(((((0.2, 0.0),),),))
    good = FunctionalSupportActionV1(
        "good",
        exact,
        np.asarray((1.0,)),
        exact,
        np.asarray((1.0,)),
    )
    bad = FunctionalSupportActionV1(
        "bad",
        exact,
        np.asarray((1.0,)),
        shifted,
        np.asarray((1.0,)),
    )
    certificate = certify_functional_support_v1(
        (good, bad),
        policy=_policy(minimum_action_count=2),
        source_artifact_ids=("source-library",),
    )
    assert not certificate.accepted
    assert certificate.action_metrics[0].accepted
    assert not certificate.action_metrics[1].accepted
    assert all(reason.startswith("bad:") for reason in certificate.reasons)


def test_certificate_identity_binds_source_provenance() -> None:
    trajectories = np.asarray(((((0.0, 0.0),),),))
    action = FunctionalSupportActionV1(
        "source-action",
        trajectories,
        np.asarray((1.0,)),
        trajectories,
        np.asarray((1.0,)),
    )
    first = certify_functional_support_v1(
        (action,),
        policy=_policy(),
        source_artifact_ids=("freeze-a",),
    )
    second = certify_functional_support_v1(
        (action,),
        policy=_policy(),
        source_artifact_ids=("freeze-b",),
    )
    assert first.certificate_id != second.certificate_id
    with pytest.raises(ValueError, match="sequence of strings"):
        certify_functional_support_v1(
            (action,),
            policy=_policy(),
            source_artifact_ids="not-a-sequence",
        )


def test_certificate_decision_must_match_action_metrics() -> None:
    trajectories = np.asarray(((((0.0, 0.0),),),))
    action = FunctionalSupportActionV1(
        "source-action",
        trajectories,
        np.asarray((1.0,)),
        trajectories,
        np.asarray((1.0,)),
    )
    certificate = certify_functional_support_v1(
        (action,),
        policy=_policy(),
        source_artifact_ids=("freeze",),
    )
    with pytest.raises(ValueError, match="exactly match"):
        replace(
            certificate,
            accepted=False,
            reasons=("invented",),
        )
    assert isinstance(certificate, FunctionalSupportCertificateV1)
    with pytest.raises(ValueError):
        action.full_trajectories_m.setflags(write=True)
