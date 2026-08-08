from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from causal4d.functional_support_v1 import (
    FunctionalSupportActionV1,
    FunctionalSupportPolicyV1,
    certify_functional_support_v1,
)
from causal4d.projected_functional_support_v1 import (
    FunctionalSupportProjectionV1,
    ProjectedFunctionalSupportActionV1,
    ProjectedFunctionalSupportPolicyV1,
    certify_projected_functional_support_v1,
)


def _base_policy(**overrides: float | int) -> FunctionalSupportPolicyV1:
    values: dict[str, float | int] = {
        "maximum_normalized_mean_error": 10.0,
        "maximum_variance_trace_relative_error": 10.0,
        "maximum_interval_endpoint_error_m": 10.0,
        "maximum_energy_distance_m": 10.0,
        "variance_floor_m2": 1e-6,
    }
    values.update(overrides)
    return FunctionalSupportPolicyV1(**values)


def _projected_policy(
    **overrides: float | int,
) -> ProjectedFunctionalSupportPolicyV1:
    values: dict[str, float | int] = {
        "maximum_projected_variance_relative_error": 0.05,
        "maximum_projected_interval_endpoint_error_m": 0.05,
        "variance_floor_m2": 1e-6,
    }
    values.update(overrides)
    return ProjectedFunctionalSupportPolicyV1(**values)


def _base_certificate(
    action: FunctionalSupportActionV1,
    *,
    policy: FunctionalSupportPolicyV1 | None = None,
):
    return certify_functional_support_v1(
        (action,),
        policy=_base_policy() if policy is None else policy,
        source_artifact_ids=("source-freeze",),
    )


def test_identical_support_passes_fixed_task_projections() -> None:
    trajectories = np.asarray(
        (
            (((-1.0, -0.5),),),
            (((1.0, 0.5),),),
        )
    )
    action = FunctionalSupportActionV1(
        action_id="source-action",
        full_trajectories_m=trajectories,
        full_weights=np.asarray((0.5, 0.5)),
        reduced_trajectories_m=trajectories.copy(),
        reduced_weights=np.asarray((0.5, 0.5)),
    )
    projection = FunctionalSupportProjectionV1(
        projection_id="endpoint-sum",
        coefficients=np.asarray((((1.0, 1.0),),)),
    )
    certificate = certify_projected_functional_support_v1(
        (action,),
        (projection,),
        policy=_projected_policy(),
        base_certificate=_base_certificate(action),
        source_artifact_ids=("projection-freeze",),
    )

    assert certificate.accepted
    assert certificate.reasons == ()
    assert certificate.metrics[0].projected_variance_relative_error == pytest.approx(
        0.0
    )
    assert certificate.metrics[0].maximum_projected_interval_endpoint_error_m == (
        pytest.approx(0.0)
    )


def test_projection_detects_dependence_change_hidden_by_marginal_metrics() -> None:
    full = np.asarray(
        (
            (((-1.0, -1.0),),),
            (((1.0, 1.0),),),
        )
    )
    reduced = np.asarray(
        (
            (((-1.0, 1.0),),),
            (((1.0, -1.0),),),
        )
    )
    action = FunctionalSupportActionV1(
        action_id="rotated-dependence",
        full_trajectories_m=full,
        full_weights=np.asarray((0.5, 0.5)),
        reduced_trajectories_m=reduced,
        reduced_weights=np.asarray((0.5, 0.5)),
    )
    base = _base_certificate(action)
    assert base.accepted
    projection = FunctionalSupportProjectionV1(
        projection_id="sum-mode",
        coefficients=np.asarray((((2**-0.5, 2**-0.5),),)),
    )

    certificate = certify_projected_functional_support_v1(
        (action,),
        (projection,),
        policy=_projected_policy(),
        base_certificate=base,
        source_artifact_ids=("projection-freeze",),
    )

    assert not certificate.accepted
    metric = certificate.metrics[0]
    assert metric.full_mean_m == pytest.approx(metric.reduced_mean_m)
    assert metric.projected_variance_relative_error > 0.9
    assert (
        "rotated-dependence:sum-mode:"
        "projected_variance_relative_error_exceeds_limit"
    ) in certificate.reasons


def test_low_rank_conditional_modes_are_projected_exactly() -> None:
    trajectories = np.asarray(((((0.0, 0.0),),),))
    action = FunctionalSupportActionV1(
        "structured-uncertainty",
        trajectories,
        np.asarray((1.0,)),
        trajectories.copy(),
        np.asarray((1.0,)),
    )
    base = _base_certificate(action)
    assert base.accepted
    projected_action = ProjectedFunctionalSupportActionV1(
        action=action,
        full_component_low_rank_factors_m=np.asarray(((((1.0, 1.0),),),)),
        reduced_component_low_rank_factors_m=np.asarray(((((1.0, -1.0),),),)),
    )
    projection = FunctionalSupportProjectionV1(
        "sum-mode",
        np.asarray((((1.0, 1.0),),)),
    )

    certificate = certify_projected_functional_support_v1(
        (projected_action,),
        (projection,),
        policy=_projected_policy(),
        base_certificate=base,
        source_artifact_ids=("projection-freeze",),
    )

    assert not certificate.accepted
    metric = certificate.metrics[0]
    assert metric.full_variance_m2 > 3.9
    assert metric.reduced_variance_m2 < 1e-4
    assert metric.projected_variance_relative_error > 0.99


def test_base_rejection_is_preserved_by_projected_certificate() -> None:
    full = np.asarray(((((0.0, 0.0),),),))
    reduced = np.asarray(((((1.0, 0.0),),),))
    action = FunctionalSupportActionV1(
        "shifted",
        full,
        np.asarray((1.0,)),
        reduced,
        np.asarray((1.0,)),
    )
    base = _base_certificate(
        action,
        policy=_base_policy(maximum_normalized_mean_error=0.0),
    )
    assert not base.accepted
    projection = FunctionalSupportProjectionV1(
        "x-readout",
        np.asarray((((1.0, 0.0),),)),
    )

    certificate = certify_projected_functional_support_v1(
        (action,),
        (projection,),
        policy=_projected_policy(
            maximum_projected_variance_relative_error=10.0,
            maximum_projected_interval_endpoint_error_m=10.0,
        ),
        base_certificate=base,
        source_artifact_ids=("projection-freeze",),
    )

    assert not certificate.accepted
    assert any(reason.startswith("base:") for reason in certificate.reasons)


def test_projection_shape_and_base_provenance_fail_closed() -> None:
    trajectories = np.asarray(((((0.0, 0.0),),),))
    action = FunctionalSupportActionV1(
        "source-action",
        trajectories,
        np.asarray((1.0,)),
        trajectories,
        np.asarray((1.0,)),
    )
    base = _base_certificate(action)
    wrong_shape = FunctionalSupportProjectionV1(
        "wrong-shape",
        np.ones((2, 1, 2)),
    )
    with pytest.raises(ValueError, match="expected"):
        certify_projected_functional_support_v1(
            (action,),
            (wrong_shape,),
            policy=_projected_policy(),
            base_certificate=base,
            source_artifact_ids=("projection-freeze",),
        )

    with pytest.raises(ValueError, match="does not bind"):
        certify_projected_functional_support_v1(
            (action,),
            (
                FunctionalSupportProjectionV1(
                    "valid",
                    np.asarray((((1.0, 0.0),),)),
                ),
            ),
            policy=_projected_policy(),
            base_certificate=replace(
                base,
                source_artifact_ids=("different-source",),
            ),
            source_artifact_ids=("projection-freeze",),
        )


def test_projection_identity_binds_coefficients() -> None:
    first = FunctionalSupportProjectionV1(
        "readout",
        np.asarray((((1.0, 0.0),),)),
    )
    second = FunctionalSupportProjectionV1(
        "readout",
        np.asarray((((0.0, 1.0),),)),
    )
    assert first.projection_artifact_id != second.projection_artifact_id
    with pytest.raises(ValueError, match="nonzero"):
        FunctionalSupportProjectionV1(
            "zero",
            np.zeros((1, 1, 2)),
        )
