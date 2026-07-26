"""Action-conditioned temporal discrepancy for counterfactual PhysTwin rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Literal, Mapping

import numpy as np

from causal4d.action_conditioned_discrepancy import (
    ActionConditionedDiscrepancyFeatures,
    ActionConditionedDiscrepancyModel,
    build_action_conditioned_features,
    forecast_action_conditioned_persistence,
)
from causal4d.contracts import (
    CounterfactualQuery,
    FactualIntervention,
    PhysicalPosterior,
    TwinBelief,
)
from causal4d.counterfactual import apply_counterfactual_operator
from causal4d.discrepancy_belief import GraphDiscrepancyBelief
from causal4d.rollout_bank import JointRolloutBank
from causal4d.stable_discrepancy_dynamics import (
    StableDiscrepancyTransitionModel,
    forecast_action_conditioned_dynamics,
)


def _readonly(values: np.ndarray, *, dtype: type | None = float) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _validated_metadata(values: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(values), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must contain finite JSON values") from error


@dataclass(frozen=True)
class ActionConditionedPhysicalPosterior:
    """Physical posterior with component-wise temporal readout uncertainty.

    ``physical`` preserves the existing provenance-complete Causal4D contract.
    The replacement readout mean and variance have shape ``(K, T, N, 3)`` and
    are produced by a separately transported graph-discrepancy belief.
    """

    physical: PhysicalPosterior
    readout_trajectories_m: np.ndarray
    readout_variance_m2: np.ndarray
    discrepancy_coefficient_covariance_m2: np.ndarray
    graph_discrepancy_belief_id: str
    discrepancy_model_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        readout = _readonly(self.readout_trajectories_m)
        variance = _readonly(self.readout_variance_m2)
        covariance = _readonly(self.discrepancy_coefficient_covariance_m2)
        expected = self.physical.state_trajectories_m.shape
        if readout.shape != expected or variance.shape != expected:
            raise ValueError(
                "temporal readout mean and variance must match physical trajectories"
            )
        if covariance.ndim != 5 or covariance.shape[:3] != (
            expected[0],
            expected[1],
            3,
        ):
            raise ValueError(
                "coefficient covariance must have shape (K, T, 3, rank, rank)"
            )
        if covariance.shape[3] != covariance.shape[4]:
            raise ValueError("coefficient covariance matrices must be square")
        if not all(
            np.all(np.isfinite(value)) for value in (readout, variance, covariance)
        ):
            raise ValueError("action-conditioned posterior arrays must be finite")
        if np.any(variance < 0.0):
            raise ValueError("temporal readout variance must be nonnegative")
        if not self.graph_discrepancy_belief_id or not self.discrepancy_model_id:
            raise ValueError("belief and model identifiers must be nonempty")
        object.__setattr__(self, "readout_trajectories_m", readout)
        object.__setattr__(self, "readout_variance_m2", variance)
        object.__setattr__(
            self,
            "discrepancy_coefficient_covariance_m2",
            covariance,
        )
        object.__setattr__(self, "metadata", _validated_metadata(self.metadata))

    @property
    def component_ids(self) -> tuple[str, ...]:
        return self.physical.component_ids

    @property
    def weights(self) -> np.ndarray:
        return self.physical.weights

    @property
    def state_trajectories_m(self) -> np.ndarray:
        return self.physical.state_trajectories_m


def _align_discrepancy_belief(
    source: GraphDiscrepancyBelief,
    physical: PhysicalPosterior,
    twin: TwinBelief,
) -> GraphDiscrepancyBelief:
    if source.component_ids == physical.component_ids:
        return source
    if source.component_ids != twin.particle_ids:
        raise ValueError(
            "graph discrepancy components must match either physical rollouts "
            "or TwinBelief particles"
        )
    indices = physical.twin_particle_indices
    return GraphDiscrepancyBelief(
        basis_sha256=source.basis_sha256,
        component_ids=physical.component_ids,
        coefficient_mean_m=source.coefficient_mean_m[indices],
        coefficient_covariance_m2=source.coefficient_covariance_m2[indices],
        projection_variance_m2=source.projection_variance_m2,
        transition_model_id=source.transition_model_id,
        innovation_model_id=source.innovation_model_id,
        source_physical_posterior_id=source.source_physical_posterior_id,
        metadata={
            **source.metadata,
            "alignment": "expanded_from_twin_particles",
            "source_graph_discrepancy_belief_id": source.artifact_id,
        },
    )


def _component_features(
    physical: PhysicalPosterior,
    query: CounterfactualQuery,
    control_anchor_m: np.ndarray,
    *,
    frame_dt_s: float,
    feature_schema: Literal["magnitude_v1", "signed_v2"],
) -> ActionConditionedDiscrepancyFeatures:
    built = [
        build_action_conditioned_features(
            query.controller_points_m,
            control_anchor_m,
            frame_dt_s=frame_dt_s,
            phi_names=physical.phi_names,
            phi=physical.phi[index],
            kappa_names=physical.kappa_names,
            kappa=physical.kappa_cf[index],
            contact_policy=query.contact_policy,
            feature_schema=feature_schema,
        )
        for index in range(len(physical.weights))
    ]
    names = built[0].names
    if any(value.names != names for value in built[1:]):
        raise RuntimeError("component feature schemas differ")
    return ActionConditionedDiscrepancyFeatures(
        names=names,
        values=np.stack([value.values for value in built], axis=0),
        component_ids=physical.component_ids,
        step_duration_s=frame_dt_s,
        schema_id=feature_schema,
    )


def apply_action_conditioned_counterfactual_operator(
    bank: JointRolloutBank,
    manifest: Mapping[str, Any],
    twin: TwinBelief,
    factual: FactualIntervention,
    query: CounterfactualQuery,
    graph_discrepancy_belief: GraphDiscrepancyBelief,
    discrepancy_model: ActionConditionedDiscrepancyModel,
    graph_basis: np.ndarray,
    control_anchor_m: np.ndarray,
    *,
    frame_dt_s: float,
    feature_schema: Literal["magnitude_v1", "signed_v2"] = "magnitude_v1",
    transition_model: StableDiscrepancyTransitionModel | None = None,
) -> ActionConditionedPhysicalPosterior:
    """Apply ``do(u_cf)`` with temporal action-conditioned readout uncertainty.

    The ordinary physical operator remains authoritative for state trajectories,
    intervention transport, contact handling, weights, and provenance. The
    extension replaces only discrepancy-aware readout moments. Supplying a stable
    transition model activates signed, physical-time discrepancy-mean transport;
    omitting it preserves graph persistence exactly.
    """

    if not np.isfinite(frame_dt_s) or frame_dt_s <= 0.0:
        raise ValueError("frame_dt_s must be finite and positive")
    basis = np.asarray(graph_basis, dtype=float)
    if basis.ndim != 2 or basis.shape[0] != bank.node_count:
        raise ValueError("graph_basis must have shape (bank.node_count, rank)")
    anchor = np.asarray(control_anchor_m, dtype=float)
    if anchor.shape != query.controller_points_m.shape[1:]:
        raise ValueError("control_anchor_m must match one controller frame")
    if not np.all(np.isfinite(anchor)):
        raise ValueError("control_anchor_m must be finite")

    physical = apply_counterfactual_operator(bank, manifest, twin, factual, query)
    aligned_belief = _align_discrepancy_belief(
        graph_discrepancy_belief,
        physical,
        twin,
    )
    features = _component_features(
        physical,
        query,
        anchor,
        frame_dt_s=frame_dt_s,
        feature_schema=feature_schema,
    )
    if transition_model is None:
        forecast = forecast_action_conditioned_persistence(
            aligned_belief,
            discrepancy_model,
            features,
            basis,
        )
        mean_transition = "graph_persistence"
    else:
        forecast = forecast_action_conditioned_dynamics(
            aligned_belief,
            discrepancy_model,
            transition_model,
            features,
            basis,
        )
        mean_transition = transition_model.model_id
    if forecast.readout_mean_m.shape != physical.state_trajectories_m.shape:
        raise ValueError(
            "counterfactual rollout must contain the endpoint plus query horizon"
        )
    readout = physical.state_trajectories_m.astype(float) + forecast.readout_mean_m
    variance = forecast.readout_variance_m2 + float(bank.variance_floor_m2)
    return ActionConditionedPhysicalPosterior(
        physical=physical,
        readout_trajectories_m=readout,
        readout_variance_m2=variance,
        discrepancy_coefficient_covariance_m2=(
            forecast.coefficient_covariance_m2
        ),
        graph_discrepancy_belief_id=graph_discrepancy_belief.artifact_id,
        discrepancy_model_id=forecast.model_id,
        metadata={
            "operator": "abduction-action-prediction",
            "discrepancy_mean_transition": mean_transition,
            "discrepancy_covariance_transition": discrepancy_model.model_id,
            "feature_schema": features.schema_id,
            "frame_dt_s": float(frame_dt_s),
            "mean_time_parameterization": (
                "persistence"
                if transition_model is None
                else transition_model.time_parameterization
            ),
            "covariance_time_parameterization": (
                discrepancy_model.time_parameterization
            ),
            "base_physical_posterior_id": physical.artifact_id,
            "aligned_graph_discrepancy_belief_id": aligned_belief.artifact_id,
            "temporal_readout_uncertainty": True,
            "future_observations_read": 0,
            "variance_floor_m2": float(bank.variance_floor_m2),
        },
    )
