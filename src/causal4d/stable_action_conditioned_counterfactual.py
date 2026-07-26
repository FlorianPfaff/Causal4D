"""Stable action-conditioned discrepancy dynamics for counterfactual readouts."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from causal4d.action_conditioned_counterfactual import (
    ActionConditionedPhysicalPosterior,
    _align_discrepancy_belief,
    _component_features,
)
from causal4d.action_conditioned_discrepancy import (
    ActionConditionedDiscrepancyModel,
)
from causal4d.contracts import (
    CounterfactualQuery,
    FactualIntervention,
    TwinBelief,
)
from causal4d.counterfactual import apply_counterfactual_operator
from causal4d.discrepancy_belief import GraphDiscrepancyBelief
from causal4d.rollout_bank import JointRolloutBank
from causal4d.stable_discrepancy_dynamics import (
    StableDiscrepancyTransitionModel,
    forecast_action_conditioned_dynamics,
)


def apply_stable_action_conditioned_counterfactual_operator(
    bank: JointRolloutBank,
    manifest: Mapping[str, Any],
    twin: TwinBelief,
    factual: FactualIntervention,
    query: CounterfactualQuery,
    graph_discrepancy_belief: GraphDiscrepancyBelief,
    innovation_model: ActionConditionedDiscrepancyModel,
    transition_model: StableDiscrepancyTransitionModel,
    graph_basis: np.ndarray,
    control_anchor_m: np.ndarray,
    *,
    frame_dt_s: float,
) -> ActionConditionedPhysicalPosterior:
    """Apply ``do(u_cf)`` with stable graph-discrepancy mean dynamics.

    The ordinary counterfactual operator remains authoritative for physical
    state, intervention transport, contact semantics, component weights, and
    provenance. This opt-in extension changes only the discrepancy-aware readout
    moments. The transition is non-expansive by construction and the identity
    model gives exact graph persistence.
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
    )
    forecast = forecast_action_conditioned_dynamics(
        aligned_belief,
        innovation_model,
        transition_model,
        features,
        basis,
    )
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
            "discrepancy_mean_transition": transition_model.model_id,
            "discrepancy_covariance_transition": innovation_model.model_id,
            "exact_persistence_fallback": (
                transition_model.model_id == "exact-graph-persistence"
            ),
            "base_physical_posterior_id": physical.artifact_id,
            "aligned_graph_discrepancy_belief_id": aligned_belief.artifact_id,
            "temporal_readout_uncertainty": True,
            "future_observations_read": 0,
            "variance_floor_m2": float(bank.variance_floor_m2),
        },
    )
