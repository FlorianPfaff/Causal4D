"""Guard action-conditioned counterfactual readouts with source-only support."""

from __future__ import annotations

from typing import Any, Literal, Mapping

import numpy as np

from causal4d.action_conditioned_counterfactual import (
    ActionConditionedPhysicalPosterior,
    _component_features,
    apply_action_conditioned_counterfactual_operator,
)
from causal4d.action_conditioned_discrepancy import (
    ActionConditionedDiscrepancyModel,
)
from causal4d.action_support import (
    ActionSupportCalibration,
    ActionSupportSelection,
    select_action_supported_candidate,
)
from causal4d.contracts import (
    CounterfactualQuery,
    FactualIntervention,
    PhysicalPosterior,
    TwinBelief,
)
from causal4d.discrepancy_belief import GraphDiscrepancyBelief
from causal4d.rollout_bank import JointRolloutBank
from causal4d.stable_discrepancy_dynamics import (
    StableDiscrepancyTransitionModel,
)


def apply_guarded_action_conditioned_counterfactual_operator(
    bank: JointRolloutBank,
    manifest: Mapping[str, Any],
    twin: TwinBelief,
    factual: FactualIntervention,
    query: CounterfactualQuery,
    graph_discrepancy_belief: GraphDiscrepancyBelief,
    discrepancy_model: ActionConditionedDiscrepancyModel,
    graph_basis: np.ndarray,
    control_anchor_m: np.ndarray,
    action_support_calibration: ActionSupportCalibration,
    *,
    frame_dt_s: float,
    feature_schema: Literal["magnitude_v1", "signed_v2"] = "magnitude_v1",
    transition_model: StableDiscrepancyTransitionModel | None = None,
) -> ActionSupportSelection[
    PhysicalPosterior,
    ActionConditionedPhysicalPosterior,
]:
    """Apply the optional readout extension with exact physical fallback.

    The ordinary physical posterior is always constructed through the existing
    counterfactual operator. The action-conditioned candidate is deployed only
    when the source-frozen feature envelope covers enough posterior mass.
    Rejection preserves ``candidate.physical`` by object identity.
    """

    candidate = apply_action_conditioned_counterfactual_operator(
        bank,
        manifest,
        twin,
        factual,
        query,
        graph_discrepancy_belief,
        discrepancy_model,
        graph_basis,
        control_anchor_m,
        frame_dt_s=frame_dt_s,
        feature_schema=feature_schema,
        transition_model=transition_model,
    )
    features = _component_features(
        candidate.physical,
        query,
        control_anchor_m,
        frame_dt_s=frame_dt_s,
        feature_schema=feature_schema,
    )
    return select_action_supported_candidate(
        action_support_calibration,
        features,
        baseline=candidate.physical,
        candidate=candidate,
        candidate_model_id=candidate.discrepancy_model_id,
        component_weights=candidate.weights,
        component_ids=candidate.component_ids,
    )


__all__ = ["apply_guarded_action_conditioned_counterfactual_operator"]
