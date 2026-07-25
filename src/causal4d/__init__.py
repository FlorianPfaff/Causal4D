"""Controlled counterfactual benchmarks for intervention-ready world models."""

from causal4d.action_conditioned_discrepancy import (
    ActionConditionedGraphDiscrepancyModel,
    fit_action_conditioned_graph_discrepancy,
    forecast_action_conditioned_graph_discrepancy,
)
from causal4d.benchmark import CounterfactualBenchmarkConfig, build_protocol
from causal4d.contact_evaluation import run_latent_contact_benchmark
from causal4d.contact_inference import LatentContactConfig
from causal4d.contracts import (
    CounterfactualQuery,
    FactualIntervention,
    PhysicalPosterior,
    TaskPosterior,
    TwinBelief,
)
from causal4d.counterfactual import apply_counterfactual_operator
from causal4d.evaluation import run_counterfactual_benchmark
from causal4d.grouped_observations import (
    ObservationGroup,
    dense_prefix_observation_groups,
    update_from_grouped_observations,
)
from causal4d.rollout_bank import JointRolloutBank, SparseTrajectoryEvidence
from causal4d.semantic_freshness import (
    SEMANTIC_TIMING_SCHEMA_VERSION,
    SEMANTIC_TIMING_SCOPE,
    SemanticFreshnessDecision,
    SemanticFreshnessLimits,
    SemanticTimingMetadata,
    apply_semantic_freshness_gate,
)

__all__ = [
    "ActionConditionedGraphDiscrepancyModel",
    "CounterfactualBenchmarkConfig",
    "CounterfactualQuery",
    "FactualIntervention",
    "LatentContactConfig",
    "JointRolloutBank",
    "ObservationGroup",
    "PhysicalPosterior",
    "SEMANTIC_TIMING_SCHEMA_VERSION",
    "SEMANTIC_TIMING_SCOPE",
    "SemanticFreshnessDecision",
    "SemanticFreshnessLimits",
    "SparseTrajectoryEvidence",
    "TaskPosterior",
    "SemanticTimingMetadata",
    "TwinBelief",
    "apply_counterfactual_operator",
    "apply_semantic_freshness_gate",
    "build_protocol",
    "dense_prefix_observation_groups",
    "fit_action_conditioned_graph_discrepancy",
    "forecast_action_conditioned_graph_discrepancy",
    "run_counterfactual_benchmark",
    "run_latent_contact_benchmark",
    "update_from_grouped_observations",
]

__version__ = "0.3.0"
