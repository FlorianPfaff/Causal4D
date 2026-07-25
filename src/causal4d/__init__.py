"""Controlled counterfactual benchmarks for intervention-ready world models."""

from causal4d.action_conditioned_discrepancy import (
    ActionConditionedDiscrepancyFeatures,
    ActionConditionedDiscrepancyForecast,
    ActionConditionedDiscrepancyModel,
    build_action_conditioned_features,
    forecast_action_conditioned_persistence,
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
from causal4d.discrepancy_belief import (
    GraphDiscrepancyBelief,
    load_graph_discrepancy_belief,
    write_graph_discrepancy_belief,
)
from causal4d.evaluation import run_counterfactual_benchmark
from causal4d.graph_mode_abduction import (
    GraphModeAbductionConfig,
    abduct_factual_intervention_graph_mode,
    graph_mode_joint_weights,
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
    "ActionConditionedDiscrepancyFeatures",
    "ActionConditionedDiscrepancyForecast",
    "ActionConditionedDiscrepancyModel",
    "CounterfactualBenchmarkConfig",
    "CounterfactualQuery",
    "FactualIntervention",
    "GraphDiscrepancyBelief",
    "GraphModeAbductionConfig",
    "LatentContactConfig",
    "JointRolloutBank",
    "PhysicalPosterior",
    "SEMANTIC_TIMING_SCHEMA_VERSION",
    "SEMANTIC_TIMING_SCOPE",
    "SemanticFreshnessDecision",
    "SemanticFreshnessLimits",
    "SparseTrajectoryEvidence",
    "TaskPosterior",
    "SemanticTimingMetadata",
    "TwinBelief",
    "abduct_factual_intervention_graph_mode",
    "apply_counterfactual_operator",
    "apply_semantic_freshness_gate",
    "build_action_conditioned_features",
    "build_protocol",
    "forecast_action_conditioned_persistence",
    "graph_mode_joint_weights",
    "load_graph_discrepancy_belief",
    "run_counterfactual_benchmark",
    "run_latent_contact_benchmark",
    "write_graph_discrepancy_belief",
]

__version__ = "0.3.0"
