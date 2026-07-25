"""Controlled counterfactual benchmarks for intervention-ready world models."""

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
from causal4d.grouped_likelihood import (
    GroupLikelihoodDiagnostics,
    grouped_component_log_likelihoods,
    posterior_weights_from_grouped_evidence,
)
from causal4d.identifiability import (
    IdentifiabilityConfig,
    InterventionIdentifiabilityResult,
    assess_intervention_identifiability,
    finite_response_sensitivity,
)
from causal4d.observation_evidence import (
    GroupedObservationEvidence,
    ObservationGroup,
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
    "CounterfactualBenchmarkConfig",
    "CounterfactualQuery",
    "FactualIntervention",
    "GroupLikelihoodDiagnostics",
    "GroupedObservationEvidence",
    "IdentifiabilityConfig",
    "InterventionIdentifiabilityResult",
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
    "assess_intervention_identifiability",
    "build_protocol",
    "finite_response_sensitivity",
    "grouped_component_log_likelihoods",
    "posterior_weights_from_grouped_evidence",
    "run_counterfactual_benchmark",
    "run_latent_contact_benchmark",
]

__version__ = "0.3.0"
