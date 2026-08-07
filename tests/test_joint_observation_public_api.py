from __future__ import annotations

import causal4d


def test_full_joint_observation_api_is_exported() -> None:
    expected = {
        "JOINT_OBSERVATION_SCHEMA_VERSION",
        "CovarianceRepresentation",
        "JointGaussianLikelihoodDiagnostics",
        "LinearJointObservationEvidence",
        "block_diagonalize_covariance",
        "joint_component_log_likelihoods",
        "posterior_weights_from_joint_observation",
        "PROB4D_JOINT_ADAPTER_SCHEMA_VERSION",
        "Prob4DJointObservationDiagnostics",
        "Prob4DReliabilityPolicy",
        "joint_observation_from_prob4d",
    }

    assert expected <= set(causal4d.__all__)
    for name in expected:
        assert getattr(causal4d, name) is not None
