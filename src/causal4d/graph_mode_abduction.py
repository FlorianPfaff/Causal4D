"""Correlation-aware graph-mode likelihood for factual intervention abduction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from causal4d.contracts import FactualIntervention, TwinBelief, array_sha256
from causal4d.intervention_abduction import physical_readout_components
from causal4d.rollout_bank import JointRolloutBank


def _readonly(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).copy()
    array.setflags(write=False)
    return array


def _coordinate_mask(observations: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    valid = np.isfinite(observations)
    if mask is None:
        return valid
    supplied = np.asarray(mask, dtype=bool)
    if supplied.shape == observations.shape[:2]:
        supplied = np.repeat(supplied[:, :, None], observations.shape[2], axis=2)
    if supplied.shape != observations.shape:
        raise ValueError("observation mask must have shape (T, N) or (T, N, 3)")
    return valid & supplied


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=float)
    if values.ndim != 2 or not np.any(np.isfinite(values)):
        raise ValueError("joint log weights must contain finite support")
    maximum = float(np.max(values[np.isfinite(values)]))
    weights = np.exp(np.where(np.isfinite(values), values - maximum, -np.inf))
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("graph-mode posterior normalization failed")
    return weights / total


def _inverse_square_root(covariance: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if float(np.min(eigenvalues)) <= 0.0:
        raise ValueError("graph-mode covariance must be positive definite")
    return (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T


@dataclass(frozen=True)
class GraphModeAbductionConfig:
    """Settings for a low-dimensional correlation-aware robust likelihood."""

    position_scale_m: float = 0.01
    dynamic_scale_m: float = 0.01
    dynamic_likelihood_weight: float = 0.25
    likelihood_temperature: float = 1.0
    degrees_of_freedom: float = 4.0
    projection_ridge: float = 1e-5
    mode_covariance_m2: np.ndarray | None = None

    def __post_init__(self) -> None:
        scalars = (
            self.position_scale_m,
            self.dynamic_scale_m,
            self.likelihood_temperature,
            self.degrees_of_freedom,
            self.projection_ridge,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in scalars):
            raise ValueError(
                "graph-mode scales, temperature, dof, and ridge must be positive"
            )
        if self.dynamic_likelihood_weight < 0.0:
            raise ValueError("dynamic_likelihood_weight must be nonnegative")
        if self.mode_covariance_m2 is not None:
            covariance = _readonly(self.mode_covariance_m2)
            if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
                raise ValueError("mode_covariance_m2 must have shape (rank, rank)")
            if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-10):
                raise ValueError("mode covariance must be symmetric")
            if float(np.min(np.linalg.eigvalsh(covariance), initial=0.0)) < -1e-10:
                raise ValueError("mode covariance must be positive semidefinite")
            object.__setattr__(self, "mode_covariance_m2", covariance)

    def metadata(self, rank: int) -> dict[str, Any]:
        covariance = self.mode_covariance_m2
        if covariance is not None and covariance.shape != (rank, rank):
            raise ValueError("mode covariance rank differs from graph basis")
        return {
            "position_scale_m": self.position_scale_m,
            "dynamic_scale_m": self.dynamic_scale_m,
            "dynamic_likelihood_weight": self.dynamic_likelihood_weight,
            "likelihood_temperature": self.likelihood_temperature,
            "degrees_of_freedom": self.degrees_of_freedom,
            "projection_ridge": self.projection_ridge,
            "mode_covariance_m2_sha256": (
                array_sha256(covariance) if covariance is not None else None
            ),
        }


def _project_component_residuals(
    residual: np.ndarray,
    valid: np.ndarray,
    basis: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    """Project (H, P, T, N, 3) residuals onto graph modes."""

    values = np.asarray(residual, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    modes = np.asarray(basis, dtype=float)
    if values.ndim != 5 or values.shape[-1] != 3:
        raise ValueError("component residuals must have shape (H, P, T, N, 3)")
    if mask.shape != values.shape[2:]:
        raise ValueError("validity mask must match residual frames, nodes, coordinates")
    if modes.ndim != 2 or modes.shape[0] != values.shape[3]:
        raise ValueError("graph basis must cover every rollout node")
    rank = modes.shape[1]
    identity = np.eye(rank)
    coefficients = np.zeros((*values.shape[:3], rank, 3), dtype=float)
    for frame in range(values.shape[2]):
        for coordinate in range(3):
            selected = mask[frame, :, coordinate]
            if not np.any(selected):
                raise ValueError("graph-mode likelihood has an empty frame/coordinate")
            design = modes[selected]
            precision = design.T @ design + ridge * identity
            right = np.einsum(
                "nr,hpn->hpr",
                design,
                values[:, :, frame, selected, coordinate],
            )
            solved = np.linalg.solve(
                precision,
                right.reshape(-1, rank).T,
            ).T.reshape(*right.shape)
            coefficients[:, :, frame, :, coordinate] = solved
    return coefficients


def _multivariate_student_t_score(
    coefficients: np.ndarray,
    covariance_m2: np.ndarray,
    *,
    degrees_of_freedom: float,
) -> np.ndarray:
    """Return mean robust score over frames and coordinates for every H/P pair."""

    inverse_root = _inverse_square_root(covariance_m2)
    whitened = np.einsum("ij,hptjc->hptic", inverse_root, coefficients)
    squared = np.sum(np.square(whitened), axis=3)
    dimension = covariance_m2.shape[0]
    terms = -0.5 * (degrees_of_freedom + dimension) * np.log1p(
        squared / degrees_of_freedom
    )
    return np.mean(terms, axis=(2, 3))


def graph_mode_joint_weights(
    bank: JointRolloutBank,
    belief: TwinBelief,
    observations_from_endpoint_m: np.ndarray,
    graph_basis: np.ndarray,
    *,
    prefix_frame_count: int,
    observation_mask: np.ndarray | None = None,
    config: GraphModeAbductionConfig | None = None,
    base_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Update a rollout bank from graph modes without changing legacy inference."""

    settings = config or GraphModeAbductionConfig()
    observations = np.asarray(observations_from_endpoint_m, dtype=float)
    basis = np.asarray(graph_basis, dtype=float)
    if observations.shape != bank.trajectories.shape[2:]:
        raise ValueError("observations must match rollout-bank frames and nodes")
    if basis.ndim != 2 or basis.shape[0] != bank.node_count or basis.shape[1] < 1:
        raise ValueError("graph_basis must have shape (node_count, rank>=1)")
    if not 2 <= prefix_frame_count < bank.frame_count:
        raise ValueError(
            "prefix_frame_count must reveal evidence and leave future frames"
        )
    if (
        settings.mode_covariance_m2 is not None
        and settings.mode_covariance_m2.shape
        != (basis.shape[1], basis.shape[1])
    ):
        raise ValueError("mode covariance rank differs from graph basis")

    components = physical_readout_components(bank, belief)
    selected_observations = observations[:prefix_frame_count]
    selected_mask = (
        None
        if observation_mask is None
        else np.asarray(observation_mask)[:prefix_frame_count]
    )
    valid = _coordinate_mask(selected_observations, selected_mask)
    residual = components[:, :, :prefix_frame_count] - selected_observations[None, None]
    coefficients = _project_component_residuals(
        residual,
        valid,
        basis,
        ridge=settings.projection_ridge,
    )
    rank = basis.shape[1]
    correlated = (
        np.zeros((rank, rank), dtype=float)
        if settings.mode_covariance_m2 is None
        else settings.mode_covariance_m2
    )
    position_covariance = correlated + settings.position_scale_m**2 * np.eye(rank)
    position_score = _multivariate_student_t_score(
        coefficients[:, :, 1:],
        position_covariance,
        degrees_of_freedom=settings.degrees_of_freedom,
    )
    score = position_score
    if settings.dynamic_likelihood_weight > 0.0:
        # Includes the endpoint-to-first-O+ increment, which the legacy positional
        # slice cannot express on its own.
        increments = np.diff(coefficients, axis=2)
        dynamic_covariance = 2.0 * correlated + settings.dynamic_scale_m**2 * np.eye(
            rank
        )
        dynamic_score = _multivariate_student_t_score(
            increments,
            dynamic_covariance,
            degrees_of_freedom=settings.degrees_of_freedom,
        )
        score = score + settings.dynamic_likelihood_weight * dynamic_score

    prior = bank.prior_joint_weights if base_weights is None else np.asarray(
        base_weights,
        dtype=float,
    )
    if prior.shape != bank.prior_joint_weights.shape:
        raise ValueError("base_weights must match the rollout bank")
    if np.any(prior < 0.0) or not np.isclose(np.sum(prior), 1.0):
        raise ValueError("base_weights must be nonnegative and sum to one")
    return _normalize_log_weights(
        np.log(np.maximum(prior, 1e-300)) + settings.likelihood_temperature * score
    )


def abduct_factual_intervention_graph_mode(
    bank: JointRolloutBank,
    belief: TwinBelief,
    observations_from_endpoint_m: np.ndarray,
    graph_basis: np.ndarray,
    *,
    prefix_frame_count: int,
    observation_mask: np.ndarray | None = None,
    config: GraphModeAbductionConfig | None = None,
) -> FactualIntervention:
    """Infer factual ``(phi, kappa_obs)`` using the opt-in graph-mode likelihood."""

    settings = config or GraphModeAbductionConfig()
    expected_stop = belief.context.o_plus.frame_start + prefix_frame_count - 1
    if expected_stop > belief.context.o_plus.frame_stop:
        raise ValueError("abduction prefix extends beyond O+")
    weights = graph_mode_joint_weights(
        bank,
        belief,
        observations_from_endpoint_m,
        graph_basis,
        prefix_frame_count=prefix_frame_count,
        observation_mask=observation_mask,
        config=settings,
    )
    hand_count = len(bank.hypothesis_metadata[0]["contact"]["attachment_shifts"])
    phi_names = ("gain_multiplier", "delay_steps", "rotation_degrees")
    kappa_names = tuple(
        f"attachment_shift_hand_{index}" for index in range(hand_count)
    ) + ("slip_fraction",)
    component_ids: list[str] = []
    phi: list[tuple[float, ...]] = []
    kappa: list[tuple[float, ...]] = []
    hypothesis_indices: list[int] = []
    particle_indices: list[int] = []
    for hypothesis_index, (hypothesis_id, metadata) in enumerate(
        zip(bank.hypothesis_ids, bank.hypothesis_metadata, strict=True)
    ):
        action = metadata["action"]
        if not bool(action["future_action_observed"]):
            raise ValueError("factual abduction requires the observed u_obs action")
        contact = metadata["contact"]
        persistent = (
            float(contact["gain_multiplier"]),
            float(contact["delay_steps"]),
            float(contact["rotation_degrees"]),
        )
        event = tuple(map(float, contact["attachment_shifts"])) + (
            float(contact["slip_fraction"]),
        )
        for particle_index, particle_id in enumerate(belief.particle_ids):
            component_ids.append(f"{hypothesis_id}::{particle_id}")
            phi.append(persistent)
            kappa.append(event)
            hypothesis_indices.append(hypothesis_index)
            particle_indices.append(particle_index)
    metadata = {
        "abduction_likelihood": {
            "family": "multivariate_student_t_graph_modes",
            **settings.metadata(np.asarray(graph_basis).shape[1]),
        },
        "graph_basis_sha256": array_sha256(np.asarray(graph_basis, dtype=float)),
        "graph_mode_count": int(np.asarray(graph_basis).shape[1]),
        "observation_prefix_frame_count_including_endpoint": prefix_frame_count,
        "o_plus_frames_used": prefix_frame_count - 1,
        "dynamic_increments_used": prefix_frame_count - 1,
        "endpoint_to_first_o_plus_increment_included": True,
        "future_frames_read_by_abduction": 0,
        "legacy_factual_abduction_unchanged": True,
        "discrepancy_scored_as_separate_readout": True,
        "discrepancy_injected_into_simulator_state": False,
    }
    json.dumps(metadata, allow_nan=False)
    return FactualIntervention(
        context=belief.context,
        component_ids=tuple(component_ids),
        phi_names=phi_names,
        kappa_names=kappa_names,
        phi=np.asarray(phi, dtype=float),
        kappa_obs=np.asarray(kappa, dtype=float),
        hypothesis_indices=np.asarray(hypothesis_indices, dtype=np.int64),
        twin_particle_indices=np.asarray(particle_indices, dtype=np.int64),
        weights=weights.reshape(-1),
        evidence_frame_stop=expected_stop,
        source_twin_belief_id=belief.artifact_id,
        metadata=metadata,
    )
