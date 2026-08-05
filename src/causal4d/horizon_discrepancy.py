"""Source-calibrated horizon discrepancy forecasts for Causal4D twin beliefs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from causal4d.belief_provider_v2_contract import (
    require_bayesian_phystwin_belief_provider_v2,
)
from causal4d.contracts import TwinBelief, array_sha256
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json, validated_json_mapping


HORIZON_DISCREPANCY_BANK_SCHEMA_VERSION = "causal4d.horizon_discrepancy_bank.v1"


def _identifier(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _identifier(value, name=name)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _identifiers(
    values: Sequence[str],
    *,
    name: str,
    expected_count: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(_identifier(value, name=name) for value in values)
    if len(result) != expected_count:
        raise ValueError(f"{name} must identify every particle")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _horizons(values: Sequence[int]) -> np.ndarray:
    if isinstance(values, (str, bytes)):
        raise ValueError("horizon_steps must be a sequence of integers")
    raw = tuple(values)
    if not raw:
        raise ValueError("horizon_steps must be nonempty")
    if any(type(value) is not int or value < 0 for value in raw):
        raise ValueError("horizon_steps must contain nonnegative integers")
    if len(set(raw)) != len(raw):
        raise ValueError("horizon_steps must not contain duplicates")
    result = np.asarray(sorted(raw), dtype=np.int64)
    if result[0] != 0:
        raise ValueError(
            "horizon_steps must include zero as an endpoint parity control"
        )
    return readonly_array(result)


def _probability_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    result = readonly_array(values, dtype=float)
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.all(np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} must contain finite probabilities")
    return result


def _particle_weights(values: np.ndarray, particle_count: int) -> np.ndarray:
    weights = _probability_vector(values, name="particle_weights")
    if weights.shape != (particle_count,) or not np.isclose(
        np.sum(weights),
        1.0,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError("particle_weights must identify every particle and sum to one")
    return weights


def _maximum_norm(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("maximum_discrepancy_m must be a positive real or null")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("maximum_discrepancy_m must be positive and finite")
    return result


def _validate_covariance(values: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(values, values.swapaxes(-1, -2), rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    if np.min(np.linalg.eigvalsh(values), initial=0.0) < -1e-12:
        raise ValueError(f"{name} must be positive semidefinite")


def _lift_map(
    indices: np.ndarray,
    weights: np.ndarray,
    *,
    tracked_count: int,
    state_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_indices = np.asarray(indices)
    if not np.issubdtype(raw_indices.dtype, np.integer):
        raise ValueError("lift_indices must contain integers")
    neighbor_indices = np.array(raw_indices, dtype=np.int64, copy=True, order="C")
    neighbor_weights = np.array(weights, dtype=np.float64, copy=True, order="C")
    extra_count = state_count - tracked_count
    if extra_count < 0:
        raise ValueError("tracked endpoint count exceeds the physical state")
    if (
        neighbor_indices.ndim != 2
        or neighbor_indices.shape != neighbor_weights.shape
        or neighbor_indices.shape[0] != extra_count
    ):
        raise ValueError("lift map must identify every untracked state node")
    if not np.all(np.isfinite(neighbor_weights)) or np.any(neighbor_weights < 0.0):
        raise ValueError("lift_weights must be finite and nonnegative")
    if extra_count:
        if neighbor_indices.shape[1] < 1:
            raise ValueError("untracked state nodes require at least one neighbor")
        if np.any(neighbor_indices < 0) or np.any(neighbor_indices >= tracked_count):
            raise ValueError("lift_indices reference an unavailable tracked node")
        if not np.allclose(
            np.sum(neighbor_weights, axis=1),
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("lift_weights must sum to one per untracked node")
        for row in neighbor_indices:
            if len(np.unique(row)) != len(row):
                raise ValueError("one lift row must not repeat a tracked node")
    return readonly_array(neighbor_indices), readonly_array(neighbor_weights)


def _lift_prediction(
    mean_m: np.ndarray,
    covariance_m2: np.ndarray,
    *,
    state_count: int,
    lift_indices: np.ndarray,
    lift_weights: np.ndarray,
    maximum_discrepancy_m: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(mean_m, dtype=float)
    covariance = np.asarray(covariance_m2, dtype=float)
    if mean.ndim != 2 or mean.shape[1] != 3 or len(mean) < 1:
        raise ValueError("provider mean_m must have shape (N>=1, 3)")
    if covariance.shape != (len(mean), 3, 3):
        raise ValueError("provider covariance_m2 must have shape (N, 3, 3)")
    if not np.all(np.isfinite(mean)):
        raise ValueError("provider discrepancy mean must be finite")
    _validate_covariance(covariance, name="provider covariance_m2")
    tracked_count = len(mean)
    result_mean = np.empty((state_count, 3), dtype=float)
    result_covariance = np.empty((state_count, 3, 3), dtype=float)
    result_mean[:tracked_count] = mean
    result_covariance[:tracked_count] = covariance
    if state_count > tracked_count:
        result_mean[tracked_count:] = np.einsum(
            "ek,ekc->ec",
            lift_weights,
            mean[lift_indices],
        )
        result_covariance[tracked_count:] = np.einsum(
            "ek,ekij->eij",
            np.square(lift_weights),
            covariance[lift_indices],
        )
    if maximum_discrepancy_m is not None:
        norms = np.linalg.norm(result_mean, axis=1)
        scale = np.minimum(
            1.0,
            maximum_discrepancy_m / np.maximum(norms, np.finfo(float).tiny),
        )
        result_mean *= scale[:, None]
    result_covariance = 0.5 * (result_covariance + result_covariance.swapaxes(-1, -2))
    _validate_covariance(result_covariance, name="lifted covariance_m2")
    return result_mean, result_covariance


@dataclass(frozen=True)
class HorizonDiscrepancyBankV1:
    """Per-particle, per-horizon discrepancy moments bound to one TwinBelief."""

    twin_belief_id: str
    particle_ids: tuple[str, ...]
    particle_weights: np.ndarray
    horizon_steps: np.ndarray
    mean_m: np.ndarray
    covariance_m2: np.ndarray
    mean_retention: np.ndarray
    additional_axis_variance_m2: np.ndarray
    calibration_id: str
    provider_revision: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        belief_id = _sha256(self.twin_belief_id, name="twin_belief_id")
        mean = readonly_array(self.mean_m, dtype=float)
        covariance = readonly_array(self.covariance_m2, dtype=float)
        if mean.ndim != 4 or mean.shape[-1] != 3:
            raise ValueError("mean_m must have shape (P, H, N, 3)")
        particle_count, horizon_count, state_count, _ = mean.shape
        if particle_count < 1 or horizon_count < 1 or state_count < 1:
            raise ValueError("discrepancy bank dimensions must be nonempty")
        particle_ids = _identifiers(
            self.particle_ids,
            name="particle_ids",
            expected_count=particle_count,
        )
        weights = _particle_weights(self.particle_weights, particle_count)
        horizons = np.asarray(self.horizon_steps)
        if not np.issubdtype(horizons.dtype, np.integer):
            raise ValueError("horizon_steps must contain integers")
        horizons = _horizons(tuple(int(value) for value in horizons.tolist()))
        if len(horizons) != horizon_count:
            raise ValueError("horizon_steps must identify every bank horizon")
        if covariance.shape != (particle_count, horizon_count, state_count, 3, 3):
            raise ValueError("covariance_m2 must have shape (P, H, N, 3, 3)")
        if not np.all(np.isfinite(mean)):
            raise ValueError("mean_m must contain only finite values")
        _validate_covariance(covariance, name="covariance_m2")
        retention = _probability_vector(
            self.mean_retention,
            name="mean_retention",
        )
        if retention.shape != (horizon_count,):
            raise ValueError("mean_retention must identify every horizon")
        additional = readonly_array(self.additional_axis_variance_m2, dtype=float)
        if additional.shape != (horizon_count, 3):
            raise ValueError("additional_axis_variance_m2 must have shape (H, 3)")
        if not np.all(np.isfinite(additional)) or np.any(additional < 0.0):
            raise ValueError(
                "additional_axis_variance_m2 must be finite and nonnegative"
            )
        if retention[0] != 1.0 or np.any(additional[0] != 0.0):
            raise ValueError("horizon zero must preserve the endpoint moments exactly")
        calibration_id = _sha256(self.calibration_id, name="calibration_id")
        provider_revision = _identifier(
            self.provider_revision,
            name="provider_revision",
        )
        metadata = validated_json_mapping(
            self.metadata,
            error_message="horizon discrepancy bank metadata must be finite JSON data",
        )
        object.__setattr__(self, "twin_belief_id", belief_id)
        object.__setattr__(self, "particle_ids", particle_ids)
        object.__setattr__(self, "particle_weights", weights)
        object.__setattr__(self, "horizon_steps", horizons)
        object.__setattr__(self, "mean_m", mean)
        object.__setattr__(self, "covariance_m2", covariance)
        object.__setattr__(self, "mean_retention", retention)
        object.__setattr__(self, "additional_axis_variance_m2", additional)
        object.__setattr__(self, "calibration_id", calibration_id)
        object.__setattr__(self, "provider_revision", provider_revision)
        object.__setattr__(self, "metadata", metadata)

    @property
    def artifact_id(self) -> str:
        """Content identity for all provenance and numerical bank values."""

        descriptor = {
            "schema_version": HORIZON_DISCREPANCY_BANK_SCHEMA_VERSION,
            "twin_belief_id": self.twin_belief_id,
            "particle_ids": list(self.particle_ids),
            "calibration_id": self.calibration_id,
            "provider_revision": self.provider_revision,
            "metadata": plain_json(self.metadata),
            "arrays": {
                "particle_weights": array_sha256(self.particle_weights),
                "horizon_steps": array_sha256(self.horizon_steps),
                "mean_m": array_sha256(self.mean_m),
                "covariance_m2": array_sha256(self.covariance_m2),
                "mean_retention": array_sha256(self.mean_retention),
                "additional_axis_variance_m2": array_sha256(
                    self.additional_axis_variance_m2
                ),
            },
        }
        encoded = json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def index_for_horizon(self, horizon_steps: int) -> int:
        """Return the canonical bank index for an exact registered horizon."""

        if type(horizon_steps) is not int or horizon_steps < 0:
            raise ValueError("horizon_steps must be a nonnegative integer")
        indices = np.flatnonzero(self.horizon_steps == horizon_steps)
        if len(indices) != 1:
            raise KeyError(f"horizon {horizon_steps} is not registered")
        return int(indices[0])

    def moments_at_horizon(
        self,
        horizon_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return immutable per-particle mean and covariance for one horizon."""

        index = self.index_for_horizon(horizon_steps)
        return self.mean_m[:, index], self.covariance_m2[:, index]


def build_horizon_discrepancy_bank(
    twin_belief: TwinBelief,
    endpoint_posteriors: Sequence[object],
    calibration: object,
    *,
    horizon_steps: Sequence[int],
    lift_indices: np.ndarray,
    lift_weights: np.ndarray,
    maximum_discrepancy_m: float | None = None,
    provider_revision: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> HorizonDiscrepancyBankV1:
    """Propagate source-frozen BPT discrepancy dynamics without target outcomes."""

    if not isinstance(twin_belief, TwinBelief):
        raise TypeError("twin_belief must be a TwinBelief")
    manifest = require_bayesian_phystwin_belief_provider_v2(
        provider_revision=provider_revision
    )
    from bayesian_phystwin.causal4d_belief_provider_v2 import (
        HorizonConditionedEndpointPredictionV1,
        HorizonDiscrepancyCalibrationV1,
        ModelAveragedEndpointPosteriorV1,
        predict_horizon_conditioned_endpoint,
    )

    if not isinstance(calibration, HorizonDiscrepancyCalibrationV1):
        raise TypeError("calibration must be a HorizonDiscrepancyCalibrationV1")
    posteriors = tuple(endpoint_posteriors)
    particle_count = len(twin_belief.particle_ids)
    if len(posteriors) != particle_count or not all(
        isinstance(value, ModelAveragedEndpointPosteriorV1) for value in posteriors
    ):
        raise ValueError("endpoint_posteriors must identify every TwinBelief particle")
    horizons = _horizons(horizon_steps)
    state_count = twin_belief.endpoint_position_m.shape[1]
    tracked_counts = {len(value.mean_m) for value in posteriors}
    if len(tracked_counts) != 1:
        raise ValueError("endpoint posteriors must use one common tracked state")
    tracked_count = tracked_counts.pop()
    neighbor_indices, neighbor_weights = _lift_map(
        lift_indices,
        lift_weights,
        tracked_count=tracked_count,
        state_count=state_count,
    )
    maximum_norm = _maximum_norm(maximum_discrepancy_m)

    mean = np.empty((particle_count, len(horizons), state_count, 3), dtype=float)
    covariance = np.empty(
        (particle_count, len(horizons), state_count, 3, 3),
        dtype=float,
    )
    retention = np.empty(len(horizons), dtype=float)
    additional = np.empty((len(horizons), 3), dtype=float)
    calibration_id: str | None = None
    for particle_index, posterior in enumerate(posteriors):
        for horizon_index, horizon in enumerate(horizons):
            prediction = predict_horizon_conditioned_endpoint(
                posterior,
                calibration,
                horizon_steps=int(horizon),
            )
            if not isinstance(prediction, HorizonConditionedEndpointPredictionV1):
                raise TypeError(
                    "Bayesian-PhysTwin returned the wrong horizon prediction type"
                )
            if prediction.calibration_id != calibration.artifact_id:
                raise ValueError("horizon prediction calibration identity changed")
            lifted_mean, lifted_covariance = _lift_prediction(
                prediction.mean_m,
                prediction.covariance_m2,
                state_count=state_count,
                lift_indices=neighbor_indices,
                lift_weights=neighbor_weights,
                maximum_discrepancy_m=maximum_norm,
            )
            mean[particle_index, horizon_index] = lifted_mean
            covariance[particle_index, horizon_index] = lifted_covariance
            if particle_index == 0:
                retention[horizon_index] = prediction.mean_retention
                additional[horizon_index] = prediction.additional_axis_variance_m2
            else:
                if prediction.mean_retention != retention[
                    horizon_index
                ] or not np.array_equal(
                    prediction.additional_axis_variance_m2,
                    additional[horizon_index],
                ):
                    raise ValueError(
                        "horizon calibration semantics changed across particles"
                    )
            if calibration_id is None:
                calibration_id = prediction.calibration_id
            elif prediction.calibration_id != calibration_id:
                raise ValueError("calibration identity changed across predictions")

    assert calibration_id is not None
    bank_metadata: dict[str, Any] = {
        "provider_manifest": manifest.as_dict(),
        "calibration_source_group_count": len(calibration.source_group_ids),
        "calibration_source_group_ids": list(calibration.source_group_ids),
        "calibration_target_outcomes_used": calibration.target_outcomes_used,
        "calibration_confirmation_outcomes_used": (
            calibration.confirmation_outcomes_used
        ),
        "maximum_discrepancy_m": maximum_norm,
        "covariance_lift_semantics": (
            "independent tracked-node covariance propagated by squared fixed "
            "readout weights"
        ),
        "future_observations_read": 0,
    }
    if metadata is not None:
        bank_metadata["consumer_metadata"] = dict(metadata)
    return HorizonDiscrepancyBankV1(
        twin_belief_id=twin_belief.artifact_id,
        particle_ids=twin_belief.particle_ids,
        particle_weights=twin_belief.weights,
        horizon_steps=horizons,
        mean_m=mean,
        covariance_m2=covariance,
        mean_retention=retention,
        additional_axis_variance_m2=additional,
        calibration_id=calibration_id,
        provider_revision=manifest.provider_revision,
        metadata=bank_metadata,
    )


__all__ = [
    "HORIZON_DISCREPANCY_BANK_SCHEMA_VERSION",
    "HorizonDiscrepancyBankV1",
    "build_horizon_discrepancy_bank",
]
