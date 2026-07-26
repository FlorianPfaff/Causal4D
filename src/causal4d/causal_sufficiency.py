"""Cross-fitted falsification test for realized-intervention sufficiency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


def _validated_metadata(values: Mapping[str, Any]) -> dict[str, Any]:
    import json

    try:
        return json.loads(json.dumps(dict(values), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must contain finite JSON values") from error


@dataclass(frozen=True)
class CausalSufficiencyResult:
    """Cross-fitted incremental predictive value of commanded-action identity."""

    baseline_rmse: float
    command_augmented_rmse: float
    relative_rmse_reduction: float
    permutation_p_value: float
    command_effect_detected: bool
    execution_count: int
    group_count: int
    command_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        numeric = (
            self.baseline_rmse,
            self.command_augmented_rmse,
            self.relative_rmse_reduction,
            self.permutation_p_value,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("sufficiency statistics must be finite")
        if self.baseline_rmse < 0.0 or self.command_augmented_rmse < 0.0:
            raise ValueError("RMSE values must be nonnegative")
        if not 0.0 <= self.permutation_p_value <= 1.0:
            raise ValueError("permutation_p_value must lie in [0, 1]")
        if min(self.execution_count, self.group_count, self.command_count) < 1:
            raise ValueError("counts must be positive")
        object.__setattr__(self, "metadata", _validated_metadata(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_rmse": self.baseline_rmse,
            "command_augmented_rmse": self.command_augmented_rmse,
            "relative_rmse_reduction": self.relative_rmse_reduction,
            "permutation_p_value": self.permutation_p_value,
            "command_effect_detected": self.command_effect_detected,
            "execution_count": self.execution_count,
            "group_count": self.group_count,
            "command_count": self.command_count,
            "metadata": dict(self.metadata),
        }


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    if array.ndim != 2 or array.shape[0] < 4 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite execution-by-feature matrix")
    return array


def _identifiers(values: Sequence[str], count: int, name: str) -> tuple[str, ...]:
    identifiers = tuple(map(str, values))
    if len(identifiers) != count or any(not value for value in identifiers):
        raise ValueError(f"{name} must contain one nonempty value per execution")
    return identifiers


def _command_features(
    command_ids: tuple[str, ...],
    categories: tuple[str, ...],
) -> np.ndarray:
    if len(categories) <= 1:
        return np.empty((len(command_ids), 0), dtype=float)
    return np.asarray(
        [
            [float(value == category) for category in categories[1:]]
            for value in command_ids
        ],
        dtype=float,
    )


def _ridge_predict(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    mean = np.mean(train_features, axis=0)
    scale = np.std(train_features, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    train = (train_features - mean) / scale
    test = (test_features - mean) / scale
    train_design = np.column_stack((np.ones(len(train)), train))
    test_design = np.column_stack((np.ones(len(test)), test))
    penalty = np.eye(train_design.shape[1], dtype=float) * float(ridge)
    penalty[0, 0] = 0.0
    normal = train_design.T @ train_design + penalty
    right = train_design.T @ train_targets
    coefficients = np.linalg.lstsq(normal, right, rcond=None)[0]
    return test_design @ coefficients


def _cross_fitted_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    group_ids: tuple[str, ...],
    *,
    ridge: float,
) -> np.ndarray:
    predictions = np.empty_like(targets)
    groups = tuple(dict.fromkeys(group_ids))
    for group in groups:
        test = np.asarray([value == group for value in group_ids], dtype=bool)
        train = ~test
        if int(np.sum(train)) < 2 or int(np.sum(test)) < 1:
            raise ValueError("every cross-fit fold needs training and test executions")
        predictions[test] = _ridge_predict(
            features[train],
            targets[train],
            features[test],
            ridge=ridge,
        )
    return predictions


def _relative_improvement(
    targets: np.ndarray,
    baseline_prediction: np.ndarray,
    augmented_prediction: np.ndarray,
) -> tuple[float, float, float]:
    baseline_rmse = float(np.sqrt(np.mean(np.square(targets - baseline_prediction))))
    augmented_rmse = float(np.sqrt(np.mean(np.square(targets - augmented_prediction))))
    if baseline_rmse <= np.finfo(float).eps:
        return baseline_rmse, augmented_rmse, 0.0
    return (
        baseline_rmse,
        augmented_rmse,
        float(1.0 - augmented_rmse / baseline_rmse),
    )


def assess_command_residual_sufficiency(
    future_residual_targets: np.ndarray,
    realization_features: np.ndarray,
    command_ids: Sequence[str],
    *,
    group_ids: Sequence[str] | None = None,
    ridge: float = 1.0e-6,
    permutation_count: int = 199,
    random_seed: int = 0,
    significance_level: float = 0.05,
    minimum_relative_improvement: float = 0.01,
) -> CausalSufficiencyResult:
    """Test whether command identity predicts residuals after conditioning on ``z``.

    A significant held-out gain from command identity is evidence that the
    supplied realization features are causally incomplete. Cross-fitting is by
    independent group, normally execution session. The permutation test is a
    finite-sample diagnostic, not a proof of conditional independence.
    """

    targets = _matrix(future_residual_targets, "future_residual_targets")
    features = _matrix(realization_features, "realization_features")
    if len(features) != len(targets):
        raise ValueError("targets and realization features must align")
    commands = _identifiers(command_ids, len(targets), "command_ids")
    groups = (
        tuple(f"execution-{index}" for index in range(len(targets)))
        if group_ids is None
        else _identifiers(group_ids, len(targets), "group_ids")
    )
    categories = tuple(sorted(set(commands)))
    unique_groups = tuple(dict.fromkeys(groups))
    if len(categories) < 2:
        raise ValueError("at least two commanded actions are required")
    if len(unique_groups) < 3:
        raise ValueError("at least three independent cross-fit groups are required")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and nonnegative")
    if permutation_count < 1:
        raise ValueError("permutation_count must be positive")
    if not 0.0 < significance_level < 1.0:
        raise ValueError("significance_level must lie in (0, 1)")
    if not np.isfinite(minimum_relative_improvement):
        raise ValueError("minimum_relative_improvement must be finite")

    command_features = _command_features(commands, categories)
    augmented_features = np.column_stack((features, command_features))
    baseline_prediction = _cross_fitted_predictions(
        features,
        targets,
        groups,
        ridge=ridge,
    )
    augmented_prediction = _cross_fitted_predictions(
        augmented_features,
        targets,
        groups,
        ridge=ridge,
    )
    baseline_rmse, augmented_rmse, observed_improvement = _relative_improvement(
        targets,
        baseline_prediction,
        augmented_prediction,
    )

    rng = np.random.default_rng(random_seed)
    command_array = np.asarray(commands, dtype=object)
    null_improvements = np.empty(permutation_count, dtype=float)
    for index in range(permutation_count):
        permuted = tuple(map(str, rng.permutation(command_array)))
        permuted_features = np.column_stack(
            (features, _command_features(permuted, categories))
        )
        permuted_prediction = _cross_fitted_predictions(
            permuted_features,
            targets,
            groups,
            ridge=ridge,
        )
        _, _, null_improvements[index] = _relative_improvement(
            targets,
            baseline_prediction,
            permuted_prediction,
        )
    p_value = float(
        (1 + np.sum(null_improvements >= observed_improvement))
        / (permutation_count + 1)
    )
    detected = bool(
        observed_improvement >= minimum_relative_improvement
        and p_value <= significance_level
    )
    return CausalSufficiencyResult(
        baseline_rmse=baseline_rmse,
        command_augmented_rmse=augmented_rmse,
        relative_rmse_reduction=observed_improvement,
        permutation_p_value=p_value,
        command_effect_detected=detected,
        execution_count=len(targets),
        group_count=len(unique_groups),
        command_count=len(categories),
        metadata={
            "cross_fit_unit": "group",
            "ridge": float(ridge),
            "permutation_count": int(permutation_count),
            "random_seed": int(random_seed),
            "significance_level": float(significance_level),
            "minimum_relative_improvement": float(minimum_relative_improvement),
            "interpretation": (
                "detected command value indicates incomplete realized-intervention "
                "features; non-detection is not proof of sufficiency"
            ),
        },
    )
