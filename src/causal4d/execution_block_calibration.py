"""Execution-block split-conformal calibration for locked real protocols.

This module is deliberately separate from the older coordinate-pooled affine
calibration.  The confirmatory unit here is one independent execution/session,
and every calibration execution contributes exactly one nonconformity score.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from causal4d.atomic_io import atomic_write_json
from causal4d.immutable_array import readonly_array as _readonly_array
from causal4d.immutable_json import plain_json, validated_json_mapping


EXECUTION_BLOCK_CALIBRATION_SCHEMA_VERSION = 1
EXECUTION_BLOCK_SCORE_KIND = "max_abs_standardized_coordinate_v1"
EXECUTION_BLOCK_CALIBRATION_UNIT = "one preregistered execution per independent session"
ExecutionBlockRole = Literal["fit", "calibration", "target"]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


@dataclass(frozen=True)
class ExecutionBlockCalibrationCase:
    """One predictive execution with an explicit independent-session identity."""

    execution_id: str
    session_id: str
    outer_fold_id: str
    split_role: ExecutionBlockRole
    prediction_case_id: str
    action_id: str
    contact_region_id: str
    mean_m: np.ndarray
    variance_m2: np.ndarray
    truth_m: np.ndarray
    valid: np.ndarray
    start_frame: int
    node_group_labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("execution_id", self.execution_id),
            ("session_id", self.session_id),
            ("outer_fold_id", self.outer_fold_id),
            ("prediction_case_id", self.prediction_case_id),
            ("action_id", self.action_id),
            ("contact_region_id", self.contact_region_id),
        ):
            if not value:
                raise ValueError(f"{name} must be nonempty")
        if self.split_role not in {"fit", "calibration", "target"}:
            raise ValueError("split_role must be fit, calibration, or target")

        mean = _readonly_array(self.mean_m, dtype=float)
        variance = _readonly_array(self.variance_m2, dtype=float)
        truth = _readonly_array(self.truth_m, dtype=float)
        valid = np.asarray(self.valid, dtype=bool).copy()
        if mean.ndim != 3 or mean.shape[2] != 3:
            raise ValueError("execution trajectories must have shape (T, N, 3)")
        if variance.shape != mean.shape or truth.shape != mean.shape:
            raise ValueError("mean, variance, and truth must share a shape")
        if valid.shape == mean.shape:
            valid = np.all(valid, axis=2)
        if valid.shape != mean.shape[:2]:
            raise ValueError("valid must have shape (T, N) or (T, N, 3)")
        if not 0 <= self.start_frame < len(mean):
            raise ValueError("start_frame must lie inside the trajectory")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
            raise ValueError("predictive moments must be finite")
        if np.any(variance <= 0.0):
            raise ValueError("predictive variance must be strictly positive")
        valid &= np.all(np.isfinite(truth), axis=2)
        valid[: self.start_frame] = False
        if not np.any(valid):
            raise ValueError("execution has no valid held-out point-frames")
        valid = _readonly_array(valid, dtype=bool)

        labels = self.node_group_labels
        if labels is not None:
            labels = tuple(map(str, labels))
            if len(labels) != mean.shape[1] or any(not value for value in labels):
                raise ValueError("node_group_labels must identify every node")

        object.__setattr__(self, "mean_m", mean)
        object.__setattr__(self, "variance_m2", variance)
        object.__setattr__(self, "truth_m", truth)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "node_group_labels", labels)

    @classmethod
    def from_real_calibration_case(
        cls,
        case: Any,
        *,
        execution_id: str,
        session_id: str,
        outer_fold_id: str,
        split_role: ExecutionBlockRole,
    ) -> ExecutionBlockCalibrationCase:
        """Bind an existing ``RealCalibrationCase`` to the locked split identity."""

        return cls(
            execution_id=execution_id,
            session_id=session_id,
            outer_fold_id=outer_fold_id,
            split_role=split_role,
            prediction_case_id=str(case.case_id),
            action_id=str(case.action_id),
            contact_region_id=str(case.contact_region_id),
            mean_m=case.mean_m,
            variance_m2=case.variance_m2,
            truth_m=case.truth_m,
            valid=case.valid,
            start_frame=int(case.start_frame),
            node_group_labels=case.node_group_labels,
        )

    @property
    def coordinate_count(self) -> int:
        return int(3 * np.sum(self.valid))


@dataclass(frozen=True)
class ExecutionBlockScore:
    """One scalar nonconformity score for one independent execution."""

    execution_id: str
    session_id: str
    outer_fold_id: str
    split_role: ExecutionBlockRole
    prediction_case_id: str
    score: float
    coordinate_count: int
    maximum_abs_residual_m: float
    mean_abs_residual_m: float

    def __post_init__(self) -> None:
        for name, value in (
            ("execution_id", self.execution_id),
            ("session_id", self.session_id),
            ("outer_fold_id", self.outer_fold_id),
            ("prediction_case_id", self.prediction_case_id),
        ):
            if not value:
                raise ValueError(f"{name} must be nonempty")
        if self.split_role not in {"fit", "calibration", "target"}:
            raise ValueError("split_role must be fit, calibration, or target")
        for name, value in (
            ("score", self.score),
            ("maximum_abs_residual_m", self.maximum_abs_residual_m),
            ("mean_abs_residual_m", self.mean_abs_residual_m),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.coordinate_count < 1:
            raise ValueError("coordinate_count must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "outer_fold_id": self.outer_fold_id,
            "split_role": self.split_role,
            "prediction_case_id": self.prediction_case_id,
            "score": float(self.score),
            "coordinate_count": int(self.coordinate_count),
            "maximum_abs_residual_m": float(self.maximum_abs_residual_m),
            "mean_abs_residual_m": float(self.mean_abs_residual_m),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExecutionBlockScore:
        return cls(
            execution_id=str(value["execution_id"]),
            session_id=str(value["session_id"]),
            outer_fold_id=str(value["outer_fold_id"]),
            split_role=str(value["split_role"]),  # type: ignore[arg-type]
            prediction_case_id=str(value["prediction_case_id"]),
            score=float(value["score"]),
            coordinate_count=int(value["coordinate_count"]),
            maximum_abs_residual_m=float(value["maximum_abs_residual_m"]),
            mean_abs_residual_m=float(value["mean_abs_residual_m"]),
        )


def _canonical_cases(
    cases: Sequence[ExecutionBlockCalibrationCase],
    *,
    role: ExecutionBlockRole,
) -> tuple[ExecutionBlockCalibrationCase, ...]:
    selected = tuple(cases)
    if not selected:
        raise ValueError(f"{role} split must contain at least one execution")
    if any(case.split_role != role for case in selected):
        raise ValueError(f"every {role} case must declare split_role={role!r}")
    folds = {case.outer_fold_id for case in selected}
    if len(folds) != 1:
        raise ValueError(f"{role} cases must belong to one outer fold")
    execution_ids = [case.execution_id for case in selected]
    session_ids = [case.session_id for case in selected]
    if len(set(execution_ids)) != len(execution_ids):
        raise ValueError(f"{role} execution ids must be unique")
    if len(set(session_ids)) != len(session_ids):
        raise ValueError(
            f"{role} calibration requires at most one execution per session"
        )
    return tuple(
        sorted(selected, key=lambda case: (case.session_id, case.execution_id))
    )


def _coordinate_residuals(case: ExecutionBlockCalibrationCase) -> np.ndarray:
    coordinate_valid = np.repeat(case.valid[:, :, None], 3, axis=2)
    return (case.mean_m - case.truth_m)[coordinate_valid]


def _mean_execution_nll(
    cases: Sequence[ExecutionBlockCalibrationCase],
    scale_a: float,
    floor_b_m2: float,
) -> float:
    scores = []
    for case in cases:
        coordinate_valid = np.repeat(case.valid[:, :, None], 3, axis=2)
        residual = (case.mean_m - case.truth_m)[coordinate_valid]
        variance = scale_a * case.variance_m2[coordinate_valid] + floor_b_m2
        scores.append(
            float(
                np.mean(
                    0.5
                    * (np.log(2.0 * np.pi * variance) + np.square(residual) / variance)
                )
            )
        )
    return float(np.mean(scores))


def _fit_equal_execution_affine_variance(
    fit_cases: Sequence[ExecutionBlockCalibrationCase],
) -> tuple[float, float, float]:
    residual_variance = float(
        np.mean([np.mean(np.square(_coordinate_residuals(case))) for case in fit_cases])
    )
    scale_grid = np.geomspace(0.01, 100.0, 25)
    maximum_floor_std = max(np.sqrt(residual_variance) * 3.0, 1e-3)
    floor_grid = np.concatenate(
        ([0.0], np.square(np.geomspace(1e-5, maximum_floor_std, 24)))
    )
    best = (float("inf"), 1.0, 0.0)
    for scale in scale_grid:
        for floor in floor_grid:
            nll = _mean_execution_nll(fit_cases, float(scale), float(floor))
            candidate = (nll, float(scale), float(floor))
            if candidate < best:
                best = candidate
    return best[1], best[2], best[0]


def score_execution_block_case(
    case: ExecutionBlockCalibrationCase,
    *,
    scale_a: float,
    floor_b_m2: float,
) -> ExecutionBlockScore:
    """Reduce one execution to its maximum standardized coordinate residual."""

    if not np.isfinite(scale_a) or scale_a <= 0.0:
        raise ValueError("scale_a must be finite and positive")
    if not np.isfinite(floor_b_m2) or floor_b_m2 < 0.0:
        raise ValueError("floor_b_m2 must be finite and nonnegative")
    coordinate_valid = np.repeat(case.valid[:, :, None], 3, axis=2)
    residual = np.abs((case.mean_m - case.truth_m)[coordinate_valid])
    variance = scale_a * case.variance_m2[coordinate_valid] + floor_b_m2
    if np.any(variance <= 0.0) or not np.all(np.isfinite(variance)):
        raise ValueError("transformed predictive variance must be finite and positive")
    standardized = residual / np.sqrt(variance)
    return ExecutionBlockScore(
        execution_id=case.execution_id,
        session_id=case.session_id,
        outer_fold_id=case.outer_fold_id,
        split_role=case.split_role,
        prediction_case_id=case.prediction_case_id,
        score=float(np.max(standardized)),
        coordinate_count=len(standardized),
        maximum_abs_residual_m=float(np.max(residual)),
        mean_abs_residual_m=float(np.mean(residual)),
    )


def _conformal_rank(calibration_units: int, confidence_level: float) -> int:
    return int(math.ceil((calibration_units + 1) * confidence_level))


def _leave_one_session_out_diagnostics(
    scores: Sequence[ExecutionBlockScore],
    *,
    confidence_level: float,
) -> list[dict[str, Any]]:
    diagnostics = []
    for removed in scores:
        remaining = [value.score for value in scores if value is not removed]
        rank = _conformal_rank(len(remaining), confidence_level)
        finite = rank <= len(remaining)
        formal_threshold = float(sorted(remaining)[rank - 1]) if finite else None
        diagnostics.append(
            {
                "removed_execution_id": removed.execution_id,
                "removed_session_id": removed.session_id,
                "remaining_units": len(remaining),
                "order_statistic_rank_one_based": rank,
                "finite_nominal_threshold": finite,
                "formal_threshold": formal_threshold,
                "diagnostic_maximum_remaining_score": (
                    float(max(remaining)) if remaining else None
                ),
            }
        )
    return diagnostics


def _fragility_diagnostics(
    scores: Sequence[ExecutionBlockScore],
    *,
    confidence_level: float,
) -> dict[str, Any]:
    values = np.asarray([value.score for value in scores], dtype=float)
    ordered = np.sort(values)
    median = float(np.median(values))
    maximum = float(ordered[-1])
    ratio: float | None
    if median > 0.0:
        ratio = maximum / median
    elif maximum == 0.0:
        ratio = 1.0
    else:
        ratio = None
    return {
        "maximum_score": maximum,
        "second_largest_score": (float(ordered[-2]) if len(ordered) >= 2 else None),
        "median_score": median,
        "maximum_to_median_score_ratio": ratio,
        "leave_one_calibration_session_out": (
            _leave_one_session_out_diagnostics(
                scores,
                confidence_level=confidence_level,
            )
        ),
        "fragility_may_select_threshold": False,
    }


@dataclass(frozen=True)
class ExecutionBlockConformalCalibration:
    """Checksummed rank-based calibration for one locked outer fold."""

    outer_fold_id: str
    confidence_level: float
    nll_scale_a: float
    nll_floor_b_m2: float
    fit_execution_ids: tuple[str, ...]
    fit_session_ids: tuple[str, ...]
    calibration_scores: tuple[ExecutionBlockScore, ...]
    expected_calibration_units: int
    order_statistic_rank_one_based: int
    threshold: float
    fit_mean_trial_gaussian_nll: float
    fragility_diagnostics: Mapping[str, Any]
    expected_fit_units: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    score_kind: str = EXECUTION_BLOCK_SCORE_KIND
    calibration_unit: str = EXECUTION_BLOCK_CALIBRATION_UNIT
    claim_ready: bool = True

    def __post_init__(self) -> None:
        if not self.outer_fold_id:
            raise ValueError("outer_fold_id must be nonempty")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1)")
        if not np.isfinite(self.nll_scale_a) or self.nll_scale_a <= 0.0:
            raise ValueError("nll_scale_a must be finite and positive")
        if not np.isfinite(self.nll_floor_b_m2) or self.nll_floor_b_m2 < 0.0:
            raise ValueError("nll_floor_b_m2 must be finite and nonnegative")
        if (
            not np.isfinite(self.fit_mean_trial_gaussian_nll)
            or not np.isfinite(self.threshold)
            or self.threshold < 0.0
        ):
            raise ValueError("fit NLL and threshold must be finite")
        if self.score_kind != EXECUTION_BLOCK_SCORE_KIND:
            raise ValueError("unsupported execution-block score kind")
        if self.calibration_unit != EXECUTION_BLOCK_CALIBRATION_UNIT:
            raise ValueError("unexpected execution-block calibration unit")
        if not self.claim_ready:
            raise ValueError("a finite execution-block calibration must be claim-ready")

        fit_executions = tuple(map(str, self.fit_execution_ids))
        fit_sessions = tuple(map(str, self.fit_session_ids))
        if (
            not fit_executions
            or len(set(fit_executions)) != len(fit_executions)
            or any(not value for value in fit_executions)
        ):
            raise ValueError("fit_execution_ids must be unique and nonempty")
        if (
            len(fit_sessions) != len(fit_executions)
            or len(set(fit_sessions)) != len(fit_sessions)
            or any(not value for value in fit_sessions)
        ):
            raise ValueError("fit_session_ids must uniquely identify fit executions")
        if self.expected_fit_units is not None:
            if self.expected_fit_units < 1:
                raise ValueError("expected_fit_units must be positive")
            if len(fit_executions) != self.expected_fit_units:
                raise ValueError("fit execution count differs from the frozen plan")

        scores = tuple(
            sorted(
                self.calibration_scores,
                key=lambda value: (value.session_id, value.execution_id),
            )
        )
        if not scores or any(value.split_role != "calibration" for value in scores):
            raise ValueError("calibration_scores must contain calibration executions")
        if any(value.outer_fold_id != self.outer_fold_id for value in scores):
            raise ValueError("calibration scores belong to the wrong outer fold")
        calibration_executions = [value.execution_id for value in scores]
        calibration_sessions = [value.session_id for value in scores]
        if len(set(calibration_executions)) != len(calibration_executions):
            raise ValueError("calibration execution ids must be unique")
        if len(set(calibration_sessions)) != len(calibration_sessions):
            raise ValueError("calibration sessions must be unique")
        if set(fit_executions) & set(calibration_executions):
            raise ValueError("an execution crosses fit and calibration")
        if set(fit_sessions) & set(calibration_sessions):
            raise ValueError("a session crosses fit and calibration")
        if self.expected_calibration_units < 1:
            raise ValueError("expected_calibration_units must be positive")
        if len(scores) != self.expected_calibration_units:
            raise ValueError("calibration execution count differs from the frozen plan")

        expected_rank = _conformal_rank(len(scores), self.confidence_level)
        if expected_rank > len(scores):
            raise ValueError("the requested coverage has no finite conformal threshold")
        if self.order_statistic_rank_one_based != expected_rank:
            raise ValueError("conformal order-statistic rank is inconsistent")
        expected_threshold = sorted(value.score for value in scores)[expected_rank - 1]
        if self.threshold != expected_threshold:
            raise ValueError("threshold does not match the registered order statistic")

        object.__setattr__(self, "fit_execution_ids", fit_executions)
        object.__setattr__(self, "fit_session_ids", fit_sessions)
        object.__setattr__(self, "calibration_scores", scores)
        object.__setattr__(
            self,
            "fragility_diagnostics",
            validated_json_mapping(self.fragility_diagnostics),
        )
        object.__setattr__(self, "metadata", validated_json_mapping(self.metadata))

    @property
    def source_execution_ids(self) -> tuple[str, ...]:
        return self.fit_execution_ids + tuple(
            value.execution_id for value in self.calibration_scores
        )

    @property
    def source_session_ids(self) -> tuple[str, ...]:
        return self.fit_session_ids + tuple(
            value.session_id for value in self.calibration_scores
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXECUTION_BLOCK_CALIBRATION_SCHEMA_VERSION,
            "artifact_kind": "ExecutionBlockConformalCalibration",
            "outer_fold_id": self.outer_fold_id,
            "confidence_level": float(self.confidence_level),
            "score_kind": self.score_kind,
            "calibration_unit": self.calibration_unit,
            "transform": {
                "formula": "variance_fit = a * variance_raw + b",
                "nll_scale_a": float(self.nll_scale_a),
                "nll_floor_b_m2": float(self.nll_floor_b_m2),
                "fit_mean_trial_gaussian_nll": float(self.fit_mean_trial_gaussian_nll),
            },
            "fit_execution_ids": list(self.fit_execution_ids),
            "fit_session_ids": list(self.fit_session_ids),
            "expected_fit_units": self.expected_fit_units,
            "calibration_scores": [
                value.as_dict() for value in self.calibration_scores
            ],
            "expected_calibration_units": self.expected_calibration_units,
            "order_statistic_rank_one_based": (self.order_statistic_rank_one_based),
            "threshold": float(self.threshold),
            "fragility_diagnostics": plain_json(self.fragility_diagnostics),
            "claim_ready": self.claim_ready,
            "coverage_claim": (
                "finite marginal execution-block split-conformal coverage under "
                "independent-session exchangeability"
            ),
            "worst_group_coverage_guarantee_claimed": False,
            "pooled_coordinate_conformal_claimed": False,
            "metadata": plain_json(self.metadata),
        }

    @property
    def calibration_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.as_dict()).encode()).hexdigest()


def fit_execution_block_conformal_calibration(
    fit_cases: Sequence[ExecutionBlockCalibrationCase],
    calibration_cases: Sequence[ExecutionBlockCalibrationCase],
    *,
    confidence_level: float = 0.90,
    expected_calibration_units: int = 9,
    expected_fit_units: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ExecutionBlockConformalCalibration:
    """Fit source-only variance, then calibrate one score per held-out session."""

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if expected_calibration_units < 1:
        raise ValueError("expected_calibration_units must be positive")
    if expected_fit_units is not None and expected_fit_units < 1:
        raise ValueError("expected_fit_units must be positive")

    fit = _canonical_cases(fit_cases, role="fit")
    calibration = _canonical_cases(calibration_cases, role="calibration")
    if fit[0].outer_fold_id != calibration[0].outer_fold_id:
        raise ValueError("fit and calibration cases must use the same outer fold")
    fit_execution_ids = tuple(case.execution_id for case in fit)
    fit_session_ids = tuple(case.session_id for case in fit)
    calibration_execution_ids = {case.execution_id for case in calibration}
    calibration_session_ids = {case.session_id for case in calibration}
    if set(fit_execution_ids) & calibration_execution_ids:
        raise ValueError("an execution crosses fit and calibration")
    if set(fit_session_ids) & calibration_session_ids:
        raise ValueError("a session crosses fit and calibration")
    if expected_fit_units is not None and len(fit) != expected_fit_units:
        raise ValueError(
            f"expected {expected_fit_units} fit executions; received {len(fit)}"
        )
    if len(calibration) != expected_calibration_units:
        raise ValueError(
            f"expected {expected_calibration_units} calibration executions; "
            f"received {len(calibration)}"
        )

    rank = _conformal_rank(len(calibration), confidence_level)
    if rank > len(calibration):
        raise ValueError(
            f"nominal coverage requires order-statistic rank {rank}, but only "
            f"{len(calibration)} independent calibration units are available"
        )

    scale_a, floor_b_m2, fit_nll = _fit_equal_execution_affine_variance(fit)
    scores = tuple(
        score_execution_block_case(
            case,
            scale_a=scale_a,
            floor_b_m2=floor_b_m2,
        )
        for case in calibration
    )
    threshold = sorted(value.score for value in scores)[rank - 1]
    return ExecutionBlockConformalCalibration(
        outer_fold_id=fit[0].outer_fold_id,
        confidence_level=confidence_level,
        nll_scale_a=scale_a,
        nll_floor_b_m2=floor_b_m2,
        fit_execution_ids=fit_execution_ids,
        fit_session_ids=fit_session_ids,
        calibration_scores=scores,
        expected_fit_units=expected_fit_units,
        expected_calibration_units=expected_calibration_units,
        order_statistic_rank_one_based=rank,
        threshold=threshold,
        fit_mean_trial_gaussian_nll=fit_nll,
        fragility_diagnostics=_fragility_diagnostics(
            scores,
            confidence_level=confidence_level,
        ),
        metadata=metadata or {},
    )


def save_execution_block_conformal_calibration(
    path: str | Path,
    calibration: ExecutionBlockConformalCalibration,
) -> None:
    payload = calibration.as_dict()
    payload["calibration_id"] = calibration.calibration_id
    atomic_write_json(path, payload)


def load_execution_block_conformal_calibration(
    path: str | Path,
) -> ExecutionBlockConformalCalibration:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != EXECUTION_BLOCK_CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported execution-block calibration schema")
    if payload.get("artifact_kind") != "ExecutionBlockConformalCalibration":
        raise ValueError("unexpected execution-block calibration artifact")
    transform = payload["transform"]
    calibration = ExecutionBlockConformalCalibration(
        outer_fold_id=str(payload["outer_fold_id"]),
        confidence_level=float(payload["confidence_level"]),
        nll_scale_a=float(transform["nll_scale_a"]),
        nll_floor_b_m2=float(transform["nll_floor_b_m2"]),
        fit_execution_ids=tuple(map(str, payload["fit_execution_ids"])),
        fit_session_ids=tuple(map(str, payload["fit_session_ids"])),
        calibration_scores=tuple(
            ExecutionBlockScore.from_dict(value)
            for value in payload["calibration_scores"]
        ),
        expected_fit_units=(
            None
            if payload.get("expected_fit_units") is None
            else int(payload["expected_fit_units"])
        ),
        expected_calibration_units=int(payload["expected_calibration_units"]),
        order_statistic_rank_one_based=int(payload["order_statistic_rank_one_based"]),
        threshold=float(payload["threshold"]),
        fit_mean_trial_gaussian_nll=float(transform["fit_mean_trial_gaussian_nll"]),
        fragility_diagnostics=payload["fragility_diagnostics"],
        metadata=payload.get("metadata", {}),
        score_kind=str(payload["score_kind"]),
        calibration_unit=str(payload["calibration_unit"]),
        claim_ready=bool(payload["claim_ready"]),
    )
    supplied_payload = dict(payload)
    supplied_payload.pop("calibration_id", None)
    if supplied_payload != calibration.as_dict():
        raise ValueError(
            "execution-block calibration payload differs from the canonical schema"
        )
    if payload.get("calibration_id") != calibration.calibration_id:
        raise ValueError("execution-block calibration checksum mismatch")
    return calibration


def evaluate_execution_block_case(
    case: ExecutionBlockCalibrationCase,
    calibration: ExecutionBlockConformalCalibration,
) -> dict[str, Any]:
    """Evaluate one target execution without altering the frozen threshold."""

    if case.split_role != "target":
        raise ValueError("execution-block evaluation requires split_role='target'")
    if case.outer_fold_id != calibration.outer_fold_id:
        raise ValueError("target execution belongs to the wrong outer fold")
    if case.execution_id in set(calibration.source_execution_ids):
        raise ValueError("target execution overlaps fit or calibration source")
    if case.session_id in set(calibration.source_session_ids):
        raise ValueError("target session overlaps fit or calibration source")

    score = score_execution_block_case(
        case,
        scale_a=calibration.nll_scale_a,
        floor_b_m2=calibration.nll_floor_b_m2,
    )
    coordinate_valid = np.repeat(case.valid[:, :, None], 3, axis=2)
    residual = case.mean_m - case.truth_m
    variance = calibration.nll_scale_a * case.variance_m2 + calibration.nll_floor_b_m2
    half_width = calibration.threshold * np.sqrt(variance)
    covered = np.abs(residual) <= half_width
    selected_covered = covered[coordinate_valid]
    selected_residual = residual[coordinate_valid]
    vectors = residual[case.valid]
    selected_half_width = half_width[coordinate_valid]
    return {
        "execution_id": case.execution_id,
        "session_id": case.session_id,
        "outer_fold_id": case.outer_fold_id,
        "action_id": case.action_id,
        "contact_region_id": case.contact_region_id,
        "score": score.as_dict(),
        "threshold": float(calibration.threshold),
        "execution_block_covered": bool(np.all(selected_covered)),
        "coordinate_count": int(len(selected_covered)),
        "covered_coordinate_count": int(np.sum(selected_covered)),
        "coordinate_coverage_diagnostic": float(np.mean(selected_covered)),
        "coordinate_rmse_m": float(np.sqrt(np.mean(np.square(selected_residual)))),
        "track_error_m": float(np.mean(np.linalg.norm(vectors, axis=1))),
        "mean_interval_width_m": float(np.mean(2.0 * selected_half_width)),
        "maximum_interval_width_m": float(np.max(2.0 * selected_half_width)),
    }


def evaluate_execution_block_cases(
    target_cases: Sequence[ExecutionBlockCalibrationCase],
    calibration: ExecutionBlockConformalCalibration,
) -> dict[str, Any]:
    """Evaluate a locked target fold and report block-level marginal coverage."""

    targets = _canonical_cases(target_cases, role="target")
    if targets[0].outer_fold_id != calibration.outer_fold_id:
        raise ValueError("target cases belong to the wrong outer fold")
    source_executions = set(calibration.source_execution_ids)
    source_sessions = set(calibration.source_session_ids)
    overlap_executions = source_executions & {case.execution_id for case in targets}
    overlap_sessions = source_sessions & {case.session_id for case in targets}
    if overlap_executions:
        raise ValueError(
            "target executions overlap source calibration: "
            + ", ".join(sorted(overlap_executions))
        )
    if overlap_sessions:
        raise ValueError(
            "target sessions overlap source calibration: "
            + ", ".join(sorted(overlap_sessions))
        )

    cases = [evaluate_execution_block_case(case, calibration) for case in targets]
    coordinate_count = sum(value["coordinate_count"] for value in cases)
    covered_coordinate_count = sum(value["covered_coordinate_count"] for value in cases)
    return {
        "schema_version": 1,
        "evaluation": "causal4d_execution_block_split_conformal_v1",
        "calibration_id": calibration.calibration_id,
        "outer_fold_id": calibration.outer_fold_id,
        "confidence_level": float(calibration.confidence_level),
        "score_kind": calibration.score_kind,
        "calibration_unit": calibration.calibration_unit,
        "target_labels_used_for_calibration": False,
        "target_execution_count": len(cases),
        "execution_block_coverage": float(
            np.mean([value["execution_block_covered"] for value in cases])
        ),
        "point_coordinate_coverage_diagnostic": float(
            covered_coordinate_count / coordinate_count
        ),
        "cases": cases,
        "claim_boundary": (
            "Marginal execution-block split-conformal coverage is interpretable only "
            "under the locked independent-session exchangeability condition. Pointwise "
            "and worst-group guarantees are not claimed."
        ),
        "worst_group_coverage_guarantee_claimed": False,
        "pooled_coordinate_conformal_claimed": False,
    }
