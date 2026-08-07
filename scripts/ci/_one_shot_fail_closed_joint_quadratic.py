#!/usr/bin/env python3
"""Fail closed when a Woodbury correction exceeds its quadratic form."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src/causal4d/joint_observation.py"
TEST = ROOT / "tests/test_joint_observation_adversarial.py"


def _replace(path: Path, old: str, new: str, *, expected_count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected_count:
        raise SystemExit(
            f"expected {expected_count} occurrence(s), found {count}: {old!r}"
        )
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    _replace(
        MODULE,
        """    return correction, log_determinant


def _joint_gaussian_log_density_dense(
""",
        """    return correction, log_determinant


def _subtract_low_rank_quadratic_correction(
    quadratic: np.ndarray,
    correction: np.ndarray,
) -> np.ndarray:
    \"\"\"Subtract a Woodbury correction without repairing inconsistent algebra.\"\"\"

    candidate = np.asarray(quadratic, dtype=float) - np.asarray(
        correction,
        dtype=float,
    )
    scale = np.maximum(np.abs(quadratic), np.abs(correction))
    tolerance = 1e-12 + 1e-10 * scale
    if not np.all(np.isfinite(candidate)):
        raise ValueError("low-rank corrected quadratic must be finite")
    if np.any(candidate < -tolerance):
        raise ValueError(
            "low-rank covariance correction exceeds the base quadratic"
        )
    return np.maximum(candidate, 0.0)


def _joint_gaussian_log_density_dense(
""",
    )
    _replace(
        MODULE,
        """        quadratic = np.maximum(quadratic - correction, 0.0)
""",
        """        quadratic = _subtract_low_rank_quadratic_correction(
            quadratic,
            correction,
        )
""",
        expected_count=2,
    )
    _replace(
        MODULE,
        """    if not np.isclose(np.sum(prior), 1.0):
        raise ValueError("prior_weights must sum to one")
""",
        """    if not np.isclose(np.sum(prior), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("prior_weights must sum to one")
""",
    )

    text = TEST.read_text(encoding="utf-8")
    if "test_low_rank_correction_cannot_exceed_base_quadratic" in text:
        raise SystemExit("joint quadratic regressions already exist")
    text = text.replace(
        """import numpy as np
import pytest

from causal4d.joint_observation import (
""",
        """import numpy as np
import pytest

import causal4d.joint_observation as joint_observation
from causal4d.joint_observation import (
""",
        1,
    )
    text = text.rstrip() + """


@pytest.mark.parametrize(
    "base_covariance",
    (
        np.eye(1),
        np.ones((1, 1, 1)),
    ),
)
def test_low_rank_correction_cannot_exceed_base_quadratic(
    base_covariance: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = LinearJointObservationEvidence(
        evidence_id="inconsistent-low-rank-correction",
        values_m=np.zeros(1),
        row_indices=np.array([0]),
        frame_indices=np.array([1]),
        node_indices=np.array([0]),
        coordinate_indices=np.array([0]),
        coefficients=np.array([1.0]),
        base_covariance_m2=base_covariance,
        shared_covariance_factor_m=np.ones((1, 1)),
        source_id="adversarial-test",
    )
    components = np.zeros((2, 2, 1, 1), dtype=float)

    def excessive_correction(
        residual: np.ndarray,
        factor: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        del factor
        shape = residual.shape[:-1]
        return np.ones(shape), np.zeros(shape)

    monkeypatch.setattr(
        joint_observation,
        "_low_rank_terms",
        excessive_correction,
    )
    with pytest.raises(ValueError, match="exceeds the base quadratic"):
        joint_observation.joint_component_log_likelihoods(
            components,
            evidence,
            prefix_frame_count=2,
        )


def test_roundoff_scale_negative_corrected_quadratic_is_zero() -> None:
    result = joint_observation._subtract_low_rank_quadratic_correction(
        np.asarray([1.0]),
        np.asarray([1.0 + 1e-13]),
    )

    np.testing.assert_array_equal(result, np.zeros(1))


def test_prior_mass_tolerance_is_absolute_not_relative() -> None:
    evidence = LinearJointObservationEvidence(
        evidence_id="strict-prior-normalization",
        values_m=np.zeros(1),
        row_indices=np.array([0]),
        frame_indices=np.array([1]),
        node_indices=np.array([0]),
        coordinate_indices=np.array([0]),
        coefficients=np.array([1.0]),
        base_covariance_m2=np.eye(1),
        source_id="adversarial-test",
    )
    components = np.zeros((2, 2, 1, 1), dtype=float)

    with pytest.raises(ValueError, match="sum to one"):
        posterior_weights_from_joint_observation(
            np.asarray([0.5, 0.500001]),
            components,
            evidence,
            prefix_frame_count=2,
        )
""" + "\n"
    TEST.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
