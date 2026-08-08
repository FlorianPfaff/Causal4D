from __future__ import annotations

import numpy as np
import pytest

from causal4d.functional_support_v1 import _weighted_energy_distance_m


def _direct_energy_distance(
    full: np.ndarray,
    full_weights: np.ndarray,
    reduced: np.ndarray,
    reduced_weights: np.ndarray,
) -> float:
    def pairwise(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        difference = (
            first.reshape(first.shape[0], -1)[:, None, :]
            - second.reshape(second.shape[0], -1)[None, :, :]
        )
        return np.sqrt(np.mean(np.square(difference), axis=-1))

    cross = pairwise(full, reduced)
    full_pair = pairwise(full, full)
    reduced_pair = pairwise(reduced, reduced)
    value = (
        2.0 * np.einsum("i,j,ij->", full_weights, reduced_weights, cross)
        - np.einsum("i,j,ij->", full_weights, full_weights, full_pair)
        - np.einsum(
            "i,j,ij->",
            reduced_weights,
            reduced_weights,
            reduced_pair,
        )
    )
    return max(float(value), 0.0)


def test_blockwise_energy_distance_matches_direct_definition() -> None:
    rng = np.random.default_rng(41)
    full = rng.normal(size=(7, 3, 2, 2))
    reduced = rng.normal(size=(5, 3, 2, 2))
    full_weights = rng.uniform(size=len(full))
    reduced_weights = rng.uniform(size=len(reduced))
    full_weights /= np.sum(full_weights)
    reduced_weights /= np.sum(reduced_weights)

    observed = _weighted_energy_distance_m(
        full,
        full_weights,
        reduced,
        reduced_weights,
    )
    expected = _direct_energy_distance(
        full,
        full_weights,
        reduced,
        reduced_weights,
    )

    gram_roundoff_rtol = float(np.sqrt(np.finfo(float).eps))
    assert observed == pytest.approx(
        expected,
        rel=gram_roundoff_rtol,
        abs=1e-12,
    )


def test_blockwise_energy_distance_is_zero_for_identical_support() -> None:
    rng = np.random.default_rng(17)
    support = rng.normal(size=(11, 4, 3, 2))
    weights = rng.uniform(size=len(support))
    weights /= np.sum(weights)

    assert _weighted_energy_distance_m(
        support,
        weights,
        support.copy(),
        weights.copy(),
    ) == pytest.approx(0.0, abs=1e-12)


def test_blockwise_energy_distance_crosses_internal_block_boundary() -> None:
    full = np.arange(300 * 2, dtype=float).reshape(300, 1, 1, 2) / 1000.0
    reduced = full[::3].copy()
    full_weights = np.full(len(full), 1.0 / len(full))
    reduced_weights = np.full(len(reduced), 1.0 / len(reduced))

    value = _weighted_energy_distance_m(
        full,
        full_weights,
        reduced,
        reduced_weights,
    )

    assert np.isfinite(value)
    assert value >= 0.0


def test_blockwise_energy_distance_rejects_overflow() -> None:
    full = np.full((2, 1, 1, 1), np.finfo(float).max)
    reduced = -full
    weights = np.asarray((0.5, 0.5))

    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="pairwise distances must be finite"):
            _weighted_energy_distance_m(full, weights, reduced, weights)
