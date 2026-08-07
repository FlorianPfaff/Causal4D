"""Claim-bearing admission for hybrid-reliability calibration artifacts.

Internal reconstruction proves that a calibration is self-consistent. Promoted
use additionally needs an independently frozen expected content identity;
otherwise a modified calibration and a recomputed checksum could remain mutually
consistent. This module supplies that separate admission boundary without
changing the exploratory loader.
"""

from __future__ import annotations

from pathlib import Path

from causal4d.hybrid_reliability import (
    HybridReliabilityCalibration,
    load_hybrid_reliability_calibration,
)


def _sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def load_claim_bearing_hybrid_reliability_calibration(
    path: str | Path,
    *,
    expected_calibration_id: str,
) -> HybridReliabilityCalibration:
    """Load one calibration and bind it to an independent frozen identity.

    ``expected_calibration_id`` must come from a protocol or source-manifest
    artifact frozen independently of the calibration file being admitted. The
    ordinary loader first validates the exact bytes, closed schema, all source
    diagnostics, every derived threshold and summary, and the embedded content
    ID. This wrapper then requires that reconstructed ID to equal the independent
    expected identity.
    """

    expected = _sha256(
        expected_calibration_id,
        name="expected_calibration_id",
    )
    calibration = load_hybrid_reliability_calibration(path)
    if calibration.calibration_id != expected:
        raise ValueError(
            "hybrid reliability calibration differs from independently frozen identity"
        )
    return calibration


__all__ = ["load_claim_bearing_hybrid_reliability_calibration"]
