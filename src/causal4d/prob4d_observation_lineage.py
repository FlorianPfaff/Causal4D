"""Compatibility import for strict Prob4D observation validation.

The implementation lives in :mod:`prob4d_observation_contract`; this module name
is retained for frozen imports and for the lightweight observation-lineage API.
"""

from .prob4d_observation_contract import (
    PROB4D_CAUSAL_LINEAGE_VERSION,
    PROB4D_CAUSAL_STREAM_ID,
    PROB4D_FIXED_LAG_GAUGE_MODEL,
    PROB4D_GAUGE_FACTOR_NAMES,
    PROB4D_JOINT_GAUGE_FACTOR_PREFIX,
    PROB4D_JOINT_GAUGE_MODEL,
    PROB4D_LEGACY_GAUGE_FACTOR_NAMES,
    PROB4D_SOURCE_REPOSITORY,
    is_prob4d_causal_observation_descriptor,
    validate_prob4d_causal_observation_metadata,
)

__all__ = [
    "PROB4D_CAUSAL_LINEAGE_VERSION",
    "PROB4D_CAUSAL_STREAM_ID",
    "PROB4D_FIXED_LAG_GAUGE_MODEL",
    "PROB4D_GAUGE_FACTOR_NAMES",
    "PROB4D_JOINT_GAUGE_FACTOR_PREFIX",
    "PROB4D_JOINT_GAUGE_MODEL",
    "PROB4D_LEGACY_GAUGE_FACTOR_NAMES",
    "PROB4D_SOURCE_REPOSITORY",
    "is_prob4d_causal_observation_descriptor",
    "validate_prob4d_causal_observation_metadata",
]
