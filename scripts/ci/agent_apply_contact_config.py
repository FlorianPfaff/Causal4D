#!/usr/bin/env python3
"""Apply the one-shot LatentContactConfig validation hardening patch."""

from __future__ import annotations

from pathlib import Path
import re


SOURCE = Path("src/causal4d/contact_inference.py")


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def _sub_once(text: str, pattern: str, replacement: str, *, label: str) -> str:
    updated, count = re.subn(
        pattern,
        lambda _match: replacement,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return updated


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    source = _replace_once(
        source,
        "from itertools import product\nfrom statistics import NormalDist\n",
        "from itertools import product\nfrom numbers import Integral, Real\n"
        "from statistics import NormalDist\n",
        label="numeric abstract base imports",
    )
    source = _replace_once(
        source,
        "    return weights\n\n\n@dataclass(frozen=True)\nclass LatentContactConfig:\n",
        '''    return weights


def _validated_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _validated_finite_real_values(
    values: Sequence[object],
    *,
    name: str,
) -> tuple[float, ...]:
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence") from exc
    if not items:
        raise ValueError(f"{name} must not be empty")
    normalized = tuple(
        _validated_finite_real(value, name=name) for value in items
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def _validated_integer_values(
    values: Sequence[object],
    *,
    name: str,
) -> tuple[int, ...]:
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence") from exc
    if not items:
        raise ValueError(f"{name} must not be empty")
    normalized: list[int] = []
    for value in items:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must contain only integers")
        normalized.append(int(value))
    result = tuple(normalized)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True)
class LatentContactConfig:
''',
        label="latent-contact validators",
    )
    source = _sub_once(
        source,
        r"    def __post_init__\(self\) -> None:\n.*?\n    @property\n",
        '''    def __post_init__(self) -> None:
        observation_fraction = _validated_finite_real(
            self.observation_fraction,
            name="observation_fraction",
        )
        if not 0.10 <= observation_fraction <= 0.20:
            raise ValueError("observation_fraction must be in [0.10, 0.20]")

        observation_noise = _validated_finite_real(
            self.observation_noise_std_m,
            name="observation_noise_std_m",
        )
        if observation_noise <= 0.0:
            raise ValueError("observation_noise_std_m must be positive")

        likelihood_scales = _validated_finite_real_values(
            self.likelihood_scales_m,
            name="likelihood_scales_m",
        )
        if min(likelihood_scales) <= 0.0:
            raise ValueError("likelihood_scales_m must be positive")

        dynamic_weights = _validated_finite_real_values(
            self.dynamic_likelihood_weights,
            name="dynamic_likelihood_weights",
        )
        if min(dynamic_weights) < 0.0:
            raise ValueError("dynamic_likelihood_weights must be non-negative")

        likelihood_powers = _validated_finite_real_values(
            self.likelihood_powers,
            name="likelihood_powers",
        )
        if min(likelihood_powers) <= 0.0 or max(likelihood_powers) > 1.0:
            raise ValueError("likelihood_powers must be in (0, 1]")

        temperatures = _validated_finite_real_values(
            self.posterior_temperatures,
            name="posterior_temperatures",
        )
        if min(temperatures) < 1.0:
            raise ValueError("posterior_temperatures must be at least one")

        if (
            isinstance(self.parameter_particle_count, bool)
            or not isinstance(self.parameter_particle_count, Integral)
            or int(self.parameter_particle_count) < 1
        ):
            raise ValueError("parameter_particle_count must be a positive integer")

        gains = _validated_finite_real_values(
            self.gain_values,
            name="gain_values",
        )
        if min(gains) <= 0.0:
            raise ValueError("gain_values must be positive")

        delays = _validated_integer_values(
            self.delay_values,
            name="delay_values",
        )
        if min(delays) < 0:
            raise ValueError("delay_values must be non-negative")

        slips = _validated_finite_real_values(
            self.slip_values,
            name="slip_values",
        )
        if min(slips) < 0.0 or max(slips) >= 1.0:
            raise ValueError("slip_values must be in [0, 1)")

        _validated_finite_real_values(
            self.rotation_values_deg,
            name="rotation_values_deg",
        )

        for name in (
            "node_prior_smoothing",
            "categorical_prior_smoothing",
            "gain_prior_bandwidth",
            "slip_prior_bandwidth",
            "rotation_prior_bandwidth_deg",
        ):
            value = _validated_finite_real(getattr(self, name), name=name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

        confidence = _validated_finite_real(
            self.confidence_level,
            name="confidence_level",
        )
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")

        variance_min = _validated_finite_real(
            self.variance_scale_min,
            name="variance_scale_min",
        )
        variance_max = _validated_finite_real(
            self.variance_scale_max,
            name="variance_scale_max",
        )
        if variance_min <= 0.0 or variance_max < variance_min:
            raise ValueError("invalid variance scale bounds")

        for name in (
            "gate_gap_closure",
            "gate_matched_degradation",
            "gate_coverage_tolerance",
            "gate_node_accuracy",
            "gate_node_credible_coverage",
            "gate_node_calibration_error",
            "gate_gain_coverage",
            "gate_delay_map_accuracy",
            "gate_delay_coverage",
            "gate_minimum_topology_gap_closure",
        ):
            value = _validated_finite_real(getattr(self, name), name=name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        for name in ("gate_gain_mae", "gate_delay_mae_steps"):
            value = _validated_finite_real(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

    @property
''',
        label="LatentContactConfig.__post_init__",
    )
    source = _sub_once(
        source,
        r"    def prefix_frame_count\(self, frame_count: int\) -> int:\n"
        r".*?\n\n    def as_dict",
        '''    def prefix_frame_count(self, frame_count: int) -> int:
        if isinstance(frame_count, bool) or not isinstance(frame_count, Integral):
            raise ValueError("frame_count must be an integer")
        normalized_frame_count = int(frame_count)
        if normalized_frame_count < 4:
            raise ValueError("frame_count must be at least four")
        return max(
            3,
            min(
                normalized_frame_count - 1,
                int(np.ceil(self.observation_fraction * normalized_frame_count)),
            ),
        )

    def as_dict''',
        label="LatentContactConfig.prefix_frame_count",
    )
    SOURCE.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
