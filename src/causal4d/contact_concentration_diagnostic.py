"""Fresh-seed diagnostic for contact-posterior concentration.

The frozen latent-contact estimator uses a multiplicative log-weight scale named
``posterior_temperature``. Its registered candidates are all at least one and can
therefore preserve or sharpen a posterior, but cannot soften it. This module keeps
that method immutable and compares the registered candidate set with a separately
labelled expanded set containing sub-unit logit scales. Every scale is selected on
source topologies only and evaluated once on fresh held-out seeds.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from causal4d.baselines import fit_baselines
from causal4d.benchmark import (
    CounterfactualBenchmarkConfig,
    Episode,
    build_protocol,
    generate_episodes,
    make_parameter_grid,
)
from causal4d.contact_evaluation import (
    FoldCalibration,
    _FittedObject,
    _calibrate_fold,
)
from causal4d.contact_inference import (
    ContactRolloutBank,
    GraphContactHypothesisModel,
    LatentContactConfig,
    build_rollout_bank,
    fit_contact_prior,
    true_contact_state,
)
from causal4d.contact_metrics import contact_recovery_metrics


@dataclass(frozen=True)
class ConcentrationPolicy:
    """One predeclared set of multiplicative log-weight candidates."""

    name: str
    logit_scales: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("concentration policy name must be nonempty")
        if not self.logit_scales:
            raise ValueError("concentration policy requires candidate scales")
        values = np.asarray(self.logit_scales, dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("logit scales must be finite and positive")
        if len(set(map(float, values))) != len(values):
            raise ValueError("logit scales must be unique")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "logit_scales": list(map(float, self.logit_scales)),
        }


@dataclass(frozen=True)
class _CalibrationCase:
    bank: ContactRolloutBank
    episode: Episode
    observations: np.ndarray


def scale_probability_weights(
    weights: np.ndarray,
    logit_scale: float,
) -> np.ndarray:
    """Scale normalized log weights while preserving exact zero support."""

    values = np.asarray(weights, dtype=float)
    scale = float(logit_scale)
    if values.size == 0:
        raise ValueError("probability weights must be nonempty")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("probability weights must be finite and nonnegative")
    if not np.isclose(np.sum(values), 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("probability weights must sum to one")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("logit_scale must be finite and positive")
    if scale == 1.0:
        return values.copy()

    positive = values > 0.0
    if not np.any(positive):
        raise ValueError("probability weights must contain positive support")
    scaled = np.zeros_like(values)
    log_weights = scale * np.log(values[positive])
    log_weights -= float(np.max(log_weights))
    scaled[positive] = np.exp(log_weights)
    scaled /= np.sum(scaled)
    return scaled


def _fit_objects(
    seed: int,
    benchmark_config: CounterfactualBenchmarkConfig,
) -> tuple[_FittedObject, ...]:
    fitted: list[_FittedObject] = []
    for object_index, protocol in enumerate(build_protocol(benchmark_config)):
        training, validation, held_out = generate_episodes(
            protocol,
            benchmark_config,
            seed=seed * 10_000 + object_index * 101,
        )
        baselines = fit_baselines(
            training,
            validation,
            make_parameter_grid(protocol.graph_object, benchmark_config),
            benchmark_config,
        )
        fitted.append(
            _FittedObject(
                protocol=protocol,
                training=training,
                validation=validation,
                held_out=held_out,
                baselines=baselines,
            )
        )
    return tuple(fitted)


def _calibration_cases(
    sources: Sequence[_FittedObject],
    model: GraphContactHypothesisModel,
    benchmark_config: CounterfactualBenchmarkConfig,
    contact_config: LatentContactConfig,
    *,
    calibration_seed: int,
) -> tuple[_CalibrationCase, ...]:
    output: list[_CalibrationCase] = []
    for source_index, source in enumerate(sources):
        bank = build_rollout_bank(
            source.protocol.graph_object,
            source.protocol.test_action,
            source.baselines.physics.posterior,
            model,
            simulator_config=benchmark_config.simulator,
            parameter_particle_count=contact_config.parameter_particle_count,
            variance_floor_m2=benchmark_config.predictive_variance_floor_m2,
            confidence_level=contact_config.confidence_level,
        )
        for condition_index, episode in enumerate(source.held_out):
            rng = np.random.default_rng(
                calibration_seed + source_index * 10_007 + condition_index * 101
            )
            observations = episode.truth + rng.normal(
                scale=contact_config.observation_noise_std_m,
                size=episode.truth.shape,
            )
            output.append(
                _CalibrationCase(
                    bank=bank,
                    episode=episode,
                    observations=observations,
                )
            )
    if not output:
        raise ValueError("concentration calibration requires source cases")
    return tuple(output)


def _raw_joint_weights(
    case: _CalibrationCase,
    calibration: FoldCalibration,
    *,
    prefix_frame_count: int,
) -> np.ndarray:
    return case.bank.update_weights(
        case.observations,
        prefix_frame_count=prefix_frame_count,
        likelihood_scale_m=calibration.likelihood_scale_m,
        likelihood_power=calibration.likelihood_power,
        dynamic_likelihood_weight=calibration.dynamic_likelihood_weight,
    )


def _candidate_scores(
    cases: Sequence[_CalibrationCase],
    calibration: FoldCalibration,
    contact_config: LatentContactConfig,
    candidates: Sequence[float],
) -> list[dict[str, Any]]:
    prefix = contact_config.prefix_frame_count(cases[0].bank.action.frame_count)
    prepared: list[tuple[_CalibrationCase, np.ndarray]] = [
        (
            case,
            _raw_joint_weights(
                case,
                calibration,
                prefix_frame_count=prefix,
            ),
        )
        for case in cases
    ]
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        recovery_rows: list[dict[str, Any]] = []
        for case, raw_weights in prepared:
            joint_weights = scale_probability_weights(raw_weights, candidate)
            recovery_rows.append(
                contact_recovery_metrics(
                    case.bank.contact_states,
                    case.bank.contact_marginal(joint_weights),
                    true_contact_state(
                        case.bank.graph_object,
                        case.episode.action,
                        case.episode.condition,
                    ),
                    confidence_level=contact_config.confidence_level,
                )
            )
        accuracy = float(np.mean([float(row["node_correct"]) for row in recovery_rows]))
        confidence = float(
            np.mean([float(row["node_confidence"]) for row in recovery_rows])
        )
        output.append(
            {
                "logit_scale": float(candidate),
                "source_case_count": len(recovery_rows),
                "mean_node_brier": float(
                    np.mean([float(row["node_brier"]) for row in recovery_rows])
                ),
                "mean_node_truth_probability": float(
                    np.mean(
                        [float(row["node_truth_probability"]) for row in recovery_rows]
                    )
                ),
                "node_accuracy": accuracy,
                "mean_node_confidence": confidence,
                "node_calibration_error": abs(confidence - accuracy),
                "node_credible_coverage": float(
                    np.mean(
                        [float(row["node_credible_covered"]) for row in recovery_rows]
                    )
                ),
            }
        )
    return output


def _select_policy_scale(
    cases: Sequence[_CalibrationCase],
    calibration: FoldCalibration,
    contact_config: LatentContactConfig,
    policy: ConcentrationPolicy,
) -> tuple[float, list[dict[str, Any]]]:
    scores = _candidate_scores(
        cases,
        calibration,
        contact_config,
        policy.logit_scales,
    )
    selected_index = min(
        range(len(scores)),
        key=lambda index: float(scores[index]["mean_node_brier"]),
    )
    return float(scores[selected_index]["logit_scale"]), scores


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty concentration panel")
    accuracy = float(np.mean([float(row["node_correct"]) for row in rows]))
    confidence = float(np.mean([float(row["node_confidence"]) for row in rows]))
    return {
        "case_count": len(rows),
        "node_accuracy": accuracy,
        "mean_node_confidence": confidence,
        "node_calibration_error": abs(confidence - accuracy),
        "mean_node_truth_probability": float(
            np.mean([float(row["node_truth_probability"]) for row in rows])
        ),
        "mean_node_brier": float(np.mean([float(row["node_brier"]) for row in rows])),
        "node_credible_coverage": float(
            np.mean([float(row["node_credible_covered"]) for row in rows])
        ),
        "mean_trajectory_rmse_m": float(
            np.mean([float(row["trajectory_rmse_m"]) for row in rows])
        ),
    }


def _grouped_aggregates(
    rows: Sequence[dict[str, Any]],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    keys = sorted({tuple(str(row[field]) for field in fields) for row in rows})
    output: list[dict[str, Any]] = []
    for key in keys:
        selected = [
            row for row in rows if tuple(str(row[field]) for field in fields) == key
        ]
        output.append(
            {
                **dict(zip(fields, key, strict=True)),
                **_aggregate(selected),
            }
        )
    return output


def _policy_comparison(
    aggregate: Sequence[dict[str, Any]],
    *,
    registered_policy: str,
    expanded_policy: str,
) -> list[dict[str, Any]]:
    worlds = sorted({str(row["world_condition"]) for row in aggregate})
    output: list[dict[str, Any]] = []
    for world in worlds:
        registered = next(
            row
            for row in aggregate
            if row["policy"] == registered_policy and row["world_condition"] == world
        )
        expanded = next(
            row
            for row in aggregate
            if row["policy"] == expanded_policy and row["world_condition"] == world
        )
        output.append(
            {
                "world_condition": world,
                "expanded_minus_registered_node_accuracy": (
                    expanded["node_accuracy"] - registered["node_accuracy"]
                ),
                "expanded_minus_registered_calibration_error": (
                    expanded["node_calibration_error"]
                    - registered["node_calibration_error"]
                ),
                "expanded_minus_registered_mean_brier": (
                    expanded["mean_node_brier"] - registered["mean_node_brier"]
                ),
                "expanded_minus_registered_trajectory_rmse_m": (
                    expanded["mean_trajectory_rmse_m"]
                    - registered["mean_trajectory_rmse_m"]
                ),
            }
        )
    return output


def run_contact_concentration_diagnostic(
    seeds: Sequence[int],
    *,
    benchmark_config: CounterfactualBenchmarkConfig | None = None,
    contact_config: LatentContactConfig | None = None,
    softening_logit_scales: Sequence[float] = (0.25, 0.50, 0.75),
) -> dict[str, Any]:
    """Run source-only scale selection and fresh held-out evaluation."""

    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be a nonempty unique sequence")
    if any(seed < 0 for seed in seed_values):
        raise ValueError("seeds must be nonnegative")
    benchmark = benchmark_config or CounterfactualBenchmarkConfig()
    contact = contact_config or LatentContactConfig()

    registered_scales = tuple(map(float, contact.posterior_temperatures))
    softening = tuple(map(float, softening_logit_scales))
    expanded_scales = tuple(dict.fromkeys((*softening, *registered_scales)))
    registered_policy = ConcentrationPolicy(
        name="registered_candidates",
        logit_scales=registered_scales,
    )
    expanded_policy = ConcentrationPolicy(
        name="expanded_with_softening",
        logit_scales=expanded_scales,
    )
    policies = (registered_policy, expanded_policy)

    evaluation_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for seed in seed_values:
        fitted = _fit_objects(seed, benchmark)
        for target_index, target in enumerate(fitted):
            sources = tuple(
                item for index, item in enumerate(fitted) if index != target_index
            )
            prior = fit_contact_prior(
                tuple(item.protocol for item in sources),
                contact,
                action_split="test",
            )
            model = GraphContactHypothesisModel(prior=prior, config=contact)
            calibration_seed = seed * 1_000_003 + target_index * 100_003 + 17
            calibration = _calibrate_fold(
                sources,
                model,
                benchmark,
                contact,
                calibration_seed=calibration_seed,
            )
            calibration_cases = _calibration_cases(
                sources,
                model,
                benchmark,
                contact,
                calibration_seed=calibration_seed,
            )
            selected_scales: dict[str, float] = {}
            for policy in policies:
                selected_scale, scores = _select_policy_scale(
                    calibration_cases,
                    calibration,
                    contact,
                    policy,
                )
                selected_scales[policy.name] = selected_scale
                selection_rows.append(
                    {
                        "seed": seed,
                        "held_out_object": target.protocol.graph_object.name,
                        "source_objects": ";".join(
                            item.protocol.graph_object.name for item in sources
                        ),
                        "policy": policy.name,
                        "selected_logit_scale": selected_scale,
                        "registered_selected_logit_scale": (
                            calibration.posterior_temperature
                        ),
                        "matches_registered_selection": bool(
                            policy.name != registered_policy.name
                            or np.isclose(
                                selected_scale,
                                calibration.posterior_temperature,
                                rtol=0.0,
                                atol=1e-15,
                            )
                        ),
                        "source_only_selection": True,
                        "target_outcomes_read_during_selection": False,
                        "candidate_scores": json.dumps(
                            scores,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                    }
                )
            if not np.isclose(
                selected_scales[registered_policy.name],
                calibration.posterior_temperature,
                rtol=0.0,
                atol=1e-15,
            ):
                raise RuntimeError(
                    "registered concentration selection was not reproduced"
                )

            bank = build_rollout_bank(
                target.protocol.graph_object,
                target.protocol.test_action,
                target.baselines.physics.posterior,
                model,
                simulator_config=benchmark.simulator,
                parameter_particle_count=contact.parameter_particle_count,
                variance_floor_m2=benchmark.predictive_variance_floor_m2,
                confidence_level=contact.confidence_level,
            )
            prefix = contact.prefix_frame_count(benchmark.frame_count)
            for condition_index, episode in enumerate(target.held_out):
                rng = np.random.default_rng(
                    seed * 1_000_003 + target_index * 10_007 + condition_index * 97
                )
                observations = episode.truth + rng.normal(
                    scale=contact.observation_noise_std_m,
                    size=episode.truth.shape,
                )
                raw_weights = bank.update_weights(
                    observations,
                    prefix_frame_count=prefix,
                    likelihood_scale_m=calibration.likelihood_scale_m,
                    likelihood_power=calibration.likelihood_power,
                    dynamic_likelihood_weight=(calibration.dynamic_likelihood_weight),
                )
                truth = true_contact_state(
                    target.protocol.graph_object,
                    episode.action,
                    episode.condition,
                )
                for policy in policies:
                    joint_weights = scale_probability_weights(
                        raw_weights,
                        selected_scales[policy.name],
                    )
                    recovery = contact_recovery_metrics(
                        bank.contact_states,
                        bank.contact_marginal(joint_weights),
                        truth,
                        confidence_level=contact.confidence_level,
                    )
                    prediction = bank.predictive_distribution(
                        joint_weights,
                        method=f"latent_contact_{policy.name}",
                        include_intervals=False,
                    )
                    trajectory_rmse = float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    prediction.mean[prefix:] - episode.truth[prefix:]
                                )
                            )
                        )
                    )
                    evaluation_rows.append(
                        {
                            "seed": seed,
                            "object": target.protocol.graph_object.name,
                            "source_objects": ";".join(
                                item.protocol.graph_object.name for item in sources
                            ),
                            "world_condition": episode.condition.name,
                            "policy": policy.name,
                            "selected_logit_scale": selected_scales[policy.name],
                            "forecast_start_frame": prefix,
                            "node_correct": recovery["node_correct"],
                            "node_confidence": recovery["node_confidence"],
                            "node_truth_probability": (
                                recovery["node_truth_probability"]
                            ),
                            "node_brier": recovery["node_brier"],
                            "node_credible_covered": (
                                recovery["node_credible_covered"]
                            ),
                            "trajectory_rmse_m": trajectory_rmse,
                        }
                    )

    aggregate = _grouped_aggregates(
        evaluation_rows,
        ("policy", "world_condition"),
    )
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DContactConcentrationDiagnostic",
        "seeds": list(seed_values),
        "benchmark_config": benchmark.as_dict(),
        "contact_config": contact.as_dict(),
        "policies": [policy.as_dict() for policy in policies],
        "selection_rows": selection_rows,
        "aggregate": aggregate,
        "by_topology": _grouped_aggregates(
            evaluation_rows,
            ("policy", "world_condition", "object"),
        ),
        "comparison": _policy_comparison(
            aggregate,
            registered_policy=registered_policy.name,
            expanded_policy=expanded_policy.name,
        ),
        "rows": evaluation_rows,
        "claim_boundary": (
            "Exploratory fresh-seed concentration diagnostic only. The frozen "
            "estimator, registered candidate set, five-seed result, thresholds, "
            "and 36-execution real protocol are unchanged. Expanded candidates "
            "are selected on source topologies only and cannot revise prior "
            "evidence."
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty diagnostic artifact: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_contact_concentration_diagnostic(
    result: dict[str, Any],
    output_directory: str | Path,
) -> dict[str, str]:
    """Write JSON, row-level CSV, source-selection CSV, and a manifest."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "contact-concentration-diagnostic.json"
    rows_path = output / "contact-concentration-rows.csv"
    selection_path = output / "contact-concentration-selection.csv"
    _write_json(
        summary_path,
        {
            key: value
            for key, value in result.items()
            if key not in {"rows", "selection_rows"}
        },
    )
    _write_csv(rows_path, result["rows"])
    _write_csv(selection_path, result["selection_rows"])
    payloads = (summary_path, rows_path, selection_path)
    manifest_path = output / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "artifact_kind": "Causal4DContactConcentrationDiagnosticManifest",
            "artifacts": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in payloads
            },
        },
    )
    return {
        "summary": str(summary_path),
        "rows": str(rows_path),
        "selection": str(selection_path),
        "manifest": str(manifest_path),
    }
