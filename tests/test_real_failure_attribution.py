import json

import pytest

from causal4d.cli.real_failure_attribution import main
from causal4d.real_failure_attribution import aggregate_real_failure_attribution


def _audit(
    case: str,
    *,
    nominal_track: float,
    bpt_track: float,
    causal_track: float,
    current_track: float,
    expanded_track: float,
    ceiling_track: float,
    nominal_coordinate: float,
    bpt_coordinate: float,
    causal_coordinate: float,
    current_coordinate: float,
    expanded_coordinate: float,
    ceiling_coordinate: float,
):
    def _gap_values(posterior, current, expanded, ceiling):
        gaps = {
            "inference_gap": posterior - current,
            "proposal_gap": current - expanded,
            "model_gap": expanded - ceiling,
        }
        total = posterior - ceiling
        dominant = max(gaps, key=gaps.get)
        return {
            "current_causal4d_posterior": posterior,
            "current_bank_oracle": current,
            "expanded_bank_oracle": expanded,
            "oracle_discrepancy_ceiling": ceiling,
            "capped_oracle_discrepancy_ceiling": ceiling + 0.001,
            **gaps,
            "capped_model_gap": expanded - ceiling - 0.001,
            "total_diagnostic_headroom": total,
            "capped_total_diagnostic_headroom": total - 0.001,
            "fraction_of_total_diagnostic_headroom": {
                name: value / total for name, value in gaps.items()
            },
            "fraction_of_capped_diagnostic_headroom": {
                "inference_gap": gaps["inference_gap"] / (total - 0.001),
                "proposal_gap": gaps["proposal_gap"] / (total - 0.001),
                "capped_model_gap": (gaps["model_gap"] - 0.001) / (total - 0.001),
            },
            "dominant_gap": dominant,
        }

    def _window():
        return {
            "case_id": case,
            "stream_id": "points",
            "frame_start": 0,
            "frame_stop": 10,
            "content_sha256": "0" * 64,
        }

    def _action(action_id):
        return {
            "action_id": action_id,
            "case_id": case,
            "frame_start": 0,
            "frame_stop": 10,
            "trajectory_sha256": "1" * 64,
            "provenance": "test",
        }

    contributions = {
        "theta_shapley": {
            "fraction_of_total_predictive_variance": 0.1,
        },
        "phi_shapley": {
            "fraction_of_total_predictive_variance": 0.2,
        },
        "kappa_shapley": {
            "fraction_of_total_predictive_variance": 0.1,
        },
        "unallocated_state_support": {
            "fraction_of_total_predictive_variance": 0.0,
        },
        "discrepancy_mean_epistemic": {
            "fraction_of_total_predictive_variance": 0.2,
        },
        "state_discrepancy_cross": {
            "fraction_of_total_predictive_variance": -0.1,
        },
        "discrepancy_conditional": {
            "fraction_of_total_predictive_variance": 0.4,
        },
        "conditional_simulator_observation_noise": {
            "fraction_of_total_predictive_variance": 0.1,
        },
    }
    return {
        "schema_version": 1,
        "case": case,
        "experiment": "real_oracle_gap_and_variance_audit",
        "information_boundary": {
            "protocol": {
                "start_frame": 7,
                "stop_frame": 10,
                "selection_metric": "track_error_m",
                "label_use": "diagnostic_only",
            },
            "o_plus_prefix_frame_interval": [0, 7],
            "holdout_labels_used_for_prediction": False,
            "holdout_labels_used_for_evaluation": True,
            "holdout_labels_used_for_oracle_selection": True,
            "oracle_outputs_deployable": False,
        },
        "causal_context": {
            "protocol_id": "sloth-v1",
            "o_minus": _window(),
            "o_plus": _window(),
            "u_obs": _action("u_obs"),
            "u_cf": _action("u_cf"),
        },
        "predictors": {
            "nominal_phystwin": {
                "track_error_m": nominal_track,
                "coordinate_rmse_m": nominal_coordinate,
                "label_use_for_prediction": False,
            },
            "bayesian_phystwin_mixture_nominal_z": {
                "track_error_m": bpt_track,
                "coordinate_rmse_m": bpt_coordinate,
                "label_use_for_prediction": False,
                "effective_component_count": 2.0,
            },
            "current_causal4d_posterior": {
                "track_error_m": causal_track,
                "coordinate_rmse_m": causal_coordinate,
                "label_use_for_prediction": False,
                "effective_component_count": 3.0,
            },
        },
        "current_bank_oracle": {
            "label_use": "diagnostic_only",
            "deployable": False,
            "selection_metric": "track_error_m",
            "discrepancy_cap_m": 0.01,
        },
        "expanded_bank_oracle": {
            "label_use": "diagnostic_only",
            "deployable": False,
            "selection_metric": "track_error_m",
            "discrepancy_cap_m": 0.01,
        },
        "gaps": {
            "track_error_m": _gap_values(
                causal_track,
                current_track,
                expanded_track,
                ceiling_track,
            ),
            "coordinate_rmse_m": _gap_values(
                causal_coordinate,
                current_coordinate,
                expanded_coordinate,
                ceiling_coordinate,
            ),
            "definitions": {
                "inference_gap": "posterior-current",
                "proposal_gap": "current-expanded",
                "model_gap": "expanded-ceiling",
            },
        },
        "bank_nesting": {"verified": True},
        "variance_decomposition": {
            "method": "weighted_shapley_variance_of_conditional_means",
            "all_holdout": {
                "total_predictive_variance_m2": 0.0004,
                "residual_mse_to_predictive_variance_ratio": 1.2,
                "contributions": contributions,
                "closure": {
                    "state_family_absolute_error_m2": 1e-16,
                    "readout_algebra_absolute_error_m2": 2e-16,
                    "predictive_allocation_absolute_error_m2": 3e-16,
                },
            },
        },
        "abduction_likelihood": {
            "observation_scale_m": 0.01,
            "likelihood_power": 12.0,
            "dynamic_likelihood_weight": 0.25,
            "degrees_of_freedom": 4.0,
        },
    }


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _audits(tmp_path):
    first = _audit(
        "case-a",
        nominal_track=0.05,
        bpt_track=0.04,
        causal_track=0.03,
        current_track=0.028,
        expanded_track=0.027,
        ceiling_track=0.020,
        nominal_coordinate=0.03,
        bpt_coordinate=0.025,
        causal_coordinate=0.020,
        current_coordinate=0.019,
        expanded_coordinate=0.0185,
        ceiling_coordinate=0.014,
    )
    second = _audit(
        "case-b",
        nominal_track=0.05,
        bpt_track=0.04,
        causal_track=0.045,
        current_track=0.035,
        expanded_track=0.032,
        ceiling_track=0.025,
        nominal_coordinate=0.03,
        bpt_coordinate=0.025,
        causal_coordinate=0.028,
        current_coordinate=0.023,
        expanded_coordinate=0.021,
        ceiling_coordinate=0.016,
    )
    return [
        _write(tmp_path / "case-a.json", first),
        _write(tmp_path / "case-b.json", second),
    ]


def test_aggregate_accounts_for_audits_and_exclusions(tmp_path):
    audits = _audits(tmp_path)
    output_json = tmp_path / "aggregate.json"
    output_csv = tmp_path / "aggregate.csv"

    result = aggregate_real_failure_attribution(
        audits,
        output_json,
        output_csv=output_csv,
        expected_case_ids=("case-a", "case-b", "case-c"),
        expected_protocol_id="sloth-v1",
        excluded_cases={"case-c": "failed preregistered synchronization gate"},
    )

    assert result["case_accounting"]["complete"] is True
    assert result["case_accounting"]["included_case_count"] == 2
    assert result["case_accounting"]["excluded_case_count"] == 1
    comparison = result["paired_comparisons"]["track_error_m"][
        "causal4d_vs_bayesian_phystwin_nominal_z"
    ]
    assert comparison["paired_delta_m"]["mean"] == pytest.approx(-0.0025)
    assert comparison["candidate_better_case_fraction"] == pytest.approx(0.5)
    dominant = result["diagnostic_gap_attribution"]["track_error_m"]["dominant_gap"]
    assert dominant["most_common"] == "mixed"
    assert len(result["evidence_fingerprint"]) == 64
    assert output_json.exists()
    assert len(output_csv.read_text(encoding="utf-8").splitlines()) == 3


def test_aggregate_fails_closed_on_unaccounted_expected_case(tmp_path):
    audits = _audits(tmp_path)

    with pytest.raises(ValueError, match="unaccounted"):
        aggregate_real_failure_attribution(
            audits,
            tmp_path / "aggregate.json",
            expected_case_ids=("case-a", "case-b", "case-c"),
        )


def test_aggregate_rejects_holdout_label_leakage(tmp_path):
    audit = _audit(
        "case-a",
        nominal_track=0.05,
        bpt_track=0.04,
        causal_track=0.03,
        current_track=0.028,
        expanded_track=0.027,
        ceiling_track=0.020,
        nominal_coordinate=0.03,
        bpt_coordinate=0.025,
        causal_coordinate=0.020,
        current_coordinate=0.019,
        expanded_coordinate=0.0185,
        ceiling_coordinate=0.014,
    )
    audit["predictors"]["current_causal4d_posterior"]["label_use_for_prediction"] = True
    path = _write(tmp_path / "leaky.json", audit)

    with pytest.raises(ValueError, match="must not use holdout labels"):
        aggregate_real_failure_attribution([path], tmp_path / "aggregate.json")


def test_cli_reads_protocol_and_explicit_exclusions(tmp_path):
    audits = _audits(tmp_path)
    protocol = {
        "protocol_id": "sloth-v1",
        "executions": [
            {"execution_id": "case-a"},
            {"execution_id": "case-b"},
            {"execution_id": "case-c"},
        ],
    }
    exclusions = {
        "protocol_id": "sloth-v1",
        "exclusions": [
            {"case_id": "case-c", "reason": "failed registered quality gate"}
        ],
    }
    protocol_path = _write(tmp_path / "protocol.json", protocol)
    exclusions_path = _write(tmp_path / "exclusions.json", exclusions)
    output_json = tmp_path / "aggregate.json"
    output_csv = tmp_path / "aggregate.csv"

    status = main(
        [
            str(output_json),
            str(output_csv),
            *(str(path) for path in audits),
            "--protocol-json",
            str(protocol_path),
            "--exclusions-json",
            str(exclusions_path),
        ]
    )

    assert status == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["case_accounting"]["complete"] is True


def test_aggregate_rejects_gap_arithmetic_mismatch(tmp_path):
    audit = _audits(tmp_path)
    payload = json.loads(audit[0].read_text(encoding="utf-8"))
    payload["gaps"]["track_error_m"]["inference_gap"] += 0.001
    path = _write(tmp_path / "bad-gap.json", payload)

    with pytest.raises(ValueError, match="does not close arithmetically"):
        aggregate_real_failure_attribution([path], tmp_path / "aggregate.json")


def test_aggregate_rejects_oracle_contract_mismatch(tmp_path):
    audit = _audits(tmp_path)
    payload = json.loads(audit[0].read_text(encoding="utf-8"))
    payload["expanded_bank_oracle"]["discrepancy_cap_m"] = 0.02
    path = _write(tmp_path / "bad-oracle.json", payload)

    with pytest.raises(ValueError, match="discrepancy caps disagree"):
        aggregate_real_failure_attribution([path], tmp_path / "aggregate.json")
