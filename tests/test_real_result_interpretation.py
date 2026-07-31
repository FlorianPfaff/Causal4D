from __future__ import annotations

from itertools import product
import json
from pathlib import Path

import pytest

from causal4d.cli.command_registry import find_command
from causal4d.cli.real_result_interpretation import main as interpretation_main
from causal4d.real_result_interpretation import (
    EXPECTED_PREACQUISITION_SHA256,
    EXPECTED_PROTOCOL_DESIGN_SHA256,
    EXPECTED_PROTOCOL_ID,
    RealResultGateSummary,
    interpret_real_result,
    load_real_result_interpretation,
    validate_real_result_interpretation,
    write_real_result_interpretation,
)

_DIGEST = "1" * 64


def _gates(**changes) -> RealResultGateSummary:
    values = {
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "protocol_design_sha256": EXPECTED_PROTOCOL_DESIGN_SHA256,
        "preacquisition_amendment_sha256": EXPECTED_PREACQUISITION_SHA256,
        "method_freeze_sha256": _DIGEST,
        "analysis_manifest_sha256": "2" * 64,
        "evidence_status": "complete",
        "factual_continuation": "passed",
        "same_grasp_transfer": "passed",
        "new_contact_transfer": "passed",
        "execution_block_calibration": "passed",
        "oracle_diagnosis": "not_estimable",
        "technical_failure_count": 0,
        "preregistered_exclusion_count": 0,
        "target_informed_selection": False,
    }
    values.update(changes)
    return RealResultGateSummary(**values)


def _gate_payload(**changes) -> dict[str, object]:
    gates = _gates(**changes)
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DRealResultGateSummary",
        **gates.as_dict(),
    }


def test_grouped_cli_registers_the_interpretation_contract() -> None:
    command = find_command("evidence/interpret-real-result")
    assert command.target.endswith("real_result_interpretation:main")
    assert command.lifecycle == "stable"


def test_full_chain_requires_all_primary_gates_and_calibration() -> None:
    result = interpret_real_result(_gates())
    assert result.rule_id == "full_chain_supported"
    assert result.paper_status == "positive"
    assert len(result.supported_claims) == 2
    assert "individual-level real counterfactual" in result.prohibited_claims[0]


def test_transfer_without_calibration_is_a_bounded_positive_result() -> None:
    result = interpret_real_result(_gates(execution_block_calibration="failed"))
    assert result.rule_id == "transfer_supported_calibration_limited"
    assert result.paper_status == "bounded_positive"
    assert all("calibration supports" not in claim for claim in result.supported_claims)


def test_same_grasp_success_does_not_rescue_new_contact_failure() -> None:
    result = interpret_real_result(_gates(new_contact_transfer="failed"))
    assert result.rule_id == "persistent_transfer_only"
    assert result.paper_status == "bounded_positive"
    assert "new-contact transfer fails" in result.headline


def test_factual_gain_does_not_imply_transfer() -> None:
    result = interpret_real_result(
        _gates(
            same_grasp_transfer="failed",
            new_contact_transfer="failed",
            execution_block_calibration="passed",
        )
    )
    assert result.rule_id == "factual_only"
    assert any("does not rescue" in value for value in result.required_limitations)


def test_failed_factual_gate_preempts_other_positive_results() -> None:
    result = interpret_real_result(_gates(factual_continuation="failed"))
    assert result.rule_id == "primary_chain_not_supported"
    assert result.paper_status == "negative"
    assert result.supported_claims == ()


def test_incomplete_evidence_and_target_selection_preempt_gate_values() -> None:
    incomplete = interpret_real_result(_gates(evidence_status="incomplete"))
    assert incomplete.rule_id == "incomplete_evidence"
    assert incomplete.paper_status == "incomplete"

    violated = interpret_real_result(_gates(target_informed_selection=True))
    assert violated.rule_id == "confirmatory_boundary_violated"
    assert violated.paper_status == "incomplete"


def test_oracle_and_failure_accounting_are_mandatory_limitations() -> None:
    result = interpret_real_result(
        _gates(
            oracle_diagnosis="model_discrepancy_dominant",
            technical_failure_count=2,
            preregistered_exclusion_count=3,
        )
    )
    limitations = " ".join(result.required_limitations)
    assert "model discrepancy" in limitations
    assert "2 technical failures" in limitations
    assert "3 preregistered exclusions" in limitations


def test_every_gate_combination_has_a_deterministic_interpretation() -> None:
    statuses = ("passed", "failed", "not_estimable")
    for values in product(statuses, repeat=4):
        gates = _gates(
            factual_continuation=values[0],
            same_grasp_transfer=values[1],
            new_contact_transfer=values[2],
            execution_block_calibration=values[3],
        )
        first = interpret_real_result(gates)
        second = interpret_real_result(gates)
        assert first.as_dict() == second.as_dict()
        assert len(first.result_sha256) == 64


def test_interpretation_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    result = interpret_real_result(_gates())
    path = tmp_path / "interpretation.json"
    write_real_result_interpretation(path, result)
    restored = load_real_result_interpretation(path)
    assert restored.as_dict() == result.as_dict()

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["headline"] = "broader claim"
    with pytest.raises(ValueError, match="differs from the locked decision tree"):
        validate_real_result_interpretation(tampered)

    with pytest.raises(FileExistsError):
        write_real_result_interpretation(path, result)


def test_cli_writes_artifact_and_can_fail_closed_on_incomplete(
    tmp_path: Path,
    capsys,
) -> None:
    input_path = tmp_path / "gates.json"
    output_path = tmp_path / "interpretation.json"
    input_path.write_text(
        json.dumps(_gate_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert interpretation_main([str(input_path), str(output_path)]) == 0
    assert load_real_result_interpretation(output_path).paper_status == "positive"
    assert json.loads(capsys.readouterr().out)["rule_id"] == "full_chain_supported"

    incomplete_input = tmp_path / "incomplete.json"
    incomplete_output = tmp_path / "incomplete-result.json"
    incomplete_input.write_text(
        json.dumps(
            _gate_payload(evidence_status="incomplete"),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        interpretation_main(
            [
                str(incomplete_input),
                str(incomplete_output),
                "--require-complete",
            ]
        )
        == 3
    )


def test_checked_in_template_cannot_pass_as_evidence() -> None:
    template_path = (
        Path(__file__).parents[1]
        / "configs"
        / "causal4d"
        / "real_result_gate_summary_v1.template.json"
    )
    payload = json.loads(template_path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="artifact kind"):
        RealResultGateSummary.from_dict(payload)


def test_gate_summary_rejects_wrong_frozen_protocol_identity() -> None:
    with pytest.raises(ValueError, match="locked design"):
        _gates(protocol_design_sha256="0" * 64)
    with pytest.raises(ValueError, match="locked v4 amendment"):
        _gates(preacquisition_amendment_sha256="0" * 64)
