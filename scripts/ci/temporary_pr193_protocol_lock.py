"""Apply the reviewed protocol-binding follow-up to PR #193."""

from __future__ import annotations

import sys
from pathlib import Path


def _replace_once(text: str, old: str, new: str, *, name: str) -> str:
    if old not in text:
        if new in text:
            return text
        raise SystemExit(f"missing patch marker: {name}")
    return text.replace(old, new, 1)


def _patch_source(root: Path) -> None:
    path = root / "src/causal4d/real_analysis_reporting.py"
    text = path.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        "from causal4d.real_result_source_verification import "
        "verify_real_result_sources\n",
        "from causal4d.real_protocol import validate_protocol\n"
        "from causal4d.real_result_source_verification import "
        "verify_real_result_sources\n",
        name="real protocol validator import",
    )
    text = _replace_once(
        text,
        "def _registered_units(\n"
        "    protocol: Mapping[str, Any],\n"
        "    endpoint: Endpoint,\n"
        ") -> tuple[dict[str, Any], ...]:\n"
        "    _require(protocol.get(\"protocol_id\") == "
        "EXPECTED_PROTOCOL_ID, \"wrong protocol\")\n",
        "def _registered_units(\n"
        "    protocol: Mapping[str, Any],\n"
        "    endpoint: Endpoint,\n"
        ") -> tuple[dict[str, Any], ...]:\n"
        "    validate_protocol(protocol)\n"
        "    _require(protocol.get(\"protocol_id\") == "
        "EXPECTED_PROTOCOL_ID, \"wrong protocol\")\n",
        name="registered protocol validation",
    )
    text = _replace_once(
        text,
        "        \"fully_crossed\": all(\n"
        "            count > 0\n"
        "            for action_counts in counts.values()\n"
        "            for count in action_counts.values()\n"
        "        ),\n"
        "        \"condition_acquisition_timing\": timing,\n",
        "        \"fully_crossed\": all(\n"
        "            count > 0\n"
        "            for action_counts in counts.values()\n"
        "            for count in action_counts.values()\n"
        "        ),\n"
        "        \"balanced_across_actions\": all(\n"
        "            len(set(action_counts.values())) == 1\n"
        "            for action_counts in counts.values()\n"
        "        ),\n"
        "        \"condition_acquisition_timing\": timing,\n",
        name="condition balance diagnostic",
    )
    path.write_text(text, encoding="utf-8")


def _patch_docs(root: Path) -> None:
    path = root / "docs/causal4d_real_analysis_reporting.md"
    text = path.read_text(encoding="utf-8")
    old_verification = (
        "The verifier reads, hashes, and parses the same exact bytes. "
        "Duplicate JSON keys,\n"
        "non-finite JSON values, symbolic links, and concurrent file "
        "replacement cannot\n"
        "silently separate the retained digest from the validated payload.\n"
    )
    new_verification = old_verification + (
        "The registered protocol is also passed through the repository's "
        "complete\n"
        "`validate_protocol` contract. Its design SHA-256 is recomputed from "
        "the full\n"
        "content, so a modified split, execution label, acquisition order, or "
        "balance\n"
        "cannot be admitted by retaining the old embedded digest.\n"
    )
    text = _replace_once(
        text,
        old_verification,
        new_verification,
        name="protocol-verification documentation",
    )
    old_balance = (
        "The acquisition-order diagnostic cannot select exclusions or revise "
        "the primary\n"
        "result. Realization-condition summaries are descriptive because "
        "condition and\n"
        "action are not fully crossed and conditions occupy different "
        "acquisition-time\n"
        "ranges.\n"
    )
    new_balance = (
        "The acquisition-order diagnostic cannot select exclusions or revise "
        "the primary\n"
        "result. Realization-condition summaries are descriptive because all "
        "condition/action\n"
        "cells exist but are unequally replicated, and conditions occupy "
        "different\n"
        "acquisition-time ranges.\n"
    )
    text = _replace_once(
        text,
        old_balance,
        new_balance,
        name="condition-balance documentation",
    )
    path.write_text(text, encoding="utf-8")


def _patch_tests(root: Path) -> None:
    path = root / "tests/test_real_analysis_reporting.py"
    text = path.read_text(encoding="utf-8")
    old_assertion = (
        "    assert (\n"
        "        report[\"design_diagnostics\"]"
        "[\"condition_comparisons_are_descriptive_only\"]\n"
        "        is True\n"
        "    )\n"
    )
    new_assertion = old_assertion + (
        "    assert report[\"design_diagnostics\"]"
        "[\"fully_crossed\"] is True\n"
        "    assert report[\"design_diagnostics\"]"
        "[\"balanced_across_actions\"] is False\n"
    )
    text = _replace_once(
        text,
        old_assertion,
        new_assertion,
        name="design-diagnostic assertions",
    )
    addition = '''

def test_reporting_recomputes_the_complete_protocol_digest(
    tmp_path: Path,
) -> None:
    freeze, analysis, freeze_sha, analysis_sha = _source_pair(tmp_path)
    payload = _factual_payload(
        freeze_sha=freeze_sha,
        analysis_sha=analysis_sha,
    )
    effect_table = tmp_path / "effects.json"
    _write_json(effect_table, payload)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["executions"][0]["command_profile_id"] = "tampered-action"
    tampered_protocol = tmp_path / "tampered-protocol.json"
    _write_json(tampered_protocol, protocol)

    with pytest.raises(ValueError, match="design SHA-256 does not match"):
        build_real_analysis_effect_report(
            effect_table,
            tampered_protocol,
            method_freeze_path=freeze,
            analysis_manifest_path=analysis,
        )
'''
    if "test_reporting_recomputes_the_complete_protocol_digest" not in text:
        marker = "\ndef test_reporting_rejects_duplicate_keys_in_bound_analysis_source(\n"
        if marker not in text:
            raise SystemExit("missing protocol-tampering test insertion marker")
        text = text.replace(marker, addition + marker, 1)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: temporary_pr193_protocol_lock.py <target-root>")
    root = Path(sys.argv[1]).resolve()
    _patch_source(root)
    _patch_docs(root)
    _patch_tests(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
