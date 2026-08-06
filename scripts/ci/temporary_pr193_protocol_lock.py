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


def _patch_verifier(root: Path) -> None:
    path = root / "src/causal4d/real_result_source_verification.py"
    text = path.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        "from typing import Any, Final, Protocol\n",
        "from typing import Any, Final, Protocol, cast\n",
        name="verifier cast import",
    )
    text = _replace_once(
        text,
        '''class RealResultSourceBinding(Protocol):
    """Minimum provenance identity needed to verify registered analysis sources."""

    protocol_id: str
    protocol_design_sha256: str
    preacquisition_amendment_sha256: str
    method_freeze_sha256: str
    analysis_manifest_sha256: str
''',
        '''class RealResultSourceBinding(Protocol):
    """Minimum provenance identity needed to verify registered analysis sources."""

    @property
    def protocol_id(self) -> str: ...

    @property
    def protocol_design_sha256(self) -> str: ...

    @property
    def preacquisition_amendment_sha256(self) -> str: ...

    @property
    def method_freeze_sha256(self) -> str: ...

    @property
    def analysis_manifest_sha256(self) -> str: ...
''',
        name="read-only source binding protocol",
    )
    text = _replace_once(
        text,
        '''    protocol = payload.get("protocol")
    _require(isinstance(protocol, Mapping), "method freeze lacks protocol provenance")
    _require(
        protocol.get("design_sha256") == binding.protocol_design_sha256,
''',
        '''    protocol_value = payload.get("protocol")
    _require(
        isinstance(protocol_value, Mapping),
        "method freeze lacks protocol provenance",
    )
    protocol = cast(Mapping[str, Any], protocol_value)
    _require(
        protocol.get("design_sha256") == binding.protocol_design_sha256,
''',
        name="method-freeze protocol narrowing",
    )
    text = _replace_once(
        text,
        '''    preacquisition = payload.get("preacquisition")
    _require(
        isinstance(preacquisition, Mapping),
        "method freeze lacks pre-acquisition provenance",
    )
    _require(
        preacquisition.get("amendment_sha256")
''',
        '''    preacquisition_value = payload.get("preacquisition")
    _require(
        isinstance(preacquisition_value, Mapping),
        "method freeze lacks pre-acquisition provenance",
    )
    preacquisition = cast(Mapping[str, Any], preacquisition_value)
    _require(
        preacquisition.get("amendment_sha256")
''',
        name="preacquisition narrowing",
    )
    text = _replace_once(
        text,
        '''    analysis = payload.get("analysis_contract")
    _require(isinstance(analysis, Mapping), "method freeze lacks an analysis contract")
    _require(
        analysis.get("target_outcomes_may_select_method_or_hyperparameters") is False,
''',
        '''    analysis_value = payload.get("analysis_contract")
    _require(
        isinstance(analysis_value, Mapping),
        "method freeze lacks an analysis contract",
    )
    analysis = cast(Mapping[str, Any], analysis_value)
    _require(
        analysis.get("target_outcomes_may_select_method_or_hyperparameters") is False,
''',
        name="analysis contract narrowing",
    )
    text = _replace_once(
        text,
        '''    for field in ("method_freeze", "registered_analysis_manifest"):
        descriptor = payload.get(field)
        _require(isinstance(descriptor, Mapping), f"{field} descriptor is missing")
        digest = descriptor.get("sha256")
''',
        '''    for field in ("method_freeze", "registered_analysis_manifest"):
        descriptor_value = payload.get(field)
        _require(
            isinstance(descriptor_value, Mapping),
            f"{field} descriptor is missing",
        )
        descriptor = cast(Mapping[str, Any], descriptor_value)
        digest = descriptor.get("sha256")
''',
        name="source descriptor narrowing",
    )
    path.write_text(text, encoding="utf-8")


def _patch_source(root: Path) -> None:
    path = root / "src/causal4d/real_analysis_reporting.py"
    text = path.read_text(encoding="utf-8")
    text = _replace_once(
        text,
        "from typing import Any, Literal\n",
        "from typing import Any, Literal, cast\n",
        name="reporting cast import",
    )
    text = _replace_once(
        text,
        "import numpy as np\n\n",
        "import numpy as np\nfrom numpy.typing import NDArray\n\n",
        name="NumPy array typing import",
    )
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
        '''def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a JSON object")
    _require(all(type(key) is str for key in value), f"{name} keys must be strings")
    return value


''',
        '''def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a JSON object")
    _require(all(type(key) is str for key in value), f"{name} keys must be strings")
    return cast(Mapping[str, Any], value)


def _json_array(value: Any, *, name: str) -> list[Any]:
    _require(isinstance(value, list), f"{name} must be a JSON array")
    return cast(list[Any], value)


''',
        name="JSON narrowing helpers",
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
        '''    raw_executions = protocol.get("executions")
    _require(isinstance(raw_executions, list), "protocol executions must be an array")
    executions: dict[str, Mapping[str, Any]] = {}
''',
        '''    raw_executions = _json_array(
        protocol.get("executions"),
        name="protocol executions",
    )
    executions: dict[str, Mapping[str, Any]] = {}
''',
        name="protocol executions narrowing",
    )
    text = _replace_once(
        text,
        '''    entries = splits.get(_ENDPOINT_SPLIT_KEYS[endpoint])
    _require(isinstance(entries, list), "endpoint split must be an array")
    units: list[dict[str, Any]] = []
''',
        '''    entries = _json_array(
        splits.get(_ENDPOINT_SPLIT_KEYS[endpoint]),
        name="endpoint split",
    )
    units: list[dict[str, Any]] = []
''',
        name="endpoint split narrowing",
    )
    text = _replace_once(
        text,
        "    array = np.asarray(values, dtype=float)\n",
        "    array: NDArray[np.float64] = np.asarray(values, dtype=np.float64)\n",
        name="summary array annotation",
    )
    text = _replace_once(
        text,
        "    array = np.asarray(values, dtype=float)\n",
        "    array: NDArray[np.float64] = np.asarray(values, dtype=np.float64)\n",
        name="bootstrap array annotation",
    )
    text = _replace_once(
        text,
        '''    x = np.asarray([indices[session] for session in sessions], dtype=float)
    y = np.asarray([effects[session] for session in sessions], dtype=float)
''',
        '''    x: NDArray[np.float64] = np.asarray(
        [indices[session] for session in sessions],
        dtype=np.float64,
    )
    y: NDArray[np.float64] = np.asarray(
        [effects[session] for session in sessions],
        dtype=np.float64,
    )
''',
        name="drift array annotations",
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
    text = _replace_once(
        text,
        '''    raw_cases = evaluation.get("cases")
    _require(
        isinstance(raw_cases, list) and bool(raw_cases),
        "evaluation cases missing",
    )
    cases = [_mapping(value, name="evaluation case") for value in raw_cases]
''',
        '''    raw_cases = _json_array(evaluation.get("cases"), name="evaluation cases")
    _require(bool(raw_cases), "evaluation cases missing")
    cases = [_mapping(value, name="evaluation case") for value in raw_cases]
''',
        name="evaluation cases narrowing",
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
    _patch_verifier(root)
    _patch_source(root)
    _patch_docs(root)
    _patch_tests(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
