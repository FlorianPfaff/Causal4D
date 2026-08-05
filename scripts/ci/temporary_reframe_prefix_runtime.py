from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(
            f"expected one patch anchor in {path}: {old[:100]!r}; "
            f"found {text.count(old)}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def replace_between(path: Path, start: str, end: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    start_index = text.find(start)
    if start_index < 0:
        raise RuntimeError(f"start marker missing in {path}: {start!r}")
    end_index = text.find(end, start_index)
    if end_index < 0:
        raise RuntimeError(f"end marker missing in {path}: {end!r}")
    path.write_text(
        text[:start_index] + replacement + text[end_index:],
        encoding="utf-8",
    )


selector = ROOT / "scripts" / "remote" / "select_deform360_prefix_kinematics_python.py"
replace_once(
    selector,
    '''_ERRATUM_PATH = (
    Path("configs")
    / "causal4d_public"
    / "deform360_source_backend_runtime_erratum_v1.json"
)
''',
    '''_REPRODUCTION_RUNTIME_PATH = (
    Path("configs")
    / "causal4d_public"
    / "deform360_source_backend_reproduction_runtime_v1.json"
)
''',
)
replace_between(
    selector,
    "    erratum_path = repository_root / _ERRATUM_PATH\n",
    "\n\ndef _expected_runtime",
    '''    contract_path = repository_root / _REPRODUCTION_RUNTIME_PATH
    contract = _strict_json_object(contract_path)
    _require_exact_fields(
        contract,
        required=frozenset(
            {
                "schema_version",
                "artifact_kind",
                "status",
                "recorded_runtime",
                "candidate_runtime",
                "evidence",
                "boundary",
                "content_sha256",
            }
        ),
        name="reproduction-runtime contract",
    )
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported reproduction-runtime schema version")
    if contract.get("artifact_kind") != (
        "Deform360SourceBackendReproductionRuntimeDeviation"
    ):
        raise ValueError("unsupported reproduction-runtime artifact kind")
    if contract.get("status") != "conditional-reproduction-runtime-deviation":
        raise ValueError("runtime deviation is not conditional reproduction evidence")
    recorded_content_sha = _require_sha256(
        contract.get("content_sha256"),
        name="reproduction-runtime content_sha256",
    )
    canonical = dict(contract)
    canonical.pop("content_sha256")
    if _canonical_sha256(canonical) != recorded_content_sha:
        raise ValueError("reproduction-runtime content checksum changed")

    recorded = contract.get("recorded_runtime")
    if not isinstance(recorded, Mapping):
        raise ValueError("recorded_runtime must be a mapping")
    _require_exact_fields(
        recorded,
        required=frozenset({"path", "sha256", "values"}),
        name="recorded runtime",
    )
    if recorded.get("path") != str(_ENVIRONMENT_PATH):
        raise ValueError("reproduction contract identifies another environment lock")
    expected_environment_sha = _require_sha256(
        recorded.get("sha256"),
        name="recorded environment sha256",
    )
    if _sha256_file(environment_path) != expected_environment_sha:
        raise ValueError("recorded source-backend environment lock changed")
    recorded_values = recorded.get("values")
    if not isinstance(recorded_values, Mapping):
        raise ValueError("recorded runtime values must be a mapping")
    _require_exact_fields(
        recorded_values,
        required=frozenset(_EXPECTED_KEYS),
        name="recorded runtime values",
    )
    for key in _EXPECTED_KEYS:
        if recorded_values.get(key) != expected[key]:
            raise ValueError(f"recorded runtime no longer matches {key}")

    candidate = contract.get("candidate_runtime")
    if not isinstance(candidate, Mapping):
        raise ValueError("candidate_runtime must be a mapping")
    _require_exact_fields(
        candidate,
        required=frozenset(_EXPECTED_KEYS),
        name="candidate runtime",
    )
    for key in _EXPECTED_KEYS:
        value = candidate.get(key)
        if type(value) is not str or not value:
            raise ValueError(f"candidate runtime {key} must be a nonempty string")
        if key != "numpy" and value != expected[key]:
            raise ValueError(f"candidate runtime unexpectedly changes {key}")
    if candidate.get("numpy") != "1.26.4":
        raise ValueError("candidate NumPy version changed")
    if candidate.get("numpy") == expected["numpy"]:
        raise ValueError("candidate runtime does not declare a NumPy deviation")

    evidence = contract.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 3:
        raise ValueError(
            "reproduction-runtime contract must contain exactly three evidence records"
        )
    command_path = _validate_file_evidence(repository_root, evidence[0])
    command_text = command_path.read_text(encoding="utf-8")
    if "/home/florianpfaff/.venvs/bpt-gpu/bin/python -m pytest" not in command_text:
        raise ValueError("runtime evidence lost the archived interpreter command")
    freeze_path = _validate_file_evidence(repository_root, evidence[1])
    freeze_lines = set(freeze_path.read_text(encoding="utf-8").splitlines())
    for line in (
        "numpy==1.26.4",
        "scipy==1.13.1",
        "torch==2.4.0+cu121",
        "warp-lang==1.15.0",
    ):
        if line not in freeze_lines:
            raise ValueError(f"runtime evidence no longer contains {line}")
    workflow_record = evidence[2]
    if not isinstance(workflow_record, Mapping):
        raise ValueError("workflow runtime evidence must be a mapping")
    _require_exact_fields(
        workflow_record,
        required=frozenset(
            {"workflow_run_id", "artifact_id", "artifact_sha256", "fact"}
        ),
        name="workflow runtime evidence",
    )
    if workflow_record.get("workflow_run_id") != 30970401038:
        raise ValueError("workflow runtime identity changed")
    if workflow_record.get("artifact_id") != 8916348471:
        raise ValueError("workflow artifact identity changed")
    _require_sha256(
        workflow_record.get("artifact_sha256"),
        name="workflow artifact sha256",
    )

    boundary = contract.get("boundary")
    expected_boundary = {
        "interpretation_permitted_only_after_zero_baseline_reproduction": True,
        "original_milestone_files_rewritten": False,
        "recorded_runtime_relabelled": False,
        "scientific_artifacts_changed": False,
        "scores_or_decisions_changed": False,
        "target_future_access_permitted": False,
        "target_prefix_access_permitted": False,
        "zero_baseline_reproduction_required": True,
    }
    if boundary != expected_boundary:
        raise ValueError("reproduction-runtime scientific boundary changed")

    candidate_runtime = {key: candidate[key] for key in _EXPECTED_KEYS}
    provenance = {
        "status": "conditional-reproduction-runtime-deviation",
        "recorded_environment_path": str(_ENVIRONMENT_PATH),
        "recorded_environment_sha256": expected_environment_sha,
        "recorded_runtime": dict(expected),
        "reproduction_runtime_contract_path": str(_REPRODUCTION_RUNTIME_PATH),
        "reproduction_runtime_contract_sha256": recorded_content_sha,
        "candidate_runtime": candidate_runtime,
        "deviation": {
            "numpy": {
                "recorded": expected["numpy"],
                "candidate": candidate_runtime["numpy"],
            }
        },
        "interpretation_permitted_only_after_zero_baseline_reproduction": True,
        "zero_baseline_reproduction_required": True,
    }
    return candidate_runtime, provenance
''',
)
selector_text = selector.read_text(encoding="utf-8")
selector_text = selector_text.replace("runtime-erratum", "reproduction-runtime")
selector.write_text(selector_text, encoding="utf-8")

runner = ROOT / "scripts" / "remote" / "run_deform360_prefix_kinematics.py"
replace_between(
    runner,
    '    provenance = payload["runtime_provenance"]\n',
    "    return payload\n",
    '''    provenance = payload["runtime_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("runtime selection omitted lock provenance")
    if provenance.get("status") != "conditional-reproduction-runtime-deviation":
        raise ValueError("runtime selection has another provenance status")
    if provenance.get("zero_baseline_reproduction_required") is not True:
        raise ValueError("runtime selection relaxed zero-baseline reproduction")
    if (
        provenance.get(
            "interpretation_permitted_only_after_zero_baseline_reproduction"
        )
        is not True
    ):
        raise ValueError("runtime selection permits premature interpretation")
    recorded = provenance.get("recorded_runtime")
    candidate = provenance.get("candidate_runtime")
    if not isinstance(recorded, Mapping) or not isinstance(candidate, Mapping):
        raise ValueError("runtime selection omitted recorded or candidate runtime")
    if dict(candidate) != dict(expected):
        raise ValueError("runtime selection candidate runtime changed")
    if recorded.get("numpy") != "2.5.1" or candidate.get("numpy") != "1.26.4":
        raise ValueError("runtime selection has another NumPy deviation")
    for key in _RUNTIME_KEYS:
        if key != "numpy" and recorded.get(key) != candidate.get(key):
            raise ValueError(f"runtime selection unexpectedly changes {key}")
    if provenance.get("deviation") != {
        "numpy": {"candidate": "1.26.4", "recorded": "2.5.1"}
    }:
        raise ValueError("runtime selection has another declared deviation")
''',
)

for relative in (
    ".github/workflows/deform360-prefix-kinematics.yml",
    ".github/workflows/temporary-deform360-prefix-kinematics-evidence.yml",
):
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "deform360_source_backend_runtime_erratum_v1.json",
        "deform360_source_backend_reproduction_runtime_v1.json",
    )
    path.write_text(text, encoding="utf-8")

selector_tests = ROOT / "tests" / "test_deform360_prefix_kinematics_python_selector.py"
replace_between(
    selector_tests,
    "def test_runtime_erratum_is_additive_and_exact() -> None:\n",
    "\n\ndef test_expected_runtime_rejects_duplicate_json_keys",
    '''def test_reproduction_runtime_deviation_is_conditional_and_exact() -> None:
    selector = _load_selector()
    original_path = ROOT / selector._ENVIRONMENT_PATH
    original = json.loads(original_path.read_text(encoding="utf-8"))

    expected, provenance = selector._load_runtime_lock(ROOT)

    assert original["numpy"] == "2.5.1"
    assert expected == {
        "python": "3.12.3",
        "numpy": "1.26.4",
        "scipy": "1.13.1",
        "torch": "2.4.0+cu121",
        "torch_cuda": "12.1",
        "warp": "1.15.0",
    }
    assert provenance["status"] == "conditional-reproduction-runtime-deviation"
    assert provenance["recorded_runtime"]["numpy"] == "2.5.1"
    assert provenance["candidate_runtime"]["numpy"] == "1.26.4"
    assert provenance["deviation"] == {
        "numpy": {"recorded": "2.5.1", "candidate": "1.26.4"}
    }
    assert provenance["zero_baseline_reproduction_required"] is True
    assert (
        provenance[
            "interpretation_permitted_only_after_zero_baseline_reproduction"
        ]
        is True
    )
    assert provenance["recorded_environment_sha256"] == (
        "2274f2a38e5b49a9e1fc5e4c49c80910d2095cf43e8b1e84928c6cc3d99b2d8c"
    )
    assert provenance["reproduction_runtime_contract_sha256"] == (
        "144ea36f828703a713ff3bc3afe49ff0518926d73544bf7b477bdf9eb5f17f98"
    )


def test_reproduction_runtime_rejects_content_identity_drift(
    tmp_path: Path,
) -> None:
    selector = _load_selector()
    for relative in (
        selector._ENVIRONMENT_PATH,
        selector._REPRODUCTION_RUNTIME_PATH,
        Path(
            "milestones/deform360-replication-source-backend-v1/verification/"
            "test-and-lint.txt"
        ),
        Path(
            "milestones/v0.3.0-causal4d-aip/environment/"
            "bpt-gpu-pip-freeze.txt"
        ),
    ):
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    contract_path = tmp_path / selector._REPRODUCTION_RUNTIME_PATH
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["candidate_runtime"]["numpy"] = "2.0.0"
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="content checksum changed"):
        selector._load_runtime_lock(tmp_path)
''',
)

runner_tests = ROOT / "tests" / "test_deform360_prefix_kinematics_runner.py"
text = runner_tests.read_text(encoding="utf-8")
text = text.replace(
    '''        "runtime_provenance": {
            "correction": {
                "numpy": {"recorded": "2.5.1", "effective": "1.26.4"}
            },
            "zero_baseline_reproduction_required": True,
        },
''',
    '''        "runtime_provenance": {
            "status": "conditional-reproduction-runtime-deviation",
            "recorded_runtime": {
                **expected,
                "numpy": "2.5.1",
            },
            "candidate_runtime": expected,
            "deviation": {
                "numpy": {"recorded": "2.5.1", "candidate": "1.26.4"}
            },
            "interpretation_permitted_only_after_zero_baseline_reproduction": True,
            "zero_baseline_reproduction_required": True,
        },
''',
)
text = text.replace(
    "def test_runner_rejects_relaxed_zero_baseline_gate(tmp_path: Path) -> None:",
    "def test_runner_rejects_relaxed_zero_baseline_gate(tmp_path: Path) -> None:",
)
runner_tests.write_text(text, encoding="utf-8")

workflow_tests = ROOT / "tests" / "test_deform360_prefix_kinematics_workflow_policy.py"
text = workflow_tests.read_text(encoding="utf-8")
text = text.replace("RUNTIME_ERRATUM", "REPRODUCTION_RUNTIME")
text = text.replace(
    "deform360_source_backend_runtime_erratum_v1.json",
    "deform360_source_backend_reproduction_runtime_v1.json",
)
text = text.replace(
    "test_gpu_jobs_reuse_only_the_exact_effective_runtime_lock",
    "test_gpu_jobs_use_only_the_conditional_reproduction_runtime",
)
workflow_tests.write_text(text, encoding="utf-8")

docs = ROOT / "docs" / "causal4d_deform360_prefix_kinematics.md"
replace_between(
    docs,
    "## Runtime provenance erratum\n",
    "## Decision gate\n",
    '''## Conditional reproduction-runtime deviation

The original source-backend milestone remains byte-for-byte unchanged. Its
`verification/environment.json` records NumPy `2.5.1`. Separate evidence shows
that the named `bpt-gpu` interpreter used by the archived validation command had
NumPy `1.26.4` in an earlier captured freeze and again when workstation2 was
probed on 5 August 2026. Those observations do not prove that the July 14
milestone record was erroneous or that the environment could not have changed
between captures.

The conditional reproduction runtime is recorded in
[`configs/causal4d_public/deform360_source_backend_reproduction_runtime_v1.json`](
../configs/causal4d_public/deform360_source_backend_reproduction_runtime_v1.json).
It preserves `2.5.1` as the recorded value and separately declares `1.26.4` as
the only candidate-runtime deviation. The contract binds the original
environment file, archived command record, earlier pip freeze, and failed
workstation2 selector artifact by SHA-256. It does not relabel or rewrite the
frozen milestone.

The runtime selector accepts the candidate environment only for this source-only
diagnostic and does not install or upgrade packages. Candidate-policy results
are interpretable only after every archived zero-velocity Chamfer and p99-strain
value reproduces within the locked tolerances. If that parity gate fails, the
workflow stops and retains the runtime deviation as an infrastructure finding;
it cannot promote or rescue a scientific result.

''',
)
doc_text = docs.read_text(encoding="utf-8")
doc_text = doc_text.replace("runtime erratum", "reproduction-runtime deviation")
docs.write_text(doc_text, encoding="utf-8")
