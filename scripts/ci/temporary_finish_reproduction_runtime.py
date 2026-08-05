from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OLD_NAME = "deform360_source_backend_runtime_erratum_v1.json"
NEW_NAME = "deform360_source_backend_reproduction_runtime_v1.json"


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


for relative in (
    ".github/workflows/deform360-prefix-kinematics.yml",
    ".github/workflows/temporary-deform360-prefix-kinematics-evidence.yml",
):
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    if OLD_NAME not in text:
        raise RuntimeError(f"obsolete runtime filename is absent from {path}")
    path.write_text(text.replace(OLD_NAME, NEW_NAME), encoding="utf-8")

policy = ROOT / "tests" / "test_deform360_prefix_kinematics_workflow_policy.py"
text = policy.read_text(encoding="utf-8")
text = text.replace("RUNTIME_ERRATUM", "REPRODUCTION_RUNTIME")
text = text.replace(OLD_NAME, NEW_NAME)
text = text.replace(
    "test_gpu_jobs_reuse_only_the_exact_effective_runtime_lock",
    "test_gpu_jobs_use_only_the_conditional_reproduction_runtime",
)
text = text.replace(
    "assert RUNTIME_ERRATUM.is_file()",
    "assert REPRODUCTION_RUNTIME.is_file()",
)
policy.write_text(text, encoding="utf-8")

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
doc_text = doc_text.replace(
    "runtime erratum, repository revisions, result, and runtime sidecar",
    "reproduction-runtime deviation, repository revisions, result, and runtime sidecar",
)
docs.write_text(doc_text, encoding="utf-8")
