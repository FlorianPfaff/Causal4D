from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# PR #230 makes these jobs independent of the former BayesianPhysTwin SSH key.
# Once credential-free, they must not execute caller-selected branch code on the
# self-hosted runner. Pull-request source remains testable in hosted contract jobs;
# operational GPU/data jobs run only from the reviewed default branch.
SELF_HOSTED_JOBS = {
    ".github/workflows/self-hosted-evaluation.yml": "evaluate",
    ".github/workflows/optional-integrations.yml": "gpu",
    ".github/workflows/deform360-contact-support.yml": "source-diagnostic",
    ".github/workflows/deform360-prefix-kinematics.yml": "source-diagnostic",
    ".github/workflows/deform360-reset-mechanics.yml": "source-diagnostic",
}


def _job_block(path: str, job_name: str) -> str:
    text = (ROOT / path).read_text(encoding="utf-8")
    marker = f"\n  {job_name}:\n"
    assert marker in text, f"{path} is missing job {job_name!r}"
    block = text.split(marker, maxsplit=1)[1]
    next_job = block.find("\n  ", 1)
    return block if next_job < 0 else block[:next_job]


def test_newly_credential_free_self_hosted_jobs_are_main_only() -> None:
    for path, job_name in SELF_HOSTED_JOBS.items():
        block = _job_block(path, job_name)
        assert "runs-on: [self-hosted" in block, (
            f"{path}:{job_name} is no longer recognized as a self-hosted job"
        )
        assert "github.ref == 'refs/heads/main'" in block, (
            f"{path}:{job_name} must reject caller-selected workflow refs before "
            "self-hosted checkout or execution"
        )


def test_main_only_guard_is_bound_to_each_job_condition() -> None:
    for path, job_name in SELF_HOSTED_JOBS.items():
        block = _job_block(path, job_name)
        runs_on = block.index("runs-on: [self-hosted")
        guard = block.index("github.ref == 'refs/heads/main'")
        # The guard must be part of job-level metadata, before steps begin. Merely
        # checking the ref in a later shell step still allocates the privileged
        # runner and permits earlier branch-controlled actions to execute.
        steps = block.index("\n    steps:")
        assert guard < runs_on < steps, (
            f"{path}:{job_name} must place the main-ref guard in the job-level "
            "condition before the self-hosted runner is allocated"
        )
