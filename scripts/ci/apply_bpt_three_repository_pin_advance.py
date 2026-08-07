#!/usr/bin/env python3
"""Synchronize the three-repository workflow with the reviewed BPT pin."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "bayesian-phystwin-provider-compatibility.yml"
PIN = ROOT / "requirements" / "ci" / "bayesian-phystwin-three-repository.sha"

OLD_REVISION = "68eaf4f4daaf7066f56f3ef8a4add65a5ae70059"
NEW_REVISION = "4832d92699129b70ffa902ee5ced07bc7200de78"


def main() -> None:
    pin = PIN.read_text(encoding="utf-8").strip()
    if pin != NEW_REVISION:
        raise SystemExit(
            "three-repository pin does not contain the reviewed BayesianPhysTwin "
            f"revision: {pin!r}"
        )

    text = WORKFLOW.read_text(encoding="utf-8")
    old_count = text.count(OLD_REVISION)
    new_count = text.count(NEW_REVISION)
    if old_count != 1 or new_count != 0:
        raise SystemExit(
            "expected exactly one old workflow revision and no new revision; "
            f"old_count={old_count}, new_count={new_count}"
        )
    WORKFLOW.write_text(text.replace(OLD_REVISION, NEW_REVISION), encoding="utf-8")


if __name__ == "__main__":
    main()
