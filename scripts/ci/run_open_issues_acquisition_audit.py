#!/usr/bin/env python3
"""Run a read-only audit of the persistent Causal4D acquisition workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


ALLOWED_STATUS_CODES = {0, 3}


class StrictJSONError(ValueError):
    """Raised when a status artifact is not strict JSON."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise StrictJSONError(f"non-finite JSON constant: {value}")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (OSError, json.JSONDecodeError, StrictJSONError):
        return None
    return value if isinstance(value, dict) else None


def _project(value: dict[str, Any] | None, keys: Iterable[str]) -> dict[str, Any] | None:
    if value is None:
        return None
    return {key: value[key] for key in keys if key in value}


def _path_info(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "configured_path": str(path),
        "exists": path.exists(),
        "is_directory": path.is_dir(),
        "is_symlink": path.is_symlink(),
    }
    if path.is_dir() and not path.is_symlink():
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            result["git_commit"] = completed.stdout.strip()
    return result


def _tree_snapshot(root: Path, *, exclude_git: bool) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    ordinary_files = 0
    directories = 0
    symlinks: list[str] = []
    other_entries: list[str] = []
    total_bytes = 0

    if not root.is_dir() or root.is_symlink():
        return {
            "available": False,
            "ordinary_file_count": 0,
            "directory_count": 0,
            "symlink_count": 0,
            "other_entry_count": 0,
            "total_file_bytes": 0,
            "metadata_sha256": None,
        }

    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if exclude_git and relative.parts and relative.parts[0] == ".git":
            continue
        try:
            entry_stat = path.lstat()
        except OSError as error:
            records.append(
                {
                    "path": relative.as_posix(),
                    "type": "unreadable",
                    "error": type(error).__name__,
                }
            )
            other_entries.append(relative.as_posix())
            continue

        mode = stat.S_IMODE(entry_stat.st_mode)
        if stat.S_ISLNK(entry_stat.st_mode):
            kind = "symlink"
            symlinks.append(relative.as_posix())
            size = entry_stat.st_size
        elif stat.S_ISREG(entry_stat.st_mode):
            kind = "file"
            ordinary_files += 1
            total_bytes += entry_stat.st_size
            size = entry_stat.st_size
        elif stat.S_ISDIR(entry_stat.st_mode):
            kind = "directory"
            directories += 1
            size = 0
        else:
            kind = "other"
            other_entries.append(relative.as_posix())
            size = entry_stat.st_size

        records.append(
            {
                "path": relative.as_posix(),
                "type": kind,
                "mode": mode,
                "size": size,
                "mtime_ns": entry_stat.st_mtime_ns,
            }
        )

    encoded = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "available": True,
        "ordinary_file_count": ordinary_files,
        "directory_count": directories,
        "symlink_count": len(symlinks),
        "other_entry_count": len(other_entries),
        "total_file_bytes": total_bytes,
        "metadata_sha256": hashlib.sha256(encoded).hexdigest(),
        "symlink_paths": symlinks[:20],
        "other_entry_paths": other_entries[:20],
    }


def _run_status(command: list[str], output_path: Path, console_path: Path) -> int:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    console_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode in ALLOWED_STATUS_CODES and not output_path.is_file():
        return 2
    return completed.returncode


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _format_fraction(value: dict[str, Any], numerator: str, denominator: str) -> str:
    return f"{value.get(numerator)}/{value.get(denominator)}"


def _build_markdown(summary: dict[str, Any]) -> str:
    source = summary.get("source_panel") or {}
    readiness = summary.get("preacquisition_readiness") or {}
    collection_gate = readiness.get("collection_gate") or {}
    evidence = summary.get("confirmatory_evidence") or {}
    next_source = source.get("next_execution")
    if isinstance(next_source, dict):
        next_source = next_source.get("execution_id") or next_source.get("id") or next_source

    lines = [
        "# Causal4D open-issue acquisition audit",
        "",
        f"- Workflow run: `{summary.get('workflow_run_id')}`",
        f"- Audit source: `{summary.get('audit_source_sha')}`",
        f"- Runner: `{summary.get('runner_name')}`",
        f"- Commands executed: `{summary.get('commands_executed')}`",
        f"- Filesystem unchanged: `{summary.get('filesystem_unchanged')}`",
        f"- Present evidence valid: `{summary.get('present_evidence_valid')}`",
        f"- Source-panel validated: `{_format_fraction(source, 'validated_executions', 'specified_executions')}`",
        f"- Next source execution: `{next_source}`",
        f"- Pre-acquisition ready: `{readiness.get('ready')}`",
        (
            "- First confirmatory execution allowed: "
            f"`{collection_gate.get('first_confirmatory_execution_allowed')}`"
        ),
        (
            "- Confirmatory acquired/validated: "
            f"`{evidence.get('acquired_executions')}/"
            f"{evidence.get('validated_executions')}`"
        ),
        f"- Confirmatory claim ready: `{evidence.get('claim_ready')}`",
        "",
        "This audit is read-only. It does not scaffold, seal, publish, repair, "
        "or count any physical source-panel or confirmatory execution.",
        "",
    ]
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-checkout", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    frozen_checkout = args.frozen_checkout.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    environment = {
        "frozen_checkout": _path_info(frozen_checkout),
        "dataset_root": _path_info(dataset_root),
    }
    paths_available = all(
        item["exists"] and item["is_directory"] and not item["is_symlink"]
        for item in environment.values()
    )
    protocol_path = (
        frozen_checkout / "configs" / "causal4d" / "sloth_multi_action_v1.json"
    )
    protocol_available = (
        protocol_path.is_file()
        and not protocol_path.is_symlink()
        and protocol_path.resolve().is_relative_to(frozen_checkout)
    )

    before = {
        "frozen_checkout": _tree_snapshot(frozen_checkout, exclude_git=True),
        "dataset_root": _tree_snapshot(dataset_root, exclude_git=False),
    }
    exit_codes: dict[str, int | None] = {
        "source_panel_status": None,
        "preacquisition_readiness": None,
        "confirmatory_evidence_status": None,
    }
    source_panel: dict[str, Any] | None = None
    readiness: dict[str, Any] | None = None
    evidence: dict[str, Any] | None = None
    commands_executed = False

    with tempfile.TemporaryDirectory(prefix="causal4d-open-issues-audit-") as raw:
        raw_root = Path(raw)
        source_path = raw_root / "source-panel-status.json"
        readiness_path = raw_root / "preacquisition-readiness.json"
        evidence_path = raw_root / "confirmatory-evidence-status.json"

        if paths_available and protocol_available:
            commands_executed = True
            exit_codes["source_panel_status"] = _run_status(
                [
                    "causal4d",
                    "protocol",
                    "readiness",
                    "source-panel-status",
                    str(frozen_checkout),
                    str(dataset_root),
                    "--verify-file-hashes",
                    "--require-complete",
                    "--output-json",
                    str(source_path),
                ],
                source_path,
                raw_root / "source-panel-console.txt",
            )
            exit_codes["preacquisition_readiness"] = _run_status(
                [
                    "causal4d",
                    "protocol",
                    "readiness",
                    "status",
                    str(frozen_checkout),
                    str(dataset_root),
                    "--verify-file-hashes",
                    "--require-ready",
                    "--output-json",
                    str(readiness_path),
                ],
                readiness_path,
                raw_root / "preacquisition-console.txt",
            )
            exit_codes["confirmatory_evidence_status"] = _run_status(
                [
                    "causal4d",
                    "protocol",
                    "real",
                    "status",
                    str(protocol_path),
                    str(dataset_root),
                    "--verify-file-hashes",
                    "--require-complete",
                    "--output-json",
                    str(evidence_path),
                ],
                evidence_path,
                raw_root / "confirmatory-console.txt",
            )
            source_panel = _load_json(source_path)
            readiness = _load_json(readiness_path)
            evidence = _load_json(evidence_path)

    after = {
        "frozen_checkout": _tree_snapshot(frozen_checkout, exclude_git=True),
        "dataset_root": _tree_snapshot(dataset_root, exclude_git=False),
    }
    filesystem_unchanged = before == after
    present_evidence_valid = (
        commands_executed
        and all(code in ALLOWED_STATUS_CODES for code in exit_codes.values())
        and source_panel is not None
        and readiness is not None
        and evidence is not None
    )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Causal4DOpenIssuesAcquisitionAuditV1",
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
        "audit_source_sha": os.environ.get("AUDIT_SOURCE_SHA"),
        "environment": environment,
        "protocol_available": protocol_available,
        "commands_executed": commands_executed,
        "exit_codes": exit_codes,
        "filesystem_before": before,
        "filesystem_after": after,
        "filesystem_unchanged": filesystem_unchanged,
        "present_evidence_valid": present_evidence_valid,
        "source_panel": _project(
            source_panel,
            (
                "valid",
                "complete",
                "specified_executions",
                "manifest_executions",
                "validated_executions",
                "completed_execution_ids",
                "missing_execution_ids",
                "invalid_execution_ids",
                "registered_prefix_valid",
                "next_execution",
                "blockers",
                "evidence_sha256",
                "status_sha256",
            ),
        ),
        "preacquisition_readiness": _project(
            readiness,
            (
                "valid",
                "ready",
                "passed",
                "collection_gate",
                "confirmatory_collection",
                "missing_prerequisites",
                "malformed_prerequisites",
                "missing_or_template_gates",
                "malformed_gates",
                "chronology_blockers",
                "blockers",
                "evidence_sha256",
                "status_sha256",
            ),
        ),
        "confirmatory_evidence": _project(
            evidence,
            (
                "specified_executions",
                "manifest_executions",
                "acquired_executions",
                "validated_executions",
                "accounting_complete",
                "complete",
                "claim_ready",
                "next_pending_execution",
                "missing_execution_ids",
                "incomplete_execution_ids",
                "invalid_execution_ids",
                "blockers",
                "status_sha256",
            ),
        ),
        "target_outcomes_used": False,
        "claim_boundary": (
            "This read-only audit does not scaffold, seal, publish, repair, or "
            "count any physical source-panel or confirmatory execution. Workflow "
            "status artifacts are not physical evidence."
        ),
    }
    _write_json(output_root / "audit-summary.json", summary)
    (output_root / "audit-summary.md").write_text(
        _build_markdown(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
