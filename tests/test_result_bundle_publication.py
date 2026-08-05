from __future__ import annotations

import json
from pathlib import Path

import pytest

import causal4d.result_bundle_publication as publication
from causal4d.result_bundle_publication import publish_result_bundle
from causal4d.result_bundle_verification import verify_embedded_result_bundle


def test_publish_result_bundle_exposes_only_verified_complete_directory(
    tmp_path: Path,
) -> None:
    target = tmp_path / "result"

    def writer(staging: Path) -> None:
        assert not target.exists()
        (staging / "metrics.json").write_text(
            '{"rmse": 0.012}\n', encoding="utf-8"
        )
        (staging / "predictions.bin").write_bytes(b"predictions")

    result = publish_result_bundle(
        target,
        benchmark="deform360-prefix-v1",
        writer=writer,
    )
    assert result == verify_embedded_result_bundle(target)
    assert result["bundle_name"] == "result"
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["benchmark"] == "deform360-prefix-v1"
    assert set(manifest["artifacts"]) == {"metrics.json", "predictions.bin"}
    assert not list(tmp_path.glob(".result.*.incomplete"))
    assert not (tmp_path / ".result.publish.lock").exists()


def test_writer_failure_never_exposes_partial_target(tmp_path: Path) -> None:
    target = tmp_path / "result"

    def writer(staging: Path) -> None:
        (staging / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        publish_result_bundle(target, benchmark="failure-test", writer=writer)
    assert not target.exists()
    assert not list(tmp_path.glob(".result.*.incomplete"))
    assert not (tmp_path / ".result.publish.lock").exists()


def test_existing_bundle_is_never_replaced(tmp_path: Path) -> None:
    target = tmp_path / "result"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        publish_result_bundle(
            target,
            benchmark="no-overwrite",
            writer=lambda staging: (staging / "new.txt").write_text(
                "new", encoding="utf-8"
            ),
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_writer_cannot_supply_manifest_or_nested_entries(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not create manifest"):
        publish_result_bundle(
            tmp_path / "manifest-result",
            benchmark="reserved-name",
            writer=lambda staging: (staging / "manifest.json").write_text(
                "{}", encoding="utf-8"
            ),
        )

    def nested_writer(staging: Path) -> None:
        (staging / "nested").mkdir()
        (staging / "nested" / "value.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="flat ordinary files"):
        publish_result_bundle(
            tmp_path / "nested-result",
            benchmark="flat-only",
            writer=nested_writer,
        )


def test_existing_publication_lock_fails_closed(tmp_path: Path) -> None:
    lock = tmp_path / ".result.publish.lock"
    lock.write_text("other publisher\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already locked"):
        publish_result_bundle(
            tmp_path / "result",
            benchmark="locked",
            writer=lambda staging: None,
        )
    assert lock.read_text(encoding="utf-8") == "other publisher\n"


def test_rename_failure_cleans_staging_and_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_rename(source, destination) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(publication.os, "rename", fail_rename)
    with pytest.raises(OSError, match="injected rename failure"):
        publish_result_bundle(
            tmp_path / "result",
            benchmark="rename-failure",
            writer=lambda staging: (staging / "value.txt").write_text(
                "value", encoding="utf-8"
            ),
        )
    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".result.*.incomplete"))
    assert not (tmp_path / ".result.publish.lock").exists()


def test_symlink_artifact_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("source", encoding="utf-8")

    def writer(staging: Path) -> None:
        try:
            (staging / "linked.txt").symlink_to(source)
        except OSError:
            pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="flat ordinary files"):
        publish_result_bundle(
            tmp_path / "result",
            benchmark="symlink-rejection",
            writer=writer,
        )
