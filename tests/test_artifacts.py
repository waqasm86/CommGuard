from __future__ import annotations

import io
import json
import tarfile

import pytest

from commguard.artifacts import ArtifactStore, restore_archive, sha256_file
from commguard.exceptions import ArtifactExistsError, ValidationError
from commguard.schemas import SCHEMA_VERSION


def artifact() -> dict[str, str]:
    return {
        "artifact_kind": "experiment_summary",
        "schema_version": SCHEMA_VERSION,
        "summary_type": "test",
    }


def test_create_only_json_and_hash(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    store.initialize()
    path = store.write_json("results/value.json", artifact())
    assert json.loads(path.read_text())["summary_type"] == "test"
    assert len(store.sha256("results/value.json")) == 64
    with pytest.raises(ArtifactExistsError):
        store.write_json("results/value.json", artifact())


def test_path_escape_is_rejected(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValidationError, match="escapes root"):
        store.resolve("../outside.json")


def test_export_contains_regular_artifacts(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    store.initialize()
    store.write_json("results/value.json", artifact())
    output = store.export(tmp_path / "bundle.tar.gz")
    with tarfile.open(output, "r:gz") as archive:
        assert "results/value.json" in archive.getnames()


def test_restore_archive_is_hash_checked_create_only_and_safe(tmp_path) -> None:
    source = ArtifactStore(tmp_path / "source")
    source.initialize()
    source.write_json("results/value.json", artifact())
    archive = source.export(tmp_path / "bundle.tar.gz")
    digest = sha256_file(archive)

    restored = restore_archive(archive, tmp_path / "restored", expected_sha256=digest)

    assert json.loads((restored / "results/value.json").read_text()) == artifact()
    with pytest.raises(ArtifactExistsError, match="existing destination"):
        restore_archive(archive, restored, expected_sha256=digest)
    with pytest.raises(ValidationError, match="SHA-256 mismatch"):
        restore_archive(archive, tmp_path / "wrong-hash", expected_sha256="0" * 64)
    assert not (tmp_path / "wrong-hash").exists()


@pytest.mark.parametrize("member_name", ["../escape", "/absolute"])
def test_restore_archive_rejects_path_traversal(tmp_path, member_name) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo(member_name)
        member.size = 1
        bundle.addfile(member, io.BytesIO(b"x"))

    with pytest.raises(ValidationError, match="unsafe"):
        restore_archive(
            archive,
            tmp_path / "restored",
            expected_sha256=sha256_file(archive),
        )
    assert not (tmp_path / "restored").exists()


def test_restore_archive_rejects_links(tmp_path) -> None:
    archive = tmp_path / "link.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("link")
        member.type = tarfile.SYMTYPE
        member.linkname = "target"
        bundle.addfile(member)

    with pytest.raises(ValidationError, match="links are prohibited"):
        restore_archive(
            archive,
            tmp_path / "restored",
            expected_sha256=sha256_file(archive),
        )
